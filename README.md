# Leadagent

An autonomous lead-enrichment agent that scrapes company websites and extracts structured intelligence using LLMs.

Given a list of company domains, Leadagent crawls their homepage and relevant subpages, cleans the HTML, and uses a two-stage LLM cascade to extract company overview, target audience, contact emails, leadership, and confidence scores.

---

## Architecture

```text
Domain Input
    |
    v
+-----------------------------------------------+
| 1. Scraper (Playwright)                       |
|    - Homepage + prioritized subpages          |
|    - Handles timeouts and 4xx/5xx errors      |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| 2. Cleaner (BeautifulSoup + Markdownify)      |
|    - Removes scripts, styles, SVG, nav        |
|    - Converts HTML to Markdown                |
|    - Extracts emails using regex              |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| 3. Compressor (MiniLM Embeddings)             |
|    - Chunks the cleaned text                  |
|    - Semantic similarity ranking              |
|    - Forces leadership-related chunks         |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| 4. Router (Two-Stage Model Cascade)           |
|    Stage 1: gpt-oss-20b                       |
|    Stage 2: gpt-oss-120b                      |
|    - Escalates on low confidence              |
|    - Escalates on high hallucination          |
|    - Verifies emails against regex whitelist  |
+-----------------------------------------------+
    |
    v
+-----------------------------------------------+
| 5. Cache (SQLite) + Output (output.json)      |
+-----------------------------------------------+
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A free Groq API key ([console.groq.com/keys](https://console.groq.com/keys))

### Installation

```bash
git clone https://github.com/alisanaik/leadagent.git
cd Leadagent

# Create and activate a virtual environment
python -m venv aenv

# Windows
aenv\Scripts\activate

# Mac/Linux
source aenv/bin/activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

### Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=gsk_your_key_here
```

The `.env` file is gitignored and will not be committed.

---

## Running

```bash
python main.py
```

The program processes the test domains:

- `postman.com`
- `supabase.com`
- `vapi.ai`

Results are saved to `output.json`. Cached results are stored in `cache.db`.

Delete `cache.db` to force a fresh run:

```bash
rm cache.db     # Mac/Linux
del cache.db    # Windows
```

---

## Project Structure

```text
Leadagent/
├── .env.example               # Template for environment variables
├── .gitignore
├── README.md
├── requirements.txt
├── schemas.py                 # Pydantic models for LLM output
├── scraper.py                 # Playwright scraping with subpage priority
├── cleaner.py                 # HTML -> Markdown + regex email extraction
├── compressor.py              # MiniLM semantic compression
├── extractor.py               # Instructor + Pydantic LLM extraction
├── router.py                  # Two-stage model cascade + verification
├── cache.py                   # SQLite caching layer
├── main.py                    # Orchestrator
└── output.json                # Sample output for the three test domains
```

---

## Technical Highlights

### 1. Token Budget Optimization with MiniLM

Raw HTML from a single company can exceed 300,000 characters. Postman's homepage plus five subpages produced **45,287 characters of cleaned text** — roughly 11,300 tokens, well over Groq's free-tier limit of 8,000 tokens per minute.

`compressor.py` splits the cleaned text into paragraph-level chunks, embeds each one using the `all-MiniLM-L6-v2` sentence transformer, and ranks chunks by cosine similarity against an extraction query. Only the top-k chunks are kept.

Result: **45,287 characters compressed to ~5,500 characters** — a 90% reduction with no loss of relevant content.

A safety net additionally force-includes any chunk containing both a strong leadership title (`CEO`, `CTO`, `founder`) and a plausible person name. This guarantees founder bios are never dropped, even if they rank low semantically. Forced chunks are capped at 15 to prevent defeating compression entirely.

### 2. Deterministic Email Extraction with Regex

LLMs hallucinate email addresses. In early testing, the model returned `info@postman.com` for a page that never contained it.

The pipeline runs a regex pass (`cleaner.py`) over **both the cleaned text and the raw HTML** before the LLM sees any content:

- Standard email regex for visible text
- Mailto-link regex for `<a href="mailto:...">` attributes that the cleaner strips out

These verified emails are injected into the prompt as a whitelist. After extraction, the LLM's emails are cross-checked against this list. Any hallucinated address is removed and the `hallucination_score` is bumped by 0.3 per offense.

Across the three test domains, regex found **10 verified emails** that the LLM would have otherwise dropped or invented.

### 3. Hallucination Score vs. Confidence Score

These two scores measure different properties:

| Score | Measures | Range | Source |
|-------|----------|-------|--------|
| `confidence_score` | Completeness of the extraction | 0.0 - 1.0 | LLM self-reported |
| `hallucination_score` | Likelihood of fabricated data | 0.0 - 1.0 | LLM self-reported + regex verification |

The LLM is prompted to penalize itself for inferred data. However, LLMs cannot reliably detect their own hallucinations — a model that invents a name will confidently report `hallucination_score = 0.0`. This is why the deterministic regex layer exists as an independent second check.

### 4. Two-Stage Model Cascade

Rather than always calling the largest model, `router.py` runs a cascade:

- **Stage 1** — `openai/gpt-oss-20b` on the compressed text. Fast and cheap.
- **Stage 2** — `openai/gpt-oss-120b` on lightly compressed text. Slower, higher accuracy.

Stage 2 only fires when Stage 1 fails to meet quality thresholds (confidence below 0.75, or hallucination above 0.5). Results from the test run:

| Domain | Stage Used | Confidence | Tokens |
|--------|-----------|------------|--------|
| postman.com | Stage 1 | 0.90 | 3,446 |
| vapi.ai | Stage 1 | 0.80 | 3,800 |
| supabase.com | Stage 2 (escalated) | 0.85 | 9,151 |

This gives smart-model accuracy on hard cases at fast-model cost on easy ones.

### 5. Subpage Priority Scoring

A homepage contains dozens of links. Scraping all of them wastes time and tokens; scraping the first five in order misses the important ones.

`scraper.py` scores each same-domain URL by keyword weight:

| Keyword | Weight |
|---------|--------|
| `about`, `press`, `founder`, `leadership` | 5 |
| `team`, `media` | 3-4 |
| `company` | 3 |
| `contact` | 2 |
| `pricing`, `careers` | 1 |

The top five scoring URLs are scraped. This is how Postman's `/company/press-media/` page (score 8) was prioritized over `/pricing/` (score 1) — the press page is where the three founders' bios live.

### 6. Why Supabase Returned Zero Leadership

Supabase's output contains an empty `leadership` array with a `confidence_score` of 0.85. This is **correct behavior, not a failure**:

- Supabase does not publish a dedicated leadership or team page.
- The founders' names appear on `/company` only inside non-leadership contexts (investor mentions, press quotes).
- The LLM correctly returned an empty list rather than hallucinating entries from unrelated text.

This is the behavior the pipeline is designed for: **honest low output beats confident fabrication**. A system that invents a CEO is worse than one that admits it doesn't know.

### 7. Cost Tracking

Every result includes a `_usage` block with per-stage token counts and estimated cost:

```json
"_usage": {
  "stages": [
    {"model": "openai/gpt-oss-20b", "total_tokens": 3446}
  ],
  "total_tokens": 3446,
  "groq_cost_usd": 0.0,
  "openai_reference_cost_usd": 0.000865
}
```

Groq's free tier costs $0, so the equivalent OpenAI gpt-4o-mini cost is computed for reference ($0.15 per 1M input tokens, $0.60 per 1M output tokens). Total across all three test domains: **~16,400 tokens, ~$0.0037 equivalent**.

### 8. Error Handling & Resilience

The pipeline is designed to never crash, even when individual steps fail:

- **Playwright timeouts** — caught per-page; the scraper logs the URL and moves on.
- **HTTP 4xx/5xx responses** — detected and skipped.
- **Groq rate limits (429) and payload-too-large (413)** — automatically retried with exponential backoff. Maximum 3 attempts per call.
- **Pydantic validation failures** — instructor retries the LLM up to 2 times with the error message appended.
- **Both models failing** — the router returns a valid `CompanyIntelligence` object with `confidence_score=0.0` and `hallucination_score=1.0` rather than raising.
- **Sequential domain processing** — domains are processed one at a time with a 10-second cooldown to avoid Groq's per-minute rate limit.

### 9. SQLite Caching

Results are cached in `cache.db`, keyed by domain. On re-run, cached domains skip both scraping and LLM calls entirely. This is useful during development when iterating on prompts — only changed domains need reprocessing.

---

## Sample Output

Full output for all three domains is in `output.json`. Example for `postman.com`:

```json
{
  "company_overview": "Postman is an API platform that enables developers to build, test, and manage APIs. It provides a collaborative environment for API development and testing.",
  "target_audience": "Developers building APIs",
  "contact_emails": [
    "info@postman.com",
    "info-jp@postman.com",
    "help@postman.com",
    "accommodations@postman.com"
  ],
  "leadership": [
    {
      "name": "Abhinav Asthana",
      "role": "CEO/Co-Founder",
      "linkedin_url": "https://www.linkedin.com/in/abhinavasthana"
    },
    {
      "name": "Ankit Sobti",
      "role": "CTO/Co-Founder",
      "linkedin_url": "https://www.linkedin.com/in/ankit-sobti"
    },
    {
      "name": "Abhijit Kane",
      "role": "Product Architect/Co-Founder",
      "linkedin_url": "https://in.linkedin.com/in/abhijitkane"
    }
  ],
  "confidence_score": 0.9,
  "hallucination_score": 0.0,
  "source_url": "https://postman.com",
  "_usage": {
    "stages": [
      {
        "model": "openai/gpt-oss-20b",
        "prompt_tokens": 2672,
        "completion_tokens": 774,
        "total_tokens": 3446
      }
    ],
    "total_prompt_tokens": 2672,
    "total_completion_tokens": 774,
    "total_tokens": 3446,
    "groq_cost_usd": 0.0,
    "openai_reference_cost_usd": 0.000865
  }
}
```

---

## Design Decisions

### Why Playwright?

Modern websites render most of their content through JavaScript. A plain HTTP request returns an empty shell. Playwright runs a real headless Chromium instance, handles JavaScript, and provides native async support. It also has built-in auto-waiting, which avoids the stale-element errors common in Selenium.

### Why a Model Cascade?

Using the largest model for every request is slow and expensive. Using the smallest is inaccurate on hard cases. The cascade runs the fast model first, checks its confidence and hallucination scores, and only escalates when the result is not good enough.

### Why Semantic Compression?

Scraped pages routinely exceed LLM token limits. Truncating arbitrarily can cut off the exact section we need. MiniLM embeddings rank chunks by relevance to the extraction query, keeping the most useful content while dropping boilerplate.

### Why Regex for Email Verification?

LLMs cannot be trusted to extract email addresses reliably. Regex is deterministic and cannot hallucinate. Extracting emails with regex before the LLM runs, then verifying the LLM's output against that list, eliminates email hallucination entirely.

### Why Two Scores Instead of One?

Confidence measures completeness. Hallucination measures truthfulness. A result can be high-confidence but hallucinated, or low-confidence but truthful. Having both lets the router make better escalation decisions and lets the user know whether the data is safe to use.

---

## Known Limitations

- **JS-heavy sites** may render content after the scraper's timeout window. Increasing the wait time would help but slows the pipeline.
- **No external search** is integrated. The agent relies entirely on the target site's own content. Adding Tavily or SerpAPI would fill in missing LinkedIn URLs, such as the VP at Vapi.
- **Groq free-tier limits** cap requests at 8,000 tokens per minute. Sequential processing with cooldowns keeps the pipeline under this limit, at the cost of wall-clock time.
- **No test suite** was included in scope. Unit tests for the cleaner and router would be the next step.

---

## Author

Alisa Naik

[LinkedIn](https://www.linkedin.com/in/alisanaik/)
