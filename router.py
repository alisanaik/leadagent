from extractor import extract_intelligence
from schemas import CompanyIntelligence
from compressor import compress_text
from cleaner import extract_emails_regex

# Groq model lineup (post Aug-2026 deprecation)
FAST_MODEL = "openai/gpt-oss-20b"
SMART_MODEL = "openai/gpt-oss-120b"

# If fast model's confidence is below this, escalate to smart model
CONFIDENCE_THRESHOLD = 0.75

# If hallucination score is above this, escalate regardless of confidence
HALLUCINATION_CEILING = 0.5

# Query for semantic compression — must include leadership keywords so
# MiniLM retrieves the bio chunks from press/about pages.
EXTRACTION_QUERY = (
    "founders CEO CTO co-founder leadership team executive bios "
    "about the company who started it who leads it "
    "contact email address support sales press company overview "
    "target audience ideal customer profile"
)


def _verify_emails(result: CompanyIntelligence, verified_emails: list[str], silent: bool = False) -> CompanyIntelligence:
    """
    Cross-checks LLM-extracted emails against the regex-verified list.
    - Removes hallucinated emails
    - If the LLM returned zero emails but regex found some, populate from the verified list
    - Bumps hallucination_score for any removed email
    """
    verified_lower = {e.lower() for e in verified_emails}
    clean_emails = []
    hallucinated_count = 0

    for email in result.contact_emails:
        if email.lower() in verified_lower:
            clean_emails.append(email)
        else:
            hallucinated_count += 1

    if not clean_emails and verified_emails:
        clean_emails = verified_emails
        if not silent:
            print(f"[Router] LLM returned no emails; using {len(verified_emails)} regex-verified email(s).")
    # Always merge in all verified emails (belt and suspenders)
    for e in verified_emails:
        if e not in clean_emails:
            clean_emails.append(e)

    result.contact_emails = clean_emails

    if hallucinated_count > 0 and not silent:
        result.hallucination_score = min(1.0, result.hallucination_score + 0.3 * hallucinated_count)
        print(f"[Router] Removed {hallucinated_count} hallucinated email(s). "
              f"hallucination_score now {result.hallucination_score:.2f}.")

    return result


def _finalize_usage(usage: dict) -> dict:
    """
    Adds an estimated USD cost to the usage dict.
    Groq is free, but we compute the equivalent OpenAI gpt-4o-mini cost
    so the metric is meaningful for comparison.
    Reference: gpt-4o-mini = $0.15/1M input, $0.60/1M output tokens.
    """
    prompt_cost = (usage["total_prompt_tokens"] / 1_000_000) * 0.15
    completion_cost = (usage["total_completion_tokens"] / 1_000_000) * 0.60

    usage["groq_cost_usd"] = 0.0  # free tier
    usage["openai_reference_cost_usd"] = round(prompt_cost + completion_cost, 6)
    return usage


def route_extraction(clean_text: str, source_url: str, raw_html: str = "") -> tuple[CompanyIntelligence, dict]:
    """
    Two-stage cascade with compression and email verification.
    Returns (CompanyIntelligence, usage_summary).
    """
    total_usage = {
        "stages": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }

    # 1. Deterministic email extraction (free, cannot hallucinate)
    verified_emails = extract_emails_regex(clean_text, raw_html=raw_html)
    print(f"[Router] Regex found {len(verified_emails)} verified email(s): {verified_emails}")

    # 2. Semantic compression — smaller top_k to stay under Groq's 8K TPM limit
    compressed_text = compress_text(clean_text, EXTRACTION_QUERY, top_k=10)

    # 3. Stage 1: Fast model on compressed text
    print(f"[Router] Stage 1: Trying {FAST_MODEL}...")
    try:
        result, usage = extract_intelligence(
            clean_text=compressed_text,
            source_url=source_url,
            model_name=FAST_MODEL,
            verified_emails=verified_emails
        )
        total_usage["stages"].append(usage)
        total_usage["total_prompt_tokens"] += usage["prompt_tokens"]
        total_usage["total_completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

        result = _verify_emails(result, verified_emails)
    except Exception as e:
        print(f"[Router] Fast model failed: {e}. Escalating.")
        result = None

    # 4. Accept fast result only if it's both confident AND clean
    if (
        result
        and result.confidence_score >= CONFIDENCE_THRESHOLD
        and result.hallucination_score < HALLUCINATION_CEILING
    ):
        print(
            f"[Router] Accepting fast result "
            f"(conf={result.confidence_score}, halluc={result.hallucination_score})."
        )
        return result, _finalize_usage(total_usage)

    # 5. Stage 2: Smart model with LIGHT compression (more context, still bounded)
    print(f"[Router] Stage 2: Escalating to {SMART_MODEL} with light compression...")
    try:
        light_compressed = compress_text(clean_text, EXTRACTION_QUERY, top_k=18)
        result, usage = extract_intelligence(
            clean_text=light_compressed,
            source_url=source_url,
            model_name=SMART_MODEL,
            verified_emails=verified_emails
        )
        total_usage["stages"].append(usage)
        total_usage["total_prompt_tokens"] += usage["prompt_tokens"]
        total_usage["total_completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]

        result = _verify_emails(result, verified_emails)
        print(
            f"[Router] Accepted smart result "
            f"(conf={result.confidence_score}, halluc={result.hallucination_score})."
        )
        return result, _finalize_usage(total_usage)
    except Exception as e:
        print(f"[Router] Both models failed: {e}")

        # 6. Graceful failure — return a valid Pydantic object, never crash
        fallback = CompanyIntelligence(
            company_overview="Extraction failed.",
            target_audience="Unknown",
            contact_emails=verified_emails,
            leadership=[],
            confidence_score=0.0,
            hallucination_score=1.0,
            source_url=source_url
        )
        return fallback, _finalize_usage(total_usage)