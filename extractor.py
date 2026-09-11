import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIStatusError
import instructor
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from schemas import CompanyIntelligence

# Set up logger so Tenacity prints retry attempts to the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# JSON mode is more reliable than tool-calling mode on Groq
client = instructor.from_openai(client, mode=instructor.Mode.JSON)


# Retry policy: only retry on rate-limit (429) and payload-too-large (413) errors.
# Validation failures are handled separately by instructor's max_retries.
@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(prompt: str, model_name: str) -> tuple[CompanyIntelligence, object]:
    """
    Makes the LLM call. Wrapped in Tenacity so rate-limit errors
    (429) and payload-too-large errors (413) trigger automatic retries
    with exponential backoff.
    """
    response, raw = client.chat.completions.create_with_completion(
        model=model_name,
        response_model=CompanyIntelligence,
        messages=[
            {"role": "system", "content": "You extract structured data from web text. Be precise and conservative."},
            {"role": "user", "content": prompt}
        ],
        max_retries=2,       # instructor-level retries for validation errors
        temperature=0.0
    )
    return response, raw


def extract_intelligence(
    clean_text: str,
    source_url: str,
    model_name: str,
    verified_emails: list[str] = None
) -> tuple[CompanyIntelligence, dict]:
    """
    Sends cleaned text to the LLM and extracts structured intelligence.
    Returns (CompanyIntelligence, usage_dict).

    Rate-limit and payload errors are retried automatically by _call_llm.
    If all retries fail, the exception propagates to router.py, which
    handles the graceful fallback.
    """
    verified_emails = verified_emails or []
    verified_email_block = (
        "\n".join(f"- {e}" for e in verified_emails)
        if verified_emails else "NONE FOUND."
    )

    prompt = f"""
    You are an expert lead enrichment analyst. Extract the required information from the provided company website text.

    CRITICAL RULES:
    1. ONLY use information explicitly stated in the text OR directly implied by the product category.
    2. If information is missing, return an empty list [] or 'Unknown' instead of guessing.
    3. For hallucination_score: If you inferred or guessed ANY information, set this closer to 1.0.
    4. For confidence_score: Rate how complete the extracted data is (0.0 = terrible, 1.0 = perfect).
    5. STRICT EMAIL RULE: You may ONLY include emails from this verified list:

    VERIFIED EMAILS (found via regex in the source):
    {verified_email_block}

    COMPANY OVERVIEW RULES:
    - Describe the COMPANY ITSELF, not a specific product feature.
    - Bad example: "Postman's AI Engineer automates API testing."
    - Good example: "Postman is an API platform for building, testing, and managing APIs."

    TARGET AUDIENCE RULES:
    - Look for phrases like "for developers", "for engineers", "for teams", "built for".
    - If the product type clearly implies an audience, state it.
    - Only return "Unknown" if the product category is genuinely unclear.

    LEADERSHIP EXTRACTION RULES:
    - Look for sections labeled "Founders", "Leadership", "Team", "Management".
    - Each entry MUST have exactly these two fields: "name" and "role". Do NOT use "title".
    - LinkedIn URLs appear in the bio text (e.g., "Connect with X on LinkedIn →").
    - LinkedIn URLs may use regional subdomains: www.linkedin.com/in/x, in.linkedin.com/in/x,
      uk.linkedin.com/in/x, de.linkedin.com/in/x, etc. Treat all of them as valid.
    - If no leadership names are present in the text, return an EMPTY LIST. Do not guess.

    Website Text to Analyze:
    ---
    {clean_text}
    ---
    """

    response, raw = _call_llm(prompt, model_name)
    response.source_url = source_url

    # Extract token usage from the raw API response
    prompt_tokens = raw.usage.prompt_tokens if raw.usage else 0
    completion_tokens = raw.usage.completion_tokens if raw.usage else 0

    usage = {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return response, usage