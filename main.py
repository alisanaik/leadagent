import asyncio
import json
import time
from typing import List

from scraper import scrape_domain
from cleaner import clean_html
from router import route_extraction
from cache import get_cached_result, save_result
from schemas import CompanyIntelligence

# Test domains required by the assignment
TEST_DOMAINS = ["postman.com", "supabase.com", "vapi.ai"]


async def process_domain(domain: str, use_cache: bool = True) -> dict:
    """
    Process a single domain end-to-end:
    Cache check -> Scrape -> Clean -> Extract -> Cache save.
    Returns a dict suitable for JSON serialization.
    """
    start_time = time.time()

    # 1. Cache check
    if use_cache:
        cached = get_cached_result(domain)
        if cached:
            elapsed = time.time() - start_time
            print(f"[Main] Cache HIT for {domain} ({elapsed:.2f}s)")
            return cached

    print(f"\n[Main] Processing {domain} (cache miss)...")

    # 2. Scrape homepage + prioritized subpages
    scraped = await scrape_domain(domain)

    if not scraped["homepage"]:
        fallback = CompanyIntelligence(
            company_overview="Extraction failed - could not fetch homepage.",
            target_audience="Unknown",
            contact_emails=[],
            leadership=[],
            confidence_score=0.0,
            hallucination_score=1.0,
            source_url=f"https://{domain}"
        ).model_dump()
        save_result(domain, fallback)
        return fallback

    # 3. Combine homepage + all subpages into a single cleaned markdown blob
    combined_text = clean_html(scraped["homepage"])
    for html in scraped["subpages"].values():
        combined_text += "\n\n" + clean_html(html)

    # 4. Extract via the router (cascade + compression + email verification)
    try:
        intelligence, usage = route_extraction(
            clean_text=combined_text,
            source_url=f"https://{domain}",
            raw_html=scraped["homepage"]
        )
    except Exception as e:
        # Last-resort catch — router should handle this, but we backstop here.
        print(f"[Main] Router raised exception for {domain}: {e}")
        intelligence = CompanyIntelligence(
            company_overview="Extraction failed after retries.",
            target_audience="Unknown",
            contact_emails=[],
            leadership=[],
            confidence_score=0.0,
            hallucination_score=1.0,
            source_url=f"https://{domain}"
        )
        usage = {
            "stages": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "groq_cost_usd": 0.0,
            "openai_reference_cost_usd": 0.0,
        }

    # 5. Serialize, attach usage, cache
    result_dict = intelligence.model_dump()
    result_dict["_usage"] = usage
    save_result(domain, result_dict)

    elapsed = time.time() - start_time
    print(
        f"[Main] Completed {domain} in {elapsed:.2f}s "
        f"({usage['total_tokens']} tokens, "
        f"${usage['openai_reference_cost_usd']:.6f} equiv.)"
    )
    return result_dict


async def process_all(domains: List[str], use_cache: bool = True) -> List[dict]:
    """
    Process all domains SEQUENTIALLY to avoid Groq's per-minute rate limits.
    Concurrent processing hits the 8K tokens/minute limit when two large
    domains share the same minute window. Sequential is slower but reliable.
    """
    results = []
    for i, domain in enumerate(domains, 1):
        print(f"\n[Main] === Domain {i}/{len(domains)}: {domain} ===")
        result = await process_domain(domain, use_cache=use_cache)
        results.append(result)
        # Cooldown between domains to let the rate-limit window reset
        if i < len(domains):
            print(f"[Main] Cooling down 10s before next domain...")
            await asyncio.sleep(10)
    return results


def main():
    """Entry point: run the pipeline on the 3 test domains and save output.json."""
    print(f"[Main] Starting lead enrichment pipeline for {len(TEST_DOMAINS)} domains")
    print(f"[Main] Domains: {TEST_DOMAINS}\n")

    overall_start = time.time()
    results = asyncio.run(process_all(TEST_DOMAINS, use_cache=True))
    overall_elapsed = time.time() - overall_start

    print(f"\n[Main] All domains processed in {overall_elapsed:.2f}s")

    # Write output.json
    output_path = "output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Main] Results written to {output_path}")


if __name__ == "__main__":
    main()
