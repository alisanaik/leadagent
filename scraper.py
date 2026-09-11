from playwright.async_api import async_playwright

SUBPAGE_PRIORITY = {
    "about": 5,
    "press": 5,
    "founder": 5,
    "leadership": 5,
    "team": 4,
    "company": 3,
    "media": 3,
    "contact": 2,
    "pricing": 1,
    "careers": 1,
}


async def fetch_page_content(page, url: str) -> str:
    """
    Fetches the HTML content of a single page.
    Handles timeouts and HTTP errors gracefully so one bad page
    does not crash the entire scraping run.
    """
    try:
        response = await page.goto(url, timeout=30000, wait_until="domcontentloaded")

        if response and response.status >= 400:
            print(f"Warning: {url} returned status {response.status}")
            return ""

        await page.wait_for_timeout(1500)
        return await page.content()

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""


async def scrape_domain(domain: str) -> dict:
    """
    Scrapes a domain's homepage plus the most relevant subpages
    (about, press, leadership, team, contact, pricing, etc.).
    Returns a dictionary with the domain, homepage HTML, and subpage HTML.
    """
    base_url = f"https://{domain}"
    scraped_data = {"domain": domain, "homepage": "", "subpages": {}}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        print(f"Scraping homepage: {base_url}")
        scraped_data["homepage"] = await fetch_page_content(page, base_url)

        if not scraped_data["homepage"]:
            await browser.close()
            return scraped_data

        all_links = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a')).map(a => a.href);
        }''')

        scored_subpages = {} 
        for link in all_links:
            if domain not in link:
                continue

            link_lower = link.lower()
            score = sum(weight for kw, weight in SUBPAGE_PRIORITY.items() if kw in link_lower)

            if score > 0:
                clean_link = link.split('#')[0]
                scored_subpages[clean_link] = max(scored_subpages.get(clean_link, 0), score)

        subpage_urls = sorted(
            scored_subpages.keys(),
            key=lambda u: scored_subpages[u],
            reverse=True
        )[:5]

        print(f"Selected subpages (by priority): {subpage_urls}")

        for sub_url in subpage_urls:
            print(f"Scraping subpage: {sub_url}")
            html_content = await fetch_page_content(page, sub_url)
            if html_content:
                scraped_data["subpages"][sub_url] = html_content

        await browser.close()

    return scraped_data