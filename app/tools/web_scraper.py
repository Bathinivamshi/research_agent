
import httpx
import trafilatura

from urllib.parse import urlparse
from ipaddress import ip_address


class WebScraperService:

    def __init__(self):

        self.timeout = 20

        self.headers = {
            "User-Agent": "ResearchAgent/1.0"
        }

    def validate_url(self, url: str):

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        if not parsed.hostname:
            raise ValueError("Invalid URL")

        if parsed.username or parsed.password:
            raise ValueError("URLs containing credentials are not allowed")

        host = parsed.hostname.lower()

        if host in ("localhost", "localhost.localdomain"):
            raise ValueError("Local URLs are not allowed")

        try:
            address = ip_address(host)

        except ValueError:
            # Normal domain name, not an IP address
            return

        if not address.is_global:
            raise ValueError(
                "Private or non-public IP addresses are not allowed"
            )

    async def scrape(self, url: str) -> dict:

        self.validate_url(url)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=False
        ) as client:

            response = await client.get(url)

            response.raise_for_status()

            content_type = response.headers.get(
                "content-type", ""
            ).lower()

            if "text/html" not in content_type:
                raise ValueError(
                    "URL does not contain an HTML webpage"
                )

            html_content = response.text

        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            favor_recall=True
        )

        if not extracted_text:
            raise ValueError(
                "Could not extract readable webpage content"
            )

        return {
            "url": url,
            "content": extracted_text,
            "content_length": len(extracted_text)
        }


web_scraper_service = WebScraperService()
