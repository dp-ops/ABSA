import scrapy
from scrapy.http import Request
from scrapy.item import Item, Field
import pandas as pd
import os

class SkroutzSpider(scrapy.Spider):
    name = "skroutz"
    allowed_domains = ["skroutz.gr"]

    # Start URL for the spider to scrape
    base_urls = [
        "https://www.skroutz.gr/c/5066/refurbished-laptops.html?page=%d"
    ]

    custom_settings = {
        "BOT_NAME": "skroutz_reviews_scraping",
        "SPIDER_MODULES": ["amazon_reviews_scraping.spiders"],
        "NEWSPIDER_MODULE": "amazon_reviews_scraping.spiders",
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "CONCURRENT_REQUESTS_PER_IP": 1,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 10,
        "RETRY_HTTP_CODES": [429, 500, 502, 503, 504, 522, 524, 408],
        "COOKIES_ENABLED": False,
        "TELNETCONSOLE_ENABLED": False,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en",
        },
        "DOWNLOADER_MIDDLEWARES": {
            "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
            "scrapy_user_agents.middlewares.RandomUserAgentMiddleware": 400,
            "scrapy.downloadermiddlewares.retry.RetryMiddleware": 90,
        },
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 5,
        "AUTOTHROTTLE_MAX_DELAY": 200,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
        "AUTOTHROTTLE_DEBUG": False,
        "HTTPCACHE_ENABLED": True,
        "HTTPCACHE_EXPIRATION_SECS": 0,
        "HTTPCACHE_DIR": "httpcache",
        "HTTPCACHE_IGNORE_HTTP_CODES": [],
        "HTTPCACHE_STORAGE": "scrapy.extensions.httpcache.FilesystemCacheStorage",
    }

    def __init__(self, name = None, **kwargs):
        super(SkroutzSpider, self).__init__(name, **kwargs)
        self.items = []
        self.current_url_index = 0
        self.page_dict = {
            i : 1 for i in range(len(self.base_urls))
        } # Dictionary to keep track of the current page for each base URL

    def start_requests(self):
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
        }

        for url_index in range(len(self.base_urls)):
            yield Request(
                self.base_urls[url_index] % self.page_dict[url_index],
                callback=self.parse,
                headers=headers,
                meta={"url_index": url_index, "page": self.page_dict[url_index]},
                errback=self.handle_error,
            )
    
    def handle_error(self, failure):
        # Handle 301 redirects and other errors
        self.logger.error(f"Request failed: {failure.request}")
        url_index = failure.request.meta["url_index"]
        self.page_dict[url_index] += 1
        if self.page_dict[url_index] > 1:
            next_url_index = (url_index + 1) % len(self.base_urls)
            self.page_dict[next_url_index] = 1
            next_base_url = self.base_urls[next_url_index]

            yield Request(
                next_base_url % 1,
                callback=self.parse,
                headers=failure.request.headers,
                meta={"url_index": next_url_index, "page": 1},
                errback=self.handle_error,
            )

    def parse(self, response):
        url_index = response.meta["url_index"]
        page = response.meta["page"]

        new_urls = response.css(".js-sku-link::attr(href)").extract()
        if not new_urls:
            self.logger.info(f"No new URLs found on page {page} of base URL {url_index}.")
            next_url_index = (url_index + 1) % len(self.base_urls)
            self.page_dict[next_url_index] = 1
            if next_url_index != 0 or self.page_dict[next_url_index] ==1:
                yield Request(
                    self.base_urls[next_url_index] % 1,
                    callback=self.parse,
                    headers=response.request.headers,
                    meta={"url_index": next_url_index, "page": 1},
                    errback=self.handle_error,
                )
            return

        for url in new_urls:
            self.items.append({"link": url})

        self.page_dict[url_index] += 1
        yield Request(
            self.base_urls[url_index] % self.page_dict[url_index],
            callback=self.parse,
            headers=response.request.headers,
            meta={"url_index": url_index, "page": self.page_dict[url_index]},
            errback=self.handle_error,
        )

    def close(self, reason):
        # Convert items to DataFrames 
        df_new = pd.DataFrame(self.items)
        df_new.drop_duplicates(subset=None, inplace=True)

        # Ensure the "data" directory exists
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        file_path = os.path.join(data_dir, "links.csv")
        if os.path.exists(file_path):
            df_existing = pd.read_csv(file_path)
            # Append new data to the existing CSV file
            df_combined = (pd.concat([df_existing, df_new]).drop_duplicates().reset_index(drop=True))
            df_combined.to_csv(file_path, index=False)
        else:
            # Save the DataFrame to a CSV file
            df_new.to_csv(file_path, index=False)

# run: scrapy runspider skroutzscraper\skroutzscraper\spiders\skroutz_urls_spider.py -o output.csv