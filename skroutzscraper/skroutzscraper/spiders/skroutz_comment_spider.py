import pandas as pd
import scrapy 
from scrapy.http import Request
from scrapy.utils.project import get_project_settings

class SkroutzCommentSpider(scrapy.Spider):
    name = "skroutz_spidy"
    allowed_domains = ["skroutz.gr"]
    myBaseUrls = "https://www.skroutz.gr"
    start = []
    file_name = "data\links.csv"

    custom_settings = {
        "BOT_NAME": "skroutz_reviews_scraping",
        "SPIDER_MODULES": ["amazon_reviews_scraping.spiders"],
        "NEWSPIDER_MODULE": "amazon_reviews_scraping.spiders",
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 1,
        "DOWNLOAD_DELAY": 2,
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

    def __init__(self, *args, **kwargs):
        super(SkroutzCommentSpider, self).__init__(**kwargs)
        for key, value in self.custom_settings.items():
            self.logger.info(f"{key}: {value}")
        
        self.df = pd.read_csv(self.file_name, sep="\t or ,")
        self.df.drop_duplicates(subset=["link"], inplace=True)
        self.df = self.df["link"].tolist()
        for i in range(len(self.df)):
            self.start_urls.append(self.myBaseUrls + self.df[i])

    def start_requests(self):
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
        }

        for url in self.start_urls:
            yield Request(
                url,
                callback=self.parse,
                headers=headers,
            )

    def parse(self, response):
        topic = response.css("#nav")
        top = topic.css("h2")
        title = response.css("#sku-details")
        titlee = title.css("h1")
        data = response.css("#sku_reviews_list")
        
        # Get all review containers
        reviews = data.css(".review-content")
        
        for review in reviews:
            # Extract review data
            star_rating = review.css(".actual-rating::text").get() or ""
            comment_text = review.css(".review-body::text").getall()
            comment_paragraphs = review.css(".review-body p::text").getall()
            comment = "".join(comment_text + comment_paragraphs).strip()
            
            # Extract aggregated rating specific to this review
            agg_rating = review.css(".review-aggregated-data.cf")
            agg_data = None
            
            # Process aggregated rating data if available for this review
            if agg_rating:
                agg_data = {
                    "pros": [],
                    "so-so": [],
                    "cons": [],
                }
                
                # Extract aspects and their text
                pros = agg_rating.css("ul.icon.pros li::text").getall()
                if pros:
                    agg_data["pros"] = [pro.strip() for pro in pros]
                    
                # Extract average/neutral aspects (so-so)
                so_so = agg_rating.css("ul.icon.so-so li::text").getall()
                if so_so:
                    agg_data["so-so"] = [item.strip() for item in so_so]
                    
                # Extract negative aspects (cons)
                cons = agg_rating.css("ul.icon.cons li::text").getall()
                if cons:
                    agg_data["cons"] = [con.strip() for con in cons]
                
                # If no aspects were found in any category, set to None
                if not any(agg_data.values()):
                    agg_data = None
            
            yield {
                "stars": star_rating,
                "comment": comment,
                "topic": "".join(top.xpath(".//text()").extract()),
                "title": "".join(titlee.xpath(".//text()").extract()),
                "agg_rating": agg_data,
            }

# run: scrapy runspider skroutzscraper\skroutzscraper\spiders\skroutz_comment_spider.py -o dirtyreview.csv