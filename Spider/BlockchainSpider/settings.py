# Scrapy settings for BlockchainSpider project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html
from scrapy.utils.reactor import install_reactor

# 下载超时和重试设置
DOWNLOAD_TIMEOUT = 60
RETRY_ENABLED = True
RETRY_TIMES = 5
RETRY_HTTP_CODES = [429, 500, 502, 503, 504]
BOT_NAME = 'BlockchainSpider'

SPIDER_MODULES = ['BlockchainSpider.spiders']
NEWSPIDER_MODULE = 'BlockchainSpider.spiders'

# settings.py
# DOWNLOAD_TIMEOUT = 30  # 已在文件顶部统一设置
DOWNLOAD_DELAY = 1  # 减少请求延迟提升速度
CONCURRENT_REQUESTS = 4  # 增加并发请求数提升速度
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # 降低并发以减少API压力

# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = 'BlockchainSpider (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Configure maximum concurrent requests performed by Scrapy (default: 16)
# CONCURRENT_REQUESTS = 4

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
# DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 5
# CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en',
    'Accept-Encoding': 'gzip',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.8 Safari/537.36',
}

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
SPIDER_MIDDLEWARES = {
    # 'contrib.mots.middlewares.MoTSMiddleware': 500,
}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 550,
}
RETRY_ENABLED = True
RETRY_TIMES = 5  # 增加重试次数提高成功率
RETRY_BACKOFF = 1.5  # 启用指数退避策略
RETRY_BACKOFF_MAX = 10
RETRY_HTTP_CODES = [500, 502, 503, 504, 400, 403, 404, 408, 599]

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
# ITEM_PIPELINES = {
    # 'contrib.mots.pipelines.MoTSPipeline': 666,
    # 'contrib.rabbit.pipelines.RabbitMQPipeline': 666,
# }

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# 初始下载延迟
AUTOTHROTTLE_START_DELAY = 1
# 最大下载延迟
AUTOTHROTTLE_MAX_DELAY = 10
# 目标并发数
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
# 显示节流统计
AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600  # 缓存1小时
HTTPCACHE_DIR = './httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [400, 403, 404, 500, 502, 503, 504]
HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
HTTPCACHE_GZIP = True

# Setting the fingerprinting algorithm is used
# See https://docs.scrapy.org/en/latest/topics/request-response.html#request-fingerprinter-implementation
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Enable asyncio
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
install_reactor('twisted.internet.asyncioreactor.AsyncioSelectorReactor')

# Log configure
LOG_LEVEL = 'INFO'

# The response size (in bytes) that downloader will start to warn.
DOWNLOAD_WARNSIZE = 33554432 * 2

# APIKey configure — set via environment variables (see .env.example)
import os


def _split_env(name: str) -> list:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


APIKEYS_BUCKET = 'BlockchainSpider.utils.bucket.StaticAPIKeyBucket'
APIKEYS = {
    "eth": _split_env("ETHERSCAN_API_KEYS"),
    "bsc": _split_env("BSCSCAN_API_KEYS"),
    "polygon": _split_env("POLYGONSCAN_API_KEYS"),
    "heco": _split_env("HECOSCAN_API_KEYS"),
}

PROVIDERS = {
    "eth": _split_env("ETH_RPC_URLS"),
}
