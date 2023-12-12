# stdlib
import sys

# tls_crawler relative
from . import logger  # noqa: F401
from .crawl import (  # noqa: F401
    load_page,
    load_page_in_chrome,
    load_page_in_firefox,
    load_pages_in_chrome,
)

logger.add(sink=sys.stderr, level="DEBUG")
