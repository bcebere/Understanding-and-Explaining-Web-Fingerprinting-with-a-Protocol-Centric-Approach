# stdlib
import sys

# tls_fingerprinting relative
# tls_crawler relative
from . import logger  # noqa: F401

logger.add(sink=sys.stderr, level="INFO")
