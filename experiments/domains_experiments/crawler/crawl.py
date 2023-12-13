# stdlib
import argparse

# import random
from pathlib import Path
import random

# third party
from tls_crawler import load_page_in_chrome

# tls_crawler absolute
import tls_crawler.logger as log
from tranco import Tranco

country = "DE"

parser = argparse.ArgumentParser(description="Tranco crawler")
parser.add_argument("--offset", type=int, default=0, help="Executation index")
args = parser.parse_args()

offset = args.offset
log.add(f"logs/chrome_trace_{offset}.log", level="DEBUG")

remote_port = 4444 + 2 * offset
docker_name = f"selenium-chrome{offset}"
docker_iface = f"veth_chrome{offset}"

t = Tranco(cache=True, cache_dir=".tranco")
latest_list = t.list()
urls = latest_list.top(10000)

repeats = 10

if offset != 0:
    random.shuffle(urls)

for url in urls:
    page_name = f"https://{url}/"
    print("Collect", page_name, flush=True)
    for use_adblocking in [False]:
        for use_http_caching in [False]:
            page = load_page_in_chrome(
                page_name=page_name,
                docker_name=docker_name,
                docker_iface=docker_iface,
                use_adblocking=use_adblocking,
                use_tracking_blocking=use_adblocking,
                use_user_cookies=use_http_caching,
                use_http_caching=use_http_caching,
                use_dns_cache=True,
                use_images=True,
                repeats=repeats,
                remote_port=remote_port,
                startup_wait_sec=1,
                http_caching_repeats=2,
                workspace=Path(f"traces_{country}"),
                country=country,
            )
