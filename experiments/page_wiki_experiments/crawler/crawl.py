# stdlib
import argparse

# import random
from pathlib import Path
import random

# third party
import pandas as pd
from tls_crawler import load_page_in_chrome

# tls_crawler absolute
import tls_crawler.logger as log

country = "DE"

parser = argparse.ArgumentParser(description="Wiki crawler")
parser.add_argument("--offset", type=int, default=0, help="Executation index")
args = parser.parse_args()

offset = args.offset
log.add(f"logs/chrome_trace_{offset}.log", level="DEBUG")

remote_port = 4444 + 2 * offset
docker_name = f"selenium-chrome{offset}"
docker_iface = f"veth_chrome{offset}"

repeats = 5

dataset = pd.read_csv("wiki_dataset.csv").sample(5000)
if offset != 0:
    dataset = dataset.sample(frac=1)

with open("user_agents.txt") as f:
    user_agents = f.read().split("\n")

random.shuffle(user_agents)

country = "DE"

for idx, row in dataset.iterrows():
    for use_http_caching in [False, True]:
        page_name = row["url"]
        for user_agent in user_agents:
            if user_agent == "":
                continue
            print(page_name, user_agent)
            use_adblocking = False
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
                user_agent=user_agent,
            )
