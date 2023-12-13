# stdlib
from typing import Dict

# third party
from datasets import load_dataset
import pandas as pd

wiki = load_dataset("wikipedia", "20220301.en", split="train")

urls: Dict[str, list] = {
    "url": [],
    "title": [],
}
for item in wiki:
    urls["url"].append(item["url"])
    urls["title"].append(item["title"])

urls_df = pd.DataFrame(urls)
urls_df.to_csv("wiki_dataset.csv", index=False)
