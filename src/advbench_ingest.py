import urllib.request
import csv
from pathlib import Path

def get_advbench_prompts(count=None):
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "advbench_harmful_behaviors.csv"
    
    if not cache_file.exists():
        urllib.request.urlretrieve(url, cache_file)
        
    prompts = []
    with open(cache_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "goal" in row:
                prompts.append(row["goal"])
                
    if count is not None:
        return prompts[:count]
    return prompts
