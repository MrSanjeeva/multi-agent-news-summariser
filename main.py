# main.py

from datasets import load_dataset
import json
import os
from agents.factual_agent import FactualAgent

DATA_PATH = "data/sample_articles.json"


def load_and_prepare_data(n=5, save=True):
    """Load Multi-News articles and save a small sample locally."""
    print("🔄 Loading Multi-News dataset...")
    dataset = load_dataset("alexfabbri/multi_news",
                           split="test[:{}]".format(n), trust_remote_code=True)

    sample_data = []
    for item in dataset:
        articles = item["document"].split("|||||")
        sample_data.append({
            "articles": articles,
            "reference_summary": item["summary"]
        })

    if save:
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump(sample_data, f, indent=2)
        print(f"✅ Sample data saved to {DATA_PATH}")

    return sample_data


def run_pipeline():
    # Load data
    data = load_and_prepare_data(n=5)

    # Instantiate factual agent
    agent = FactualAgent()

    # Run each article through the factual agent
    for i, sample in enumerate(data):
        print(f"\n🗞️  Processing Article {i+1}")
        all_text = " ".join(sample["articles"])
        facts = agent.extract_facts(all_text)
        print("📌 Extracted Key Facts:\n", facts)
        print("📋 Reference Summary:\n", sample["reference_summary"])


if __name__ == "__main__":
    run_pipeline()
