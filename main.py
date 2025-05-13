# main.py

from datasets import load_dataset
import json
import os
from agents.factual_agent import FactualAgent
from agents.bias_agent import BiasAgent
from agents.rewrite_agent import RewriteAgent
import pandas as pd

DATA_PATH = "data/sample_articles.json"


def load_and_prepare_data(n=5, save=True):
    """Load Multi-News articles and save a small sample locally."""
    print("Loading Multi-News dataset...")
    dataset = load_dataset("alexfabbri/multi_news",
                           split="test[:{}]".format(n), trust_remote_code=True)
    """
    @misc{alex2019multinews,
    title={Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model},
    author={Alexander R. Fabbri and Irene Li and Tianwei She and Suyi Li and Dragomir R. Radev},
    year={2019},
    eprint={1906.01749},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
    }
    """

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
            print(f"Sample data saved to {DATA_PATH}")

        return sample_data


def run_pipeline():
    # Load data
    data = load_and_prepare_data(n=5)

    # Intiate agents
    agent = FactualAgent()
    bias = BiasAgent()
    rewrite = RewriteAgent()

    # Run each article through the factual agent
    for i, sample in enumerate(data):
        print(f"\n Processing Article {i+1}")
        all_text = " ".join(sample["articles"])

        # Agent 1: Extract facts
        facts = agent.extract_facts(all_text)
        print("Extracted Key Facts:\n", facts)  # ✅ PRINT TAGGED FACTS HERE

        # Agent 2: Filter bias
        unbiased = bias.filter_bias(facts)

        # Agent 3: Rewrite
        final_summary = rewrite.rewrite(unbiased)

        print("Final Summary:\n", final_summary)
        # print("Reference Summary:\n", sample["reference_summary"])

        bias_scores = bias.get_bias_confidence_score(final_summary)
        print("Bias Scores:", bias_scores)
    # print("Extracted Key Facts:\n", facts)
    # print("Reference Summary:\n", sample["reference_summary"])


if __name__ == "__main__":
    run_pipeline()
