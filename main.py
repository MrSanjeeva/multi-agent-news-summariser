from datasets import load_dataset
import json
import os
from agents.factual_agent import FactualAgent
from agents.bias_agent import BiasAgent
from agents.rewrite_agent import RewriteAgent
from agents.bart_summariser import BARTSummariser
from detoxify import Detoxify
from evaluation.metrics import evaluate_all
from evaluation.logger import EvaluationLogger

DATA_PATH = "data/sample_articles.json"


def load_and_prepare_data(n=5, save=True):
    print("Loading Multi-News dataset...")
    dataset = load_dataset("alexfabbri/multi_news",
                           split=f"test[:{n}]", trust_remote_code=True)

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
    data = load_and_prepare_data(n=5)

    agent = FactualAgent()
    bias = BiasAgent()
    rewrite = RewriteAgent()
    bart_agent = BARTSummariser()
    detox = Detoxify("original")
    logger = EvaluationLogger()

    for i, sample in enumerate(data):
        print(f"\n Processing Article {i+1}")
        all_text = " ".join(sample["articles"])
        reference = sample["reference_summary"]

        # Agent pipeline
        facts = agent.extract_facts(all_text)
        print("Extracted Key Facts:\n", facts)

        unbiased = bias.filter_bias(facts)
        final_summary = rewrite.rewrite(unbiased)
        bart_summary = bart_agent.summarise(all_text)

        # Evaluate both summaries
        results = evaluate_all(reference, bart_summary,
                               final_summary, bias, detox)

        print("ROUGE (Multi-agent):", results["rouge"]["multi_agent"])
        print("ROUGE (BART):", results["rouge"]["bart"])

        print("BERTScore (Multi-agent):", results["bertscore"]["multi_agent"])
        print("BERTScore (BART):", results["bertscore"]["bart"])

        print("Bias Scores (multi-agent):", results["bias"]["multi_agent"])
        print("Bias Scores (BART):", results["bias"]["bart"])

        print("\nFinal Summary:\n", final_summary)
        print("BART Summary:\n", bart_summary)
        print("Reference Summary:\n", reference)

        # Save log entry
        logger.log_entry(
            article_id=i + 1,
            reference=reference,
            bart_summary=bart_summary,
            multiagent_summary=final_summary,
            metrics_dict=results
        )

    logger.save()


if __name__ == "__main__":
    run_pipeline()
