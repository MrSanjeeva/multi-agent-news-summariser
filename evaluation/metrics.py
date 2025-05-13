from rouge_score import rouge_scorer
import bert_score
from detoxify import Detoxify


def compute_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


def compute_bertscore(reference, summary):
    _, _, f1 = bert_score.score(
        [summary], [reference], lang="en", verbose=False)
    return round(f1[0].item(), 4)


def compute_bias_scores(summary, bias_agent):
    return bias_agent.get_bias_confidence_score(summary)


def evaluate_all(reference, bart_summary, multi_summary, bias_agent, detox=None):
    detox = detox or Detoxify("original")

    return {
        "rouge": {
            "bart": compute_rouge(reference, bart_summary),
            "multi_agent": compute_rouge(reference, multi_summary),
        },
        "bertscore": {
            "bart": compute_bertscore(reference, bart_summary),
            "multi_agent": compute_bertscore(reference, multi_summary),
        },
        "bias": {
            "bart": compute_bias_scores(bart_summary, bias_agent),
            "multi_agent": compute_bias_scores(multi_summary, bias_agent),
        }
    }
