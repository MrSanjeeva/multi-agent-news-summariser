import os
import json
from datetime import datetime


class EvaluationLogger:
    def __init__(self, log_dir="evaluation/logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"run_{timestamp}.json")
        self.data = []

    def log_entry(self, article_id, reference, bart_summary, multiagent_summary, metrics_dict):
        entry = {
            "article_id": article_id,
            "reference_summary": reference,
            "bart_summary": bart_summary,
            "multiagent_summary": multiagent_summary,
            "metrics": metrics_dict
        }
        self.data.append(entry)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
