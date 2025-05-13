from transformers import pipeline


class BARTSummariser:
    def __init__(self):
        print("Loading BART summarisation pipeline...")
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn")

    def summarise(self, text, max_length=200, min_length=50):
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
