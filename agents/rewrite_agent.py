from transformers import pipeline


class RewriteAgent:
    def __init__(self):
        print("Loading rewriting model (BART)...")
        self.rewriter = pipeline(
            "summarization", model="facebook/bart-large-cnn")
        """
        @article{DBLP:journals/corr/abs-1910-13461,
        author    = {Mike Lewis and
                    Yinhan Liu and
                    Naman Goyal and
                    Marjan Ghazvininejad and
                    Abdelrahman Mohamed and
                    Omer Levy and
                    Veselin Stoyanov and
                    Luke Zettlemoyer},
        title     = {{BART:} Denoising Sequence-to-Sequence Pre-training for Natural Language
                    Generation, Translation, and Comprehension},
        journal   = {CoRR},
        volume    = {abs/1910.13461},
        year      = {2019},
        url       = {http://arxiv.org/abs/1910.13461},
        eprinttype = {arXiv},
        eprint    = {1910.13461},
        timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},
        biburl    = {https://dblp.org/rec/journals/corr/abs-1910-13461.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
      }
      """

    def rewrite(self, text):
        """Rewrites the text to improve fluency and coherence using BART."""
        cleaned_text = " ".join(
            text.strip().split())  # Remove extra whitespace/newlines
        if not cleaned_text or len(cleaned_text) < 30:
            return "Not enough content to rewrite."

        try:
            result = self.rewriter(
                cleaned_text, max_length=200, min_length=40, do_sample=False)
            return result[0]['summary_text'].strip()
        except Exception as e:
            return f"Rewrite failed: {str(e)}"
