# # agents/bias_agent.py

import re
from detoxify import Detoxify
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)


class BiasAgent:
    def __init__(self, toxicity_threshold=0.3):
        print("Loading Detoxify model...")
        self.model = Detoxify('original')
        self.threshold = toxicity_threshold

        # Expanded lexical bias cues
        self.bias_lexicon = set([
            # Ideological
            "progressive", "equity", "inclusive", "social justice", "deep state", "patriot",
            "traditional values", "family values", "woke agenda",
            # Loaded language
            "radical", "heroic", "reckless", "corrupt", "disastrous", "outrageous",
            # Framing
            "mainstream media", "cancel culture", "elitist", "fake news",
            # Speculative
            "crisis", "catastrophe", "conspiracy", "weaponized", "apocalypse",
            # Bias verbs
            "admits", "blasts", "defends", "mocked", "slams", "lashes out", "praises"
        ])

    def filter_bias(self, text, max_sentences=5):
        """Split text into sentences, score toxicity and lexical bias, and return least biased ones."""
        sentences = sent_tokenize(text)
        scored = []

        for sent in sentences:
            toxicity = self.model.predict(sent)['toxicity']
            bias_hits = sum(1 for word in sent.lower().split()
                            if word.strip('.,!?"') in self.bias_lexicon)
            scored.append((sent, toxicity, bias_hits))

        # Filter out highly toxic or lexically biased sentences
        filtered = [s for s in scored if s[1] < self.threshold and s[2] == 0]

        if not filtered:
            return "All content filtered out for bias."

        # Sort by lowest toxicity then return top N
        sorted_filtered = sorted(filtered, key=lambda x: x[1])[:max_sentences]
        return "\n".join([s[0] for s in sorted_filtered]) or "No unbiased content found."

    def get_bias_confidence_score(self, text):
        """Compute overall bias confidence score from toxicity + lexical cues."""
        sentences = sent_tokenize(text)
        toxic_scores = [self.model.predict(s)["toxicity"] for s in sentences]
        lexical_hits = sum(
            any(re.search(rf"\b{re.escape(word)}\b", s.lower())
                for word in self.bias_lexicon)
            for s in sentences
        )

        avg_toxicity = sum(toxic_scores) / \
            len(toxic_scores) if toxic_scores else 0
        lexical_ratio = lexical_hits / len(sentences) if sentences else 0

        # Weighted score: 70% from Detoxify, 30% from lexical cues
        final_score = round(0.7 * avg_toxicity + 0.3 * lexical_ratio, 3)

        return {
            "toxicity_score": round(avg_toxicity, 3),
            "lexical_bias_ratio": round(lexical_ratio, 3),
            "bias_confidence_score": final_score
        }
