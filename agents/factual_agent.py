# agents/factual_agent.py

import spacy


class FactualAgent:
    def __init__(self):
        print("🔍 Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

    def extract_facts(self, text, max_sentences=5):
        """Extracts named entities and ranks top sentences."""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        # Basic scoring: count named entities in each sentence
        scored = []
        for sent in sentences:
            score = len([ent for ent in sent.ents])
            scored.append((score, sent.text.strip()))

        # Sort by score and return top N
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
        return "\n".join([s[1] for s in top if s[0] > 0]) or "No strong facts found."
