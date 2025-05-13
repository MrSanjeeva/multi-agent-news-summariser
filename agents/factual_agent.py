import spacy
import torch
import warnings
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer, util
import os

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add your API Key and CSE ID here or use environment variables
GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY", "AIzaSyARLIvAdzUc0-FQa0V9rwUrFoTLBhFaTFc")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "c498910b36e094dfb")


class FactualAgent:
    def __init__(self, use_sentence_transformers=False):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")

        print("Loading entailment model (DeBERTa MNLI)...")
        self.entailment_model = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli",
            device=-1
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "MoritzLaurer/DeBERTa-v3-base-mnli")
        self.max_length = 512

        print("Loading embedding model...")
        self.use_sentence_transformers = use_sentence_transformers
        if use_sentence_transformers:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
            self.bert_model.eval()

        print("Loading sentence similarity model for Google entailment...")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.google_service = build(
            "customsearch", "v1", developerKey=GOOGLE_API_KEY)

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        if self.use_sentence_transformers:
            return self.embedding_model.encode(sentences, convert_to_tensor=True).cpu().numpy()
        else:
            encoded_input = self.bert_tokenizer(
                sentences, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_output = self.bert_model(**encoded_input)
            sentence_embeddings = self._mean_pool(
                model_output, encoded_input['attention_mask'])
            return sentence_embeddings.numpy()

    def deduplicate(self, sentences, similarity_threshold=0.90):
        if len(sentences) <= 1:
            return sentences

        embeddings = self.get_embeddings(sentences)
        retained = []
        seen_indices = set()

        for i, sent in enumerate(sentences):
            if i in seen_indices:
                continue
            retained.append(sent)
            for j in range(i + 1, len(sentences)):
                if cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] > similarity_threshold:
                    seen_indices.add(j)

        return retained

    def prepare_entailment_input(self, premise, hypothesis):
        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)

        available_premise_len = self.max_length - len(hypothesis_tokens) - 3
        if available_premise_len <= 0:
            raise ValueError(
                "Hypothesis too long to fit into max_length limit.")

        truncated_premise = self.tokenizer.convert_tokens_to_string(
            premise_tokens[:available_premise_len])
        return f"{truncated_premise} </s> {hypothesis}"

    def google_fact_check(self, sentence):
        try:
            res = self.google_service.cse().list(
                q=sentence, cx=GOOGLE_CSE_ID, num=3).execute()
            snippets = [item['snippet'] for item in res.get('items', [])]
            return snippets
        except Exception as e:
            print(f"Google search failed: {e}")
            return []

    def google_entailment_verification(self, sentence, snippets, threshold=0.75):
        if not snippets:
            return False
        sent_emb = self.sim_model.encode(sentence, convert_to_tensor=True)
        for snip in snippets:
            snip_emb = self.sim_model.encode(snip, convert_to_tensor=True)
            sim_score = util.cos_sim(sent_emb, snip_emb).item()
            if sim_score > threshold:
                return True
        return False

    def extract_facts(self, text, max_sentences=5):
        doc = self.nlp(text)
        sentences = list(doc.sents)

        scored = []
        for sent in sentences:
            s = sent.text.strip()
            if len(s.split()) < 5 or s.endswith(":"):
                continue
            entity_score = len([ent for ent in sent.ents])
            length_score = len(s) / 100
            score = entity_score + length_score
            scored.append((score, s))

        top_candidates = [text for _, text in sorted(
            scored, key=lambda x: x[0], reverse=True)[:max_sentences]]
        filtered_sentences = self.deduplicate(top_candidates)

        verified_facts = []
        for sent in filtered_sentences:
            try:
                input_text = self.prepare_entailment_input(text, sent)
                result = self.entailment_model(input_text)[0]
                label = result["label"].upper()
                score = result["score"]

                tag_map = {
                    "ENTAILMENT": "✔ Supported",
                    "NEUTRAL": "❓ Unclear",
                    "CONTRADICTION": "❌ Possibly False"
                }
                tag = tag_map.get(label, f"Unknown label: {label}")

                sources = self.google_fact_check(sent)
                if label == "NEUTRAL" and self.google_entailment_verification(sent, sources):
                    tag = "✅ Verified by Google"

                if sources:
                    snippet_text = " | ".join(sources[:2])
                    source_info = f"\n Sources: {snippet_text}"
                else:
                    source_info = "\n No supporting sources found"

                verified_facts.append(
                    f"{tag} ({score:.2f}): {sent}{source_info}")
            except Exception as e:
                print(f"Verification error:\n{sent}\nError: {str(e)}")
                verified_facts.append(f"Failed to verify: {sent}")

        return "\n".join(verified_facts) if verified_facts else "No facts extracted."
