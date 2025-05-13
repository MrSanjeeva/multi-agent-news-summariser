# Multi-Agent News Summarisation System

This project implements a multi-agent AI system for generating concise, coherent, and unbiased summaries from multiple news sources reporting on the same topic.

### Project Description

The system simulates multiple specialized language agents:

- **Factual Agent**: Extracts key facts and named entities from the articles.
- **Bias Detection Agent**: Detects and mitigates bias in the extracted content.
- **Rewriting Agent**: Refines the summary for fluency and readability.

The goal is to compare this multi-agent pipeline against traditional single-agent summarisation models like BART and T5.

### Research Question

Does a multi-agent approach improve the factual accuracy, neutrality, and coherence of AI-generated news summaries compared to a single-agent summarisation system?

### Folder Structure

```
.
├── agents/         # Agent implementations (factual, bias, rewrite)
├── data/           # News datasets and test articles
├── evaluation/     # Evaluation scripts and metrics
├── main.py         # Pipeline runner
├── requirements.txt
└── README.md
```

### How to Run

```bash
# Activate virtual environment
source news-summariser-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```
