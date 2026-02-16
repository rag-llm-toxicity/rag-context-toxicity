# RAG Context Toxicity

This repository contains the code and experimental pipeline for an anonymous submission studying:

**The impact of knowledge base composition on toxicity in Retrieval-Augmented Generation (RAG) systems.**

## Overview

Retrieval-Augmented Generation (RAG) integrates external knowledge bases into large language model (LLM) generation. While RAG improves factual grounding, the safety implications of retrieved context remain underexplored.

This project conducts a controlled, data-centric analysis of how different knowledge base types influence toxic language generation.

We isolate the effect of retrieval by holding prompts and model settings constant while varying the knowledge base content.

## Knowledge Base Types

Three knowledge bases are constructed:

- **Neutral KB** — curated neutral factual content (e.g., Wikipedia-style)
- **Toxic KB** — adversarial toxic corpora
- **Mixed KB** — heterogeneous social media discourse

## Experimental Pipeline

The pipeline consists of five stages:

1. **Prompt Preparation**
2. **Knowledge Base Construction**
3. **Generation**
   - Prompt-only inference
   - Retrieval-Augmented Generation (RAG)
4. **Toxicity Evaluation**
5. **Statistical Analysis**

## Repository Structure

```
rag-context-toxicity/
│
├── input_prompts/           # Prompt sets for each KB condition
│   ├── input_prompts_toxic.py
│   ├── input_prompts_mixed.py
│   ├── input_prompts_neutral.py
│
├── kb_construction/         # Scripts to build knowledge bases
│   ├── build_neutral_kb.py
│   ├── build_toxic_kb.py
│   ├── build_mixed_kb.py
│
├── generation/              # Inference pipelines
│   ├── prompt_generation.py
│   ├── rag_inference.py
│   ├── prompt_only_inference.py
│
├── evaluation/              # Toxicity scoring
│   ├── detoxify_scoring.py
│   ├── perspective_api_scoring.py
│
├── analysis/                # Statistical analysis and visualization
│   ├── toxicity_statistics.py
│   ├── plotting.py
│
├── requirements.txt
└── README.md
```

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### 1. Construct Knowledge Bases

```
python kb_construction/build_neutral_kb.py
python kb_construction/build_toxic_kb.py
python kb_construction/build_mixed_kb.py
```

### 2. Run Generation

Prompt-only:

```
python generation/prompt_only_inference.py
```

RAG:

```
python generation/rag_inference.py
```

### 3. Evaluate Toxicity

```
python evaluation/detoxify_scoring.py
python evaluation/perspective_api_scoring.py
```

### 4. Analyze Results

```
python analysis/toxicity_statistics.py
python analysis/plotting.py
```

## Evaluation Metrics

Toxicity is assessed using complementary classifiers measuring multiple dimensions, including:

- General toxicity
- Severe toxicity
- Obscenity
- Threat
- Insult
- Identity attack

## Reproducibility

All experiments are conducted using identical prompts across conditions to isolate the effect of retrieved context.

Sample prompts and configuration scripts are included. Full datasets and outputs will be released upon acceptance.

## Ethical Considerations

This project studies toxic language generation and includes adversarial content solely for research purposes aimed at improving safety in language models.

## License

MIT License
