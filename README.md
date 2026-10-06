---
title: Applied Data Science with Python
permalink: /
---

# Applied Data Science with Python

> [!INFO] Course Information
> - **Instructor:** Christopher Seaman
> - **Dates:** January 15th - March 11th, 2026 _(10 class meetings)_
> - **Lecture:** Wednesday, 1:00 PM - 3:00 PM, Mission Hall 1407
> - **Lab:** Wednesday, 3:00 PM - 4:30 PM, Mission Hall 1407
> - **GitHub:** [https://github.com/christopherseaman/datasci_223](https://github.com/christopherseaman/datasci_223)
> - **Course Website:** [https://christopherseaman.github.io/datasci_223/](https://christopherseaman.github.io/datasci_223/)

This repository contains the course materials for UCSF DataSci 223: Applied Data Science with Python.

## Course Topics

### Foundational (L01-L04)

1. **Defensive Programming & Debugging**
    - Secrets and config with `.env` and YAML
    - Input validation, logging, and linting with Ruff
    - VS Code debugger with Jupyter: breakpoints, logpoints, watches
2. **Larger-than-memory Data with Polars**
    - Why memory limits sneak up on you
    - Lazy execution and streaming with `scan_*` / `collect`
    - Config-driven Polars pipelines
3. **SQL for Self-service Datasets & Analysis**
    - SQL in notebooks with JupySQL + DuckDB/SQLite
    - Filtering, aggregation, joins, and execution order
    - Subqueries, CTEs, window functions, and views
    - SQL + Python workflows
4. **NLP: Extracting Meaning from Clinical Text**
    - Tokenization, POS tagging, and named entity recognition
    - Text extraction with regex and spaCy matchers
    - Bag-of-words, TF-IDF, and document similarity
    - Clinical text pipelines

### ML/AI Progression (L05-L08)

5. **Classification: Predicting Labels with Classic ML**
    - Train/test splits, cross-validation, and evaluation metrics
    - Supervised model tour (logistic regression, SVM, Random Forest, XGBoost)
    - How models fail, interpretation, and data preparation
6. **Neural Networks: From Neurons to Deep Learning**
    - Neurons, activation functions, and how networks learn
    - Architectures: dense, CNN, RNN for images, signals, and text
    - Training in practice: callbacks, checkpoints, custom loops
7. **Transformers: the Building Blocks of LLMs**
    - Self-attention and building a GPT from scratch
    - Embeddings, cosine similarity, and vector databases
    - Fine-tuning and prompt engineering
    - LLM API integration
8. **LLM Applications: Agents, RAG & Workflows**
    - When to use LLMs and common failure modes
    - Agentic LLMs with tools and structured output
    - Retrieval-augmented generation (RAG) and Model Context Protocol (MCP)
    - Workflow orchestration patterns

### Applied (L09-L10)

9. **Computer Vision with Pretrained Models**
    - CNNs in PyTorch and torchvision transforms
    - Transfer learning with pretrained models
    - Object detection and image segmentation
10. **A/B Testing & Causal Inference**
    - Stats refresher: CLT, hypothesis tests, confidence intervals
    - Potential outcomes, confounding, and why randomization works
    - Running A/B tests in practice: peeking, CUPED
