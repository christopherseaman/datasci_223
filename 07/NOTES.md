# Lecture 07 Notes — Transformers: More than Meets the Eye

# Links & Self-Guided Review

Jay Alammar's Illustrated Transformer walks through the same Q/K/V mechanism with step-by-step animated diagrams. Karpathy's microGPT blog post matches the Demo 1 code directly; the full nanoGPT repo goes further.

UCSF Versa is sunsetting soon and ChatGPT Enterprise is replacing it in March 2026. These aren't abstract research tools — they're things our institution is actively deploying.

# From Neural Networks to Transformers

Every model on this timeline uses the same core algorithm. ELMo, BERT, GPT-3, ChatGPT, Llama — they all use attention. The differences are scale (parameters, data, compute) and training strategy (bidirectional vs autoregressive, RLHF), not architecture. We're about to teach the one algorithm underneath all of it.

GPT (2018) had 117M parameters. GPT-3 (2020) had 175B — a 1,500x increase in two years. ChatGPT hit 100M users in two months, the fastest consumer product adoption in history. The open-weight explosion (Llama, Mistral, DeepSeek) since 2023 is what makes the rest of this lecture practical — you can run and fine-tune these models on a laptop now.

The model family tree has three branches worth knowing: encoder-only (BERT family — classification, NER), decoder-only (GPT family — generation), and encoder-decoder (T5 family — translation, summarization). Modern LLMs have converged on decoder-only. That's the architecture we'll build in Demo 1.

# Transformer Architecture

## The Problem: Processing Everything at Once

The original 2017 paper used encoder-decoder, but modern LLMs (GPT-4, Claude, Llama) are decoder-only. No separate "understanding" step — just predict the next token given everything so far. Counterintuitive, but it works because the decoder's self-attention already builds rich representations of the input without a dedicated encoder.

The pipeline: Tokenize → Embed → Add positional encodings → Stack attention layers → Generate output. Every section that follows fills in one step of this pipeline.

## Self-Attention: Letting Tokens Talk

"The animal didn't cross the street because *it* was too tired." What does "it" refer to? You resolve it instantly — but try to explain *how*. Self-attention is the mechanism that makes this explicit: each token computes how much it should attend to every other token, resolving references in a single parallel step.

For each token, three vectors from learned weight matrices:
- **Query**: "What am I looking for?" — cat's query encodes what information it needs from the rest of the sentence
- **Key**: "What do I have to offer?" — each token advertises its relevance
- **Value**: "Here's my actual content" — what gets retrieved when attention fires

The "cat, sat, mat" example makes it concrete. Cat's query dot-producted against every key: scores 112, 96, 78. Scale by √d_k, softmax to probabilities: 0.73, 0.22, 0.05. Cat attends mostly to itself and somewhat to sat. That weighting is *learned* — different sentences produce different attention patterns. This is what makes self-attention adaptive.

The code snippet is six lines of numpy. The entire attention mechanism is matrix multiplication, scaling, softmax, and another matrix multiplication. No magic.

## Multi-Head Attention

If single-head attention is one person reading a sentence, multi-head is eight people reading the same sentence, each noticing different things — syntax, semantics, coreference, tense. Then they pool what they found. 8 heads × 64 dims = 512 dims. Same total dimension, but the specialization lets the model capture multiple relationship types simultaneously.

## Putting It Together

The cross-attention block in the decoder is where the decoder reads the encoder's output. It's asking: "Given what I've generated so far, what parts of the input should I focus on next?" This is the component that modern decoder-only models remove — they concatenate everything into one sequence and let self-attention handle it all.

Cross-entropy loss measures prediction error, gradients flow back through the whole stack, Adam optimizer updates weights. Repeat over billions of examples.

## Beyond Text

Transformers are not a text-specific architecture. Anything that can be sequenced works: image patches (Vision Transformers), protein residues (AlphaFold), EHR events, audio spectrograms. Clinical EHR modeling is directly relevant — sequences of diagnosis codes, medications, and lab values over time are structurally identical to word sequences, and attention learns which prior events matter for predicting outcomes.

# Building a GPT from Scratch

Karpathy's microGPT implements everything we just covered in 200 lines of Python with zero dependencies. The architecture diagram maps band-by-band to the transformer components from the previous section: autograd engine, token + position embeddings, multi-head self-attention with residual connections, feed-forward MLP, output head, softmax.

GPT-4 uses the same algorithm. Different tokenizer, terabytes of data, thousands of GPUs — but the core algorithm is the same.

# LIVE DEMO!

Build a complete GPT from scratch: autograd engine, embeddings, attention, MLP, training, generation. Character-level name generation — the model goes from random characters to plausible-sounding names in ~1000 steps on CPU.

The autograd verification cell (gradients of f(x,y) = xy + x²) is backpropagation demystified. The training loss dropping from ~3.3 to ~1.8 — each step makes the model slightly better at predicting the next character. Temperature comparison: 0.2 produces common, safe names; 1.0 produces creative, sometimes broken ones. Same temperature parameter that shows up in the API later.


# Embeddings

Embeddings map discrete tokens to continuous vectors where meaning is geometry. Similar items cluster together; relationships become directions. Every layer of a transformer produces embeddings — they're the model's internal representation of meaning. LLMs produce rich, high-dimensional embeddings internally, but for practical tasks like search we use smaller purpose-built models (Sentence Transformers) because their embeddings are compact enough to store and compare at scale.

The idea generalizes beyond text — recommendation systems, drug interactions, diagnostic codes, categorical variables can all be embedded. The geometry arises from patterns in the data in an unsupervised, self-organizing way. Nobody labels which words should be near each other.

Word2vec: "king - man + woman = queen" — directions in the space encode relationships. The distributed embedding diagram shows how "bank" near "river" and "bank" near "money" land in different neighborhoods.

## Sentence Transformers

One vector per sentence (not per word), and it's contextual — "bank" gets a different embedding depending on the surrounding sentence. Three lines of code: load model, encode sentences, get a (3, 384) array. Production-ready embedding.

## Cosine Similarity

Same concept as Lecture 04 (document similarity with TF-IDF), now applied to dense embeddings. Measures angle between vectors, ignoring magnitude — a 10-page discharge summary and a 1-paragraph note about the same condition have different magnitudes but similar directions.

Encode documents, encode query, compute similarities, sort. Documents ranked by semantic relevance, no keyword matching required.

## Vector Databases

For small datasets (hundreds to low thousands), brute-force cosine similarity against everything works fine. For millions of documents, you need approximate nearest-neighbor indexing — that's what vector databases provide. ChromaDB handles embedding automatically with a built-in model, so the code is even simpler than the manual approach.

# General Models → Getting the Details Right

LLMs are general-purpose — the same model translates, summarizes, classifies, writes code, and reasons. Open-weight models that would cost millions to train from scratch are freely available. The practical question isn't "how do I build a model?" but "how do I get an existing model to do what I need?"

Two approaches: prompting (minutes to test, lower cost — the recommended default) and fine-tuning (days to weeks, higher cost — for specialized vocabulary or domain patterns). Most people will only ever need prompting.

## Fine-Tuning

Continue training on domain data. Save it for specialized vocabulary or patterns where you have hundreds+ labeled examples. The code snippet shows a complete pipeline: load pre-trained GPT-2, tokenize clinical text, set up Trainer, train.

## Making Fine-Tuning Practical

Full fine-tuning updates every weight — expensive and often unnecessary. Layer freezing locks early layers and trains only the later task-specific ones. Head replacement swaps the final layers while using the frozen base as a feature extractor — the most common transfer-learning pattern. LoRA inserts small trainable adapter modules (~1-5% of parameters) into the frozen model; swap adapters for different tasks without retraining from scratch. Pruning removes redundant weights after training for faster inference.

In practice, most teams start with prompting, move to LoRA if needed, and rarely do full fine-tuning.

## Hallucination

No general solution. The model confidently generates plausible-sounding text that may be completely wrong. Three partial mitigations: RAG (ground responses in actual documents — Lecture 8), prompt and output design (structured outputs, schema enforcement, require citations), and human-in-the-loop (expert review for high-stakes decisions). None are foolproof.

# LIVE DEMO!!

Part 1: semantic search on PMC-Patients data. Embed with SentenceTransformer, compute similarity heatmap, build a search function, index in ChromaDB. The heatmap shows that embedding similarity captures medical relatedness — cardiac cases near cardiac cases — without any domain-specific rules.

Part 2: fine-tune GPT-2 on clinical text. Before fine-tuning, generic output. After, clinical-sounding text. Then the hallucination demo: prompts outside the training distribution produce confident, detailed, completely fabricated patient scenarios. The fabricated text is disturbingly convincing — patient IDs, specific lab values, plausible timelines, all invented.

# Prompt Engineering

"Programming" the model without retraining. Every prompt has the same building blocks:

- **ROLE**: Who the model should act as
- **TASK**: What needs to be done
- **FORMAT**: How to structure the output
- **CONSTRAINTS**: Boundaries and requirements
- **EXAMPLES**: Concrete input/output pairs

## Zero-Shot, One-Shot, Few-Shot

More examples = more consistent output. Zero-shot works for simple, well-defined tasks. One-shot establishes the pattern. Few-shot (2-5 examples) is needed for complex output formats or domain-specific conventions. The more structured the expected output, the more examples help.

## System Prompts

System prompts set persistent behavior — persona, constraints, default format. User prompts contain per-request content. The clinical documentation assistant example: "Use ICD-10 codes when identifying diagnoses. Flag findings that need follow-up. Never provide treatment recommendations." System prompts are how you enforce safety constraints in healthcare LLM deployments.

## Explicit Structure and Grounding

XML tags or section markers separate instructions from data from format requirements. Reduces errors when the model handles multiple components simultaneously. Grounding — asking the model to extract and cite relevant quotes before answering — reduces hallucination by forcing the model to anchor its response in the source material.

## Self-Verification and Chain-of-Thought

Chain-of-thought: "Think through each pair step by step." Self-verification: "After completing your analysis, verify you checked every combination." Both improve accuracy on multi-step reasoning. Chain-of-thought in particular is the foundation of reasoning models like o1 and DeepSeek-R1 — they do this internally at inference time.

## Prompt Chaining

Break complex tasks into sequential steps: extract medications → check interactions → summarize findings. Each prompt's output feeds into the next. This is the foundation of agentic workflows in Lecture 8.

## Structured Responses

JSON, XML, or table output instead of free text. Put the schema in the prompt, validate programmatically. LLMs sometimes wrap JSON in markdown code fences or add commentary around it — defensive parsing is necessary. Try direct `json.loads()` first, strip markdown fences second, find outermost braces third. Return None if all fail.

The lecture schema (diagnosis, confidence, icd_code, reasoning), Demo 3 schema (diagnosis, key_findings, treatment, outcome), and assignment schema (diagnosis, medications, lab_values, confidence) are all different. The prompting pattern is the same, the specific fields change per task.

# LLM API Integration

OpenRouter uses the same OpenAI SDK with a different `base_url` — one import, one client initialization change, access to GPT-4o-mini, Claude, Llama, and every other major model. Same `client.chat.completions.create()` call, same message format with system/user roles, same `response.choices[0].message.content` to extract the result.

API keys go in environment variables. `python-dotenv` loads them from a `.env` file; the raw `os.environ` pattern is what's actually happening underneath.

# LIVE DEMO!!!

Systematic comparison of five prompting techniques on five clinical cases from PMC-Patients:
1. Zero-shot (free text)
2. Few-shot (free text)
3. Schema JSON (structured)
4. Few-shot + JSON (structured with examples)
5. Chain-of-thought + JSON (reasoning then structured)

The `clean_json()` function uses three parsing strategies: direct parse, strip markdown fences, find outermost braces. The JSON validity heatmap shows that few-shot + JSON and chain-of-thought produce more reliable structured output than schema-only. The side-by-side diagnosis comparison shows where technique choice matters and where it doesn't.
