# 🌪️ Whirlwind Tour of LLMs

Practical LLM Tooling for Research Applications

### 🐿️ [chris@badmath.org](mailto:chris@badmath.org)

### 🤖 [not.badmath.org/llm-tour](https://not.badmath.org/llm-tour)

# History

## Word Embeddings (2013)

**word2vec**: Represent words as vectors in high-dimensional space

- Similar words cluster together (“insulin” near “glucose”)
- Used **Continuous Bag of Words** and **Skip-gram** algorithms for building context
    - **CBOW**: surrounding words used to predict word in the middle
    - **Skip-gram**: input word used to predict context
    - Precursors to **Attention**

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image.png)

## Sequence-to-Sequence & RNNs (2014)

**Encoder-decoder architecture**: Transform one sequence into another

![Encoder-Decoder](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaencoder_decoder.png)

Encoder-Decoder

- Encoder processes input into fixed representation
- Decoder generates output from that representation
- Used for translation, summarization
- Built on **RNNs** (Recurrent Neural Networks) with sequential processing

## RNNs, LSTM, and Limitations

Introduce “memory” to neural networks

![RNN Problems](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediarnn_problems.png)

**Two critical problems**:

**1. Vanishing gradients**: Error signals shrink as they propagate backward through time. Early words in sequence get minimal learning signal.

**2. Sequential bottleneck**: Must process word-by-word (word 1 → word 2 → word 3…). Cannot parallelize training. Slow and doesn’t scale.

## Attention Mechanism (2015)

**Key innovation**: Decoder focuses on specific input parts at each step

![Attention 2015](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaattention_2015.png)

- Dynamically weights which inputs matter most
- Solves information bottleneck

## Transformers (2017)

**“Attention is All You Need”**: Eliminated sequential processing entirely

![Attention Paper](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/f4b45abc-91c2-400a-b0ac-8f60462ae898.png)

**Transformer solution**:

- Process entire sequence simultaneously (parallel)
- All tokens relate to all others via attention
- No vanishing gradient problem
- 100x+ training speedup enables web-scale datasets

## The Scale-Up Era (2018-2024)

**2018**: GPT (170M parameters), BERT
**2020**: GPT-3 (175B parameters), few-shot learning
**2022**: ChatGPT with RLHF, 100M users in 2 months
**2024**: GPT-4, Claude 3.5, Gemini, Llama 3 (200K+ token context windows)

![Timeline](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediatimeline_llm.webp)

## Activation Functions

You’re familiar with sigmoid from logistic regression

![Sigmoid vs ReLU](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediasigmoid_vs_relu.png)

**Sigmoid problem**: In deep networks, gradients vanish as they propagate. Early layers learn almost nothing.

**ReLU innovation**: f(x) = max(0, x)

- Flat below zero, linear above
- Gradients flow through deep networks
- Cheap to compute (just comparison)

**Impact**: ReLU + massive data + GPUs = practical deep learning

## Tokenization

![Word Embeddings](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaword2vec_concept.png)

Breaking text into processable chunks:

- Subword units handle rare terms efficiently
- ~4 characters per token on average
- Models process 64K-200K+ tokens per request
- Each token gets a number for mathematical processing

## Embeddings

Map tokens into continuous high-dimensional space:

![Word Embedding Space](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaword_embedding_distributed.webp)

- Semantic similarity → spatial proximity
- “king” - “man” + “woman” ≈ “queen” (geometry captures analogies)

**Embedding use cases**: Semantic search, document clustering, similarity matching, anomaly detection

## Attention Mechanism

Every token attends to every other token:

- Each token queries all others for relevance
- Weighted combination forms context-awareness
- Captures long-range dependencies
- Fully parallelizable (no sequential bottleneck)

![Attention](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaattention_2015%201.png)

## Multi-Head Attention

Run multiple attention mechanisms in parallel:

![Multi-Head Attention](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediamultihead_attention_1.png)

## **Why multiple heads?**

![Multi-Head Detail](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediamultihead_attention_2.png)

- Each learns different relationships
- One head: syntactic structure
- Another head: semantic meaning
- Another head: entity references

## Transformer Architecture

![Transformer Overview](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediatransformer_overview.png)

**Key components**:

- Multi-head self-attention (all-to-all token relationships)
- Residual connections (preserve information through layers)
- Layer normalization (stabilize activations)
- Feed-forward transformations (adds complexity beyond linear combinations)

**Process**:

1. Tokenize and embed input
2. Add positional encodings (order information)
3. Stack attention + transformation layers
4. Generate output tokens

## Transformer Explainer

[Transformer Explainer: LLM Transformer Model Visually Explained](https://poloclub.github.io/transformer-explainer/)

# Capabilities

## Text Transformation

**📝 Summarization**: Condense while preserving key information

**🔍 Extraction**: Unstructured text → structured data

**🏷️ Classification**: Categorize by type, topic, sentiment

**🌐 Translation**: Language-to-language, format-to-format

## Beyond Text

### **LLM’s / transformers can work on any kind of sequential data:**

**👁️ Vision Transformers**: Images split into patches, processed as token sequences

**📊 Time-series**: Electronic health records, sensor readings, financial data

**🎨 Multimodal**: Combine text, images, audio (GPT-4V, Gemini, Claude)

**Key principle**: Attention generalizes to any sequence where order and relationships matter

## The Current Landscape

![](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediallmtools_landscape.webp)

## Zoom in on Agents

![](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediaagents_landscape.png)

## When to Use LLMs

**✅ Good fits**:

- Text summarization and transformation
- Structured data extraction
- Content classification
- Question answering over documents
- Draft generation with review

**❌ Poor fits**:

- Precise calculations (use tools)
- Factual retrieval without verification
- Real-time data without external connection
- High-stakes autonomous decisions
- Deterministic logic (use rule engines)

**Decision framework**: Can you describe the task clearly? Are errors catchable? Can you validate outputs?

## Common Failure Modes

**🎭 Hallucinations**: Fabricated citations, confident incorrect answers

- **Mitigation**: RAG, fact-checking, require citations

**💉 Prompt injection**: User input overrides instructions

- **Mitigation**: Separate user content from system instructions

**🎲 Inconsistency**: Same input → different outputs (temperature > 0)

- **Mitigation**: temperature=0 for extraction tasks

**📦 Context overflow**: Important information at edges gets lost

- **Mitigation**: Place critical info at start/end

**⚠️ Task/expertise mismatch**: User lacks domain knowledge to identify LLM errors

- **Mitigation**: Require expert review, provide reference materials, limit autonomous decisions

# Practical Skills & Tools

## Fine-Tuning vs Prompt Engineering

**Prompting** (recommended default):

- Fast iteration (minutes to test)
- No data collection needed
- Works across many tasks
- Lower cost

**Fine-tuning** (specialized cases only):

- Adapt pre-trained model on your specific data
- Requires 100s-1000s labeled examples
- Length dataset preparation and training

## Prompt Engineering: Few-Shot Learning

**Traditional ML**: Thousands of labeled examples + training

**LLM approach**: Examples in prompt, **no retraining needed!**

![Few-Shot Learning](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/mediafew_shot.png)

## Retrieval-Augmented Generation (RAG)

Ground model responses in retrieved documents by adding them to context before processing

**Process**:

1. Chunk documents, create embeddings
2. Store in vector database (via MCP)
3. Query → retrieve relevant chunks
4. Include chunks in prompt
5. Model generates response with citations

**Advantages**:

- May reduce hallucinations
- Provides verifiable sources
- Keeps information current

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%201.png)

## Prompt Engineering Example

```
[ROLE]
You are an expert data analyst.

[TASK]
Extract key metrics from the following report.

[FORMAT]
Return JSON: {metric_name: value, confidence: 0-1}

[CONSTRAINTS]
- Only include explicitly stated data
- Cite source sentences

[EXAMPLES]
Input: "Q3 revenue grew 15% to $2M"
Output: {"revenue": 2000000, "growth": 0.15, "cite": [1]}
```

## Agentic LLMs

**Traditional LLM use**: Single request → single response

**Agentic LLMs**: Multi-turn, self-guided iterations with tool access

**Key differences**:

- **Autonomy**: Agent decides next steps based on observations
- **Tool use**: Can invoke external functions (search, database queries, calculators)
- **Iteration**: Loops until task complete or max steps reached
- **State management**: Maintains context across multiple actions

**Example flow**:

1. Task: “Find recent papers on treatment X and summarize findings”
2. Agent searches literature database (tool call)
3. Agent reads top 3 papers (tool call)
4. Agent synthesizes findings
5. Agent checks if answer is complete → if not, searches for more specific info

## Prompting Techniques for Agentic LLMs

For complex agentic workflows, several advanced prompting patterns exist:

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%202.png)

**Common patterns**:

- **Chain-of-thought**: Make reasoning explicit step-by-step
- **Self-consistency**: Generate multiple reasoning paths, vote on answer
- **ReAct**: Interleave reasoning and tool actions
- **Reflection**: Surface uncertainty and assumptions
- **Decision trees**: Explicit conditional logic

**Resources**: See [Anthropic’s prompt engineering docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) and [OpenAI’s prompt examples](https://platform.openai.com/docs/examples)

**NOTE**: LLM “reasoning” is not the same as thinking and does NOT always achieve better results or fewer hallucinations. It IS always more expensive.

https://machinelearning.apple.com/research/illusion-of-thinking

## Structured Outputs

**Traditional**: Parse free text with regex → fragile, high error rate

**Structured outputs**: Model guarantees JSON schema conformance

**Benefits**:

- Simpler prompts (no formatting instructions)
- Reliable downstream processing
- Enforcable via “function calling” evaluation

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%203.png)

## Model Context Protocol (MCP)

**What MCP does**: Standardized way to connect LLMs to data sources and tools

**Common MCP servers**:

- **File systems**: Google Drive, Dropbox, local files
- **Databases**: Postgres, MySQL, SQLite
- **Communication**: Slack, Gmail, Microsoft Teams
- **Development**: GitHub, GitLab, code repositories, Context7 (docs for LLMs)
- **Web**: Puppeteer (web scraping), search APIs
- **Business tools**: Notion, Jira, Salesforce

## Workflow Orchestration

**The problem:** Real tasks span multiple steps and decision points:

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%204.png)

**Workflows provide:**

- **Sequencing**: Chain LLM calls with conditional logic (if this result, then that step)
- **State management**: Maintain context, remember decisions, handle partial failures
- **Tool integration**: Connect LLMs to databases, APIs, file systems, validation rules
- **Error handling**: Retries, fallbacks, human-in-the-loop checkpoints
- **Observability**: Track which step failed, inspect intermediate outputs, debug chains

**Common patterns**: Sequential chains, parallel fan-out/fan-in, retry loops, human approval
gates, RAG + generation, multi-agent delegation.

**Implementation approaches:**

- **Visual builders** (OpenAI Agent Builder): Non-engineers, rapid demos and iteration
- **Code-first** (LangGraph, AI SDK): Best for complex logic, version control, CI/CD, production

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%205.png)

## Pattern: Prompt Chaining ⛓️

**Concept**: Each LLM call processes output from previous call, well-defined tasks broken into verifiable steps

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%206.png)

**Benefit**: Allows complex tasks with each step as simple, testable, debuggable

## Pattern: Parallelization ⚡

**Concept**: Run independent LLM tasks simultaneously

![Parallelization](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/medialangchain_parallelization.png)

- **Speed:**
    - **Divide-and-conquer:** Split subtasks, execute in parallel, combine results
    - **First-to-finish:** Start task multiple times with different criteria, accept first completed
- **Confidence:**
    1. Run task multiple times with different criteria
    2. Choose winner or synthesize results

## Pattern: Orchestrator-Workers 🎯

**Concept**: Orchestrator breaks task into subtasks, delegates to workers, synthesizes results

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%207.png)

**Use cases**: Dynamic subtasks that cannot be predefined

- Generate multi-section reports (each section = one worker)
- Update content across unknown number of documents
- Research tasks requiring multiple queries

## Pattern: Evaluator-Optimizer 🔄

**Concept**: Generate → evaluate → refine loop until criteria met

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%208.png)

**Use cases**: Tasks with quality criteria requiring iteration

- Translation with semantic equivalence checking
- Clinical note generation with completeness validation
- Data extraction with accuracy scoring

## Pattern: Guardrails 🛡️

**Concept**: Input/output monitors that enforce safety and compliance rules

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%209.png)

**Common guardrails**:

- **PII/PHI detection**: Flag or redact protected health information
- **Hallucination detection**: Check if claims are grounded in source text
- **Jailbreak detection**: Identify prompt injection attempts
- **Format validation**: Ensure structured outputs meet schema

## Pattern: Routing & Logic 🔀

**Concept**: Conditional branching based on content or criteria

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%2010.png)

**Logic nodes**:

- **If/else**: Route based on classification or criteria
- **While loops**: Repeat until condition met (e.g., all sections complete)
- **Human approval**: Pause for review before high-stakes action

## Pattern: Deterministic Steps 🧮

**Concept**: Integrate rule-based logic alongside LLM calls

![image.png](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/image%2011.png)

**Use cases**: Known logic that doesn’t require LLM flexibility

- Dose calculations (use formulas, not LLMs)
- Date arithmetic
- Database lookups
- API calls with fixed parameters

## OpenAI Agent Builder Hands-on

[OpenAI Platform](https://platform.openai.com/agent-builder)

## Getting Started: Start Small

**Input $/1M tokens:**

![Layer.tiff](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/Layer.tiff)

**Output $/1M tokens:**

![Layer.tiff](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/Layer%201.tiff)

**“Baby” models** (low cost, quick):

- Mini/Nano tiers from major providers
- ~10x cheaper than flagship models
- Good for well-defined tasks

**Self-hosted options** (free, private):

- Ollama (desktop) & PocketPal (iOS)
- No API costs, no usage limits
- Ideal for sensitive data prototyping

## Getting Started: Testing & Validation

**Start simple**:

- Test on 5-10 representative examples first
- Manually review outputs - does it do what you need?
- Try edge cases (missing data, unusual formats)
- Incorporate failures into few-shot examples done correctly

**Red flags to watch for**:

- Inconsistent outputs for similar inputs
- Made-up citations or facts
- Missing required information
- Wrong format or structure

**Remember**: Choose tasks that you can meaningfully oversee, think of LLMs as prolific interns

# Resources

- **Transformer Explainer**: [https://poloclub.github.io/transformer-explainer/](https://poloclub.github.io/transformer-explainer/) - Visualize how transformers process text

### **Prompt engineering guides**:

- **Anthropic Prompt Engineering**: [https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) - Comprehensive guide with examples
- **OpenAI Prompt Examples**: [https://platform.openai.com/docs/examples](https://platform.openai.com/docs/examples) - Use case library

### **Learning from scratch**:

- **nanoGPT**: [https://github.com/karpathy/nanochat](https://github.com/karpathy/nanochat) - Andrej Karpathy’s GPT implementation from scratch (educational)

### **Workflow builders**:

- **OpenAI Workflows**: [https://platform.openai.com/workflows](https://platform.openai.com/workflows) - Visual workflow canvas
- **LangChain Documentation**: [https://python.langchain.com/docs](https://python.langchain.com/docs) - Code-based workflows and agents

### **Self-hosting**:

- **Ollama**: Server for LLMs [https://ollama.com](https://ollama.com/)
- **PocketPal**: iOS LLM host [https://github.com/a-ghorbani/pocketpal-ai](https://github.com/a-ghorbani/pocketpal-ai)
- **IBM Granite 4.0**: IBM model fits on your phone [https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- **gpt-oss**: OpenAI open-source models [https://openai.com/index/introducing-gpt-oss/](https://openai.com/index/introducing-gpt-oss/)
- **GLM-4.6**: high-scoring open-source model [https://huggingface.co/zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6)

### **Model Context Protocol**:

- **MCP Documentation**: [https://modelcontextprotocol.io](https://modelcontextprotocol.io/) - Protocol specification and getting started
- **MCP Servers Repository**: [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) - Pre-built connectors

### Tools I like or want to try:

- **OpenAI’s GUI Agent Builder**: [https://platform.openai.com/agent-builder](https://platform.openai.com/agent-builder)
- **Prompt management**: [latitude.so](https://latitude.so/)
- **Agent management:**
    - **Agents**: [https://platform.openai.com/docs/guides/agents](https://platform.openai.com/docs/guides/agents)
    - **AutoGen**: [https://microsoft.github.io/autogen/stable//index.html](https://microsoft.github.io/autogen/stable//index.html)
    - **LangGraph**: [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)
    - **smolagents**: [https://huggingface.co/docs/smolagents/index](https://huggingface.co/docs/smolagents/index)
    - **AI-SDK**: [https://ai-sdk.dev/docs/agents/overview](https://ai-sdk.dev/docs/agents/overview)
- **Orchestrators**
    - **Kestra**: [https://kestra.io](https://kestra.io/)
    - **Inngest**: ****[https://www.inngest.com](https://www.inngest.com/)
    - **Temporal**: [https://temporal.io](https://temporal.io/)
- **Command Line Tools**
    - **Claude Code:** [https://www.claude.com/product/claude-code](https://www.claude.com/product/claude-code)
        - **claude-flow**: [https://github.com/ruvnet/claude-flow](https://github.com/ruvnet/claude-flow)
        - **superclaude**: [https://www.superclaude.sh/](https://www.superclaude.sh/)
        - **awesome-claude-code** (list of resources): [https://github.com/hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
    - **OpenAI Codex**: [https://openai.com/codex/](https://openai.com/codex/)
    - **Cursor**: [https://cursor.com/](https://cursor.com/)

### **Academic discounts** (students and educators):

- **GitHub Education**: [https://github.com/education](https://github.com/education) - Free Pro account with Copilot access
- **ChatGPT for Teachers**: [https://openai.com/index/chatgpt-for-teachers/](https://openai.com/index/chatgpt-for-teachers/) - Free ChatGPT Plus for educators
- **Claude for Education**: [https://www.claude.com/solutions/education](https://www.claude.com/solutions/education) - Academic pricing and resources
- **Gemini for Students**: [https://gemini.google/students/](https://gemini.google/students/) - Student access to Gemini Advanced
- **Oracle Cloud Free Tier**: [https://www.oracle.com/cloud/free/](https://www.oracle.com/cloud/free/) - Best deal in free hosting, but not powerful enough to run models

![Layer.tiff](%F0%9F%8C%AA%EF%B8%8F%20Whirlwind%20Tour%20of%20LLMs/Layer%202.tiff)