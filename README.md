# deep-agent-template

A production-ready LangGraph + DeepAgents template for building AI agents with vector memory, human-in-the-loop review, and modular skills.

## What's included

- **LangGraph orchestration** — 5-node graph with auto-retry and HITL support
- **Vector memory** — local ChromaDB + sentence-transformers (no API key needed)
- **Universal skills** — debug-mantra, planning, reflection, scrutinize, post-mortem
- **Tools** — web search, file read/write, shell execution
- **Modular backends** — swap vector DB via `.env`

## Project structure

```
├── agent.py          — system prompt, model, create_agent()
├── main.py           — LangGraph graph + CLI
├── tools/            — agent tools
├── skills/           — SKILL.md files
├── memory/           — vector memory (ChromaDB + embeddings + chunking)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quickstart

### Local

```bash
# 1. Clone
git clone https://github.com/your-org/deep-agent-template my-agent
cd my-agent

# 2. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 4. Run
python main.py --task "your task here"
python main.py --task "your task here" --hitl
```

### Docker

```bash
# 1. Configure
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY

# 2. Build and run
docker compose run agent --task "your task here"

# With HITL
docker compose run -it agent --task "your task here" --hitl
```

## Customization

**1. Define your output schema** — edit `AgentOutput` in `agent.py`:

```python
class AgentOutput(BaseModel):
    title: str
    result: str
    sources: list[str]
```

**2. Write your system prompt** — edit `SYSTEM_PROMPT` in `agent.py`:

```python
SYSTEM_PROMPT = """
You are a research analyst...
"""
```

**3. Pick your tools** — edit `create_agent()` in `agent.py`:

```python
tools=[web_search]                              # research agent
tools=[read_file, write_file, run_command]      # coding agent
tools=[web_search, read_file, write_file]       # mixed
```

> Note: `read_file` and `write_file` are sandboxed to the `workspace/` folder. The agent cannot read or write outside it.

**4. Add project-specific skills** — drop a folder into `skills/`:

```
skills/
├── debug-mantra/   ← universal (included)
├── planning/       ← universal (included)
├── reflection/     ← universal (included)
├── scrutinize/     ← universal (included)
├── post-mortem/    ← universal (included)
└── your-skill/     ← add yours here
    └── SKILL.md
```

## Optional

**Remove memory** — if your agent doesn't need persistent memory:

1. Delete `memory/`
2. Remove `load_memory_node` and `save_node` memory calls in `main.py`
3. Remove `sentence-transformers` and `chromadb` from `pyproject.toml`

**Add a vector DB backend** — implement `BaseMemoryStore` in `memory/backends/`:

```python
class PineconeMemoryStore(BaseMemoryStore):
    ...
```

Then add `elif backend == "pinecone"` in `memory/backends/__init__.py` and set `MEMORY_BACKEND=pinecone` in `.env`.

**Enable subagents** — configure `get_subagent()` in `agent.py`, then pass `use_subagents=True` to `create_agent()`.

## Environment variables

See `.env.example` for all available options.

## Observability (optional)

Enable LangSmith tracing to monitor every agent run:

1. Get a free API key at **smith.langchain.com**
2. Uncomment in `.env`:

```dotenv
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your-project-name
```

3. Run your agent — traces appear automatically in **Projects** on LangSmith

Each trace shows every node, tool call, skill load, and token usage per run.
