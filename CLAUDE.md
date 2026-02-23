# DIVAN: Personal Advisory Council CLI

## What is this?

Divan is a CLI tool where you pose a question and 4-5 AI advisors with distinct worldviews deliberate on it in parallel, then a Bas Vezir (Grand Vizier) synthesizes all perspectives into a decision brief.

The name comes from the Ottoman "Divan-i Humayun" (Imperial Council), where advisors with different roles would deliberate on matters of state before the Sultan made a decision.

## Why this exists

The builder (Alp) is an AI consultant and aspiring founder who struggles with decision paralysis when evaluating ideas, career moves, and technical choices. He has strong analytical thinking but gets trapped in his own perspective. This tool externalizes his decision-making process into a structured multi-perspective deliberation.

This is a personal tool first, potential product second. Build for one user: Alp.

## Core interaction

```
$ divan "Should I leave my consulting job to build a genomics startup?"

╭─────────────────────────────────────────────╮
│  DİVAN - Personal Advisory Council          │
│  "Should I leave my consulting job to       │
│   build a genomics startup?"                │
╰─────────────────────────────────────────────╯

[All 4 advisors deliberate in PARALLEL, streaming simultaneously]

⚔️  The Contrarian (Muhalif)     ⚙️  The Operator (Sadrazam)
🔭  The Visionary (Kahin)        👤  The Customer (Müşteri)

👁  Bas Vezir is synthesizing...
[final synthesis in a gold-accented card]
```

## Architecture

### Overview

```
divan/
├── CLAUDE.md              # This file
├── README.md              # User-facing docs
├── pyproject.toml         # Project config, dependencies
├── divan/
│   ├── __init__.py
│   ├── cli.py             # Entry point, Click CLI
│   ├── engine.py          # LangGraph deliberation engine (parallel fan-out/fan-in)
│   ├── advisor.py         # Advisor model + persona loader
│   ├── synthesis.py       # Bas Vezir synthesis prompt builder
│   ├── config.py          # Settings (API keys, model config)
│   ├── models.py          # Model factory (provider-agnostic via init_chat_model)
│   ├── display.py         # Rich-based streaming display
│   └── tools/
│       ├── __init__.py    # Tool registry (TOOL_REGISTRY, get_tools_for_advisor)
│       └── base.py        # Core tools: web_search, read_file, list_files, grep_search, run_command
├── personas/
│   ├── contrarian.md      # System prompt for The Contrarian
│   ├── operator.md        # System prompt for The Operator
│   ├── visionary.md       # System prompt for The Visionary
│   ├── customer.md        # System prompt for The Customer
│   └── bas_vezir.md       # System prompt for Bas Vezir (synthesizer)
├── docs/
│   └── ROADMAP.md         # Feature roadmap
└── tests/
    └── test_engine.py
```

### Tech stack

- **Python 3.11+**
- **LangGraph** for orchestration (parallel fan-out to advisors, fan-in to synthesis)
- **LangChain** with `init_chat_model()` for provider-agnostic model creation
- **langchain-anthropic**, **langchain-openai**, **langchain-google-genai** as LLM providers
- **Rich** for terminal UI (cards, panels, markdown rendering, live streaming)
- **Click** for CLI argument parsing
- **Pydantic** + **pydantic-settings** for config and data models
- No database. No web framework. Async only for LangGraph parallel execution.

### Model configuration

Models use `"provider:model_name"` format. Supported providers:
- `anthropic:` (Claude models, e.g., `anthropic:claude-sonnet-4-6`)
- `openai:` (GPT models, e.g., `openai:gpt-5.2`)
- `google_genai:` (Gemini models, e.g., `google_genai:gemini-2.5-flash`)

Defaults:
- Advisors: `google_genai:gemini-2.5-flash` (fast, cheap, good quality)
- Synthesis: `anthropic:claude-sonnet-4-6` (best reasoning for Bas Vezir)

## Detailed design

### Personas (THE MOST IMPORTANT PART)

Each persona is a markdown file in `personas/` with this structure:

```markdown
---
name: The Contrarian
title: Muhalif
icon: ⚔️
color: red
order: 1
---

[System prompt content here]
```

The YAML frontmatter defines display properties. The markdown body IS the system prompt sent to the LLM. The synthesizer persona has `is_synthesizer: true` in frontmatter.

**Persona design principles:**
- Each advisor MUST produce meaningfully different output from the others. If two advisors say roughly the same thing, one of them is poorly designed.
- Personas should be opinionated and specific, not balanced. The balance comes from having multiple perspectives, not from each advisor being balanced.
- Each persona should have a signature question they always answer.
- Keep personas to 300-500 words max. Shorter system prompts produce more consistent behavior.
- Personas should speak in first person, directly to the user, as if they're a real advisor sitting across the table.

### Persona definitions

**1. The Contrarian (Muhalif), ⚔️, Red, Order 1**
Finds every flaw, plays devil's advocate, stress-tests assumptions. Signature: "Why will this fail?" / "What are you not seeing?"

**2. The Operator (Sadrazam), ⚙️, Blue, Order 2**
Only cares about execution. Signature: "Can you ship a v0.1 in one weekend?" / "What's step 1, literally today?"

**3. The Visionary (Kahin), 🔭, Purple, Order 3**
Thinks 3-5 years out, connects to larger trends. Signature: "What does this become at scale?" / "What larger shift does this ride?"

**4. The Customer (Müşteri), 👤, Green, Order 4**
Role-plays as the potential buyer/user. Signature: "Why would I pay for this?" / "What's the alternative I'm using today?"

**5. Bas Vezir (Grand Vizier), 👁, Gold, Order 99 (always last)**
Synthesizes ALL advisor responses into a structured decision brief with: Verdict, Agreements, Disagreements, Recommended Action, and "The one thing everyone missed."

### Deliberation engine (engine.py)

Uses LangGraph for parallel fan-out/fan-in:

1. **DivanState** TypedDict with `operator.add` reducer for accumulating parallel advisor responses
2. **Fan-out**: START -> all advisor nodes run in parallel (LangGraph superstep)
3. **Fan-in**: All advisor nodes -> synthesis node -> END
4. Each advisor node wraps LLM call in try/except for error isolation

**Key design decisions:**

1. **Parallel, not sequential.** LangGraph supersteps execute all advisors concurrently for speed.
2. **Advisors do NOT see each other's responses.** Each advisor gets only the user's question. This prevents groupthink. Only Bas Vezir sees everything.
3. **Streaming is mandatory.** The display layer streams each advisor's tokens as they arrive using Rich Live.
4. **Error isolation.** One advisor failing does not kill the whole deliberation.

### Display layer (display.py)

Uses Rich library with parallel streaming:

1. **Header panel:** Shows "DİVAN" title and the question in a bordered panel.
2. **Advisor cards:** All 4 advisors' responses stream simultaneously in Rich Panels with colored borders, icons, and Markdown rendering.
3. **Bas Vezir panel:** Gold-bordered, appears last after all advisors complete.

### CLI interface (cli.py)

```bash
# Basic usage
divan "Should I build X?"

# Pipe from stdin
echo "Should I build X?" | divan

# Save output to file
divan "Should I build X?" --output brief.md

# Use specific personas only
divan "Should I build X?" --advisors contrarian,operator

# List available personas
divan --list

# Specify model for advisors
divan "Should I build X?" --model openai:gpt-5.2

# Specify model for synthesis
divan "Should I build X?" --synthesis-model anthropic:claude-opus-4-6
```

### Config (config.py)

Uses pydantic-settings with .env file support:

```python
class DivanSettings(BaseSettings):
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    advisor_model: str = "google_genai:gemini-2.5-flash"
    synthesis_model: str = "anthropic:claude-sonnet-4-6"
    max_tokens: int = 1500
    synthesis_max_tokens: int = 2000
    personas_dir: str = "personas"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DIVAN_")
```

### Tool system

Advisors can use tools to ground their advice in real information. Tools are defined per-persona via a `tools:` list in the YAML frontmatter. The tool system has three layers:

1. **Tool Registry** (`divan/tools/__init__.py`): Global `TOOL_REGISTRY` dict mapping tool names to LangChain `BaseTool` instances. Functions: `register_tool()`, `get_tools_for_advisor()`, `list_available_tools()`, `ensure_tools_registered()`.

2. **Core Tools** (`divan/tools/base.py`): Five tools registered on import:
   - `web_search`: DuckDuckGo search for current information
   - `read_file`: Read local file contents (line-limited)
   - `list_files`: List directory contents with glob patterns
   - `grep_search`: Regex search across files
   - `run_command`: Shell command execution (timeout-limited)

3. **Persona frontmatter**: Each persona specifies which tools it can use. Example: `tools: [web_search, read_file]`. Bas Vezir never has tools (synthesizer only).

**Display behavior**: Advisors with tools use an invoke loop (no streaming during tool calls). Tool usage is shown inline in the panel (`🔍 Searching: "query"...`). The final response appears below the tool lines. Advisors without tools stream normally.

**TUI integration**: The interactive setup includes a "Tools" step between advisor selection and model selection. Users can use defaults, customize per-advisor, or disable all tools.

## Important constraints

- **TUI is preferred over dashed CLI commands.** The interactive TUI menu is the primary interface. CLI flags exist for scripting and automation, but interactive users should go through the TUI flow. Do not add new `--flag` commands when a TUI prompt is more appropriate.
- **Never use em dashes in any text.** Use commas, periods, or other punctuation instead. This applies to persona prompts, CLI output, README, everything. The user has a strong preference against em dashes.
- **Streaming is non-negotiable.** The watching-them-think experience is core to the product feel.
- **Each persona file is self-contained.** A user can add new advisors by dropping a new .md file in the personas/ directory. The engine auto-discovers them and orders by the `order` field in frontmatter.
- **The Bas Vezir always runs last and always sees all other responses.** This is not configurable.
- **Model-agnostic.** Any LangChain-supported provider works via `init_chat_model()`.

## Future phases (DO NOT BUILD NOW, just be aware)

See `docs/ROADMAP.md` for the full roadmap.

**v0.2: War Room mode (tmux)**
- `divan --war-room "question"` opens tmux with split panes, one per advisor, all streaming simultaneously.

**v0.3: Web UI**
- Ottoman-themed dark web interface with advisor cards arranged visually.
- React + Tailwind. WebSocket streaming from a FastAPI backend.

**v0.4: Custom Divan compositions**
- Users create "Divan templates" for different decision types.
- `divan --template startup "My idea is..."`

**v0.5: Memory and learning**
- The Divan remembers past deliberations and can reference them.
