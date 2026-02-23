# Divan Roadmap

## Tier 1: Smarter Advisors (Done)

### 1. Tool-Enabled Advisors ✅
Give advisors tools (web search, file reading, grep, shell commands) so they can ground advice in real information instead of reasoning only from training data. Tools are defined per-persona in frontmatter and managed via the TUI.

### 2. Advisor Memory ✅
Advisors remember past deliberations and can reference them. "Last time you asked about X, The Contrarian warned about Y, and that turned out to be right."

### 3. Custom Divan Templates ✅
Pre-configured advisor compositions for different decision types. `divan --template startup "My idea is..."` loads a startup-focused set of advisors. Templates are YAML files in `templates/`, with TUI picker and CLI flag support.

## Tier 2: Better Interface

### 1. War Room Mode (tmux)
`divan --war-room "question"` opens tmux with split panes, one per advisor, all streaming simultaneously. Full terminal real estate.

### 2. Web UI
Ottoman-themed dark web interface with advisor cards arranged visually. React + Tailwind with WebSocket streaming from a FastAPI backend.

### 3. Export and Sharing
Export deliberations as formatted PDFs, share via URL, or push to Notion/Obsidian.

## Tier 3: Platform

### 1. MCP Tool Marketplace
Community-contributed tool packs for specific domains: financial analysis, codebase review, competitive intelligence.

### 2. Multi-User Divans
Invite collaborators to observe or participate in deliberations. Each person gets their own follow-up thread.

### 3. Divan API
Expose deliberation as an API for integration into other workflows, Slack bots, CI/CD pipelines, or other AI agent systems.
