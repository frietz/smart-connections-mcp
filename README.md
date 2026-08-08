# Smart Connections MCP Server

Exposes your Obsidian Smart Connections vector database to Claude Code via Model Context Protocol (MCP).

## What This Does

Instead of using text-based `Grep`, Claude Code can now perform **semantic search** across your vault:

- **semantic_search**: Find notes by meaning, not keywords
- **find_related**: Get related notes (like Smart Connections sidebar)
- **get_context_blocks**: Get best context for RAG queries

## Architecture

```
Smart Connections Plugin
    ↓ (creates)
.smart-env/multi/*.ajson
    ↓ (reads)
This MCP Server
    ↓ (exposes via)
MCP Protocol
    ↓ (consumed by)
Claude Code
```

## Installation

### Quick Install (Recommended)

```bash
cd ~/smart-connections-mcp
./install.sh
```

The script will:
- ✅ Install UV package manager (if needed)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Auto-detect your Obsidian vault
- ✅ Configure `~/.mcp.json`
- ✅ Verify installation

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

#### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create Virtual Environment and Install Dependencies

```bash
cd ~/smart-connections-mcp
uv venv
uv pip install -r requirements.txt
```

**Important dependencies:**
- `mcp>=1.0.0` - Official Model Context Protocol SDK
- `sentence-transformers>=2.2.0` - For semantic search
- `numpy<2.0.0` - Version 1.x required (2.x breaks compatibility)
- `torch>=2.0.0` and `transformers>=4.30.0` - ML dependencies

#### 3. Configure Claude Code

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "smart-connections": {
      "command": "/Users/YOUR_USERNAME/smart-connections-mcp/.venv/bin/python",
      "args": ["/Users/YOUR_USERNAME/smart-connections-mcp/server.py"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/path/to/your/obsidian/vault"
      }
    }
  }
}
```

**Note:** Use the virtual environment Python, not system Python!

#### 4. Verify Installation

```bash
claude mcp list
```

Expected output:
```
smart-connections: .venv/bin/python server.py - ✓ Connected
```

</details>

### Migration to New Machine

**See [DEPLOYMENT.md](DEPLOYMENT.md)** for detailed migration guide.

Quick migration:
```bash
# On new machine
git clone https://github.com/dan6684/smart-connections-mcp.git ~/smart-connections-mcp
cd ~/smart-connections-mcp
./install.sh
```

**Important:** Keep this MCP server in a **separate repository** from your Obsidian vault. See [DEPLOYMENT.md](DEPLOYMENT.md) for rationale and best practices.

### Troubleshooting

If you see timeout issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Usage Examples

### Semantic Search

**Old way (Grep):**
```
Grep pattern: "self-compassion"
→ Only finds notes with exact word "self-compassion"
```

**New way (Semantic Search):**
```
semantic_search(query: "recognizing self-worth and releasing shame")
→ Finds: Ann Shulgin note ("I am a treasure")
        BM playa note ("I am beautiful, playa saved me")
        Therapy notes (related concepts)
```

### Find Related Notes

**Like Smart Connections sidebar:**
```
find_related(file_path: "DailyNotes/2025-10-25.md")
→ Returns top 10 semantically similar notes
```

### Get Context for RAG

**Build context for complex queries:**
```
get_context_blocks(query: "transformation through embodiment")
→ Returns actual text blocks most relevant to query
→ Claude can use these for grounded answers
```

## How It Works

1. **Reads existing embeddings** written by the Smart Connections plugin - no
   re-indexing of what it has already done
2. **Handles both store layouts.** Smart Connections 4.7.2 moved from
   `.smart-env/multi/*.ajson` (one file per note, vectors inline as JSON) to
   `.smart-env/smart_sources/` plus flat float32 blobs addressed by row index.
   The old tree is left in place when the plugin migrates, so reading it keeps
   succeeding against a frozen corpus - the modern layout is preferred and the
   legacy one warns when it is used
3. **Uses whichever model the store was built with.** The model is identified
   from the vectors themselves, by re-encoding one note and matching. The
   plugin's `smart_env.json` records a model key that goes stale, and the
   query prefix depends on getting this right - an asymmetric model queried
   without its prefix scores worse than the one it replaced
4. **Embeds what the plugin has not.** Smart Connections only runs while
   Obsidian is open, so notes written in between are chunked per heading and
   indexed here, into a separate cache merged at query time. Nothing is ever
   written into `.smart-env/`
5. **Cosine similarity** over one normalized matrix to rank results
6. **Returns JSON** with file paths, similarity scores, and block text

## Testing

```bash
./run-tests.sh          # everything
./run-tests.sh units    # hermetic only - no vault, no model, no network
```

Two suites, stdlib `unittest`, no test dependency to install.

- **`tests/test_server_units.py`** - 30 tests, ~0.03s. Synthetic blobs and
  metadata; runs anywhere. Each case is a regression that actually happened and
  says which one in its docstring.
- **`tests/test_server_live.py`** - integration against a real vault and model.
  Covers all four tools directly and again over the MCP protocol in a separate
  process. Skipped automatically when no store is present, so the hermetic
  suite still runs on a machine without Obsidian.

Set `OBSIDIAN_VAULT_PATH` to test against a vault other than
`~/obsidian/vault-obsidian`.

The three `test_*.py` scripts this replaced pointed at absolute paths on the
original author's machine, contained no assertions, and printed
"All semantic search tests completed successfully" after finding zero results
for every query.

## Tools Provided

### `semantic_search`
```python
semantic_search(
    query: str,           # Natural language query
    limit: int = 10,      # Max results
    min_similarity: float = 0.3  # Threshold
)
```

Returns:
```json
{
  "query": "self-compassion",
  "results_count": 5,
  "results": [
    {
      "path": "DailyNotes/2025-08-29.md",
      "similarity": 0.87,
      "key": "smart_sources:DailyNotes/2025-08-29.md",
      "metadata": {"tags": ["#Dream", "#grateful"]}
    }
  ]
}
```

### `find_related`
```python
find_related(
    file_path: str,      # e.g., "DailyNotes/2025-10-25.md"
    limit: int = 10
)
```

### `get_context_blocks`
```python
get_context_blocks(
    query: str,
    max_blocks: int = 5
)
```

Returns actual text content (not just paths) for RAG.

## Performance

- **Initial load:** ~2-3 seconds (loads 3,249 embeddings)
- **Query time:** ~100-200ms (cosine similarity across all embeddings)
- **Memory:** ~50MB (cached embeddings)

## Troubleshooting

**See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed debugging guide.**

### Common Issues

#### Server Timeout on `claude mcp list`
**Symptoms:** Connection hangs, no response after 30+ seconds

**Fixes:**
1. Ensure using virtual environment Python (not system Python)
2. Verify NumPy version is <2.0.0: `uv pip list | grep numpy`
3. Check server starts manually:
   ```bash
   OBSIDIAN_VAULT_PATH="/path/to/vault" .venv/bin/python server.py
   ```

#### Import Errors
**Error:** `ImportError: numpy.core.multiarray failed to import`

**Fix:** Reinstall with NumPy 1.x:
```bash
uv pip install "numpy<2.0.0" --force-reinstall
```

#### No Results Returned
- Check `.smart-env/multi/` has .ajson files
- Verify Smart Connections is enabled in Obsidian
- Lower `min_similarity` threshold (try 0.2 instead of 0.3)

#### Wrong Results
- Smart Connections may need to re-index
- Check embedding model matches (BGE-micro-v2)
- Restart server to reload embeddings

## Development

**Update embeddings:**
- Smart Connections auto-updates `.smart-env/`
- MCP server reads on startup (restart to refresh)
- Future: Add file watcher for auto-reload

**Add new tools:**
Edit `handle_request()` in `server.py`

## License

MIT - Use freely for personal PKM workflows
