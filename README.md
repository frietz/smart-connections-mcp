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
.smart-env/smart_sources/smart_sources.ajson   metadata
.smart-env/smart_sources/mf_<id>               float32 vectors
.smart-env/smart_blocks/mf_<id>                float32 vectors
    ↓ (reads, never writes)
This MCP Server  ──  ~/.cache/smart-connections-mcp/   its own top-up vectors
    ↓ (exposes via)                                    merged at query time
MCP Protocol
    ↓ (consumed by)
Claude Code
```

Smart Connections 4.7.2 replaced the older `.smart-env/multi/*.ajson` layout
(one file per note, vectors inline as JSON) with the tree above. The legacy
path is still readable and still supported, but it warns when it is used - the
plugin leaves the old tree in place on migration, so reading it silently
succeeds against a corpus frozen at the migration date.

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
./run-tests.sh          # everything - 76 tests
./run-tests.sh units    # hermetic only - no vault, no model, no network
./run-tests.sh paths    # hermetic, against a synthetic 4.7.2 store on disk
./run-tests.sh live     # integration against the real vault and model
```

Three suites, stdlib `unittest`, no test dependency to install.

- **`tests/test_server_units.py`** - 40 tests, ~0.03s. Pure functions against
  synthetic blobs and metadata; runs anywhere.
- **`tests/test_server_paths.py`** - 17 tests, ~1s. Builds a miniature 4.7.2
  store in a temp directory and drives `load_embeddings` end to end against it.
  This module exists because the suite above it tested pure functions well and
  call sites not at all - and every production-severity defect this project has
  had was a call site.
- **`tests/test_server_live.py`** - 19 tests against a real vault and model.
  Covers all four tools directly and again over the MCP protocol in a separate
  process.

Each case is a regression that actually happened and names it in its docstring,
so a failure says which bug came back. The standard for adding one: revert the
fix it guards and confirm the test goes RED. A test that stays green with the
fix removed was never checking the requirement it claims.

**A missing store fails the live suite rather than skipping it.**
`run-tests.sh` sets `SCMCP_REQUIRE_LIVE` whenever live tests were asked for.
Green-because-nothing-ran is the exact lie told by the three `test_*.py`
scripts this suite replaced - they pointed at absolute paths on the original
author's machine, asserted nothing, and printed "All semantic search tests
completed successfully" after finding zero results for every query. Set
`SCMCP_ALLOW_NO_STORE=1` to opt out on a machine without Obsidian.

Set `OBSIDIAN_VAULT_PATH` to test against a vault other than
`~/obsidian/vault-obsidian`.

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

Measured on a 956-note vault indexed as 19,012 vectors (956 sources, 18,056
blocks, 384 dims). Your numbers scale with vault size - `index_stats` reports
the live ones rather than these.

- **First load:** ~8s. Parses the 40MB metadata file, then loads the embedding
  model to identify which one the store was built with.
- **Later loads:** ~0.8s. The parsed matrix is cached under
  `~/.cache/smart-connections-mcp/`, keyed to the store state it was read at.
- **Query:** ~20ms median - one matmul against the normalized matrix.
- **Memory:** ~29MB for the matrix itself, plus the model.

The first query of a session may also spend up to `SMART_CONNECTIONS_TOPUP_SECONDS`
(12s default) embedding notes the plugin has not reached yet. That budget is
wall clock, not a chunk count, so it holds whichever model is selected.

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
- Check `.smart-env/smart_sources/` has `smart_sources.ajson` and an `mf_*`
  blob beside it (or `.smart-env/multi/*.ajson` on a pre-4.7.2 plugin)
- Verify Smart Connections is enabled in Obsidian
- Run `index_stats` - it reports the live vector count and the identified model
- Lower `min_similarity` threshold (try 0.2 instead of 0.3)

#### Wrong Results
- Run `index_stats` and check the reported model is the one selected in
  Obsidian. **Do not trust `smart_env.json`'s `model_key`** - it goes stale
  under 4.7.2 and can name a model the store was not built with. The server
  ignores it and identifies the model from the vectors instead, by re-encoding
  one note and matching against its stored vector.
- An asymmetric model queried without its query prefix scores *worse* than the
  model it replaced, and the failure looks like "that model is bad" rather than
  "a string is missing". Prefixes are keyed by model name in `SEMANTIC_PROFILES`.
- Smart Connections may need to re-index
- Reconnect the MCP server (`/mcp` in Claude Code) to reload after a model change

## Development

**Update embeddings:**
- Smart Connections auto-updates `.smart-env/` while Obsidian is open
- The server picks up notes written since, without a restart: search entry
  points re-check the vault for stale and deleted notes, throttled to once per
  5s and guarded by a signature so an unchanged vault costs ~0.4us
- `--reindex` drains a whole backlog in one pass, ignoring the time budget
- A reconnect is still needed after a **model** change, which rebuilds both caches

**Add new tools:**
Edit `handle_request()` in `server.py`

## License

MIT - Use freely for personal PKM workflows
