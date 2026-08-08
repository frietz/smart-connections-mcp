#!/usr/bin/env python3
"""
Smart Connections MCP Server
Exposes Smart Connections vector database to Claude Code via MCP protocol
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server


CACHE_VERSION = 2
# Results are serialized straight into the model's context, so an oversized
# `limit` is a context hazard, not just a slow response. Clamp it.
MAX_RESULTS = 50
MAX_BLOCKS = 20
DEFAULT_CACHE_DIR = Path(
    os.getenv("SMART_CONNECTIONS_CACHE_DIR", Path.home() / ".cache" / "smart-connections-mcp")
)


class SmartConnectionsDatabase:
    """Interface to Smart Connections .smart-env vector database"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.smart_env_path = self.vault_path / ".smart-env"
        self.multi_path = self.smart_env_path / "multi"

        # Lazy load embedding model (same as Smart Connections uses)
        self.model = None
        self.model_name = 'TaylorAI/bge-micro-v2'

        # Vector store, built once then reused.
        # matrix rows are L2-normalized, so cosine similarity is a plain dot
        # product and the whole search collapses to one matmul.
        self.matrix: Optional[np.ndarray] = None   # (N, dim) float32, normalized
        self.keys: List[str] = []
        self.paths: List[Optional[str]] = []
        self.lines: List[Optional[list]] = []
        self.is_source: Optional[np.ndarray] = None  # bool mask
        self.is_block: Optional[np.ndarray] = None   # bool mask
        self.key_index: Dict[str, int] = {}
        self.skipped = 0
        self.embeddings_loaded = False

        cache_root = DEFAULT_CACHE_DIR
        vault_id = hashlib.sha256(str(self.vault_path.resolve()).encode()).hexdigest()[:16]
        self.cache_dir = cache_root / vault_id

    def ensure_model_loaded(self):
        """Lazy load the embedding model on first use.

        torch/sentence_transformers are imported here, not at module top,
        so a server instance that never runs a search never pays the
        ~6s CPU / ~860MB RAM import cost. Threads capped to 1: encoding a
        single short query against an in-memory cache gains nothing from
        intra-op parallelism and only contends across instances.
        """
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(1)
            self.model = SentenceTransformer(self.model_name)

    # ------------------------------------------------------------------
    # Vector store construction
    # ------------------------------------------------------------------

    def _fingerprint(self) -> str:
        """Cheap signature of the .ajson corpus.

        Rebuilds the cache when Smart Connections re-embeds. Uses count,
        newest mtime, and total size - enough to catch any real change
        without hashing ~200MB on every boot.
        """
        count = 0
        newest = 0.0
        total = 0
        for f in self.multi_path.glob("*.ajson"):
            try:
                st = f.stat()
            except OSError:
                continue
            count += 1
            total += st.st_size
            if st.st_mtime > newest:
                newest = st.st_mtime
        raw = f"v{CACHE_VERSION}:{count}:{newest:.0f}:{total}:{self.model_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _load_from_cache(self, fingerprint: str) -> bool:
        meta_path = self.cache_dir / "meta.json"
        vec_path = self.cache_dir / "vectors.npy"
        if not meta_path.exists() or not vec_path.exists():
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != fingerprint:
                return False
            # mmap so a cold process pays page-in cost only for rows it touches
            matrix = np.load(vec_path, mmap_mode="r")
            keys = meta["keys"]
            if matrix.shape[0] != len(keys):
                return False
        except Exception:
            return False

        self.matrix = matrix
        self.keys = keys
        self.paths = meta["paths"]
        self.lines = meta["lines"]
        self.skipped = meta.get("skipped", 0)
        self._finalize_masks()
        return True

    def _save_to_cache(self, fingerprint: str):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_vec = self.cache_dir / "vectors.tmp.npy"
            tmp_meta = self.cache_dir / "meta.json.tmp"
            # Pass a handle, not a path: np.save silently appends ".npy" to any
            # path that lacks it, which would break the atomic replace below.
            with open(tmp_vec, "wb") as f:
                np.save(f, self.matrix)
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fingerprint": fingerprint,
                        "model": self.model_name,
                        "built_at": time.time(),
                        "skipped": self.skipped,
                        "keys": self.keys,
                        "paths": self.paths,
                        "lines": self.lines,
                    },
                    f,
                )
            tmp_vec.replace(self.cache_dir / "vectors.npy")
            tmp_meta.replace(self.cache_dir / "meta.json")
        except Exception:
            # A cache we cannot write is a slow start, not a failure.
            pass

    def _finalize_masks(self):
        keys = self.keys
        self.is_source = np.fromiter(
            (k.startswith("smart_sources:") for k in keys), dtype=bool, count=len(keys)
        )
        self.is_block = np.fromiter(
            (k.startswith("smart_blocks:") for k in keys), dtype=bool, count=len(keys)
        )
        self.key_index = {k: i for i, k in enumerate(keys)}
        self.embeddings_loaded = True

    def load_embeddings(self):
        """Build (or restore) the normalized vector matrix.

        Smart Connections leaves `vec: null` on entries it has not embedded
        (blocks under min_chars, stale placeholders) - roughly a fifth of the
        corpus. Those must be filtered here: np.array(None, dtype=float32)
        yields a 0-d array, which silently turns a later dot product into a
        vector and blows up float() conversion for the WHOLE query.
        """
        if self.embeddings_loaded:
            return
        if not self.multi_path.exists():
            self.matrix = np.zeros((0, 1), dtype=np.float32)
            self._finalize_masks()
            return

        fingerprint = self._fingerprint()
        if self._load_from_cache(fingerprint):
            return

        vectors = []
        keys: List[str] = []
        paths: List[Optional[str]] = []
        lines: List[Optional[list]] = []
        skipped = 0
        dim = None

        for ajson_file in self.multi_path.glob("*.ajson"):
            try:
                with open(ajson_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # .ajson files are append-only fragments:
                #   "key1": {value1},
                #   "key2": {value2},
                # so wrap in braces and drop the trailing comma.
                if content and not content.startswith("{"):
                    content = "{" + content.rstrip(",").strip() + "}"
                if not content:
                    continue
                data = json.loads(content)
            except Exception:
                continue  # malformed shard - skip, keep the rest

            for key, item in data.items():
                if not isinstance(item, dict):
                    continue
                embeddings = item.get("embeddings")
                if not isinstance(embeddings, dict):
                    continue
                model_entry = embeddings.get(self.model_name)
                if not isinstance(model_entry, dict):
                    continue

                vec = model_entry.get("vec")
                # --- the guard that keeps one bad row from killing every query
                if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                    skipped += 1
                    continue
                arr = np.asarray(vec, dtype=np.float32)
                if arr.ndim != 1:
                    skipped += 1
                    continue
                if dim is None:
                    dim = arr.shape[0]
                elif arr.shape[0] != dim:
                    skipped += 1
                    continue
                norm = float(np.linalg.norm(arr))
                if not np.isfinite(norm) or norm == 0.0:
                    skipped += 1
                    continue

                vectors.append(arr / norm)
                keys.append(key)
                paths.append(self._resolve_path(key, item))
                lines.append(item.get("lines"))

        if vectors:
            self.matrix = np.stack(vectors).astype(np.float32, copy=False)
        else:
            self.matrix = np.zeros((0, dim or 1), dtype=np.float32)

        self.keys = keys
        self.paths = paths
        self.lines = lines
        self.skipped = skipped
        self._finalize_masks()
        self._save_to_cache(fingerprint)

    @staticmethod
    def _resolve_path(key: str, item: dict) -> Optional[str]:
        """Vault-relative path for an entry.

        Block entries carry `path: null`; their vault path is the part of the
        key before the first heading anchor.
        """
        path = item.get("path")
        if path:
            return path
        if ":" in key:
            _, rest = key.split(":", 1)
        else:
            rest = key
        return rest.split("#", 1)[0] or None

    # ------------------------------------------------------------------
    # Text resolution
    # ------------------------------------------------------------------

    def _read_text(self, row: int, max_chars: int = 2000) -> str:
        """Read the note text a row points at.

        Smart Connections stores no note text in .smart-env - only paths and
        line ranges - so text is read from the vault on demand, for the
        handful of rows that actually made the result set.
        """
        path = self.paths[row]
        if not path:
            return ""
        full = self.vault_path / path
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                content = f.readlines()
        except OSError:
            return ""

        span = self.lines[row]
        if isinstance(span, (list, tuple)) and len(span) == 2:
            try:
                start = max(int(span[0]) - 1, 0)      # stored 1-indexed, inclusive
                end = min(int(span[1]), len(content))
                if end > start:
                    content = content[start:end]
            except (TypeError, ValueError):
                pass

        text = "".join(content).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
        return text

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _top_rows(self, sims: np.ndarray, mask: Optional[np.ndarray], limit: int,
                  threshold: float) -> List[int]:
        """Rank rows by similarity, honouring a subset mask and a floor."""
        limit = max(1, min(int(limit), MAX_RESULTS))
        scores = sims
        if mask is not None:
            scores = np.where(mask, sims, -np.inf)
        eligible = int(np.count_nonzero(scores >= threshold))
        if eligible == 0:
            return []
        k = min(limit, eligible)
        # argpartition keeps this O(N) instead of a full sort of ~30k rows
        candidates = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
        candidates = candidates[scores[candidates] >= threshold]
        return list(candidates[np.argsort(-scores[candidates])])

    def _encode(self, query: str) -> np.ndarray:
        self.ensure_model_loaded()
        qv = np.asarray(self.model.encode(query), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(qv))
        if norm:
            qv = qv / norm
        return qv

    def semantic_search(self, query: str, limit: int = 10, min_similarity: float = 0.3) -> List[Dict]:
        """
        Perform semantic search against the vector database

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of results with path, score, and metadata
        """
        self.load_embeddings()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []

        query_vec = self._encode(query)
        if query_vec.shape[0] != self.matrix.shape[1]:
            return []

        sims = np.asarray(self.matrix @ query_vec)

        results = []
        for row in self._top_rows(sims, None, limit, min_similarity):
            results.append({
                'key': self.keys[row],
                'path': self.paths[row],
                'similarity': round(float(sims[row]), 4),
                'lines': self.lines[row],
                'text_preview': self._read_text(row, max_chars=200),
            })
        return results

    def find_related(self, file_path: str, limit: int = 10) -> List[Dict]:
        """
        Find notes related to a specific file

        Args:
            file_path: Path to file (relative to vault)
            limit: Maximum number of results

        Returns:
            List of related files
        """
        self.load_embeddings()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []

        target_key = f"smart_sources:{file_path}"
        row = self.key_index.get(target_key)
        if row is None:
            return []

        target_vec = np.asarray(self.matrix[row], dtype=np.float32)
        sims = np.asarray(self.matrix @ target_vec)

        mask = self.is_source.copy()
        mask[row] = False  # never return the source note as its own relation

        results = []
        for r in self._top_rows(sims, mask, limit, -1.0):
            results.append({
                'key': self.keys[r],
                'path': self.paths[r],
                'similarity': round(float(sims[r]), 4),
            })
        return results

    def get_context_blocks(self, query: str, max_blocks: int = 5) -> List[Dict]:
        """
        Get best context blocks for a query (for RAG)

        Args:
            query: Query string
            max_blocks: Maximum number of blocks to return

        Returns:
            List of block contents with metadata
        """
        self.load_embeddings()
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []

        query_vec = self._encode(query)
        if query_vec.shape[0] != self.matrix.shape[1]:
            return []

        sims = np.asarray(self.matrix @ query_vec)

        # Blocks carry full text (up to 2k chars each), so they cap lower.
        max_blocks = max(1, min(int(max_blocks), MAX_BLOCKS))

        results = []
        for row in self._top_rows(sims, self.is_block, max_blocks, 0.4):
            results.append({
                'key': self.keys[row],
                'path': self.paths[row],
                'similarity': round(float(sims[row]), 4),
                'lines': self.lines[row],
                'text': self._read_text(row),
            })
        return results

    def stats(self) -> Dict:
        """Vector store health - vector count, dropped rows, cache state."""
        self.load_embeddings()
        return {
            'vault_path': str(self.vault_path),
            'model': self.model_name,
            'vectors': int(self.matrix.shape[0]) if self.matrix is not None else 0,
            'dimensions': int(self.matrix.shape[1]) if self.matrix is not None and self.matrix.ndim == 2 else 0,
            'sources': int(self.is_source.sum()) if self.is_source is not None else 0,
            'blocks': int(self.is_block.sum()) if self.is_block is not None else 0,
            'skipped_unembedded': self.skipped,
            'cache_dir': str(self.cache_dir),
            'cache_present': (self.cache_dir / 'vectors.npy').exists(),
        }


async def main():
    import sys
    import logging

    # Setup logging to stderr
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.debug("Starting smart-connections-mcp server...")

    # Get vault path from environment
    vault_path = os.getenv('OBSIDIAN_VAULT_PATH')

    if not vault_path:
        raise ValueError("OBSIDIAN_VAULT_PATH environment variable not set")

    logger.debug(f"Vault path: {vault_path}")

    # Initialize database
    db = SmartConnectionsDatabase(vault_path)
    logger.debug("Database initialized")

    # Create MCP server
    server = Server("smart-connections-mcp")
    logger.debug("MCP server created")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="semantic_search",
                description="Search vault using semantic similarity (not keyword matching). Finds notes related to query meaning, not just exact words.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10
                        },
                        "min_similarity": {
                            "type": "number",
                            "description": "Minimum similarity threshold 0-1 (default: 0.3)",
                            "default": 0.3
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="find_related",
                description="Find notes related to a specific file path. Like Smart Connections sidebar in Obsidian.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path relative to vault root (e.g., 'DailyNotes/2025-10-25.md')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            types.Tool(
                name="get_context_blocks",
                description="Get best text blocks for a query (for RAG/context building). Returns actual text content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to find relevant context for"
                        },
                        "max_blocks": {
                            "type": "integer",
                            "description": "Maximum number of blocks (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="index_stats",
                description="Report vector index health: vector count, dimensions, sources vs blocks, rows skipped as unembedded, and cache state.",
                inputSchema={"type": "object", "properties": {}}
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool execution requests"""

        if arguments is None:
            arguments = {}

        try:
            if name == "semantic_search":
                results = db.semantic_search(
                    query=arguments['query'],
                    limit=arguments.get('limit', 10),
                    min_similarity=arguments.get('min_similarity', 0.3)
                )

                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({
                            "query": arguments['query'],
                            "results_count": len(results),
                            "results": results
                        }, indent=2)
                    )
                ]

            elif name == "find_related":
                results = db.find_related(
                    file_path=arguments['file_path'],
                    limit=arguments.get('limit', 10)
                )

                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({
                            "source_file": arguments['file_path'],
                            "related_count": len(results),
                            "related_files": results
                        }, indent=2)
                    )
                ]

            elif name == "get_context_blocks":
                results = db.get_context_blocks(
                    query=arguments['query'],
                    max_blocks=arguments.get('max_blocks', 5)
                )

                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({
                            "query": arguments['query'],
                            "blocks_count": len(results),
                            "blocks": results
                        }, indent=2)
                    )
                ]

            elif name == "index_stats":
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(db.stats(), indent=2)
                    )
                ]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            raise RuntimeError(f"Tool execution error: {str(e)}")

    # Run the server using stdin/stdout streams
    logger.debug("Starting stdio server...")
    async with stdio_server() as (read_stream, write_stream):
        logger.debug("stdio server started, running MCP server...")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="smart-connections-mcp",
                server_version="0.2.0",
                capabilities=types.ServerCapabilities(
                    tools=types.ToolsCapability(),
                ),
            ),
        )
        logger.debug("MCP server finished")


if __name__ == "__main__":
    asyncio.run(main())
