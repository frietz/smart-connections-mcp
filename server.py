#!/usr/bin/env python3
"""
Smart Connections MCP Server
Exposes Smart Connections vector database to Claude Code via MCP protocol
"""

import asyncio
import hashlib
import json
import os
import sys
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

# --- top-up indexing ---------------------------------------------------------
# Smart Connections only embeds while Obsidian is running. Everything an agent
# session writes to the vault - daily logs, hook output, autolinked backlinks -
# stays invisible to search until Obsidian next opens. Measured 2026-08-08:
# 187 of 955 notes stale, 5 absent, the index 4.2 hours behind, and a note
# authored that same evening could not be retrieved at all.
#
# So we embed the gap ourselves. Nothing is ever written into .smart-env -
# Obsidian remains the sole owner of its own store. These vectors live in a
# separate cache and are merged at query time; once Obsidian catches up, its
# vector wins and the top-up row is dropped.
TOPUP_ENABLED = os.getenv("SMART_CONNECTIONS_TOPUP", "1") != "0"
TOPUP_MAX_NOTES = int(os.getenv("SMART_CONNECTIONS_TOPUP_MAX", "400"))
# Per-run ceiling, expressed in SECONDS rather than chunks. The first query of
# a session pays it, so what matters is wall clock, and encode speed varies by
# almost 5x across the models Smart Connections offers - bge-micro-v2 runs
# ~21ms/chunk, arctic-embed-s ~118ms. A fixed chunk count silently becomes a
# 5x longer stall when the model changes; a time budget holds the latency
# ceiling no matter which model is selected, with no knob to re-tune.
#
# Targets are taken NEWEST FIRST, because the notes an agent session needs are
# the ones just written. Clearing a large backlog is what `--reindex` is for.
TOPUP_TIME_BUDGET_SECONDS = float(os.getenv("SMART_CONNECTIONS_TOPUP_SECONDS", "12"))
TOPUP_ENCODE_BATCH = int(os.getenv("SMART_CONNECTIONS_TOPUP_BATCH", "64"))
TOPUP_ENCODE_THREADS = int(os.getenv("SMART_CONNECTIONS_TOPUP_THREADS", "4"))
# How often a query may re-check the vault for edits. Indexing once at process
# start is not enough: an agent session writes notes and then asks about them
# in the same session, and load_embeddings() runs exactly once. Verified
# 2026-08-09 - a note written 30 seconds earlier was unfindable.
#
# The check is a 955-file stat walk at ~28ms, so it is throttled rather than
# run per query. Nothing changed means an early return before the ~127ms
# matrix rebuild, so the steady-state cost of a query is unchanged.
TOPUP_RECHECK_SECONDS = float(os.getenv("SMART_CONNECTIONS_TOPUP_RECHECK", "5"))
# How often a pass that deferred work may be retried. The stale set cannot
# change on its own - top-up never writes embed_at, that is Obsidian's field -
# so a backlog left by the time budget would otherwise sit behind the signature
# guard until some unrelated note is edited, i.e. never drain inside a session.
# Retrying it on the 5s edit cadence would instead put the whole time budget
# back on the latency path of every query, so backlog retries get their own,
# slower clock. Edits are unaffected: a new mtime changes the signature.
TOPUP_BACKLOG_RETRY_SECONDS = float(os.getenv("SMART_CONNECTIONS_TOPUP_RETRY", "60"))
TOPUP_MAX_CHARS = 2000   # bge-micro-v2 truncates near 512 tokens; this clears it
TOPUP_MIN_CHARS = 50     # below this a chunk carries no retrievable signal
TOPUP_MAX_CHUNKS = 80    # per note, so one huge daily log cannot dominate a run
TOPUP_SKIP_DIRS = {".git", ".obsidian", ".smart-env", ".trash"}


class SmartConnectionsDatabase:
    """Interface to Smart Connections .smart-env vector database"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.smart_env_path = self.vault_path / ".smart-env"
        self.multi_path = self.smart_env_path / "multi"

        # Lazy load embedding model. The name is READ FROM Smart Connections'
        # own config rather than hardcoded: our top-up vectors have to live in
        # the same space as the plugin's, so the two must never diverge.
        #
        # This is also the upgrade path. bge-micro-v2 is the smallest BGE
        # variant (384-dim) and it is the ceiling on retrieval precision here -
        # measured 2026-08-08, a question answered verbatim in a note still
        # ranked ~500th. Switching the model in Obsidian's Smart Connections
        # settings and letting it re-embed is what raises that ceiling; this
        # server then follows automatically.
        self.model = None
        self.model_name = self._configured_model()

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
        # vault-relative path -> epoch seconds Smart Connections last embedded it
        self.embed_at: Dict[str, float] = {}
        self._last_check = 0.0
        self._last_signature = None
        self._last_drain = 0.0     # when a top-up pass last ran
        self._deferred = False     # last pass left notes un-encoded
        self.topup_added = 0      # notes embedded here because SC was behind
        self.topup_superseded = 0 # stale SC rows dropped in favour of a top-up
        self.topup_skipped = 0    # over TOPUP_MAX_NOTES this run

        cache_root = DEFAULT_CACHE_DIR
        vault_id = hashlib.sha256(str(self.vault_path.resolve()).encode()).hexdigest()[:16]
        self.cache_dir = cache_root / vault_id

    DEFAULT_MODEL = 'TaylorAI/bge-micro-v2'

    # Asymmetric retrieval models expect a fixed instruction on the QUERY side
    # and (sometimes) a different one on the document side. The prefix is
    # applied by the host application, not baked into the model, so it has to
    # be reproduced here or the two vector spaces do not line up.
    #
    # This is not a tuning detail. Benchmarked on this vault, arctic-embed-s
    # scored MRR 0.760 with its prefix and 0.215 without - three and a half
    # times WORSE than the bge-micro-v2 it would be replacing. Selecting an
    # asymmetric model without this table is a downgrade, not an upgrade.
    #
    # Values mirror `transformers_models` in the Smart Connections plugin, so
    # our embeddings stay in the same space as the ones it writes.
    SEMANTIC_PROFILES = {
        'TaylorAI/bge-micro-v2': {'query_prefix': '', 'document_prefix': ''},
        'Snowflake/snowflake-arctic-embed-s': {
            'query_prefix': 'Represent this sentence for searching relevant passages: ',
            'document_prefix': '',
        },
        'Snowflake/snowflake-arctic-embed-xs': {
            'query_prefix': 'Represent this sentence for searching relevant passages: ',
            'document_prefix': '',
        },
        # Loadable only via the canonical repo; the Xenova ONNX port that the
        # plugin lists cannot be opened by sentence-transformers.
        'intfloat/multilingual-e5-small': {
            'query_prefix': 'query: ', 'document_prefix': 'passage: ',
        },
    }

    def _profile(self) -> dict:
        return self.SEMANTIC_PROFILES.get(
            self.model_name, {'query_prefix': '', 'document_prefix': ''}
        )

    def _configured_model(self) -> str:
        """Model Smart Connections is actually using, from its own settings."""
        override = os.getenv('SMART_CONNECTIONS_MODEL')
        if override:
            return override
        try:
            with open(self.smart_env_path / 'smart_env.json', 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            key = (((cfg.get('smart_sources') or {}).get('embed_model') or {})
                   .get('transformers') or {}).get('model_key')
            if isinstance(key, str) and key.strip():
                return key.strip()
        except Exception:
            pass
        return self.DEFAULT_MODEL

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
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                # Several models the plugin offers are ONNX-only ports
                # (Xenova/*, onnx-community/*). Smart Connections runs them
                # through transformers.js; sentence-transformers cannot open
                # them at all. Say which model and what it costs, rather than
                # surfacing a bare OSError from deep in the loader.
                raise RuntimeError(
                    f"cannot load embedding model '{self.model_name}'. Smart "
                    f"Connections can use ONNX-only repos (Xenova/*, "
                    f"onnx-community/*) but this server cannot, so semantic "
                    f"search and top-up indexing are unavailable while it is "
                    f"selected. Pick a model with PyTorch weights - "
                    f"TaylorAI/bge-micro-v2 or Snowflake/snowflake-arctic-embed-s - "
                    f"or set SMART_CONNECTIONS_MODEL to an equivalent "
                    f"non-ONNX repo. Underlying error: {type(e).__name__}: {e}"
                ) from e

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
        self.embed_at = meta.get("embed_at", {})
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
                        "embed_at": self.embed_at,
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
            self._apply_topup()
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
                resolved_path = self._resolve_path(key, item)
                paths.append(resolved_path)
                lines.append(item.get("lines"))

                # Note-level embed time, used to detect what Obsidian has not
                # caught up on yet. Only source rows carry a trustworthy one.
                if key.startswith("smart_sources:") and resolved_path:
                    at = (model_entry.get("last_embed") or {}).get("at") \
                        or (item.get("last_embed") or {}).get("at")
                    if isinstance(at, (int, float)):
                        self.embed_at[resolved_path] = at / 1000.0

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
        self._apply_topup()

    # ------------------------------------------------------------------
    # Top-up: cover what Obsidian has not embedded yet
    # ------------------------------------------------------------------

    def _stale_or_missing(self) -> List[tuple]:
        """(path, mtime) for notes whose content is newer than SC's vector.

        Runs on a throttle before every search, so it is on the latency path.
        os.scandir is used rather than rglob + .stat(): DirEntry carries the
        stat result from the directory read itself, halving the syscalls -
        measured 28ms to 13ms across 955 files.
        """
        out = []
        stack = [(str(self.vault_path), "")]
        while stack:
            abs_dir, rel_dir = stack.pop()
            try:
                entries = list(os.scandir(abs_dir))
            except OSError:
                continue
            for entry in entries:
                if entry.name in TOPUP_SKIP_DIRS:
                    continue
                rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((entry.path, rel))
                        continue
                    if not entry.name.endswith(".md"):
                        continue
                    mtime = entry.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                embedded = self.embed_at.get(rel)
                # 2s slack absorbs filesystem/clock granularity between the two.
                if embedded is None or mtime > embedded + 2:
                    out.append((rel, mtime))
        return out

    def _refresh_if_stale(self):
        """Pick up vault edits made since this process started.

        Called before every search. Two guards keep it off the hot path: a
        wall-clock throttle, and a signature over the stale set so an unchanged
        vault returns before the matrix rebuild.
        """
        if not TOPUP_ENABLED or not self.embeddings_loaded:
            return
        now = time.time()
        if now - self._last_check < TOPUP_RECHECK_SECONDS:
            return
        self._last_check = now

        targets = self._stale_or_missing()
        signature = hashlib.sha256(
            "|".join(f"{r}:{m:.3f}" for r, m in sorted(targets)).encode()
        ).hexdigest()
        if signature == self._last_signature:
            # Vault unchanged since the last pass. Return - unless that pass
            # deferred work, in which case the signature will NEVER change on
            # its own: the stale set is derived from Obsidian's embed_at, which
            # top-up does not write. Recording the signature after an
            # incomplete pass froze the retry outright; retrying it on this 5s
            # clock would put the full time budget on every query instead. So
            # the backlog gets its own slower cadence.
            if (not self._deferred
                    or now - self._last_drain < TOPUP_BACKLOG_RETRY_SECONDS):
                return
        self._last_signature = signature
        self._last_drain = now
        self._apply_topup(targets=targets)
        self._deferred = bool(self.topup_skipped)

    def _chunk_note(self, rel: str) -> List[tuple]:
        """Split a note into heading-scoped chunks: (key_suffix, text, [start, end]).

        Chunking is not a refinement here, it is the whole point. A single
        whole-note vector loses to Smart Connections' own heading blocks every
        time, and truncating a long note to fit the encoder silently drops
        whatever sits past the cut. Measured 2026-08-08: the vault CLAUDE.md
        embedded whole scored 0.533 and ranked 21,796th for a question its own
        text answers, because the answer lived 4,000 characters in - past the
        truncation point. Per-heading chunks fix both problems at once.

        Mirrors Smart Connections' own `path#Heading` key shape so the rows
        merge into the block mask and read back through the same line-range
        path as native blocks.
        """
        try:
            raw = (self.vault_path / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        lines = raw.split("\n")
        start_idx = 0
        # Skip frontmatter: tags and ids are not what anyone searches for.
        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    start_idx = i + 1
                    break

        chunks: List[tuple] = []
        heading = ""
        buf: List[str] = []
        buf_start = start_idx + 1
        fence = False

        def flush(end_line: int):
            if not buf:
                return
            text = "\n".join(buf).strip()
            if len(text) < TOPUP_MIN_CHARS:
                return
            # Prepend the heading so a chunk carries its own topic.
            body = (f"{heading}\n{text}" if heading else text)[:TOPUP_MAX_CHARS]
            chunks.append((heading, body, [buf_start, end_line]))

        for i in range(start_idx, len(lines)):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = not fence
            # A heading inside a fence is code, not structure.
            if not fence and stripped.startswith("#") and " " in stripped[:8]:
                flush(i)
                if len(chunks) >= TOPUP_MAX_CHUNKS:
                    return chunks
                heading = stripped.lstrip("#").strip()
                buf = []
                buf_start = i + 1
                continue
            buf.append(line)
        flush(len(lines))
        return chunks[:TOPUP_MAX_CHUNKS]

    @staticmethod
    def _with_source_chunk(chunks: List[dict]) -> List[dict]:
        """Guarantee one note-level (`smart_sources:`) row per topped-up note.

        Heading-scoped chunking only produces an empty-heading chunk when a
        note opens with prose. Notes here open with frontmatter and then their
        H1, so nearly every one yields blocks and nothing else - and since a
        topped-up note supersedes its ENTIRE Smart Connections representation,
        that dropped the source row Obsidian had written. find_related both
        anchors on that row and ranks only source rows, so a topped-up note
        became invisible to it in both directions; index_stats' source count
        fell with it. Verified 2026-08-09: CLAUDE.md, README.md, _MOC.md and a
        daily note all chunk to 0 empty-heading chunks.

        The note vector is pooled from the chunk vectors rather than encoded a
        second time. That costs nothing, and it avoids re-imposing the encoder
        truncation that per-heading chunking exists to escape - a whole-note
        encode of this file scored 0.533 at rank 21,796 for a question its own
        text answers. Pooled unit vectors stay in the same space, so the row
        remains comparable with Obsidian's own source rows.
        """
        if not chunks or any((c.get("heading") or "") == "" for c in chunks):
            return chunks
        pooled = np.mean(
            [np.asarray(c["vec"], dtype=np.float32) for c in chunks], axis=0
        )
        n = float(np.linalg.norm(pooled))
        if not np.isfinite(n) or n == 0.0:
            return chunks
        # lines None: _read_text falls back to the whole note, which is what a
        # note-level row should return.
        return chunks + [{"heading": "", "lines": None, "vec": pooled / n}]

    def _load_topup_cache(self):
        """Return (index, matrix). Vectors live in .npy, never in the JSON.

        Storing them as JSON lists cost 8.7s on warm start - deserializing
        ~4k x 384 floats through Python objects. mmap'd .npy makes the same
        load ~0.1s, which is the whole point of caching them.
        """
        idx_p = self.cache_dir / "topup.json"
        vec_p = self.cache_dir / "topup.npy"
        if not idx_p.exists() or not vec_p.exists():
            return {}, None
        try:
            with open(idx_p, "r", encoding="utf-8") as f:
                index = json.load(f)
            if not isinstance(index, dict):
                return {}, None
            matrix = np.load(vec_p, mmap_mode="r")
            if index.get("_model") != self.model_name:
                return {}, None   # model changed: the vector space did too
            return index.get("notes", {}), matrix
        except Exception:
            return {}, None

    def _save_topup_cache(self, notes: Dict[str, dict], matrix):
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_vec = self.cache_dir / "topup.tmp.npy"
            tmp_idx = self.cache_dir / "topup.json.tmp"
            with open(tmp_vec, "wb") as f:
                np.save(f, matrix)
            with open(tmp_idx, "w", encoding="utf-8") as f:
                json.dump({"_model": self.model_name, "notes": notes}, f)
            tmp_vec.replace(self.cache_dir / "topup.npy")
            tmp_idx.replace(self.cache_dir / "topup.json")
        except Exception:
            pass

    def _apply_topup(self, unlimited: bool = False, targets=None):
        """Embed stale/missing notes and merge them into the live matrix.

        Note-level only. A stale note's *block* vectors stay stale, so its
        top-up row restores whole-note recall but not per-heading precision -
        that is Obsidian's job when it next runs. Stated plainly rather than
        implied: this closes the "cannot find it at all" gap, not every gap.
        """
        if not TOPUP_ENABLED:
            return
        # Reset before the early return, not after it: the counters describe
        # THIS pass, and _refresh_if_stale reads topup_skipped to decide
        # whether work is still outstanding. Leaving a previous pass's value
        # standing after a no-op would report a backlog that no longer exists.
        self.topup_added = self.topup_superseded = self.topup_skipped = 0
        if targets is None:
            targets = self._stale_or_missing()
        if not targets:
            return

        cached_notes, cached_matrix = self._load_topup_cache()
        pending, done = [], {}
        for rel, mtime in targets:
            hit = cached_notes.get(rel)
            if (hit and cached_matrix is not None
                    and abs(hit.get("mtime", -1) - mtime) < 0.001 and hit.get("chunks")):
                rows = [c.get("row") for c in hit["chunks"]]
                if all(isinstance(r, int) and 0 <= r < cached_matrix.shape[0] for r in rows):
                    done[rel] = {
                        "mtime": mtime,
                        "chunks": [
                            {"heading": c.get("heading"), "lines": c.get("lines"),
                             "vec": np.asarray(cached_matrix[c["row"]], dtype=np.float32)}
                            for c in hit["chunks"]
                        ],
                    }
                    continue
            pending.append((rel, mtime))

        # Newest first: the note written five minutes ago is the one this
        # session will be asked about.
        pending.sort(key=lambda x: x[1], reverse=True)
        if not unlimited and len(pending) > TOPUP_MAX_NOTES:
            self.topup_skipped += len(pending) - TOPUP_MAX_NOTES
            pending = pending[:TOPUP_MAX_NOTES]

        if pending:
            doc_prefix = self._profile()['document_prefix']
            groups = []
            for rel, mtime in pending:
                chunks = self._chunk_note(rel)
                if chunks:
                    groups.append((rel, mtime, chunks))

            owners, vec_parts = [], []
            if groups:
                try:
                    self.ensure_model_loaded()
                except RuntimeError as e:
                    # Top-up is an enhancement; Smart Connections' own vectors
                    # still work. Degrade to those rather than take search down,
                    # but say so - a silent fallback here is the failure mode
                    # this whole server has been fixing all night.
                    print(f"WARNING: top-up indexing disabled - {e}", file=sys.stderr)
                    return
                import torch
                # Batch encoding is throughput-bound, unlike the single-query
                # path the 1-thread default was chosen for. Measured on 8 cores:
                # 30.1ms/chunk at 1 thread, 20.8 at 4, 26.1 at 8.
                torch.set_num_threads(max(1, TOPUP_ENCODE_THREADS))
                deadline = None if unlimited else time.time() + TOPUP_TIME_BUDGET_SECONDS
                i = 0
                try:
                    while i < len(groups):
                        if deadline is not None and time.time() > deadline:
                            self.topup_skipped += len(groups) - i
                            break
                        # Fill a batch whole-notes-at-a-time. A note is never
                        # split across the deadline, because a half-embedded
                        # note would be cached as complete and never revisited.
                        batch_texts, batch_owners = [], []
                        while i < len(groups) and len(batch_texts) < TOPUP_ENCODE_BATCH:
                            rel, mtime, chunks = groups[i]
                            for heading, body, span in chunks:
                                batch_texts.append(doc_prefix + body)
                                batch_owners.append((rel, mtime, heading, span))
                            i += 1
                        vec_parts.append(np.asarray(
                            self.model.encode(batch_texts, batch_size=32,
                                              show_progress_bar=False),
                            dtype=np.float32,
                        ))
                        owners.extend(batch_owners)
                finally:
                    torch.set_num_threads(1)

            if owners:
                vecs = np.vstack(vec_parts)
                for (rel, mtime, heading, span), vec in zip(owners, vecs):
                    v = np.asarray(vec, dtype=np.float32).reshape(-1)
                    n = float(np.linalg.norm(v))
                    if not np.isfinite(n) or n == 0.0:
                        continue
                    entry = done.setdefault(rel, {"mtime": mtime, "chunks": []})
                    entry["chunks"].append(
                        {"heading": heading, "lines": span, "vec": v / n}
                    )

        if not done:
            return

        # A topped-up note replaces its ENTIRE Smart Connections representation -
        # the source row and every stale block row. Keeping the old blocks would
        # let outdated text outrank the current text of the same note, which is
        # the exact failure this whole feature exists to remove.
        touched = set(done)
        keep = [
            i for i, k in enumerate(self.keys)
            if not (self.paths[i] in touched
                    and (k.startswith("smart_sources:") or k.startswith("smart_blocks:")))
        ]
        self.topup_superseded = len(self.keys) - len(keep)

        add_keys, add_paths, add_lines, add_vecs = [], [], [], []
        new_index: Dict[str, dict] = {}
        for rel, entry in done.items():
            rows = []
            chunks = self._with_source_chunk(entry["chunks"])
            for c in chunks:
                h = c.get("heading") or ""
                add_keys.append(f"smart_blocks:{rel}#{h}" if h else f"smart_sources:{rel}")
                add_paths.append(rel)
                add_lines.append(c.get("lines"))
                rows.append({"heading": h, "lines": c.get("lines"), "row": len(add_vecs)})
                add_vecs.append(np.asarray(c["vec"], dtype=np.float32))
            new_index[rel] = {"mtime": entry["mtime"], "chunks": rows}
        if not add_vecs:
            return

        add_rows = np.stack(add_vecs).astype(np.float32, copy=False)
        base = np.asarray(self.matrix)[keep]
        if base.shape[0] and base.shape[1] != add_rows.shape[1]:
            return  # dimension mismatch: leave the base index untouched

        self.matrix = np.vstack([base, add_rows]) if base.shape[0] else add_rows
        self.keys = [self.keys[i] for i in keep] + add_keys
        self.paths = [self.paths[i] for i in keep] + add_paths
        self.lines = [self.lines[i] for i in keep] + add_lines
        self.topup_added = len(add_keys)
        self._finalize_masks()
        # Row indices in new_index are positions in add_rows, so the two are
        # written together and stay consistent by construction.
        self._save_topup_cache(new_index, add_rows)

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
        query = self._profile()['query_prefix'] + query
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
        self._refresh_if_stale()
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
        self._refresh_if_stale()
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
        self._refresh_if_stale()
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
        self._refresh_if_stale()
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
            'topup': {
                'enabled': TOPUP_ENABLED,
                'notes_added': self.topup_added,
                'stale_rows_superseded': self.topup_superseded,
                'deferred_over_cap': self.topup_skipped,
                'note': ('Notes Obsidian has not embedded yet, re-chunked and indexed '
                         'here per heading. Obsidian remains the primary indexer; '
                         'nothing is written into .smart-env.'),
            },
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


def reindex_cli() -> int:
    """Clear the whole top-up backlog in one pass, without a per-run cap.

    The MCP path deliberately caps work so the first query of a session stays
    responsive. That means a large backlog - the state after Obsidian has been
    closed for a while - drains over several sessions. Run this once to drain
    it now:

        OBSIDIAN_VAULT_PATH=... python3 server.py --reindex
    """
    vault = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault:
        print("OBSIDIAN_VAULT_PATH is not set", file=sys.stderr)
        return 1

    db = SmartConnectionsDatabase(vault)
    start = time.time()
    db.load_embeddings()          # bounded pass, plus whatever the cache holds
    db.embeddings_loaded = True
    db._apply_topup(unlimited=True)   # then drain the remainder

    s = db.stats()
    t = s["topup"]
    print(f"vectors        : {s['vectors']}")
    print(f"chunks added   : {t['notes_added']}")
    print(f"stale dropped  : {t['stale_rows_superseded']}")
    print(f"deferred       : {t['deferred_over_cap']}")
    print(f"elapsed        : {time.time() - start:.1f}s")
    remaining = len(db._stale_or_missing())
    print(f"still stale    : {remaining}"
          + ("  <- all covered by the top-up cache" if remaining else ""))
    return 0


if __name__ == "__main__":
    if "--reindex" in sys.argv:
        sys.exit(reindex_cli())
    asyncio.run(main())
