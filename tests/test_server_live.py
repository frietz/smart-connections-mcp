#!/usr/bin/env python3
"""Integration tests against a real vault and a real embedding model.

Skipped entirely when no store is present, so the hermetic suite still runs on
a machine without Obsidian. Set OBSIDIAN_VAULT_PATH to point at one.

These are slow (the model loads once) and they read a live store that Obsidian
may be writing, so they assert on invariants rather than on counts.

    OBSIDIAN_VAULT_PATH=~/obsidian/vault-obsidian \\
      ~/smart-connections-mcp/.venv/bin/python -m unittest discover -s tests
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import SmartConnectionsDatabase as DB

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
VAULT = Path(os.path.expanduser(
    os.getenv("OBSIDIAN_VAULT_PATH", "~/obsidian/vault-obsidian")))


def store_present():
    env = VAULT / ".smart-env"
    return (env / "smart_sources" / "smart_sources.ajson").exists() or \
           (env / "multi").is_dir()


PRESENT = store_present()
# Skipping the entire live suite and exiting 0 is the same shape of lie the
# tests this replaced told: green because nothing ran. run-tests.sh sets this
# whenever live tests were actually asked for, so an absent store is a failure
# rather than eighteen quiet skips. Set SCMCP_ALLOW_NO_STORE=1 to opt out.
REQUIRED = os.getenv("SCMCP_REQUIRE_LIVE") == "1"

needs_store = unittest.skipUnless(
    PRESENT, f"no Smart Connections store under {VAULT}")


class StoreAvailability(unittest.TestCase):
    def test_the_live_store_is_present_when_live_tests_were_requested(self):
        if not REQUIRED:
            self.skipTest("live tests not required (SCMCP_REQUIRE_LIVE unset)")
        self.assertTrue(
            PRESENT,
            f"live tests were requested but no Smart Connections store exists "
            f"under {VAULT}. Every LiveStore and MCPProtocol test would skip "
            f"and the run would report success having verified nothing. Set "
            f"OBSIDIAN_VAULT_PATH, or SCMCP_ALLOW_NO_STORE=1 to accept it.")


@needs_store
class LiveStore(unittest.TestCase):
    """One database for the class - the model load is the expensive part."""

    @classmethod
    def setUpClass(cls):
        cls.db = DB(str(VAULT))
        cls.db.load_embeddings()

    def test_the_index_is_not_empty(self):
        self.assertIsNotNone(self.db.matrix)
        self.assertGreater(self.db.matrix.shape[0], 0)
        self.assertGreater(self.db.matrix.shape[1], 0)

    def test_row_lists_stay_the_same_length_as_the_matrix(self):
        n = self.db.matrix.shape[0]
        self.assertEqual(len(self.db.keys), n)
        self.assertEqual(len(self.db.paths), n)
        self.assertEqual(len(self.db.lines), n)
        self.assertEqual(self.db.is_source.shape[0], n)
        self.assertEqual(self.db.is_block.shape[0], n)

    def test_every_row_is_a_unit_vector(self):
        import numpy as np
        step = max(1, self.db.matrix.shape[0] // 200)
        norms = np.linalg.norm(np.asarray(self.db.matrix[::step]), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    def test_no_indexed_path_is_missing_from_disk(self):
        # A note deleted since load used to keep its vectors and surface as a
        # result with a score and no text.
        missing = [p for p in self.db.path_set if not (VAULT / p).exists()]
        self.assertEqual(missing, [], f"indexed but absent: {missing[:5]}")

    def test_search_returns_scored_results_in_range(self):
        results = self.db.semantic_search("what did we decide about caching",
                                          limit=5, min_similarity=0.3)
        self.assertTrue(results)
        for r in results:
            self.assertGreaterEqual(r["similarity"], 0.3)
            self.assertLessEqual(r["similarity"], 1.0001)
            self.assertTrue(r["key"])

    def test_search_honours_its_limit(self):
        self.assertLessEqual(len(self.db.semantic_search("note", limit=3)), 3)

    def test_context_blocks_carry_real_text(self):
        # Smart Connections stores no note text, only paths and line ranges;
        # this used to return "" for every block.
        blocks = self.db.get_context_blocks("vault", max_blocks=3)
        self.assertTrue(blocks)
        for b in blocks:
            self.assertTrue(b["text"].strip(), f"empty block text for {b['path']}")

    def test_find_related_works_on_a_note_that_topup_may_have_superseded(self):
        # An H1-first note chunks to blocks only, so top-up used to strip the
        # source row find_related both anchors on and ranks by.
        target = "CLAUDE.md" if (VAULT / "CLAUDE.md").exists() else None
        if target is None:
            self.skipTest("no CLAUDE.md in this vault")
        related = self.db.find_related(target, limit=5)
        self.assertTrue(related, "find_related lost the note's source row")
        self.assertNotIn(target, [r["path"] for r in related],
                         "a note must not be its own relation")

    def test_find_related_on_an_unknown_path_is_empty_not_an_error(self):
        self.assertEqual(self.db.find_related("no/such/note.md"), [])

    def test_stats_describe_the_matrix_it_just_built(self):
        s = self.db.stats()
        self.assertEqual(s["vectors"], self.db.matrix.shape[0])
        self.assertEqual(s["dimensions"], self.db.matrix.shape[1])
        self.assertIn("rows_added", s["topup"])
        self.assertIn("notes_added", s["topup"])

    def test_the_query_prefix_matches_the_identified_model(self):
        # Selecting an asymmetric model without its prefix is a downgrade, not
        # an upgrade: measured MRR 0.215 against bge's 0.641 on this vault.
        profile = self.db._profile()
        self.assertEqual(set(profile), {"query_prefix", "document_prefix"})
        if "arctic" in self.db.model_name:
            self.assertTrue(profile["query_prefix"],
                            f"{self.db.model_name} needs a query prefix")

    def test_a_second_load_is_served_from_cache(self):
        again = DB(str(VAULT))
        again.load_embeddings()
        self.assertEqual(again.matrix.shape, self.db.matrix.shape)
        self.assertEqual(again.model_name, self.db.model_name)


@needs_store
@unittest.skipUnless(VENV_PY.exists(), "no .venv interpreter")
class MCPProtocol(unittest.TestCase):
    """All four tools over the real MCP protocol, in a separate process.

    The in-repo tests this replaced drove a hand-rolled JSON-RPC dialogue at a
    hardcoded path on the original author's machine, asserted nothing, and
    printed success after finding zero results.
    """

    @classmethod
    def setUpClass(cls):
        cls.out = cls._run_client()

    @staticmethod
    def _run_client():
        script = f'''
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command={str(VENV_PY)!r}, args=[{str(REPO / "server.py")!r}],
        env={{**os.environ, "OBSIDIAN_VAULT_PATH": {str(VAULT)!r}}})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            out = {{"tools": [t.name for t in (await s.list_tools()).tools]}}
            for name, args in (
                ("index_stats", {{}}),
                ("semantic_search", {{"query": "vault", "limit": 3}}),
                ("get_context_blocks", {{"query": "vault", "max_blocks": 2}}),
                ("find_related", {{"file_path": "CLAUDE.md", "limit": 3}}),
                ("semantic_search", {{"query": "vault", "limit": 99999}}),
            ):
                res = await s.call_tool(name, args)
                out.setdefault(name, []).append(json.loads(res.content[0].text))
    print("@@RESULT@@" + json.dumps(out))

asyncio.run(main())
'''
        proc = subprocess.run([str(VENV_PY), "-c", script],
                              capture_output=True, text=True, timeout=900,
                              env={**os.environ,
                                   "OBSIDIAN_VAULT_PATH": str(VAULT)})
        marker = proc.stdout.find("@@RESULT@@")
        if marker < 0:
            raise AssertionError(
                f"client produced no result (rc={proc.returncode})\n"
                f"stdout tail: {proc.stdout[-800:]}\n"
                f"stderr tail: {proc.stderr[-800:]}")
        return json.loads(proc.stdout[marker + len("@@RESULT@@"):].strip())

    def test_all_four_tools_are_advertised(self):
        self.assertLessEqual(
            {"semantic_search", "find_related", "get_context_blocks", "index_stats"},
            set(self.out["tools"]))

    def test_index_stats_reports_a_populated_index(self):
        st = self.out["index_stats"][0]
        self.assertGreater(st["vectors"], 0)
        self.assertGreater(st["dimensions"], 0)

    def test_search_over_the_protocol_returns_results_in_range(self):
        res = self.out["semantic_search"][0]
        self.assertGreater(res["results_count"], 0)
        for r in res["results"]:
            self.assertTrue(0.0 <= r["similarity"] <= 1.0001)

    def test_blocks_over_the_protocol_carry_text(self):
        blocks = self.out["get_context_blocks"][0]["blocks"]
        self.assertTrue(blocks)
        self.assertTrue(all(b.get("text") for b in blocks))

    def test_find_related_over_the_protocol_is_not_empty(self):
        self.assertGreater(self.out["find_related"][0]["related_count"], 0)

    def test_an_oversized_limit_is_clamped(self):
        # Results are serialized into the model's context, so an unbounded
        # limit is a context hazard rather than a slow response.
        self.assertLessEqual(self.out["semantic_search"][1]["results_count"], 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
