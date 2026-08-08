#!/usr/bin/env python3
"""Hermetic tests: no vault, no embedding model, no network.

Every case here is a regression that actually happened. The comment on each
names it, so a future change that reintroduces one fails with a description of
the bug rather than an assertion number.

    ~/smart-connections-mcp/.venv/bin/python -m unittest discover -s tests
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server
from server import SmartConnectionsDatabase as DB


def unit_rows(n, dim=384, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def write_blob(path, rows):
    path.write_bytes(np.asarray(rows, dtype="<f4").tobytes())


def bare_db(tmp):
    """A database with no store behind it, for pure-logic calls."""
    db = DB(str(tmp / "no-such-vault"))
    db.cache_dir = tmp / "cache"
    return db


class BlobGeometry(unittest.TestCase):
    """_open_blob must PROVE a shape, never infer one.

    Two failures, in opposite directions. A blob whose row count the metadata
    under-reports used to fail exact division and return None, which took the
    whole modern load down to an empty index. And among divisors that pass an
    exact-division check, several give a legal-looking dimension whose rows are
    windows straddling two real vectors - they normalize fine and rank like
    retrieval.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-blob-"))
        self.db = bare_db(self.tmp)

    def test_exact_metadata_opens_at_the_true_dim(self):
        rows = unit_rows(50)
        p = self.tmp / "mf_exact"
        write_blob(p, rows)
        mat = self.db._open_blob(p, 50)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.shape, (50, 384))
        np.testing.assert_allclose(np.asarray(mat[0]), rows[0], rtol=1e-6)

    def test_blob_ahead_of_metadata_still_opens(self):
        # The plugin writes metadata and vectors separately, so the blob is
        # routinely a few rows ahead. Refusing here empties the whole index.
        rows = unit_rows(53)
        p = self.tmp / "mf_ahead"
        write_blob(p, rows)
        mat = self.db._open_blob(p, 50)
        self.assertIsNotNone(mat, "a blob ahead of its metadata must still open")
        self.assertEqual(mat.shape[1], 384)
        np.testing.assert_allclose(np.asarray(mat[0]), rows[0], rtol=1e-6)

    def test_wrong_row_count_does_not_produce_a_misaligned_window(self):
        rows = unit_rows(64)
        p = self.tmp / "mf_factor"
        write_blob(p, rows)
        # 64*384 floats also divides as 32 rows of 768 - a legal-looking shape
        # whose rows straddle two real vectors.
        mat = self.db._open_blob(p, 32)
        self.assertIsNotNone(mat)
        self.assertEqual(mat.shape[1], 384,
                         "must land on the true dimension, not a factorization")

    def test_unnormalized_blob_is_refused(self):
        rng = np.random.default_rng(1)
        p = self.tmp / "mf_raw"
        write_blob(p, rng.standard_normal((20, 384)) * 7.0)
        self.assertIsNone(self.db._open_blob(p, 20),
                          "vectors that are not unit length prove nothing")

    def test_truncated_and_empty_blobs_are_refused(self):
        p = self.tmp / "mf_odd"
        p.write_bytes(b"\x00" * 13)          # not a whole number of float32s
        self.assertIsNone(self.db._open_blob(p, 1))
        empty = self.tmp / "mf_empty"
        empty.write_bytes(b"")
        self.assertIsNone(self.db._open_blob(empty, 1))
        self.assertIsNone(self.db._open_blob(self.tmp / "mf_missing", 1))


class SourceRowSynthesis(unittest.TestCase):
    """Every topped-up note needs a note-level row.

    Heading-scoped chunking only emits an empty-heading chunk when a note opens
    with prose. Notes here open with frontmatter then an H1, so top-up produced
    blocks and nothing else - and since a topped-up note supersedes its entire
    prior representation, find_related lost the note in both directions: it
    anchors on the source row AND ranks only source rows.
    """

    def test_pooled_row_is_added_when_no_chunk_is_note_level(self):
        chunks = [{"heading": "A", "lines": [1, 5], "vec": unit_rows(1, seed=1)[0]},
                  {"heading": "B", "lines": [6, 9], "vec": unit_rows(1, seed=2)[0]}]
        out = DB._with_source_chunk(chunks)
        self.assertEqual(len(out), 3)
        synthesized = out[-1]
        self.assertEqual(synthesized["heading"], "")
        self.assertIsNone(synthesized["lines"], "a note-level row spans the note")
        self.assertAlmostEqual(float(np.linalg.norm(synthesized["vec"])), 1.0, places=5)

    def test_existing_note_level_chunk_is_not_duplicated(self):
        # Round-trips through the top-up cache would otherwise add one per pass.
        chunks = [{"heading": "", "lines": [1, 5], "vec": unit_rows(1, seed=3)[0]},
                  {"heading": "B", "lines": [6, 9], "vec": unit_rows(1, seed=4)[0]}]
        self.assertEqual(len(DB._with_source_chunk(chunks)), 2)

    def test_pooled_row_of_one_chunk_is_that_chunk(self):
        v = unit_rows(1, seed=5)[0]
        out = DB._with_source_chunk([{"heading": "A", "lines": [1, 2], "vec": v}])
        np.testing.assert_allclose(out[-1]["vec"], v, rtol=1e-5)

    def test_degenerate_input_is_passed_through(self):
        self.assertEqual(DB._with_source_chunk([]), [])
        z = np.zeros(384, dtype=np.float32)
        chunks = [{"heading": "A", "lines": None, "vec": z},
                  {"heading": "B", "lines": None, "vec": -z}]
        self.assertEqual(len(DB._with_source_chunk(chunks)), 2,
                         "a zero pooled vector must not be added")


class Chunking(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-chunk-"))
        self.db = DB(str(self.tmp))

    def write(self, name, text):
        (self.tmp / name).write_text(text, encoding="utf-8")
        return name

    def test_h1_first_note_yields_no_note_level_chunk(self):
        # This is why source-row synthesis exists; if this ever stops being
        # true the synthesis is still correct, but the reason has changed.
        rel = self.write("h1.md", "---\ntags: [x]\n---\n# Title\n\n" + "body. " * 40)
        chunks = self.db._chunk_note(rel)
        self.assertTrue(chunks)
        self.assertEqual([c for c in chunks if c[0] == ""], [])

    def test_frontmatter_is_not_embedded(self):
        rel = self.write("fm.md", "---\ntags: [secret-token-xyz]\n---\n# T\n\n" + "body. " * 40)
        self.assertNotIn("secret-token-xyz",
                         " ".join(c[1] for c in self.db._chunk_note(rel)))

    def test_headings_inside_a_fence_are_code_not_structure(self):
        body = "body. " * 40
        rel = self.write("fence.md",
                         f"# Real\n\n{body}\n\n```\n# Not A Heading\n```\n\n{body}")
        self.assertEqual([c[0] for c in self.db._chunk_note(rel)].count("Not A Heading"), 0)

    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(self.db._chunk_note("nope.md"), [])


class ModernMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-meta-"))
        self.db = bare_db(self.tmp)

    def test_later_lines_win_and_bad_lines_are_skipped(self):
        p = self.tmp / "smart_sources.ajson"
        p.write_text(
            '"smart_sources:a.md": {"path":"a.md","v":1},\n'
            '{ this is not json ,\n'
            '"smart_sources:a.md": {"path":"a.md","v":2},\n'
            '"smart_sources:b.md": {"path":"b.md","v":1},\n',
            encoding="utf-8")
        self.db.sources_ajson = p
        entries = self.db._read_modern_entries()
        self.assertEqual(entries["smart_sources:a.md"]["v"], 2)
        self.assertIn("smart_sources:b.md", entries)

    def _with_blobs(self, *names):
        self.db.sources_dir = self.tmp / "smart_sources"
        self.db.sources_dir.mkdir(exist_ok=True)
        for n in names:
            write_blob(self.db.sources_dir / n, unit_rows(2))

    def test_newest_fingerprint_wins_on_timestamp_not_order(self):
        self._with_blobs("mf_old", "mf_new")
        entries = {
            "a": {"embedding": {"default": {"mf_old": {"at": 100},
                                            "mf_new": {"at": 900}}}},
            "b": {"embedding": {"default": {"mf_old": {"at": 500}}}},
            "c": {"embedding": None},
            "d": {},
        }
        self.assertEqual(self.db._newest_fingerprint(entries), "mf_new")

    def test_a_future_timestamp_cannot_hijack_the_selection(self):
        # One corrupt or clock-skewed `at` in 33MB of metadata used to flip the
        # whole load to another space - or to one with no blob, failing the load.
        self._with_blobs("mf_old", "mf_new")
        entries = {"a": {"embedding": {"default": {
            "mf_old": {"at": 1e15}, "mf_new": {"at": 1e12}}}}}
        self.assertEqual(self.db._newest_fingerprint(entries), "mf_new")

    def test_a_space_with_no_blob_on_disk_is_not_selected(self):
        self._with_blobs("mf_present")
        entries = {"a": {"embedding": {"default": {
            "mf_present": {"at": 100}, "mf_absent": {"at": 900}}}}}
        self.assertEqual(self.db._newest_fingerprint(entries), "mf_present")

    def test_newest_fingerprint_is_none_when_nothing_is_embedded(self):
        self._with_blobs()
        self.assertIsNone(self.db._newest_fingerprint({"a": {"embedding": {"default": {}}}}))


class CacheKeying(unittest.TestCase):
    """The cache key must describe the bytes that were actually read.

    _save_to_cache used to be handed a freshly computed fingerprint, which
    re-stats the store AFTER the load - so a matrix built from state T0 was
    published under the key for T1 and read as a valid hit on the next start.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-key-"))
        self.db = bare_db(self.tmp)

    def test_a_captured_signature_reproduces_its_key(self):
        sig = "m:frozen-store-state"
        self.assertEqual(self.db._fingerprint(sig), self.db._fingerprint(sig))

    def test_key_changes_with_the_model_even_on_one_store(self):
        sig = "m:frozen-store-state"
        first = self.db._fingerprint(sig)
        self.db.model_name = "some/other-model"
        self.assertNotEqual(first, self.db._fingerprint(sig))

    def test_signature_carries_no_model_so_it_can_be_captured_early(self):
        before = self.db._store_signature()
        self.db.model_name = "some/other-model"
        self.assertEqual(before, self.db._store_signature())


class TopupCachePairing(unittest.TestCase):
    """Vectors and index are two files published by two renames.

    A reader can land between them. Row numbers alone do not catch it - they
    are validated only against the matrix bounds, so an index paired with a
    LARGER matrix passes and silently attaches other notes' vectors.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-pair-"))
        self.db = bare_db(self.tmp)
        self.db.cache_dir.mkdir(parents=True, exist_ok=True)

    def test_round_trip(self):
        rows = unit_rows(4)
        notes = {"a.md": {"mtime": 1.0, "chunks": [{"heading": "", "lines": None, "row": 0}]}}
        self.db._save_topup_cache(notes, rows)
        got, mat = self.db._load_topup_cache()
        self.assertIn("a.md", got)
        self.assertEqual(mat.shape, (4, 384))

    def test_mismatched_pair_is_rejected(self):
        self.db._save_topup_cache({"a.md": {"mtime": 1.0, "chunks": []}}, unit_rows(4))
        idx_path = self.db.cache_dir / "topup.json"
        idx = json.loads(idx_path.read_text())
        idx["_rows"] = 99                      # index from a different write
        idx_path.write_text(json.dumps(idx))
        self.assertEqual(self.db._load_topup_cache(), ({}, None))

    def test_model_change_invalidates(self):
        self.db._save_topup_cache({"a.md": {"mtime": 1.0, "chunks": []}}, unit_rows(4))
        self.db.model_name = "some/other-model"
        self.assertEqual(self.db._load_topup_cache(), ({}, None))


class DeletionHandling(unittest.TestCase):
    """A deleted note kept its vectors until the process restarted.

    Search went on returning it, and _read_text supplies "" for the missing
    file, so it surfaced as a result with a score and no content. The embed
    time has to go with the rows: kept, a note restored with its original
    mtime is in neither the matrix nor the stale set.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-del-"))
        (self.tmp / "kept.md").write_text("# kept\n\n" + "body. " * 40, encoding="utf-8")
        self.db = DB(str(self.tmp))
        self.db.cache_dir = self.tmp / "cache"
        rows = unit_rows(3)
        self.db.matrix = rows
        self.db.keys = ["smart_sources:kept.md", "smart_sources:gone.md",
                        "smart_blocks:gone.md#h"]
        self.db.paths = ["kept.md", "gone.md", "gone.md"]
        self.db.lines = [None, None, [1, 2]]
        self.db.embed_at = {"kept.md": 10.0, "gone.md": 10.0}
        self.db._finalize_masks()

    def test_absent_paths_are_dropped_with_their_rows(self):
        _, present = self.db._scan_vault()
        self.assertEqual(self.db._drop_deleted(present), 2)
        self.assertEqual(self.db.matrix.shape[0], 1)
        self.assertEqual(self.db.paths, ["kept.md"])
        self.assertNotIn("smart_sources:gone.md", self.db.key_index)
        self.assertEqual(len(self.db.keys), self.db.matrix.shape[0])
        self.assertEqual(len(self.db.lines), self.db.matrix.shape[0])

    def test_embed_time_is_dropped_with_the_rows(self):
        _, present = self.db._scan_vault()
        self.db._drop_deleted(present)
        self.assertNotIn("gone.md", self.db.embed_at)
        self.assertIn("kept.md", self.db.embed_at)

    def test_a_restored_note_is_seen_as_stale_again(self):
        _, present = self.db._scan_vault()
        self.db._drop_deleted(present)
        restored = self.tmp / "gone.md"
        restored.write_text("# back\n\n" + "body. " * 40, encoding="utf-8")
        os.utime(restored, (10, 10))           # its original mtime, as a backup would
        stale, _ = self.db._scan_vault()
        self.assertIn("gone.md", dict(stale))

    def test_a_path_the_walk_missed_but_which_exists_is_kept(self):
        # "not seen by the walk" is not "not on disk" - the walk skips
        # directories and counts only .md, and the cost of being wrong here is
        # deleting live vectors.
        _, present = self.db._scan_vault()
        self.assertEqual(self.db._drop_deleted(present - {"kept.md"}), 2)
        self.assertIn("kept.md", self.db.path_set)


class DeferredTopupRetry(unittest.TestCase):
    """A deferred pass must retry, and must not retry on the query clock.

    The stale set comes from the plugin's embed_at, which top-up never writes,
    so recording the signature after an incomplete pass froze the retry
    outright - the backlog could never drain inside a process. Retrying it on
    the 5s edit clock instead puts the whole time budget on every query.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-retry-"))
        self.db = bare_db(self.tmp)
        self.db.embeddings_loaded = True
        self.calls = []
        self.db._apply_topup = lambda **kw: self.calls.append(kw)
        self.db._scan_vault = lambda: ([("a.md", 1.0)], {"a.md"})

    def test_unchanged_vault_with_no_backlog_does_not_rebuild(self):
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 1)
        self.db._last_check = 0.0
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 1, "an idle vault must cost nothing")

    def test_deferred_work_is_retried_but_only_after_the_backlog_window(self):
        self.db.topup_skipped = 5               # the pass left work behind
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.db._deferred)

        self.db._last_check = 0.0               # query clock open, backlog clock not
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 1, "must not put the budget on every query")

        self.db._last_check = 0.0
        self.db._last_drain -= server.TOPUP_BACKLOG_RETRY_SECONDS + 1
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 2, "a backlog must eventually drain")

    def test_an_edit_is_picked_up_inside_the_backlog_window(self):
        self.db._refresh_if_stale()
        self.db._scan_vault = lambda: ([("a.md", 2.0)], {"a.md"})   # mtime moved
        self.db._last_check = 0.0
        self.db._refresh_if_stale()
        self.assertEqual(len(self.calls), 2)


class AtomicPublish(unittest.TestCase):
    """A failed load must leave the previous state exactly as it was.

    _load_modern used to populate embed_at while collecting refs. A retry that
    failed then left the first attempt's matrix paired with a half-written map
    of embed times - and the caller, seeing a matrix, carried on. Every note
    looked unembedded, so the first query spent the whole top-up budget
    re-encoding an already-indexed vault.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-atomic-"))
        self.db = bare_db(self.tmp)
        self.db.matrix = unit_rows(2)
        self.db.keys = ["smart_sources:a.md", "smart_sources:b.md"]
        self.db.paths = ["a.md", "b.md"]
        self.db.lines = [None, None]
        self.db.embed_at = {"a.md": 10.0, "b.md": 20.0}
        self.db.model_name = "kept/model"
        self.db._finalize_masks()

    def test_a_failed_load_touches_nothing(self):
        self.db.sources_ajson = self.tmp / "missing.ajson"     # forces False
        self.assertFalse(self.db._load_modern())
        self.assertEqual(self.db.embed_at, {"a.md": 10.0, "b.md": 20.0})
        self.assertEqual(self.db.matrix.shape[0], 2)
        self.assertEqual(self.db.model_name, "kept/model")

    def test_a_failed_load_after_metadata_parses_still_touches_nothing(self):
        # Gets past _read_modern_entries, then finds no usable refs.
        p = self.tmp / "smart_sources.ajson"
        p.write_text('"smart_sources:x.md": {"path":"x.md"},\n', encoding="utf-8")
        self.db.sources_ajson = p
        self.assertFalse(self.db._load_modern())
        self.assertEqual(self.db.embed_at, {"a.md": 10.0, "b.md": 20.0})
        self.assertEqual(self.db.paths, ["a.md", "b.md"])

    def test_a_load_that_fails_AFTER_reading_embed_times_touches_nothing(self):
        # The two cases above both return before the ref loop runs, so neither
        # actually exercises the bug: they pass just as happily against the
        # version that wrote embed_at incrementally. This one has valid refs -
        # so the buggy version records embed times - and then fails at the blob,
        # which is the real shape of a retry losing to a mid-write store.
        fp = "mf_probe"
        (self.tmp / "smart_sources").mkdir(exist_ok=True)
        rng = np.random.default_rng(3)
        write_blob(self.tmp / "smart_sources" / fp,
                   rng.standard_normal((4, 384)) * 9.0)   # not unit: refused
        p = self.tmp / "smart_sources.ajson"
        p.write_text(json.dumps("smart_sources:x.md")[:-1] + '": ' + json.dumps({
            "path": "x.md",
            "embedding": {"default": {fp: {"file_i": 0, "at": 12345.0}}},
        }) + ",\n", encoding="utf-8")
        self.db.sources_ajson = p
        self.db.sources_dir = self.tmp / "smart_sources"
        self.db.blocks_dir = self.tmp / "smart_blocks"

        self.assertFalse(self.db._load_modern())
        self.assertEqual(self.db.embed_at, {"a.md": 10.0, "b.md": 20.0},
                         "a failed load recorded embed times for a matrix it "
                         "never published")
        self.assertNotIn("x.md", self.db.embed_at)
        self.assertEqual(self.db.paths, ["a.md", "b.md"])


class ModelReverification(unittest.TestCase):
    """The re-check must arm on the warm path and survive a transient error.

    It was only armed by a cold rebuild - the one path that had just identified
    the model anyway - so the safety net was dead for any process starting from
    cache, which is most of them.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-verify-"))
        (self.tmp / "note.md").write_text("# n\n\n" + "body text. " * 60,
                                          encoding="utf-8")
        self.db = DB(str(self.tmp))
        self.db.cache_dir = self.tmp / "cache"
        self.db.matrix = unit_rows(1)
        self.db.keys = ["smart_sources:note.md"]
        self.db.paths = ["note.md"]
        self.db.lines = [None]
        self.db._finalize_masks()

    def test_probe_is_armed_from_loaded_state(self):
        self.db.active_space = "mf_x"
        self.db._arm_model_probe()
        self.assertIsNotNone(self.db._probe_material)
        self.assertEqual(self.db._probe_material[0], "mf_x")

    def test_no_probe_without_a_known_space(self):
        self.db.active_space = None
        self.db._arm_model_probe()
        self.assertIsNone(self.db._probe_material)

    def test_an_encode_failure_does_not_permanently_skip_the_check(self):
        # The flag used to be set before the check could complete, turning a
        # transient error into a permanent skip of the safety net.
        class Boom:
            def encode(self, *a, **k):
                raise RuntimeError("no")
        self.db._probe_material = ("mf_x", "text", unit_rows(1)[0])
        self.db.model = Boom()
        self.db._verify_recorded_model()
        self.assertFalse(self.db._verified_model, "must be retried, not skipped")

    def test_a_matching_model_marks_the_check_done(self):
        vec = unit_rows(1, seed=9)[0]
        class Same:
            def encode(self, *a, **k):
                return vec
        self.db._probe_material = ("mf_x", "text", vec)
        self.db.model = Same()
        self.db.model_name = "right/model"
        self.db._verify_recorded_model()
        self.assertTrue(self.db._verified_model)
        self.assertEqual(self.db.model_name, "right/model")

    def test_content_drift_is_not_treated_as_a_wrong_model(self):
        # Right model, note rewritten since it was embedded, measured at 0.631.
        # A 0.7 floor rejected that; the model-vs-model gap is what matters.
        a = unit_rows(1, seed=11)[0]
        b = unit_rows(1, seed=12)[0]
        drifted = a * 0.65 + b * float(np.sqrt(1 - 0.65 ** 2))
        drifted = drifted / np.linalg.norm(drifted)
        class Drifted:
            def encode(self, *x, **k):
                return drifted
        self.db._probe_material = ("mf_x", "text", a)
        self.db.model = Drifted()
        self.db.model_name = "right/model"
        self.db._verify_recorded_model()
        self.assertEqual(self.db.model_name, "right/model",
                         "an edited probe note must not unseat a correct model")


class Counters(unittest.TestCase):
    def test_notes_and_rows_are_reported_separately(self):
        # A two-heading note contributes three rows once a note-level row is
        # synthesized, so one number cannot mean both.
        tmp = Path(tempfile.mkdtemp(prefix="scmcp-count-"))
        db = bare_db(tmp)
        db.matrix = np.zeros((0, 1), dtype=np.float32)
        db._finalize_masks()
        topup = db.stats()["topup"]
        self.assertIn("rows_added", topup)
        self.assertIn("notes_added", topup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
