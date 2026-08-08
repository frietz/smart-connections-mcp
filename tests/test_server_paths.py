#!/usr/bin/env python3
"""Call-site tests: the whole load path, driven against a synthetic store.

The unit suite covers pure functions well and call sites not at all, which is
backwards - every production-severity defect in this project's history was fixed
by changing a call site or a save path, not a pure function. Reverting those
fixes left the suite green.

So this module builds a complete miniature 4.7.2 store on disk and drives
`load_embeddings` through it. No vault, no network, no embedding model: the
model map is pre-seeded so identification never runs, and the few tests that
need an encoder inject a fake one.
"""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server
from server import SmartConnectionsDatabase as DB

DIM = 32                      # smallest dimension _open_blob will accept
MODEL = "test/model"


def unit(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def make_store(tmp, fp="mf_test", n_notes=3, seed=0):
    """A minimal but complete modern store. Returns (vault, fingerprint)."""
    vault = tmp / "vault"
    env = vault / ".smart-env"
    src_dir, blk_dir = env / "smart_sources", env / "smart_blocks"
    for d in (vault, src_dir, blk_dir):
        d.mkdir(parents=True, exist_ok=True)

    body = "This is a note body with enough characters to be a usable probe. " * 12
    lines = []
    for i in range(n_notes):
        rel = f"note{i}.md"
        (vault / rel).write_text(f"# Note {i}\n\n{body}\n\n## Second\n\n{body}",
                                 encoding="utf-8")
        entry = {
            "path": rel,
            "embedding": {"default": {fp: {"file_i": i, "at": 1000.0 + i}}},
            "blocks_data": {
                f"#Note {i}": {
                    "key": f"{rel}#Note {i}", "lines": [1, 4],
                    "should_embed": True,
                    "embedding": {"default": {fp: {"file_i": i}}},
                },
            },
        }
        lines.append(json.dumps(f"smart_sources:{rel}") + ": " +
                     json.dumps(entry) + ",")
    (src_dir / "smart_sources.ajson").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    (src_dir / fp).write_bytes(unit(n_notes, seed=seed).tobytes())
    (blk_dir / fp).write_bytes(unit(n_notes, seed=seed + 1).tobytes())
    return vault, fp


def db_for(vault, cache, fp):
    """A database whose model is already known, so no probe is needed."""
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "fingerprint-models.json").write_text(json.dumps({fp: MODEL}))
    db = DB(str(vault))
    db.cache_dir = cache
    db.model_name = MODEL
    return db


class FakeEncoder:
    """Records what it was asked to encode; returns deterministic unit vectors."""

    def __init__(self, dim=DIM):
        self.seen = []
        self.dim = dim

    def encode(self, text, **kwargs):
        items = text if isinstance(text, list) else [text]
        self.seen.extend(items)
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, s in enumerate(items):
            rng = np.random.default_rng(abs(hash(s)) % (2 ** 32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out if isinstance(text, list) else out[0]


class CacheKeyIsPinnedToTheReadStore(unittest.TestCase):
    """The save path, not the key function.

    _save_to_cache used to be handed a freshly computed fingerprint, which
    re-stats the store AFTER the load - publishing a matrix built from state T0
    under the key for T1, which reads as a valid hit on the next start. The unit
    tests for _fingerprint could not see this: the defect was in the caller.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-toctou-"))
        self.vault, self.fp = make_store(self.tmp)
        self.cache = self.tmp / "cache"
        server.TOPUP_ENABLED = False

    def tearDown(self):
        server.TOPUP_ENABLED = True

    def test_a_quiet_store_is_cached_under_the_signature_it_was_read_at(self):
        db = db_for(self.vault, self.cache, self.fp)
        before = db._store_signature()
        db.load_embeddings()
        meta = json.loads((self.cache / "meta.json").read_text())
        self.assertEqual(meta["fingerprint"], db._fingerprint(before))

    def test_the_saved_key_is_the_one_captured_before_the_read(self):
        # The discriminating case is a store that is quiet across the read and
        # then moves before the save. Re-stating at save time keys the matrix
        # to a state it never saw; using the captured signature does not. The
        # signature is scripted so the difference is a single value rather than
        # a race the test would have to win.
        db = db_for(self.vault, self.cache, self.fp)
        real_sig = db._store_signature
        seen = {"n": 0}

        def scripted():
            seen["n"] += 1
            return "A" if seen["n"] <= 2 else "B"   # quiet for the read, then moves

        db._store_signature = scripted
        saved = []
        real_save = db._save_to_cache
        db._save_to_cache = lambda fp: (saved.append(fp), real_save(fp))[1]

        db.load_embeddings()

        db._store_signature = real_sig
        self.assertEqual(len(saved), 1, "expected exactly one cache publish")
        self.assertEqual(saved[0], db._fingerprint("A"),
                         "the matrix was published under a store state it was "
                         "never read at")

    def test_a_store_that_never_settles_is_never_cached(self):
        db = db_for(self.vault, self.cache, self.fp)
        blob = self.vault / ".smart-env" / "smart_sources" / self.fp
        stamp = {"t": 6_000_000}
        real = db._load_modern

        def always_moving():
            ok = real()
            stamp["t"] += 1000
            os.utime(blob, (stamp["t"], stamp["t"]))
            return ok

        db._load_modern = always_moving
        db.load_embeddings()
        self.assertTrue(db.embeddings_loaded, "the read is still usable")
        self.assertFalse((self.cache / "meta.json").exists(),
                         "a read the store moved under must not be cached")


class WarmStartArmsTheModelCheck(unittest.TestCase):
    """The wiring, not the helper.

    The round-3 HIGH was that the model re-check was armed only by a cold
    rebuild - the one path that had just identified the model anyway - so it was
    dead for any process starting from cache. A test that calls the helper
    directly cannot see that.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-warm-"))
        self.vault, self.fp = make_store(self.tmp)
        self.cache = self.tmp / "cache"
        server.TOPUP_ENABLED = False
        db_for(self.vault, self.cache, self.fp).load_embeddings()   # warm it

    def tearDown(self):
        server.TOPUP_ENABLED = True

    def test_a_cache_hit_arms_the_probe(self):
        db = db_for(self.vault, self.cache, self.fp)
        db.load_embeddings()
        self.assertTrue((self.cache / "meta.json").exists())
        self.assertIsNotNone(db._probe_material,
                             "warm start left the model re-check disarmed")
        self.assertEqual(db._probe_material[0], self.fp)

    def test_the_active_space_survives_the_cache(self):
        # Without it the probe cannot be armed on the warm path at all.
        db = db_for(self.vault, self.cache, self.fp)
        db.load_embeddings()
        self.assertEqual(db.active_space, self.fp)


class QueryPrefixReachesTheEncoder(unittest.TestCase):
    """Asserting the profile table has a prefix is not the same as using it.

    Benchmarked on the real vault: arctic with its prefix scores MRR 0.760,
    without it 0.215 - worse than the model it replaced.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-prefix-"))
        self.db = DB(str(self.tmp))
        self.db.cache_dir = self.tmp / "cache"
        self.fake = FakeEncoder()
        self.db.model = self.fake

    def test_an_asymmetric_model_prefixes_the_query(self):
        self.db.model_name = "Snowflake/snowflake-arctic-embed-s"
        self.db._encode("keyset pagination")
        self.assertEqual(len(self.fake.seen), 1)
        self.assertTrue(
            self.fake.seen[0].startswith(
                "Represent this sentence for searching relevant passages: "),
            f"query reached the encoder unprefixed: {self.fake.seen[0]!r}")

    def test_a_symmetric_model_does_not_prefix(self):
        self.db.model_name = "TaylorAI/bge-micro-v2"
        self.db._encode("keyset pagination")
        self.assertEqual(self.fake.seen[0], "keyset pagination")


class TopupCountersAndSupersede(unittest.TestCase):
    """Counts, and find_related surviving a supersede, on the real top-up path.

    The counter test this replaces only asserted that two keys existed, so
    merging them back into one number left it green. And the find_related fix
    was only covered at the level of the pooling helper, never through an
    actual supersede.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-topup-"))
        self.vault, self.fp = make_store(self.tmp, n_notes=2)
        self.cache = self.tmp / "cache"
        self._topup = server.TOPUP_ENABLED
        server.TOPUP_ENABLED = False          # the initial load must not encode
        self.db = db_for(self.vault, self.cache, self.fp)
        self.db.load_embeddings()
        self.db.model = FakeEncoder()
        self.db.ensure_model_loaded = lambda: None
        server.TOPUP_ENABLED = True
        server.TOPUP_TIME_BUDGET_SECONDS = 60

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup

    def test_notes_and_rows_are_counted_as_different_things(self):
        rel = "note0.md"
        mtime = (self.vault / rel).stat().st_mtime
        self.db._apply_topup(targets=[(rel, mtime)])
        # Assert through stats(), not the attributes. The original defect was
        # in the mapping - rows reported under the notes label - so a test that
        # reads the attributes directly cannot see it. Hold the freshness
        # throttle closed so stats() does not start another pass and reset them.
        self.db._last_check = time.time()
        topup = self.db.stats()["topup"]
        self.assertEqual(topup["notes_added"], 1, "one note was topped up")
        self.assertEqual(topup["rows_added"], self.db.topup_added)
        self.assertGreater(topup["rows_added"], topup["notes_added"],
                           "a note with two headings contributes more rows "
                           "than notes - the two counts cannot be one number")

    def test_find_related_survives_a_note_being_superseded(self):
        # An H1-first note chunks to blocks only, so top-up used to strip the
        # source row find_related both anchors on and ranks by.
        rel = "note0.md"
        mtime = (self.vault / rel).stat().st_mtime
        self.db._apply_topup(targets=[(rel, mtime)])
        self.assertIn(f"smart_sources:{rel}", self.db.key_index,
                      "supersede removed the note's source row")
        self.assertTrue(self.db.find_related(rel, limit=3),
                        "a topped-up note became invisible to find_related")


class LoudFailures(unittest.TestCase):
    """Silence was the original bug in this codebase; keep the warnings tested."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-loud-"))
        self.vault, self.fp = make_store(self.tmp, n_notes=2)
        self.cache = self.tmp / "cache"
        self._topup = server.TOPUP_ENABLED
        server.TOPUP_ENABLED = False          # the initial load must not encode
        self.db = db_for(self.vault, self.cache, self.fp)
        self.db.load_embeddings()
        # No test in this module may reach the network. Every path that would
        # load an encoder gets a fake one here, explicitly.
        self.db.model = FakeEncoder()
        self.db.ensure_model_loaded = lambda: None
        server.TOPUP_ENABLED = True
        server.TOPUP_TIME_BUDGET_SECONDS = 60

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup

    def test_a_dimension_mismatch_says_so_and_leaves_the_index_alone(self):
        self.db.model = FakeEncoder(dim=DIM * 2)      # wrong-width vectors
        before = self.db.matrix.shape
        rel = "note0.md"
        err = io.StringIO()
        with redirect_stderr(err):
            self.db._apply_topup(targets=[(rel, (self.vault / rel).stat().st_mtime)])
        message = err.getvalue().lower()
        self.assertIn("not merged", message,
                      "the top-up silently discarded encoded vectors")
        self.assertIn(f"{DIM * 2}d", message, "the message must name both widths")
        self.assertIn(f"{DIM}d", message)
        self.assertEqual(self.db.matrix.shape, before)

    def test_a_model_change_under_a_running_process_is_announced(self):
        self.db.model_name = "some/other-model"       # store still says MODEL
        self.db._last_check = 0.0
        err = io.StringIO()
        with redirect_stderr(err):
            self.db._refresh_if_stale()
        self.assertIn("reconnect", err.getvalue().lower(),
                      "a model change went unmentioned")

    def test_reindex_reports_what_it_did(self):
        # The operator line used to be annotated backwards - it claimed the
        # remainder was "all covered by the top-up cache" precisely when some
        # were not, and said nothing when the count was zero.
        os.environ["OBSIDIAN_VAULT_PATH"] = str(self.vault)
        os.environ["SMART_CONNECTIONS_MODEL"] = MODEL      # no identification
        real_root, real_topup = server.DEFAULT_CACHE_DIR, server.TOPUP_ENABLED
        server.DEFAULT_CACHE_DIR = self.tmp / "reindex-cache"
        server.TOPUP_ENABLED = False                       # no encoder needed
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = server.reindex_cli()
        finally:
            os.environ.pop("OBSIDIAN_VAULT_PATH", None)
            os.environ.pop("SMART_CONNECTIONS_MODEL", None)
            server.DEFAULT_CACHE_DIR, server.TOPUP_ENABLED = real_root, real_topup

        self.assertEqual(rc, 0)
        text = out.getvalue()
        for expected in ("vectors", "behind Obsidian",
                         "covered by the top-up cache"):
            self.assertIn(expected, text)
        self.assertNotIn("still stale", text, "the inverted label is back")


class FailedVerificationRepairsTheProcess(unittest.TestCase):
    """Fixing the map file while the process keeps querying the wrong space is
    not a fix - it leaves this run exactly as wrong, with a warning."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-verify2-"))
        self.db = DB(str(self.tmp))
        self.db.cache_dir = self.tmp / "cache"
        self.db.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stored = unit(1, seed=7)[0]
        self.db._probe_material = ("mf_x", "some note text", self.stored)
        self.db._record_model("mf_x", "wrong/model")
        self.db.model_name = "wrong/model"
        self.db.model = FakeEncoder()          # never reproduces `stored`

    def test_a_mismatch_swaps_the_model_in_this_process_not_just_on_disk(self):
        fake_module = type(sys)("sentence_transformers")
        fake_module.SentenceTransformer = lambda name: FakeEncoder()
        sys.modules["sentence_transformers"] = fake_module
        self.db._identify_model = lambda text, stored: "right/model"
        err = io.StringIO()
        try:
            with redirect_stderr(err):
                self.db._verify_recorded_model()
        finally:
            sys.modules.pop("sentence_transformers", None)

        self.assertEqual(self.db.model_name, "right/model",
                         "the process kept querying in the wrong space")
        self.assertEqual(self.db._model_map().get("mf_x"), "right/model")

    def test_a_mismatch_with_no_usable_model_says_search_is_wrong(self):
        self.db._identify_model = lambda text, stored: None
        err = io.StringIO()
        with redirect_stderr(err):
            self.db._verify_recorded_model()
        self.assertIn("mismatched", err.getvalue().lower())
        self.assertIsNone(self.db._model_map().get("mf_x"),
                          "a disproven mapping must not be kept")


class ProbeThresholdIsPinned(unittest.TestCase):
    """The floor has to sit between two measured cases, so lock both sides.

    Measured on the real vault: right model, unchanged text 0.980; right model,
    note rewritten since 0.631; wrong model -0.076.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-floor-"))
        self.db = DB(str(self.tmp))
        self.db.cache_dir = self.tmp / "cache"
        self.db.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _at_cosine(target, cos, seed):
        """A unit vector at exactly `cos` from `target`."""
        rng = np.random.default_rng(seed)
        r = rng.standard_normal(target.shape[0]).astype(np.float32)
        r -= target * float(r @ target)
        r /= np.linalg.norm(r)
        v = target * cos + r * float(np.sqrt(1 - cos ** 2))
        return v / np.linalg.norm(v)

    def _run(self, cos):
        stored = unit(1, seed=21)[0]
        drifted = self._at_cosine(stored, cos, seed=22)
        self.assertAlmostEqual(float(stored @ drifted), cos, places=4)
        self.db._probe_material = ("mf_x", "text", stored)
        self.db._record_model("mf_x", "right/model")
        self.db.model_name = "right/model"
        self.db.model = type("M", (), {"encode": lambda s, t, **k: drifted})()
        self.db._identify_model = lambda *a: "SHOULD-NOT-BE-CALLED"
        err = io.StringIO()
        with redirect_stderr(err):
            self.db._verify_recorded_model()
        return self.db.model_name

    def test_the_measured_drift_case_is_accepted(self):
        # 0.631 is a real measurement: right model, probe note rewritten. A
        # floor above it rejects a correct identification.
        self.assertEqual(self._run(0.631), "right/model")

    def test_the_floor_is_where_it_is_documented(self):
        self.assertLess(server.MODEL_PROBE_MIN_COSINE, 0.631,
                        "the floor must sit below the measured drift case")
        self.assertGreater(server.MODEL_PROBE_MIN_COSINE, 0.05,
                           "the floor must sit above the wrong-model case")

    def test_a_wrong_model_score_is_rejected(self):
        self.db._probe_material = ("mf_x", "text", unit(1, seed=31)[0])
        self.db._record_model("mf_x", "wrong/model")
        self.db.model_name = "wrong/model"
        self.db.model = type("M", (), {
            "encode": lambda s, t, **k: unit(1, seed=32)[0]})()   # unrelated
        self.db._identify_model = lambda *a: None
        with redirect_stderr(io.StringIO()):
            self.db._verify_recorded_model()
        self.assertIsNone(self.db._model_map().get("mf_x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
