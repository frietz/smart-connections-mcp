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

DIM = 64                      # see make_store: wide enough to leave _open_blob
MODEL = "test/model"          # more than one legal factorization to reject

# Real `at` values are epoch MILLISECONDS (~1.78e12 in 2026), and the server
# divides by 1000 to get the seconds it compares against file mtimes. A fixture
# using small numbers agrees with that code by accident rather than by modelling
# it, so a regression in the units stays green here. This sits deliberately in
# the past relative to the notes the fixture writes, which keeps them looking
# un-embedded exactly as they did before.
BASE_AT_MS = 1785000000000.0                       # 2026-07-25, in millis


def unit(n, dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def make_store(tmp, fp="mf_test", n_notes=3, seed=0, spaces=(),
               blob_extra_rows=0):
    """A minimal but complete modern store. Returns (vault, fingerprint).

    `spaces` adds sibling embedding spaces beside `fp` under
    `embedding.default`, which is what the real store looks like: several `mf_*`
    names side by side, the plugin having simply stopped writing the old ones.
    Each entry is a dict with `name` and `at`, optionally `blob` (write one, so
    the space exists on disk) and `mtime` (touch it, so blob recency and
    metadata recency can be made to disagree).

    `blob_extra_rows` writes a source blob with more rows than the metadata
    knows about. The plugin writes metadata and blob separately, so this is the
    normal transient state, and it is the one that used to take the whole load
    to an empty index.
    """
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
        slots = {fp: {"file_i": i, "at": BASE_AT_MS + i}}
        for extra in spaces:
            slots[extra["name"]] = {"file_i": i, "at": extra["at"]}
        entry = {
            "path": rel,
            "embedding": {"default": dict(slots)},
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
    (src_dir / fp).write_bytes(
        unit(n_notes + blob_extra_rows, seed=seed).tobytes())
    (blk_dir / fp).write_bytes(unit(n_notes, seed=seed + 1).tobytes())

    for extra in spaces:
        if not extra.get("blob"):
            continue
        for d in (src_dir, blk_dir):
            (d / extra["name"]).write_bytes(
                unit(n_notes, seed=extra.get("seed", 90)).tobytes())
        if extra.get("mtime") is not None:
            for d in (src_dir, blk_dir):
                os.utime(d / extra["name"], (extra["mtime"], extra["mtime"]))
    return vault, fp


def db_for(vault, cache, fp, also=()):
    """A database whose model is already known, so no probe is needed."""
    cache.mkdir(parents=True, exist_ok=True)
    mapping = {name: MODEL for name in (fp,) + tuple(also)}
    (cache / "fingerprint-models.json").write_text(json.dumps(mapping))
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
        self._topup = server.TOPUP_ENABLED
        server.TOPUP_ENABLED = False

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup

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
        #
        # Scripted on the settle check, not on a call count. An earlier version
        # returned "A" for the first two calls, which encoded how many times
        # load_embeddings happens to read the signature today - a third read
        # added anywhere would have turned a quiet store into a moving one and
        # gone red without a bug.
        #
        # The structural reads are the pre-load capture and the one settle check
        # that follows the load. Both must see a quiet store. Any read after
        # that is the defect itself - re-stating the store to build the cache
        # key - so it gets a different answer.
        db = db_for(self.vault, self.cache, self.fp)
        real_sig = db._store_signature
        state = {"loaded": False, "settled": False}
        real_load = db._load_modern

        def load():
            ok = real_load()
            state["loaded"] = True
            return ok

        def scripted():
            if not state["loaded"]:
                return "A"                    # pre-load capture
            if not state["settled"]:
                state["settled"] = True
                return "A"                    # the settle check - still quiet
            return "B"                        # anything later: the store moved

        db._load_modern = load
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


class ActiveSpaceSelectionOnTheLoadPath(unittest.TestCase):
    """Which embedding space the load picks, driven through load_embeddings.

    Round 5 found the fixture wrote exactly one space, so `_newest_fingerprint`
    and the mtime fallback could never disagree and the whole selection branch
    at the call site was untestable. The real store carries several `mf_*` names
    on almost every source with only some of them present as blobs, so this is
    the normal shape, not an edge case.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="scmcp-space-"))
        self._topup = server.TOPUP_ENABLED
        server.TOPUP_ENABLED = False

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup

    def test_the_newest_metadata_wins_over_the_newest_blob(self):
        # An older space whose blob was touched last - a backup or a sync tool
        # is enough. Choosing by blob mtime picks the stale space and every
        # score afterwards is computed in the wrong vector space, silently,
        # because every model here is the same width.
        vault, fp = make_store(
            self.tmp, fp="mf_new", n_notes=3,
            spaces=[{"name": "mf_old", "at": BASE_AT_MS - 500_000,
                     "blob": True, "mtime": time.time() + 5_000}])
        db = db_for(vault, self.tmp / "cache", fp, also=("mf_old",))
        db.load_embeddings()
        self.assertEqual(db.active_space, "mf_new",
                         "the load followed blob mtime instead of the "
                         "metadata's embedding time")

    def test_a_space_with_no_blob_on_disk_is_not_selected(self):
        # Exactly the live vault on 2026-08-09: the metadata still names the
        # pre-migration space long after its blob was deleted. Selecting it
        # takes the whole load to an empty index.
        vault, fp = make_store(
            self.tmp, fp="mf_live", n_notes=3,
            spaces=[{"name": "mf_ghost", "at": BASE_AT_MS + 900_000}])
        db = db_for(vault, self.tmp / "cache", fp, also=("mf_ghost",))
        db.load_embeddings()
        self.assertEqual(db.active_space, "mf_live")
        self.assertGreater(db.matrix.shape[0], 0, "the load returned nothing")

    def test_a_future_embedding_time_does_not_win(self):
        # One corrupt `at` in 33MB of metadata should not be able to select a
        # space. The future space also holds the newest blob, so a horizon check
        # that stops working cannot be rescued by the mtime fallback.
        vault, fp = make_store(
            self.tmp, fp="mf_real", n_notes=3,
            spaces=[{"name": "mf_future",
                     "at": (time.time() + 86400 * 365) * 1000.0,
                     "blob": True, "mtime": time.time() + 9_000}])
        db = db_for(vault, self.tmp / "cache", fp, also=("mf_future",))
        db.load_embeddings()
        self.assertEqual(db.active_space, "mf_real",
                         "a timestamp in the future selected the space")

    def test_embedding_times_are_read_as_epoch_millis(self):
        # `at` is millis on disk and seconds in embed_at. Pin the conversion
        # here: without it the only guard is a pure-function unit test, and the
        # load path could stop dividing without anything going red.
        vault, fp = make_store(self.tmp, n_notes=2)
        db = db_for(vault, self.tmp / "cache", fp)
        db.load_embeddings()
        self.assertAlmostEqual(db.embed_at["note0.md"], BASE_AT_MS / 1000.0,
                               delta=5.0)
        self.assertLess(db.embed_at["note0.md"], time.time(),
                        "an embedding time landed in the future - `at` was "
                        "read as seconds when it is written as millis")

    def test_a_blob_ahead_of_the_metadata_still_opens(self):
        # The plugin writes metadata and blob separately, so the blob running a
        # few rows ahead is routine. This shape is chosen so the WIDEST legal
        # factorization is the wrong one: 4 rows of 64 against a metadata that
        # knows of 2 offers 128 first, whose rows are two real vectors glued
        # together. Only the unit-norm proof rejects it. With the blob exactly
        # as long as the metadata claims there is a single candidate and the
        # proof is never asked anything.
        vault, fp = make_store(self.tmp, n_notes=2, blob_extra_rows=2)
        db = db_for(vault, self.tmp / "cache", fp)
        with redirect_stderr(io.StringIO()):
            db.load_embeddings()
        self.assertTrue(db.embeddings_loaded)
        self.assertGreater(db.matrix.shape[0], 0,
                           "a blob longer than the metadata emptied the index")
        self.assertEqual(db.matrix.shape[1], DIM,
                         "the proof accepted a wrong factorization")


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
        self._topup = server.TOPUP_ENABLED
        server.TOPUP_ENABLED = False
        db_for(self.vault, self.cache, self.fp).load_embeddings()   # warm it

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup

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
        self._budget = server.TOPUP_TIME_BUDGET_SECONDS
        server.TOPUP_TIME_BUDGET_SECONDS = 60

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup
        server.TOPUP_TIME_BUDGET_SECONDS = self._budget

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
        self._budget = server.TOPUP_TIME_BUDGET_SECONDS
        server.TOPUP_TIME_BUDGET_SECONDS = 60

    def tearDown(self):
        server.TOPUP_ENABLED = self._topup
        server.TOPUP_TIME_BUDGET_SECONDS = self._budget

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
        # Restore, do not pop: run-tests.sh exports OBSIDIAN_VAULT_PATH, so
        # deleting it here would leave every later test in this process reading
        # a different vault than the one the run was pointed at.
        saved_env = {k: os.environ.get(k)
                     for k in ("OBSIDIAN_VAULT_PATH", "SMART_CONNECTIONS_MODEL")}
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
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
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
