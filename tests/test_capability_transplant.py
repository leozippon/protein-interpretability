"""The transplant contract: what an empty and a full selection have to reproduce.

Every localisation number this experiment reports is read against two endpoints,
so the endpoints are the thing that has to be exact. A selection of no groups has
to leave the destination checkpoint bit-for-bit, a selection of every group has
to install the source checkpoint bit-for-bit, and a group together with its
complement has to be the whole difference and nothing twice. These are checked on
real checkpoints built and loaded here rather than on a stub, because the failure
mode they guard against -- a tensor no group claims, or a destination the loaded
model does not expose -- only exists once a checkpoint is serialized and read
back.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.transfer import capability_transplant as ct
from src.transfer.profile_increment import correlation, standardized_rank

STAGE = ROOT/"scripts/transfer/d2_transplant_likelihood.py"
ANALYSIS = ROOT/"scripts/transfer/analyse_d2_transplant.py"
ADMITTED = ROOT/("results/transfer/readout_expansion_20260923/complete/results/external_baseline/"
                 "20260923105347_9e1188dbfcda")
ADMITTED_NATIVE = {
    "llama-2-7b": ADMITTED/"readout_analysis_llama2_parent_native/readout_llama-2-7b_native.json",
    "prollama-stage-1": ADMITTED/"readout_analysis_prollama_stage1_native/readout_prollama-stage-1_native.json",
    "prollama": ADMITTED/"readout_analysis_prollama_stage2_native/readout_prollama_native.json",
}

#: The admitted 211-assay native support of the Llama lineage, and the endpoints
#: every transplant is read between.
ADMITTED_SUPPORT = dict(n_assays=211, n_families=169, n_variants=26943)
ADMITTED_RAW_M = {
    "llama-2-7b": -0.007404251607431609,
    "prollama-stage-1": 0.14604217316024715,
    "prollama": 0.15864682053456233,
}

LAYERS = 8
TAILS = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
)


def production_names(n_layers=32):
    """The 323 tensor names the two staged checkpoints carry, without reading them."""
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for layer in range(n_layers):
        for tail in (*TAILS, "self_attn.rotary_emb.inv_freq"):
            names.append(f"model.layers.{layer}.{tail}")
    return names


def build_pair(root: Path):
    """Two tiny Llama checkpoints that differ on every projection and nothing else."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=LAYERS, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         tie_word_embeddings=False)
    torch.manual_seed(20260923)
    first = LlamaForCausalLM(config)
    second = LlamaForCausalLM(config)
    second.load_state_dict(first.state_dict())
    with torch.no_grad():
        for name, parameter in second.named_parameters():
            if name.endswith("_proj.weight") or name in ("model.embed_tokens.weight", "lm_head.weight"):
                parameter.add_(torch.full_like(parameter, 0.125))
    paths = []
    for index, model in enumerate((first, second)):
        path = root/f"checkpoint_{index}"
        model.to(torch.float16).save_pretrained(path, safe_serialization=True, max_shard_size="20KB")
        (path/"tokenizer.model").write_bytes(b"tokenizer fixture")
        if not (path/"model.safetensors.index.json").exists():
            raise AssertionError("fixture must be sharded so the census reads an index")
        paths.append(path)
    return paths


def load_fp32(path):
    import torch
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return model.eval().requires_grad_(False)


def state_digests(model):
    live = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    return {name: ct.fp32_digest(tensor) for name, tensor in live.items()}


class ThePartitionCoversTheCheckpoint(unittest.TestCase):
    def test_every_production_tensor_belongs_to_exactly_one_declared_group(self):
        names = production_names()
        self.assertEqual(len(names), 323)
        members = ct.partition(names, 32)
        self.assertEqual(set(members), set(ct.group_names()))
        flattened = [name for group in ct.group_names() for name in members[group]]
        self.assertEqual(sorted(flattened), sorted(names))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual({group: len(members[group]) for group in ct.group_names()},
                         {"embedding": 1, "head": 1, "norm": 65, "rope": 32,
                          "attention_q1": 32, "attention_q2": 32, "attention_q3": 32,
                          "attention_q4": 32, "mlp_q1": 24, "mlp_q2": 24, "mlp_q3": 24,
                          "mlp_q4": 24})

    def test_the_depth_blocks_are_contiguous_and_equal(self):
        assignment = [ct.depth_block(layer, 32) for layer in range(32)]
        self.assertEqual(assignment, sorted(assignment))
        self.assertEqual([assignment.count(index) for index in range(ct.DEPTH_BLOCKS)], [8, 8, 8, 8])

    def test_an_unrecognised_tensor_is_refused_rather_than_dropped(self):
        with self.assertRaises(ValueError):
            ct.group_of("model.layers.0.self_attn.rope_scaling", 32)
        with self.assertRaises(ValueError):
            ct.group_of("transformer.h.0.attn.c_attn.weight", 32)
        with self.assertRaises(ValueError):
            ct.partition([*production_names(), "model.adapter.weight"], 32)

    def test_a_group_and_its_complement_are_disjoint_and_exhaustive(self):
        for spec in ("attention_q1", "embedding,head", "mlp_q1,mlp_q2,mlp_q3,mlp_q4"):
            selected = set(ct.resolve_groups(spec))
            complement = set(ct.resolve_groups("complement:"+spec))
            self.assertEqual(selected & complement, set())
            self.assertEqual(selected | complement, set(ct.group_names()))
        self.assertEqual(ct.resolve_groups("none"), ())
        self.assertEqual(set(ct.resolve_groups("all")), set(ct.group_names()))
        for bad in ("", "attention_q5", "attention_q1,attention_q1", "complement:all", "complement:"):
            with self.assertRaises(ValueError):
                ct.resolve_groups(bad)


class TheTransplantReproducesItsEndpoints(unittest.TestCase):
    """The two endpoints, on checkpoints written and read back the production way."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment gate
            raise unittest.SkipTest(f"torch/transformers unavailable: {error}")
        cls._root = tempfile.TemporaryDirectory()
        root = Path(cls._root.name)
        cls.destination, cls.source = build_pair(root)
        cls.census = ct.census(cls.destination, cls.source)

    @classmethod
    def tearDownClass(cls):
        cls._root.cleanup()

    def test_the_census_finds_the_differences_it_was_given(self):
        changed = {row["group"] for row in self.census["tensors"] if row["differing_elements_fp32"]}
        self.assertEqual(changed, {"embedding", "head", "attention_q1", "attention_q2",
                                   "attention_q3", "attention_q4", "mlp_q1", "mlp_q2",
                                   "mlp_q3", "mlp_q4"})
        self.assertTrue(self.census["groups"]["norm"]["fp32_bytes_equal"])
        for row in self.census["tensors"]:
            self.assertEqual(row["fp32_bytes_equal"], row["differing_elements_fp32"] == 0)

    def test_transplanting_no_group_reproduces_the_destination_bit_identically(self):
        model = load_fp32(self.destination)
        before = state_digests(model)
        receipt = ct.transplant(model, self.source, self.census, ())
        self.assertEqual(receipt["n_transplanted_tensors"], 0)
        self.assertEqual(state_digests(model), before)
        self.assertEqual(state_digests(model), state_digests(load_fp32(self.destination)))

    def test_transplanting_every_group_reproduces_the_source_bit_identically(self):
        model = load_fp32(self.destination)
        receipt = ct.transplant(model, self.source, self.census, ct.group_names())
        self.assertEqual(receipt["absent_destinations"], [])
        self.assertEqual(receipt["n_transplanted_tensors"], len(self.census["tensors"]))
        self.assertEqual(state_digests(model), state_digests(load_fp32(self.source)))
        self.assertNotEqual(state_digests(model), state_digests(load_fp32(self.destination)))

    def test_a_group_and_its_complement_together_are_the_full_transplant(self):
        parts = ct.transplant(load_fp32(self.destination), self.source, self.census,
                              ct.resolve_groups("attention_q1,mlp_q1"))
        rest = ct.transplant(load_fp32(self.destination), self.source, self.census,
                             ct.resolve_groups("complement:attention_q1,mlp_q1"))
        whole = ct.transplant(load_fp32(self.destination), self.source, self.census,
                              ct.group_names())
        self.assertEqual(set(parts["transplanted_tensors"]) & set(rest["transplanted_tensors"]), set())
        self.assertEqual(sorted(parts["transplanted_tensors"]+rest["transplanted_tensors"]),
                         sorted(whole["transplanted_tensors"]))
        self.assertEqual(parts["n_transplanted_differing_elements"]
                         + rest["n_transplanted_differing_elements"],
                         whole["n_transplanted_differing_elements"])

    def test_a_partial_transplant_installs_exactly_the_named_tensors(self):
        model = load_fp32(self.destination)
        ct.transplant(model, self.source, self.census, ("mlp_q2",))
        observed = state_digests(model)
        expected = state_digests(load_fp32(self.destination))
        source = state_digests(load_fp32(self.source))
        moved = set(ct.partition([row["name"] for row in self.census["tensors"]], LAYERS)["mlp_q2"])
        self.assertTrue(moved)
        for name, digest in observed.items():
            self.assertEqual(digest, source[name] if name in moved else expected[name], name)

    def test_a_state_that_does_not_match_the_census_is_refused(self):
        import torch

        model = load_fp32(self.destination)
        with torch.no_grad():
            model.model.layers[0].mlp.up_proj.weight.add_(1.0)
        with self.assertRaises(ValueError):
            ct.transplant(model, self.source, self.census, ("embedding",))

    def test_a_differing_tensor_without_a_live_destination_is_refused(self):
        model = load_fp32(self.destination)
        tampered = json.loads(json.dumps(self.census))
        tampered["tensors"].append(dict(
            name="model.layers.0.self_attn.rotary_emb.inv_freq", group="rope", shape=[4],
            n_elements=4, dtypes=["torch.float32", "torch.float32"],
            native_sha256=["a"*64, "b"*64], fp32_sha256=["a"*64, "b"*64],
            differing_elements_fp32=4, native_bytes_equal=False, fp32_bytes_equal=False))
        with self.assertRaises(ValueError):
            ct.transplant(model, self.source, tampered, ("embedding",))

    def test_an_architecture_or_tokenizer_mismatch_is_refused(self):
        root = Path(self._root.name)
        clone = root/"clone"
        clone.mkdir()
        for item in self.source.iterdir():
            (clone/item.name).write_bytes(item.read_bytes())
        (clone/"tokenizer.model").write_bytes(b"a different tokenizer")
        with self.assertRaises(ValueError):
            ct.architecture_record(self.destination, clone)
        config = json.loads((clone/"config.json").read_text())
        config["num_hidden_layers"] = LAYERS+1
        (clone/"config.json").write_text(json.dumps(config))
        (clone/"tokenizer.model").write_bytes(b"tokenizer fixture")
        with self.assertRaises(ValueError):
            ct.architecture_record(self.destination, clone)


class TheScreenPanelIsLabelBlind(unittest.TestCase):
    @staticmethod
    def rows(n=60):
        rng = np.random.default_rng(7)
        return [dict(assay=f"ASSAY_{i:03d}", cluster=f"family_{i // 2}",
                     measured=rng.normal(size=5).tolist()) for i in range(n)]

    def test_permuting_every_measured_effect_leaves_the_selection_unchanged(self):
        rows = self.rows()
        rng = np.random.default_rng(11)
        shuffled = [dict(row, measured=list(rng.permutation(row["measured"]))) for row in rows]
        for row in shuffled:
            rng.shuffle(row["measured"])
        first = ct.select_screen_panel(rows, families=12, seed=20260923)
        second = ct.select_screen_panel(shuffled, families=12, seed=20260923)
        self.assertEqual(ct.panel_digest(first), ct.panel_digest(second))
        self.assertFalse(first["uses_outcomes"])

    def test_the_panel_takes_one_assay_from_each_selected_family(self):
        rows = self.rows()
        panel = ct.select_screen_panel(rows, families=12, seed=20260923)
        families = {row["assay"]: row["cluster"] for row in rows}
        self.assertEqual(len(panel["selected_assay_ids"]), 12)
        self.assertEqual(sorted({families[a] for a in panel["selected_assay_ids"]}, key=str),
                         sorted(panel["selected_family_ids"], key=str))
        self.assertEqual(panel["selected_assay_ids"], sorted(panel["selected_assay_ids"]))

    def test_a_panel_larger_than_the_eligible_families_is_refused(self):
        with self.assertRaises(ValueError):
            ct.select_screen_panel(self.rows(), families=999, seed=20260923)


class TheEndpointEstimatorIsTheAdmittedOne(unittest.TestCase):
    def test_the_within_assay_correlation_is_the_readout_study_spelling(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("analyse_d2", ANALYSIS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(3)
        for _ in range(20):
            scores = rng.normal(size=40)
            measured = rng.normal(size=40)
            mine = module.assay_spearman(scores, measured)
            study = correlation(rankdata(standardized_rank(scores)), standardized_rank(measured))
            self.assertEqual(mine, study)
            self.assertAlmostEqual(mine, float(spearmanr(scores, measured).statistic), places=12)

    def test_a_constant_score_vector_is_undefined_rather_than_zero(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("analyse_d2", ANALYSIS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIsNone(module.assay_spearman([1.0]*6, list(range(6))))


@unittest.skipUnless(all(path.exists() for path in ADMITTED_NATIVE.values()),
                     "admitted readout reports are not staged in this checkout")
class TheSupportIsTheAdmittedReadoutSupport(unittest.TestCase):
    def reports(self):
        return {arm: json.loads(path.read_text()) for arm, path in ADMITTED_NATIVE.items()}

    def test_the_three_lineage_arms_share_one_211_assay_support(self):
        digests = set()
        for arm, report in self.reports().items():
            self.assertEqual(report["support"]["definition"], "native")
            self.assertEqual(
                dict(n_assays=report["n_assays"], n_families=report["n_families"],
                     n_variants=report["n_variants"]), ADMITTED_SUPPORT, arm)
            digests.add(ct.support_digest(report["support"]["assay_ids"]))
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests.pop(),
                         "81432626b0fb4e711fccea81c0aa786c7d1d69b2dee7f935e2f3aee6ffb282fe")

    def test_the_raw_likelihood_endpoint_is_fold_free_and_excludes_no_assay(self):
        for arm, report in self.reports().items():
            record = report["summaries"]["raw_M_spearman"]
            self.assertEqual(record["point"], ADMITTED_RAW_M[arm])
            self.assertEqual(record["excluded_assays"], 0)
            self.assertEqual(record["resamples"], 2000)
            self.assertEqual(record["n_units"], 169)
            self.assertEqual(record["unit"], "wild-type family at 50% identity")
            self.assertEqual(report["bootstrap_seed"], 20260923)
            self.assertNotIn("raw_M", report["folds"])


class NoLabelReachesTheModelSideStage(unittest.TestCase):
    """The stage that runs the models never reads an effect, and nothing is tuned."""

    FITTING = frozenset(("nested_predict", "ridge_predict", "family_folds", "evaluate_readouts",
                         "row_weights", "sequence_features", "readout_analysis"))

    def test_the_transplant_stage_never_reads_a_measured_effect(self):
        tree = ast.parse(STAGE.read_text())
        literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
        self.assertNotIn("measured", literals)
        self.assertNotIn("profile_scores", literals)
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn("measured", attributes)

    def test_neither_entry_point_imports_a_fitting_or_tuning_symbol(self):
        for path in (STAGE, ANALYSIS):
            tree = ast.parse(path.read_text())
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
            leaked = {name for name in imported if name.rsplit(".", 1)[-1] in self.FITTING}
            self.assertEqual(leaked, set(), f"{path.name} imports {leaked}")


if __name__ == "__main__":
    unittest.main()
