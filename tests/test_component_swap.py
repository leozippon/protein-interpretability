"""What the component-swap stage must get right, on CPU stubs.

The stage takes a declared component group from one checkpoint of a lineage,
keeps everything else from another, and measures the chimera in
``21_joint_mode_qualification.py``'s estimand. That turns a two-point correlation
-- the ProLLaMA stage that costs 4.69 nats of text is the stage that retrains
``embed_tokens,lm_head`` -- into an attribution, but only if four things hold, and
every one of them can be wrong while the run still completes:

**What moved is what was declared.** A group resolved by pattern rather than
declared would name a different set of tensors on the next architecture, and a
cell would silently stop being the cell its name says.

**Nothing was coerced.** A moved tensor whose shape differs between donor and
host is limitation L24's failure shape reached from a new direction: the model
loads, runs and generates plausible text while being meaningless. So does an
embedding row moved between two different vocabularies. Both are refused before
any write, which also means a refusal leaves the host intact rather than half
written.

**A tied pair is one tensor.** Where the embedding and the output head share
storage, ``embedding`` and ``lm_head`` are not two groups: writing either writes
both. The stage refuses those two names there and says so, rather than reporting
two cells that are the same cell.

**The identity really is the identity.** Swapping a checkpoint into itself must
reproduce it bit for bit -- which is also how the two unmodified reference cells
are produced, so each of them carries its own anchor.

Everything runs on randomly initialised two-layer models built from config
objects with stub tokenizers: no GPU, no network, no 7B checkpoint, as
``tests/test_neuron_basis_circuit.py`` and ``tests/test_perturbation_sensitivity.py``
do. The 7B path is unit-tested here and has never been executed on this host.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/transfer"))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)

from src.transfer import arms as A  # noqa: E402
from src.transfer import joint_modes as JM  # noqa: E402
from src.transfer.budget import MEASURABLE, UNMEASURABLE  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_swap_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("24_component_swap.py")


# ------------------------------------------------------------------ the stubs

D_MODEL, D_MLP, N_LAYER, N_HEAD = 8, 24, 2, 2

DOCUMENTS = [
    "the harbour opened onto a shallow bay and the tide withdrew",
    "a compiler translates one language into another before it runs",
    "rainfall is concentrated in two short seasons of the year",
    "iron rusts when exposed to oxygen and water for long enough",
]

REFERENCE_DOCUMENTS = [
    "the mountain pass is closed to traffic every winter",
    "a ledger records each transaction in the order it arrived",
    "salt marshes buffer the coast against a storm surge",
]

SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
    "MSDTLTRLAEVLEARKGAAPDSSYVASLYHKG",
    "MQPNDITFFQRFQNDILAGRKTITIRDASESH",
]

REFERENCE_SEQUENCES = [
    "GVLTKPQNAWKLFEDSHRTMCYIVGAKLPDEF",
    "TTKLNDWSFQAERGHIPMCYVKLNDWSFQAER",
    "PLKMNVFDSAQWERTYHIGCPLKMNVFDSAQW",
]


class StubJointTokenizer:
    """A vocabulary that merges residues and whose delimiters are not its tokens.

    The two properties of the staged ProLLaMA tokenizer that decide this stage's
    numbers, in stub form: ``Seq=<`` is spelled out of ordinary pieces rather than
    carried as one token, and a residue run comes back as multi-residue pieces, so
    the family's declared symbol unit is the token. ``get_vocab`` is present
    because the stage digests the whole id-to-token map: two checkpoints are only
    interchangeable at the embedding if row *i* means the same symbol in both.
    """

    def __init__(self, *, extra_tokens: tuple[str, ...] = ()) -> None:
        self.bos_id = 2
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {self.bos_id: "<s>"}
        self._add("<unk>")
        for token in ("Seq", "=", "<", ">", "[", "]"):
            self._add(token)
        for residue in A.AA20:
            self._add(residue)
        for left in A.AA20:
            for right in A.AA20:
                self._add(left + right)
        for character in " abcdefghijklmnopqrstuvwxyz.,\n#":
            self._add(character)
        for token in extra_tokens:
            self._add(token)
        self._longest = max(len(token) for token in self._vocab)
        self.unk_id = self._vocab["<unk>"]

    def _add(self, token: str) -> None:
        if token in self._vocab:
            return
        index = len(self._vocab) + 10
        self._vocab[token] = index
        self._inverse[index] = token

    def __len__(self) -> int:
        return len(self._vocab) + 10

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token)

    def convert_ids_to_tokens(self, index: int):
        return self._inverse.get(int(index))

    def _fragment(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            for length in range(min(self._longest, len(text) - cursor), 0, -1):
                candidate = text[cursor : cursor + length]
                if candidate in self._vocab:
                    ids.append(self._vocab[candidate])
                    cursor += length
                    break
            else:
                ids.append(self.unk_id)
                cursor += 1
        return ids

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        return {"input_ids": [self.bos_id, *self._fragment(text)]}


def _llama(vocab: int, seed: int, *, tie: bool = False) -> LlamaForCausalLM:
    """One randomly initialised LLaMA stub. Never cached: the tests mutate it."""

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=D_MODEL,
            intermediate_size=D_MLP,
            num_hidden_layers=N_LAYER,
            num_attention_heads=N_HEAD,
            num_key_value_heads=N_HEAD,
            max_position_embeddings=256,
            tie_word_embeddings=tie,
        )
    )


def _states_equal(left, right) -> bool:
    first, second = left.state_dict(), right.state_dict()
    if set(first) != set(second):
        return False
    return all(torch.equal(first[name], second[name]) for name in first)


VOCAB = len(StubJointTokenizer())


# ------------------------------------------------------- the group declaration


class TheComponentGroupsAreDeclared(unittest.TestCase):
    """A swap is an attribution only if what moved was decided in advance."""

    def setUp(self) -> None:
        self.model = _llama(VOCAB, seed=1)

    def test_an_undeclared_group_is_refused_by_name(self):
        with self.assertRaises(ValueError) as raised:
            STAGE.group_tensor_names(self.model, "mlp")
        message = str(raised.exception)
        self.assertIn("unknown component group", message)
        self.assertIn("vocabulary_interface", message)

    def test_the_four_declared_groups_are_the_ones_the_stage_offers(self):
        self.assertEqual(
            sorted(STAGE.COMPONENT_GROUPS),
            ["body", "embedding", "lm_head", "vocabulary_interface"],
        )
        choices = {
            action.dest: action.choices for action in STAGE.build_parser()._actions
        }
        self.assertEqual(
            sorted(choices["component_group"]), sorted(STAGE.COMPONENT_GROUPS)
        )

    def test_the_interface_is_the_embedding_and_the_head_and_nothing_else(self):
        embedding = set(STAGE.group_tensor_names(self.model, "embedding"))
        head = set(STAGE.group_tensor_names(self.model, "lm_head"))
        interface = set(STAGE.group_tensor_names(self.model, "vocabulary_interface"))
        self.assertEqual(embedding | head, interface)
        self.assertEqual(embedding & head, set())

    def test_the_body_is_the_exact_complement_of_the_interface(self):
        names = set(self.model.state_dict())
        interface = set(STAGE.group_tensor_names(self.model, "vocabulary_interface"))
        body = set(STAGE.group_tensor_names(self.model, "body"))
        self.assertEqual(interface & body, set())
        self.assertEqual(interface | body, names)
        self.assertTrue(body, "a body group that is empty would swap nothing")

    def test_the_names_are_read_off_the_model_and_not_spelled_for_one_family(self):
        # The same declaration resolved against a different architecture. Nothing
        # in the stage spells `embed_tokens` or `wte`; both are located through
        # get_input_embeddings/get_output_embeddings.
        self.assertEqual(
            STAGE.group_tensor_names(self.model, "vocabulary_interface"),
            ("lm_head.weight", "model.embed_tokens.weight"),
        )
        torch.manual_seed(3)
        gpt2 = GPT2LMHeadModel(
            GPT2Config(vocab_size=32, n_positions=64, n_embd=D_MODEL, n_layer=1, n_head=2)
        )
        self.assertEqual(
            STAGE.group_tensor_names(gpt2, "vocabulary_interface"),
            ("lm_head.weight", "transformer.wte.weight"),
        )


# --------------------------------------------------------------- the swap itself


class TheSwapMovesExactlyTheDeclaredTensors(unittest.TestCase):
    def setUp(self) -> None:
        self.host = _llama(VOCAB, seed=1)
        self.donor = _llama(VOCAB, seed=2)
        self.original = copy.deepcopy(self.host)

    def _logits(self, model) -> torch.Tensor:
        ids = torch.tensor([[3, 17, 41, 12, 8]], dtype=torch.long)
        with torch.no_grad():
            return model(ids).logits.clone()

    def test_swapping_a_checkpoint_into_itself_reproduces_it_bit_identically(self):
        # The identity anchor, and the same run that produces a reference cell.
        before = self._logits(self.host)
        for group in sorted(STAGE.COMPONENT_GROUPS):
            record = STAGE.swap_component(self.host, copy.deepcopy(self.original), group)
            self.assertFalse(record["swap_changed_weights"], group)
            self.assertEqual(
                len(record["tensors_identical_before_swap"]),
                record["n_distinct_tensors_written"],
                group,
            )
        self.assertTrue(_states_equal(self.host, self.original))
        self.assertTrue(torch.equal(self._logits(self.host), before))

    def test_swapping_the_vocabulary_interface_and_back_returns_the_original_weights(self):
        record = STAGE.swap_component(self.host, self.donor, "vocabulary_interface")
        self.assertTrue(record["swap_changed_weights"])
        self.assertFalse(_states_equal(self.host, self.original))
        back = STAGE.swap_component(
            self.host, copy.deepcopy(self.original), "vocabulary_interface"
        )
        self.assertTrue(back["swap_changed_weights"])
        self.assertTrue(_states_equal(self.host, self.original))

    def test_only_the_declared_tensors_move(self):
        moved = set(STAGE.group_tensor_names(self.host, "vocabulary_interface"))
        STAGE.swap_component(self.host, self.donor, "vocabulary_interface")
        chimera = self.host.state_dict()
        base, other = self.original.state_dict(), self.donor.state_dict()
        for name in chimera:
            source = other if name in moved else base
            self.assertTrue(torch.equal(chimera[name], source[name]), name)

    def test_a_body_swap_is_the_mirror_image_of_an_interface_swap(self):
        # base body + donor interface, reached from both directions. If the two
        # groups did not partition the state dict these would differ.
        forward = copy.deepcopy(self.original)
        STAGE.swap_component(forward, copy.deepcopy(self.donor), "vocabulary_interface")
        mirror = copy.deepcopy(self.donor)
        STAGE.swap_component(mirror, copy.deepcopy(self.original), "body")
        self.assertTrue(_states_equal(forward, mirror))

    def test_the_chimera_shares_no_storage_with_the_donor(self):
        STAGE.swap_component(self.host, self.donor, "vocabulary_interface")
        after = self.host.state_dict()["model.embed_tokens.weight"].clone()
        with torch.no_grad():
            self.donor.model.embed_tokens.weight.add_(1.0)
        self.assertTrue(
            torch.equal(self.host.state_dict()["model.embed_tokens.weight"], after),
            "the swap aliased the donor's tensor instead of copying its values",
        )

    def test_a_shape_mismatch_is_refused_and_the_host_is_left_intact(self):
        # A late tensor, so an implementation that wrote as it walked would
        # already have modified an early one before it raised.
        self.donor.model.layers[1].mlp.up_proj.weight = torch.nn.Parameter(
            torch.randn(D_MLP + 1, D_MODEL)
        )
        with self.assertRaises(ValueError) as raised:
            STAGE.swap_component(self.host, self.donor, "body")
        message = str(raised.exception)
        self.assertIn("model.layers.1.mlp.up_proj.weight", message)
        self.assertIn(str(D_MLP + 1), message)
        self.assertIn("L24", message)
        self.assertTrue(
            _states_equal(self.host, self.original),
            "a refused swap must leave the host intact, not half written",
        )

    def test_a_dtype_mismatch_is_refused_rather_than_requantised(self):
        weight = self.donor.model.layers[0].mlp.up_proj.weight
        self.donor.model.layers[0].mlp.up_proj.weight = torch.nn.Parameter(
            weight.detach().double()
        )
        with self.assertRaises(ValueError) as raised:
            STAGE.swap_component(self.host, self.donor, "body")
        self.assertIn("dtype", str(raised.exception))
        self.assertTrue(_states_equal(self.host, self.original))

    def test_a_group_already_equal_in_both_checkpoints_is_reported_as_unchanged(self):
        # The swap that runs, produces a complete artefact, and moved nothing.
        with torch.no_grad():
            self.donor.model.embed_tokens.weight.copy_(
                self.host.model.embed_tokens.weight
            )
            self.donor.lm_head.weight.copy_(self.host.lm_head.weight)
        record = STAGE.swap_component(self.host, self.donor, "vocabulary_interface")
        self.assertEqual(
            sorted(record["tensors_identical_before_swap"]),
            sorted(record["tensors_moved"]),
        )
        self.assertFalse(record["swap_changed_weights"])

    def test_a_donor_without_the_named_tensor_is_refused(self):
        thin = copy.deepcopy(self.donor)
        complete = thin.state_dict

        def narrowed():
            state = dict(complete())
            state.pop("model.embed_tokens.weight")
            return state

        thin.state_dict = narrowed
        with self.assertRaises(ValueError) as raised:
            STAGE.swap_component(self.host, thin, "vocabulary_interface")
        self.assertIn("model.embed_tokens.weight", str(raised.exception))


# ------------------------------------------------------------------ tied weights


class TiedEmbeddingsAreOneTensor(unittest.TestCase):
    """Where the embedding and the head share storage they are not two groups."""

    def test_tying_is_read_from_the_tensors_and_not_from_the_config(self):
        model = _llama(VOCAB, seed=1, tie=False)
        self.assertFalse(STAGE.locate_vocabulary_interface(model).tied)
        model.lm_head.weight = model.model.embed_tokens.weight
        location = STAGE.locate_vocabulary_interface(model)
        self.assertTrue(location.tied)
        self.assertFalse(location.declared_tie_word_embeddings)
        self.assertFalse(location.record()["declaration_matches_observation"])

    def test_a_tied_checkpoint_refuses_the_separable_groups_by_name(self):
        model = _llama(VOCAB, seed=1, tie=True)
        for group in ("embedding", "lm_head"):
            with self.assertRaises(ValueError) as raised:
                STAGE.group_tensor_names(model, group)
            message = str(raised.exception)
            self.assertIn("ties its input embedding", message)
            self.assertIn("vocabulary_interface", message)
        self.assertEqual(
            STAGE.locate_vocabulary_interface(model).record()["separable_groups"],
            ["body", "vocabulary_interface"],
        )

    def test_a_tied_checkpoint_still_moves_the_pair_as_one_object(self):
        host = _llama(VOCAB, seed=1, tie=True)
        donor = _llama(VOCAB, seed=2, tie=True)
        record = STAGE.swap_component(host, donor, "vocabulary_interface")
        # Both names are reported because both were declared to move; one write
        # happened because there is only one tensor there.
        self.assertEqual(
            sorted(record["tensors_moved"]),
            ["lm_head.weight", "model.embed_tokens.weight"],
        )
        self.assertEqual(record["n_distinct_tensors_written"], 1)
        self.assertTrue(record["swap_changed_weights"])
        self.assertTrue(
            torch.equal(host.model.embed_tokens.weight, donor.model.embed_tokens.weight)
        )
        self.assertTrue(STAGE.locate_vocabulary_interface(host).tied)

    def test_an_untied_checkpoint_separates_the_two_groups(self):
        model = _llama(VOCAB, seed=1, tie=False)
        location = STAGE.locate_vocabulary_interface(model)
        self.assertTrue(location.separable)
        self.assertEqual(
            location.record()["separable_groups"], sorted(STAGE.COMPONENT_GROUPS)
        )
        self.assertEqual(
            STAGE.group_tensor_names(model, "embedding"), ("model.embed_tokens.weight",)
        )
        self.assertEqual(STAGE.group_tensor_names(model, "lm_head"), ("lm_head.weight",))

    def test_a_body_group_is_still_available_under_tying(self):
        model = _llama(VOCAB, seed=1, tie=True)
        body = set(STAGE.group_tensor_names(model, "body"))
        self.assertNotIn("lm_head.weight", body)
        self.assertNotIn("model.embed_tokens.weight", body)
        self.assertTrue(body)


# ------------------------------------------------------- interchangeability gates


def _facts(**overrides) -> dict:
    facts = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "n_layers": N_LAYER,
        "d_model": D_MODEL,
        "n_heads": N_HEAD,
        "vocab_size": VOCAB,
        "max_position_embeddings": 256,
        "dtype_observed": ["float32"],
    }
    facts.update(overrides)
    return facts


class TheTwoCheckpointsMustBeInterchangeable(unittest.TestCase):
    def test_two_checkpoints_of_one_lineage_pass(self):
        record = STAGE.assert_interchangeable(
            _llama(VOCAB, seed=1),
            _llama(VOCAB, seed=2),
            host_facts=_facts(),
            donor_facts=_facts(),
        )
        self.assertEqual(record["verdict"], "INTERCHANGEABLE")
        self.assertFalse(record["host_vocabulary_interface"]["tied"])

    def test_a_declared_shape_difference_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            STAGE.assert_interchangeable(
                _llama(VOCAB, seed=1),
                _llama(VOCAB, seed=2),
                host_facts=_facts(),
                donor_facts=_facts(n_layers=N_LAYER + 1),
            )
        self.assertIn("n_layers", str(raised.exception))

    def test_a_differing_state_dict_is_refused_even_when_the_facts_agree(self):
        torch.manual_seed(4)
        deeper = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=VOCAB,
                hidden_size=D_MODEL,
                intermediate_size=D_MLP,
                num_hidden_layers=N_LAYER + 1,
                num_attention_heads=N_HEAD,
                num_key_value_heads=N_HEAD,
                max_position_embeddings=256,
                tie_word_embeddings=False,
            )
        )
        with self.assertRaises(ValueError) as raised:
            STAGE.assert_interchangeable(
                _llama(VOCAB, seed=1),
                deeper,
                host_facts=_facts(),
                donor_facts=_facts(),
            )
        self.assertIn("same tensors", str(raised.exception))

    def test_a_tying_mismatch_is_its_own_refusal(self):
        with self.assertRaises(ValueError) as raised:
            STAGE.assert_interchangeable(
                _llama(VOCAB, seed=1, tie=True),
                _llama(VOCAB, seed=2, tie=False),
                host_facts=_facts(),
                donor_facts=_facts(),
            )
        self.assertIn("ties its vocabulary interface", str(raised.exception))

    def test_two_different_vocabularies_are_refused(self):
        with self.assertRaises(ValueError) as raised:
            STAGE.assert_same_vocabulary(
                StubJointTokenizer(), StubJointTokenizer(extra_tokens=("Superfamily",))
            )
        self.assertIn("not the same vocabulary", str(raised.exception))

    def test_one_vocabulary_is_recorded_by_digest_rather_than_assumed(self):
        record = STAGE.assert_same_vocabulary(StubJointTokenizer(), StubJointTokenizer())
        self.assertEqual(record["verdict"], "IDENTICAL")
        self.assertEqual(record["vocabulary_size"], VOCAB)
        self.assertEqual(len(record["vocabulary_sha256"]), 64)
        self.assertNotEqual(
            record["vocabulary_sha256"],
            STAGE.vocabulary_digest(StubJointTokenizer(extra_tokens=("Superfamily",))),
        )


# ----------------------------------------------------------------- the driver


def _stub_draw(kind: str, scored: list[str], reference: list[str]):
    """``arms.protein_cohort``/``text_cohort`` as two disjoint seeded windows."""

    def draw(n, *positional, skip=0, name="", with_ec=False, seed=None):
        records = scored if skip == 0 else reference
        return A.Cohort(
            name,
            kind,
            list(records)[:n],
            8,
            400,
            {
                "sampling": A.sampling_record(
                    seed=seed, skip=skip, requested=n, eligible=64, corpus=f"stub_{kind}"
                )
            },
        )

    return draw


@contextlib.contextmanager
def _patched_corpus():
    """Stage 21 owns the draw, so stage 21 is where the corpus is replaced."""

    saved = (STAGE.STAGE21.protein_cohort, STAGE.STAGE21.text_cohort)
    STAGE.STAGE21.protein_cohort = _stub_draw("protein", SEQUENCES, REFERENCE_SEQUENCES)
    STAGE.STAGE21.text_cohort = _stub_draw("text", DOCUMENTS, REFERENCE_DOCUMENTS)
    try:
        yield
    finally:
        STAGE.STAGE21.protein_cohort, STAGE.STAGE21.text_cohort = saved


def _run_driver(host_model, donor_model, *, host: str, donor: str, group: str) -> dict:
    """``main`` end to end, with only the two loaders and the corpus replaced."""

    tokenizer = StubJointTokenizer()
    # By call order rather than by path, so that a run naming one directory twice
    # can still be handed two different sets of weights -- which is how the
    # loader-nondeterminism refusal is reachable at all.
    loads = iter((host_model, donor_model))
    digests = {host: "a" * 64, donor: ("a" if host == donor else "b") * 64}
    saved = (
        STAGE.STAGE21.load_tokenizer,
        STAGE.STAGE21.load_model,
        STAGE.checkpoint_weights_digest,
        sys.argv,
    )
    STAGE.STAGE21.load_tokenizer = lambda path: (Path(path), tokenizer)
    STAGE.STAGE21.load_model = lambda resolved, tok, *, device, dtype: (
        next(loads),
        _facts(),
    )
    STAGE.checkpoint_weights_digest = lambda path: digests[Path(path).name]
    try:
        with tempfile.TemporaryDirectory() as directory:
            sys.argv = [
                "24_component_swap.py",
                "--host", f"/nowhere/{host}",
                "--donor", f"/nowhere/{donor}",
                "--component-group", group,
                "--rendering", "prollama",
                "--device", "cpu",
                "--donor-device", "cpu",
                "--dtype", "float32",
                "--sequences", str(len(SEQUENCES)),
                "--unigram-sequences", "3",
                "--protein-min-len", "8",
                "--protein-max-len", "400",
                "--text-min-chars", "8",
                "--max-tokens", "128",
                "--out", directory,
            ]
            with _patched_corpus(), contextlib.redirect_stdout(io.StringIO()):
                STAGE.main()
            written = sorted(Path(directory).glob("*.json"))
            assert len(written) == 1, written
            return {
                "name": written[0].name,
                "payload": json.loads(written[0].read_text(encoding="utf-8")),
            }
    finally:
        (
            STAGE.STAGE21.load_tokenizer,
            STAGE.STAGE21.load_model,
            STAGE.checkpoint_weights_digest,
            sys.argv,
        ) = saved


class TheDriverWritesOneArtefactForOneCell(unittest.TestCase):
    """The whole driver on stubs: the part no CPU host can otherwise reach.

    The lineage's checkpoints are 7B and do not fit the free memory of the shared
    cards this repository validates on, so the driver's wiring -- the two loaders,
    the interchangeability gates, the swap and the two-mode measurement -- would
    otherwise be exercised for the first time on a cluster.
    """

    def test_the_artefact_carries_the_moved_tensors_and_both_checkpoint_digests(self):
        host = _llama(VOCAB, seed=1)
        donor = _llama(VOCAB, seed=2)
        result = _run_driver(
            host, donor, host="base", donor="stage1", group="vocabulary_interface"
        )
        payload = result["payload"]

        self.assertEqual(payload["schema_version"], STAGE.SCHEMA_VERSION)
        self.assertEqual(
            result["name"],
            "component_swap__host-base__vocabulary_interface-from-stage1.json",
        )

        chimera = payload["chimera"]
        self.assertEqual(chimera["component_group"], "vocabulary_interface")
        self.assertEqual(
            sorted(chimera["tensors_moved"]),
            ["lm_head.weight", "model.embed_tokens.weight"],
        )
        self.assertTrue(chimera["swap_changed_weights"])
        self.assertGreater(chimera["n_tensors_kept_from_host"], 0)

        self.assertEqual(payload["host"]["weights_sha256"], "a" * 64)
        self.assertEqual(payload["donor"]["weights_sha256"], "b" * 64)
        self.assertEqual(payload["host"]["role"], "host")
        self.assertEqual(payload["donor"]["role"], "donor")
        self.assertFalse(payload["cell"]["is_reference_cell"])

        # Tying is a reported fact, not an assumption.
        self.assertFalse(
            payload["interchangeability"]["host_vocabulary_interface"]["tied"]
        )
        self.assertEqual(payload["interchangeability"]["verdict"], "INTERCHANGEABLE")
        self.assertEqual(payload["tokenizer_vocabulary"]["verdict"], "IDENTICAL")

        # Stage 21's estimand, its cohort record and its verdict convention.
        self.assertEqual(sorted(payload["modes"]), ["protein", "text"])
        for mode, record in payload["modes"].items():
            self.assertIn(record["verdict"], (MEASURABLE, UNMEASURABLE))
            self.assertIn("context_information_nats", record)
            self.assertIn("clean_nll_nats_per_scored_token", json.dumps(record))
            self.assertIn("cross_entropy_nats", record["unigram_reference"])
            self.assertEqual(record["cohort"]["kind"], mode)
            self.assertEqual(
                record["cohort"]["sampling_record"]["seed"], A.DEFAULT_CORPUS_DRAW_SEED
            )
            self.assertEqual(record["cohort"]["band"], [8, 400])
        self.assertEqual(
            payload["modes"]["protein"]["reference_limitation"],
            STAGE.PROTEIN_REFERENCE_LIMITATION,
        )
        self.assertIn("PRE-ADAPTATION REFERENCE", payload["limitations"]["protein_mode_reference"])
        self.assertTrue(
            any("L24" in entry for entry in payload["limitations"]["swap"]),
            "the artefact must carry the shape-coercion limitation",
        )
        self.assertEqual(
            payload["rendering"]["symbol_unit"], JM.TOKEN_UNIT
        )
        self.assertEqual(
            sorted(payload["provenance"]["modules"]), sorted(STAGE.PROVENANCE_MODULES)
        )
        self.assertEqual(
            payload["thresholds"]["minimum_context_information_nats"],
            STAGE.JOINT_MODE_QUALIFICATION_FLOOR_NATS,
        )
        # The magnitude is this lineage's own declared floor and the artefact has
        # to say so: it is deliberately not the calibrated identification floor,
        # and a reader who cannot tell them apart cannot read the verdict.
        self.assertIn(
            "UNDERIVED",
            payload["thresholds"]["minimum_context_information_status"],
        )

    def test_the_reference_cell_is_its_own_identity_anchor(self):
        model = _llama(VOCAB, seed=1)
        result = _run_driver(
            model,
            copy.deepcopy(model),
            host="base",
            donor="base",
            group="vocabulary_interface",
        )
        payload = result["payload"]
        self.assertTrue(payload["cell"]["is_reference_cell"])
        self.assertFalse(payload["chimera"]["swap_changed_weights"])
        self.assertEqual(
            payload["host"]["weights_sha256"], payload["donor"]["weights_sha256"]
        )

    def test_the_measurement_is_taken_on_the_chimera_and_not_on_the_host(self):
        # The swap is in place, so a driver that measured before it -- or that
        # measured a copy -- would report the host's numbers under a chimera's
        # name, and every cell of the campaign would read as the reference.
        model = _llama(VOCAB, seed=1)
        reference = _run_driver(
            copy.deepcopy(model),
            copy.deepcopy(model),
            host="base",
            donor="base",
            group="vocabulary_interface",
        )["payload"]
        chimera = _run_driver(
            copy.deepcopy(model),
            _llama(VOCAB, seed=2),
            host="base",
            donor="stage1",
            group="vocabulary_interface",
        )["payload"]
        for mode in ("text", "protein"):
            self.assertNotEqual(
                reference["modes"][mode]["context_information_nats"],
                chimera["modes"][mode]["context_information_nats"],
                mode,
            )
        # The held-out reference is a property of the cohort, not of the weights,
        # so it must be the same on both sides of the comparison.
        self.assertEqual(
            reference["modes"]["text"]["unigram_reference"]["cross_entropy_nats"],
            chimera["modes"]["text"]["unigram_reference"]["cross_entropy_nats"],
        )

    def test_a_swap_that_moved_something_under_identical_weights_stops_the_run(self):
        # The L24 shape: byte-identical weight files that do not load to identical
        # tensors mean part of the model was initialised at load time.
        host = _llama(VOCAB, seed=1)
        donor = _llama(VOCAB, seed=2)
        with self.assertRaises(RuntimeError) as raised:
            _run_driver(host, donor, host="base", donor="base", group="vocabulary_interface")
        self.assertIn("byte-identical", str(raised.exception))


# ------------------------------------------------------------------- the wiring


class StageWiring(unittest.TestCase):
    def test_the_stage_is_not_registered_in_the_panel_contract(self):
        import panel_contract

        self.assertNotIn("component_swap", panel_contract.STAGE_CONTRACTS)

    def test_the_provenance_modules_exist_and_name_the_qualification_stage(self):
        self.assertIn(
            "scripts/transfer/21_joint_mode_qualification.py", STAGE.PROVENANCE_MODULES
        )
        for name in STAGE.PROVENANCE_MODULES:
            self.assertTrue((REPO / name).exists(), name)

    def test_the_estimand_is_imported_from_stage_21_and_not_restated(self):
        # Appendix B rule 12. A second copy of the estimand, the cohort draw or the
        # loader would make this stage's numbers a different measurement from the
        # qualification figures they are read against.
        source = Path(STAGE.__file__).read_text(encoding="utf-8")
        for name in (
            "def protein_mode",
            "def text_mode",
            "def mode_cohorts",
            "def draw_cohort",
            "def score_positions",
            "def unigram_record",
            "def verdict_record",
            "def load_model",
            "def load_tokenizer",
        ):
            self.assertNotIn(name, source, f"{name} is restated rather than imported")
        for attribute in (
            "protein_mode",
            "text_mode",
            "mode_cohorts",
            "load_tokenizer",
            "load_model",
            "VERDICT_NOTE",
        ):
            self.assertTrue(hasattr(STAGE.STAGE21, attribute), attribute)

    def test_the_cohort_defaults_are_the_qualification_stage_s_own(self):
        # A chimera measured on a different cohort than the checkpoints it is
        # compared with would not be comparable with them at all.
        mine = {action.dest: action.default for action in STAGE.build_parser()._actions}
        theirs = {
            action.dest: action.default for action in STAGE.STAGE21.build_parser()._actions
        }
        for name in (
            "sequences",
            "unigram_sequences",
            "protein_min_len",
            "protein_max_len",
            "text_min_chars",
            "max_tokens",
            "protein_context",
            "cohort_draw_seed",
            "min_context_information",
        ):
            self.assertEqual(mine[name], theirs[name], name)
        self.assertEqual(mine["cohort_draw_seed"], A.DEFAULT_CORPUS_DRAW_SEED)

    def test_the_donor_is_kept_off_the_scoring_device_by_default(self):
        defaults = {action.dest: action.default for action in STAGE.build_parser()._actions}
        self.assertEqual(defaults["donor_device"], "cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
