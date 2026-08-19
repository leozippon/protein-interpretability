"""A *new* experiment id must not go backwards, because concurrent appends make ids collide.

Agents append to `docs/EXPERIMENT_LOG.md` at the same time. Each reads the tail,
takes what looks like the next free id, and writes — and two of them reading the
same tail take the same id. That has happened repeatedly, and it corrupts the
record in the worst possible way: two unrelated experiments answer to one name,
so a later reference resolves to whichever one the reader finds first.

**Uniqueness is the wrong invariant.** One experiment legitimately owns several
entries — a pre-registration, its amendments, its dispatch record, its reading, a
correction — and forcing distinct ids on them would break the link the shared id
exists to make. A reading *must* carry the id of the pre-registration it reads;
that is the whole point of freezing a decision rule under an id.

**Every occurrence is the wrong sequence.** Whenever two pre-registrations are
open at once — the normal state of this programme, not an accident — the earlier
one's later entries land *after* the later one's id, because they are appended
when they were written. Charging that as an out-of-order id makes the check fire
on the convention it is supposed to protect. Between 2026-08-10 and 2026-08-18 it
did so thirty-odd times and detected nothing, and each firing was answered by
another named exception; the list was on its way to becoming the hole its own
convention forbids.

So the rule is monotonicity over the ids in the order each is **first
introduced**: that sequence never decreases and never rises by more than one. A
later entry re-using an id already in the log is a continuation of the experiment
that owns it, and is not a position in the sequence at all.

What that gives up, stated plainly rather than left to be re-derived: an id that
has already been introduced is indistinguishable, by id order alone, from a
legitimate continuation, so a genuinely new experiment that files itself under a
stale id already in the log is not caught here. That case was never caught
reliably anyway — the common collision, where two agents read the same tail and
both take the same next id, sits *at* the running maximum and violated nothing
under either rule. What this file checks is what id order can actually decide:
introductions never go backwards and never skip.

The rule is checked against a frozen list of the violations that already exist.
Historical entries are **not** rewritten to satisfy it — the chronology is the
evidence, and editing it to please a guard would destroy more than the guard
protects. The list is the accepted-exception convention this codebase already
uses for `DELIBERATE_FILE_ORDER` and `STAGED_BUT_NOT_ADMITTED`: every entry is
named, so a new violation cannot hide among the old ones, and the exceptions are
checked to still exist so the list cannot quietly go stale.
"""

from __future__ import annotations

import collections
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_LOG = REPO_ROOT / "docs" / "EXPERIMENT_LOG.md"

#: A section heading that opens an experiment entry.
#:
#: Only the FIRST id of a combined heading is read. `EXP-R2-077/078/079` is one
#: entry covering three experiments, and the ids it does not lead with are not
#: positions in the sequence.
HEADING = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s+—\s+EXP-R2-(\d+)", re.M)

#: Violations of the rule that already exist, as `(previous id, id, why)`.
#:
#: Recorded rather than repaired. All of them are the numbering drift of the
#: 2026-07-30 to 2026-08-05 period, when ids were reserved in blocks and many
#: were assigned to work that never headed an entry of its own. Nothing later
#: appears here: every out-of-order id since then has been a continuation of an
#: experiment already in the log, which the first-occurrence rule admits by
#: construction rather than by declaration.
KNOWN_VIOLATIONS: tuple[tuple[int, int, str], ...] = (
    (66, 68, "ids reserved in a block during the 2026-07-30 audit"),
    (68, 72, "as above"),
    (74, 77, "as above"),
    (77, 86, "as above; EXP-R2-080 to 085 are named inside entries rather than heading them"),
    (89, 91, "as above"),
    (91, 93, "as above"),
    (93, 92, "EXP-R2-092 was launched under a sub-heading inside EXP-R2-091's entry, so its id first heads an entry at its results"),
    (93, 115, "the 2026-08-04 block; 094 to 114 head no entry of their own"),
    (115, 128, "as above, 116 to 127 likewise"),
)


def heading_ids() -> list[tuple[str, int]]:
    text = EXPERIMENT_LOG.read_text(encoding="utf-8")
    return [(m.group(1), int(m.group(2))) for m in HEADING.finditer(text)]


def introductions(ids: list[int]) -> list[int]:
    """The ids in the order each is first introduced, with later re-use dropped.

    An experiment's amendments, dispatch records, readings and corrections all
    carry its id, appended where they were written. Only the entry that first
    puts an id in the log claims a position in the numbering sequence.
    """

    seen: set[int] = set()
    order: list[int] = []
    for identifier in ids:
        if identifier not in seen:
            seen.add(identifier)
            order.append(identifier)
    return order


def violations(ids: list[int]) -> list[tuple[int, int]]:
    """Every id introduced below the running maximum, or more than one above it.

    Against the running maximum rather than the immediate predecessor, and the
    difference is not cosmetic. An id introduced out of order is one event, but
    it breaks an adjacent-pair rule twice: once stepping down into it and once
    stepping back up out of it. The second break then falls on whichever innocent
    entry happens to follow, and excusing it would mean excusing a real collision
    if one ever landed there. The running maximum is also exactly what the rule
    is trying to say: take the next id above everything already used.
    """

    introduced = introductions(ids)
    found: list[tuple[int, int]] = []
    running = introduced[0]
    for current in introduced[1:]:
        if current < running or current > running + 1:
            found.append((running, current))
        running = max(running, current)
    return found


class TheRuleBites(unittest.TestCase):
    """The rule is checked on sequences it must accept and reject, not on the log.

    A guard whose only exercise is the file it guards passes for as long as that
    file happens to be clean, and says nothing about what it would catch. These
    cases are the ones the guard exists for.
    """

    def test_only_the_first_occurrence_of_an_id_enters_the_sequence(self):
        self.assertEqual(introductions([170, 171, 170, 172, 171]), [170, 171, 172])

    def test_a_continuation_appended_after_the_log_moved_on_is_allowed(self):
        # EXP-R2-171 is pre-registered, EXP-R2-172 is registered while it runs,
        # and 171's reading is then appended under its own id. The reading is the
        # answer to a rule frozen under 171 and must carry that id.
        #
        # This is also the blind spot named in the module docstring: a *new*
        # experiment mis-filed under 171 would be admitted here too, because id
        # order cannot tell the two apart. See the docstring for why that is the
        # right trade rather than an oversight.
        self.assertEqual(violations([170, 171, 172, 171, 173]), [])

    def test_a_new_id_introduced_below_the_running_maximum_is_caught(self):
        # The defect itself, and the reason this file exists: an agent reads a
        # stale tail at 168, takes 169 for a NEW experiment, and appends after
        # the log has already reached 172.
        self.assertEqual(violations([170, 171, 172, 169]), [(172, 169)])

    def test_a_skipped_id_is_caught(self):
        self.assertEqual(violations([170, 171, 174]), [(171, 174)])

    def test_a_continuation_neither_absorbs_a_skip_nor_manufactures_one(self):
        # Dropping the continuation from the sequence must not change which
        # transitions are charged: 174 is still a skip past 172, and the 170 in
        # between neither hides it nor adds a spurious step of its own.
        self.assertEqual(violations([170, 171, 172, 170, 174]), [(172, 174)])

    def test_the_rule_reads_against_the_running_maximum(self):
        # Why the rule uses the running maximum. An id introduced out of order is
        # one event; an adjacent-pair rule would charge it twice and land the
        # second charge on the innocent entry after it, which is where a real
        # collision would otherwise be excused.
        self.assertEqual(violations([170, 171, 172, 169, 173]), [(172, 169)])

    def test_a_clean_run_is_accepted(self):
        self.assertEqual(violations([25, 26, 27, 28]), [])

    def test_a_combined_heading_contributes_only_its_leading_id(self):
        text = "## 2026-08-01 — EXP-R2-077/078/079: two owed checks\n\n## 2026-08-01 — EXP-R2-080: x\n"
        self.assertEqual([int(m.group(2)) for m in HEADING.finditer(text)], [77, 80])


class ExperimentIdsAreMonotonic(unittest.TestCase):
    def setUp(self):
        self.entries = heading_ids()
        self.ids = [identifier for _, identifier in self.entries]

    def test_the_log_is_parsed_at_all(self):
        # Guards every assertion below against a heading-format change silently
        # emptying the check: a regex that matches nothing passes vacuously.
        self.assertGreater(len(self.entries), 100, "no experiment headings were parsed")
        # Not a sortedness check: the exception list exists precisely because the
        # sequence is not sorted. What must hold is that the parse reached both
        # ends of the log, so a regex matching only a prefix cannot pass.
        self.assertLess(min(self.ids), 30, "the parse did not reach the earliest entries")
        self.assertGreater(max(self.ids), 150, "the parse did not reach the latest entries")

    def test_no_new_id_goes_backwards_or_skips(self):
        # Counted, not membership-tested. An id is introduced once, so the same
        # transition cannot arise twice and a declaration cannot silently excuse
        # more occurrences than exist; counting also makes a duplicated
        # declaration show up as stale below instead of being absorbed.
        found = collections.Counter(violations(self.ids))
        known = collections.Counter((p, c) for p, c, _ in KNOWN_VIOLATIONS)
        unexpected = sorted((found - known).elements())
        self.assertEqual(
            unexpected,
            [],
            "a new experiment id was introduced out of order. Agents append "
            "concurrently, so a stale tail read gives two experiments one id; "
            "take the next id above the current maximum. A later entry of an "
            "experiment already in the log keeps that experiment's id and is not "
            "checked here. Offending introductions: "
            + ", ".join(f"{previous} -> {current}" for previous, current in unexpected),
        )

    def test_every_declared_exception_still_exists(self):
        # The other half of the guard. An exception list that outlives what it
        # excuses stops being a record and starts being a hole.
        found = collections.Counter(violations(self.ids))
        surplus = collections.Counter((p, c) for p, c, _ in KNOWN_VIOLATIONS) - found
        stale = [
            f"{previous} -> {current} ({why})"
            for previous, current, why in KNOWN_VIOLATIONS
            if surplus[(previous, current)]
        ]
        self.assertEqual(
            stale,
            [],
            "these declared exceptions no longer occur; remove them so the list "
            "keeps excusing only what is really there: " + "; ".join(stale),
        )

    def test_the_dates_never_go_backwards_either(self):
        # The same defect one axis over, and cheaper to state than to argue
        # about: an entry dated before the one above it was appended from a stale
        # tail even if its id happens to fit.
        dates = [date for date, _ in self.entries]
        offenders = [
            f"{previous} then {current}"
            for previous, current in zip(dates, dates[1:])
            if current < previous
        ]
        self.assertEqual(offenders, [], "entry dates go backwards: " + "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
