"""Experiment ids must not go backwards, because concurrent appends make them collide.

Agents append to `docs/EXPERIMENT_LOG.md` at the same time. Each reads the tail,
takes what looks like the next free id, and writes — and two of them reading the
same tail take the same id. That has happened repeatedly, and it corrupts the
record in the worst possible way: two unrelated experiments answer to one name,
so a later reference resolves to whichever one the reader finds first.

**Uniqueness is the wrong invariant.** One experiment legitimately owns several
entries — a pre-registration, its reading, a repair — and forcing distinct ids on
them would break the link the shared id exists to make. What actually fails is
*ordering*: an id assigned from a stale tail is smaller than one already used, or
skips over ids taken in between. So the rule is monotonicity, checked in file
order: the id sequence never decreases and never rises by more than one.

The rule is checked against a frozen list of the violations that already exist.
Historical entries are **not** rewritten to satisfy it — the chronology is the
evidence, and editing it to please a guard would destroy more than the guard
protects. The list is the accepted-exception convention this codebase already
uses for `DELIBERATE_FILE_ORDER` and `STAGED_BUT_NOT_ADMITTED`: every entry is
named, so a new violation cannot hide among the old ones, and the exceptions are
checked to still exist so the list cannot quietly go stale.
"""

from __future__ import annotations

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
#: Recorded rather than repaired. The first eight are the numbering drift of the
#: 2026-07-30 to 2026-08-05 period, when ids were reserved in blocks and several
#: were assigned to work that was never logged under them; the four in the
#: 2026-08-10 band are pre-registration/reading pairs that were separated by
#: other entries appended in between, which is the same shape as the last one and
#: is not a defect.
KNOWN_VIOLATIONS: tuple[tuple[int, int, str], ...] = (
    (66, 68, "ids reserved in a block during the 2026-07-30 audit"),
    (68, 72, "as above"),
    (74, 77, "as above"),
    (77, 86, "as above; EXP-R2-080 to 085 are named inside entries rather than heading them"),
    (89, 91, "as above"),
    (91, 93, "as above"),
    (93, 92, "EXP-R2-092's results entry follows its launch, which 093 was appended between"),
    (93, 86, "EXP-R2-086's completion entry, appended after later work"),
    (93, 115, "the 2026-08-04 block; 094 to 114 head no entry of their own"),
    (115, 128, "as above, 116 to 127 likewise"),
    (152, 150, "EXP-R2-150's reading, appended after 151 and 152 were launched"),
    (154, 152, "EXP-R2-152's reading, appended after 153 and 154"),
    # EXP-R2-171 ran across a session in which other agents appended six entries,
    # so its pre-declarations and readings are interleaved with higher ids. Each
    # continuation is appended in the order it was written, and every one shows up
    # here as a violation against whatever the maximum had reached. The order is
    # the evidence: a pre-declaration counts only because it was written before
    # the numbers existed, and reordering the file to tidy the sequence would
    # destroy exactly the property that makes it worth anything. Four entries,
    # four exceptions, each declared rather than excused by a looser rule --
    # nothing here can distinguish a continuation from a collision structurally,
    # because both are an id that has been seen before, so the separation has to
    # be a human statement and this is it.
    (
        172,
        171,
        "EXP-R2-171's repaired reading, deliberately left after EXP-R2-172",
    ),
    (172, 171, "EXP-R2-171's studentised pre-declaration, left after 172 for the same reason"),
    (178, 171, "EXP-R2-171's studentised reading, appended after 173 to 178 had landed"),
)


def heading_ids() -> list[tuple[str, int]]:
    text = EXPERIMENT_LOG.read_text(encoding="utf-8")
    return [(m.group(1), int(m.group(2))) for m in HEADING.finditer(text)]


def violations(ids: list[int]) -> list[tuple[int, int]]:
    """Every id that is below the running maximum, or more than one above it.

    Against the running maximum rather than the immediate predecessor, and the
    difference is not cosmetic. An entry deliberately left out of order — a
    reading appended after a later experiment, because reordering it would
    falsify the chronology that makes it evidence — is one event, but it breaks
    an adjacent-pair rule twice: once stepping down into it and once stepping
    back up out of it. The second break then falls on whichever innocent entry
    happens to follow, and excusing it would mean excusing a real collision if
    one ever landed there. The running maximum is also exactly what the rule is
    trying to say: take the next id above everything already used.
    """

    found: list[tuple[int, int]] = []
    running = ids[0]
    for current in ids[1:]:
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

    def test_a_repeated_id_is_allowed_because_one_experiment_owns_several_entries(self):
        self.assertEqual(violations([170, 171, 171, 171, 172]), [])

    def test_a_stale_tail_read_that_reuses_an_earlier_id_is_caught(self):
        # The defect itself: two agents read the tail at 170, both take 171, and
        # the second appends after the first has already moved the log to 172.
        self.assertEqual(violations([170, 171, 172, 171]), [(172, 171)])

    def test_a_skipped_id_is_caught(self):
        self.assertEqual(violations([170, 171, 174]), [(171, 174)])

    def test_one_endorsed_interleave_costs_exactly_one_exception(self):
        # Why the rule reads against the running maximum. An entry left out of
        # order on purpose is one event; an adjacent-pair rule would charge it
        # twice and land the second charge on the innocent entry after it, which
        # is where a real collision would otherwise be excused.
        self.assertEqual(violations([170, 171, 172, 171, 173]), [(172, 171)])

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
        found = violations(self.ids)
        known = [(previous, current) for previous, current, _ in KNOWN_VIOLATIONS]
        unexpected = [pair for pair in found if pair not in known]
        self.assertEqual(
            unexpected,
            [],
            "an experiment id was appended out of order. Agents append "
            "concurrently, so a stale tail read gives two experiments one id; "
            "take the next id above the current maximum. Offending transitions: "
            + ", ".join(f"{previous} -> {current}" for previous, current in unexpected),
        )

    def test_every_declared_exception_still_exists(self):
        # The other half of the guard. An exception list that outlives what it
        # excuses stops being a record and starts being a hole.
        found = violations(self.ids)
        stale = [
            f"{previous} -> {current} ({why})"
            for previous, current, why in KNOWN_VIOLATIONS
            if (previous, current) not in found
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
