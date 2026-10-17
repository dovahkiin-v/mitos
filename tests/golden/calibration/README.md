# Layer C calibration record

Mitos compares similarity scores against fixed thresholds: the surface band's two lines,
the record pause floor, and others. This folder holds the measurements those thresholds
are set from.

- `fixtures.json`: labelled fixtures. Each one names a class, a form (`query` or `draft`)
  and the Harbor slugs it targets. The form decides the embedding space, so a value is
  recorded in the space its threshold compares.
- `runs/<run_id>.json`: one recorded run, with every fixture's raw hits, scores and class
  statistics (nearest-rank percentiles), plus the model, commit and constants in force.
- `record.json` (schema 2): the embedding model the record is calibrated for, an index of
  runs, and one entry per calibrated constant. An entry's `boundaries` keys are the runs
  it cites.

## The shape of a run

**Classes nest class → space → sub-block.** A class posed both ways (retired-precedent
as a query and as a draft) has two sub-blocks, and the space is the sub-block's key, not
a field, so a constant reads `classes[class][its space]` and can never read the other
space's values. Each sub-block carries its `measure`, nearest-rank statistics over its
values, and `negatives`: the same statistics over its negative values (slug negatives, or
the per-item `negative_items` of the rejection-target class). Every floor sits between a
class's targets and its hard negatives, so the record keeps both sides, and a calibrated
threshold's boundary must lie inside the range they span together.

**The rejection-target class measures items.** Its fixtures use `measure: "item"` and name
rejected options as `{owner, option}` refs: `option` is a case-sensitive prefix of exactly
one of the owner's items, read without its `(N)` marker. An ordinal is never authored,
because an edit of the line can shift it. Write the colon (`"Postgres:"`) when a shorter
prefix would match two items. `targets` names the owner and any live decision that adopted
the same option; their decision-ranking values are recorded as context. `rejection_form`
says how the query meets the option: for the question its owner settled, or elsewhere with
the owner's reason carrying over or not. Each `negative_items` ref also names its `kind`:
`generic` (do nothing, defer, status quo) or `vocabulary` (another decision's non-generic
item, rejected for another question, naming a different mechanism while sharing the
query's subject words). Both are hard negatives: the rejection-match floor must keep them
out of a section. A re-proposal elsewhere whose reason does not carry
(`elsewhere-does-not-carry`) is never a negative; it stays a target under its form, and the
floor's recall is read over the other two forms only.

**Items are read through the shipped client** (since 5b). The per-item shape is the index
that ships, so it is written and searched through `QdrantRejectionIndex` in production's
shape (payload `{owner_id, model}`), one search over the whole collection that must return
every point, and every hit is labelled by the harness's own map from
`rejections.rejection_points`, never through the code it measures. The same client is bound
as the query reads' rejection index, so every query-form envelope carries the shipped
rejection section, recorded per fixture as `section` and `section_state`, and the read
fails unless that section equals the instrument's own ranking cut at the floor in force and
the section bound. A dense run over the measurement copy binds the copy's own rejection
collection the same way.

**`shapes` compares the two rejection-index shapes** (ruling 2's evidence): one point per
rejected item part, and one point per decision holding its whole field. Each is a throwaway
collection read whole, so no low score is censored. The per-decision shape has no shipped
reader and is read raw with `exact` set, on Harbor only. Per shape the block gives target and
negative statistics, targets by `rejection_form`, the share of target refs ranked first,
and the margin over the best non-target, and the per-item negatives again by `kind`. The
class statistics take the per-item values, the ruled index. Two facts to read it with: a one-item prose field embeds identically in both
shapes (its item text is its whole field), so the shapes differ only on the multi-item
fields; and a numbered item's embedded text is its identity text, which keeps its
`: REJECTED —`. No query contains that word, so it slightly lowers a numbered item's
value relative to a prose one.

## Recording a run

A run names the commit it read, so record it from a committed tree: commit the fixtures,
corpus edits and code first, then run
`MITOS_UPDATE_CALIBRATION=1 ./venv/bin/python -m pytest tests/golden/test_calibration_live.py -k record_run`
serially with Qdrant up, review the run file, and commit it with `record.json` second. Its
provenance then names the first commit with `dirty_tree: false`. Re-running
`test_calibration_classes` afterwards must soft-diff empty: the vectors come from the cache.

`tests/golden/test_calibration_live.py` reads the fixtures (live, Gemini and Qdrant).
`tests/golden/test_calibrated_constants.py` runs offline in CI: every name in its
`CALIBRATED_CONSTANTS` must equal its record value, the record may hold no other
constant, and the record must name the model `mitos/models.py` configures. A second row
derives from `mitos/` source every module-level constant named `*_THRESHOLD`, `*_FLOOR`
or `*_MARGIN`, and requires each to be calibrated, pending or exempt with a reason.
A threshold named any other way escapes that row, so name new ones with those suffixes.

## When to re-run

1. **The embedding model changes.** The offline row fails. Run
   `_calibration_record.set_record_model("<new model>")`, re-run Layer C, and re-derive
   every constant. Old runs stay as history; each constant fails until it cites a run on
   the new model.
2. **A calibrated constant moves.** The offline row fails until the record agrees.
3. **The dense corpus doubles** since the dense run the record cites. Nothing in the
   suite can see this one, so it is tracked by hand.

## Re-deriving a constant

1. Run Layer C (and the dense set) under the current constants.
2. Read each class's distribution and place the boundary between the classes the
   constant separates, in the constant's own space. Where Harbor and the dense set place
   it differently, take the side on which the claim holds on both: the higher boundary
   for `strong` and the rejection floor, the lower for the weak/none line, the dense
   run's for a margin. Record that choice as the entry's `side`.
3. **A floor that arms an escalation** (`_calibration_record.ESCALATION_FLOORS`: the
   rejection-match floor; 7a adds the tombstone floor) also carries a `cry_wolf` block:
   its `ceiling`, the `ratio` of sections shown to sections earned measured on the copy's
   replay at the floor (`appearances / earned`, 4 places), `earned` with its two signals
   `earned_label` and `earned_proxy`, the `reads` measured, and the ids that let anyone
   re-read it (`replay_run`, `copy_state`, `labels_sha256`). A ratio over its ceiling is
   refused: the floor rises to the smallest value at which it is not, recorded as side
   `ceiling` with `raised_from` naming `max(boundaries)`; an unraised floor keeps side
   `higher` and `raised_from: null`. A raised value must still lie inside every cited run's
   range. The rejection floor's boundary is code:
   `_calibration_record.rejection_floor_boundary`, just above every hard negative of the
   corpus, recomputed from the committed Harbor run by an offline row.
4. In one commit: append the run(s) with `MITOS_UPDATE_CALIBRATION=1` (a run is appended,
   never rewritten), write the constant's record entry, add its dotted name to
   `CALIBRATED_CONSTANTS`, remove it from `PENDING`, and replace the comment above the
   constant with a pointer to this record rather than restating numbers.

## What the record does not cover

- `mitos.conflict.CONFLICT_SIMILARITY_FLOOR` keeps its own calibration block in
  `conflict.py`.
- A workspace's `MITOS_MODEL_OVERRIDE_EMBEDDING` reads bands calibrated for the default
  model, and nothing on a read says so.
