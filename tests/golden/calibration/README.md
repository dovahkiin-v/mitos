# Layer C calibration record

Mitos compares similarity scores against fixed thresholds: the surface band's two lines,
the record pause floor, and others. This folder holds the measurements those thresholds
are set from.

- `fixtures.json`: labelled fixtures. Each one names a class, a form (`query` or `draft`)
  and the Harbor slugs it targets. The form decides the embedding space, so a value is
  recorded in the space its threshold compares.
- `runs/<run_id>.json`: one recorded run, with every fixture's raw hits, scores and class
  statistics (nearest-rank percentiles), plus the model, commit and constants in force.
- `record.json`: the embedding model the record is calibrated for, an index of runs, and
  one entry per calibrated constant. An entry's `boundaries` keys are the runs it cites.

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
3. In one commit: append the run(s) with `MITOS_UPDATE_CALIBRATION=1` (a run is appended,
   never rewritten), write the constant's record entry, add its dotted name to
   `CALIBRATED_CONSTANTS`, remove it from `PENDING`, and replace the comment above the
   constant with a pointer to this record rather than restating numbers.

## What the record does not cover

- `mitos.conflict.CONFLICT_SIMILARITY_FLOOR` keeps its own calibration block in
  `conflict.py`.
- A workspace's `MITOS_MODEL_OVERRIDE_EMBEDDING` reads bands calibrated for the default
  model, and nothing on a read says so.
