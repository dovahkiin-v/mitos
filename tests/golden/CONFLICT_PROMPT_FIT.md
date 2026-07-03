# Conflict judgment — SONNET prompt-fit artifact

*Phase 4b of `v-mitos-conflict-sensor-20260613` (§6.2). Records how the shipped SONNET
judge fits the batched ≤`CONFLICT_TOP_K` judgment shape over the §6.3 golden fixtures.*

**This file is a readout, not the source of truth.** It is regenerated from a live probe
run — do not hand-edit the numbers. To refresh:

```bash
./venv/bin/python -m pytest tests/golden/test_conflict_eval_live.py::test_conflict_floor_calibration -s
```

(needs `GEMINI_API_KEY` + `ANTHROPIC_API_KEY` + Qdrant on `:7333`; skips loudly otherwise).
The machine-readable run lands in `tests/golden/reports/conflict-calibration-probe.json`
(gitignored). The reviewed regression numbers live in the committed
`tests/golden/conflict.baseline.metrics.json` and are soft-diffed by
`test_conflict_baseline_diff_soft_gate` (warns, never reds — the Layer B law).

The §6.3 fixtures + this artifact are also the **input to the scheduled Sonnet-vs-Flash
role eval** (CONF-D5, ROADMAP) — 4b records SONNET's fit; it runs no comparison and
builds nothing in `bridge/Mitos/model_eval/`.

---

## Snapshot (probe run, 2026-07-03 · SONNET `claude-sonnet-4-6` · prompt `conflict-tenability-v1`)

Run at the **probe floor `0.0`** (nothing screened by S5) so every candidate is scored.
One batched `messages.create` per proposal; six proposals, batch ≤ `CONFLICT_TOP_K` (5).

### Aggregate quality

| metric | value | reading |
|---|---|---|
| `not_tenable_recall` | **1.00** | every judged genuine contradiction surfaced |
| `not_tenable_precision` | **0.75** | 1 false positive of 4 surfaced — the narrows FP (below) |
| `same_polarity_fp_rate` | **0.00** | the #34 must-not-flag guard held (config base ✗ strict-schema judged tenable) |

### Per-fixture verdict + retrieval similarity + batch cost/latency

| kind | sim | judged | tenable | surfaced | conf | in/out tok | ms |
|---|---|---|---|---|---|---|---|
| genuine-contradiction | 0.8681 | yes | False | **yes** | 0.99 | 1204 / 541 | 11153 |
| same-polarity-agreement | 0.8325 | yes | True | no | 0.98 | 1183 / 446 | 8960 |
| cross-domain-structural | 0.7972 | yes | False | **yes** | 0.95 | 1199 / 521 | 10189 |
| declared-contradiction | — | no (S4 drop) | — | no | — | 1196 / 534 | 10230 |
| global-vs-scoped-narrows | 0.7870 | yes | False | **yes ⚠** | 0.97 | 1178 / 490 | 9559 |
| multilingual | 0.9311 | yes | False | **yes** | 0.99 | 1258 / 466 | 9297 |

`sim` is the candidate's document-space similarity (is_query=False, 2a) — `null` for the
declared-drop (its named candidate was screened by S4 before judgment; the batch still
fired for that proposal's other over-fetched candidates, hence non-null token/latency).

### Token budget

Input ≈ **1180–1260** tokens, output ≈ **446–541** tokens per batched judgment — well
under the CONF-D8 ~3K/batch budget. `cache_read = cache_creation = 0`: prompt caching is
**off** at this surface by design (RF-3 — the sync surface flips it on later; the static
`system` prefix is the cache anchor already in place from 3a).

### Latency (honest fit note)

At the **full ≤5-candidate batch width** (this probe runs at floor 0.0, so batches are
maximal), the judgment call takes ≈ **9–11 s** — **above** the P95 ≤5 s budget (CONF-D8),
though comfortably under the `CONFLICT_LLM_TIMEOUT_S = 15` hard cap ("slow AI is failed
AI", P14). In production the calibrated floor (`0.76`) admits far fewer candidates per
proposal, so real batches are smaller and faster; but the ≤5-width latency is a genuine
prompt-fit datapoint, and per §6.2 the lever (smaller batches / tier change) is the
**scheduled Sonnet-vs-Flash eval's** call, not a 4b tuning act.

---

## The 0.85 surface threshold — validated, not re-derived (D3)

CONF-D4 *pins* `CONFLICT_SURFACE_THRESHOLD = 0.85` by decision; 4b does not move it. The
confidence-calibration curve is the validator — and at n=6 it is **degenerate**: every
judged confidence clusters in the top bin `[0.75, 1.0]` (mean ≈ 0.98). So the curve
**cannot empirically re-derive** 0.85 — it can only confirm 0.85 is *not contradicted*
(no populated discriminating region argues for a different pin) and that denser validation
awaits corpus growth.

**The load-bearing sub-finding:** the narrows false positive surfaced at confidence
**0.97 — above 0.85**. So the threshold does **not** protect against a *confidently-wrong*
judgment; precision protection is the judge's quality, not the gate. Moving 0.85 or
asserting a tight calibration would both be fiction at n=6.

---

## The narrows false positive (D4) — tracked, not screenable by the floor

The SONNET judge ruled the **tenable** global-vs-scoped-narrows pair
(`harbor-all-endpoints-authenticated` ✗ `harbor-health-endpoint-public`) **not-tenable and
surfaced it** (confidence 0.97) — a false positive (`not_tenable_precision = 0.75`).

- **Root cause:** the resolving `Narrows` edge lives on the **candidate**
  (`health-endpoint-public` declares `Narrows: all-endpoints-authenticated`). The judge
  prompt feeds the proposal's + candidate's axiom / rejected_paths / scope, but **not** the
  candidate's declared strong-relationship edges, so the judge is blind to the candidate
  declaring itself a scoped carve-out. `declared_strong_targets` is proposal-forward-only,
  so 2b correctly does not screen it either.
- **The floor cannot screen it (recall-first).** The tenable candidate retrieves at
  similarity **0.7870** — only **0.0102** below the binding cross-domain contradiction
  (**0.7972**, which sets the floor). A floor high enough to screen 0.7870 would leave the
  genuine contradiction with no recall-first drift margin — forbidden.
- **Disposition (tracked, all landed in 4b):**
  - ADR `conflict-narrows-fp-is-judge-gap-not-floor-screenable` (recorded via mitos).
  - ROADMAP follow-on — the structural fix (feed the candidate's declared strong edges to
    the judge: a 3a-renderer + 5a-runtime change) and/or the scheduled Sonnet-vs-Flash tier
    eval (does a different tier judge scope-narrowing correctly?).
  - `VINGA_QUESTIONS` #6 — the precision-invariant flag for review.
- The seeded baseline records the **honest** numbers (the FP included), banded-soft, so it
  neither hides the gap nor reds CI.

---

## The floor calibration (D1/D2) — for reference

The reviewed value + full derivation live in the **Calibration block** on
`mitos/conflict.py`'s `CONFLICT_SIMILARITY_FLOOR`. In brief: recall-first — the highest
cutoff admitting every judged contradiction = `min(0.7972, 0.8681, 0.9311) − margin`
= `0.7972 − 0.03` ≈ `0.7672`, landed **`0.76`** (rounded down, err low). **D2:** the
cross-domain pair *is* the binding minimum (0.7972) but clears the old provisional 0.55
comfortably — the embedding-recall ceiling is narrower than §9 hypothesized; it retrieves
fine in document space, so no embedding-only-recall design signal is logged.
