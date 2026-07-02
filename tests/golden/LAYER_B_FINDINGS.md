# Layer-B retrieval eval — measurement findings

Non-obvious results from the first live runs of the retrieval eval (`test_retrieval_live.py`
over the frozen Harbor corpus, `gemini-embedding-2`). These are the things a future
maintainer needs that the code and fixture files don't say on their own. The per-fixture
rationale lives in the `notes` fields of `oracle.semantic.json`; this file records what we
*learned by measuring*.

## 1. `harbor-storage-is-sqlite` is a corpus hub — never use it as a hard negative

On the first run it cracked the top-5 of **5 of 6 fixtures** (ranks 5, 4, 3, 3, —, 4)
regardless of query topic. Its text ("Harbor keeps file and account metadata in a single
embedded SQLite database") is short, generic, and infra-flavored, so in a ~28-node corpus
the embedding parks it near almost everything. An `expect_absent` slug that fires on nearly
any query measures **corpus hubness, not query-specific discrimination** — it would produce
a permanent, meaningless false-positive floor.

It was the original config-fixture hard negative; it's now **demoted to a `measure_only`
twin** (`probe: config-storage-entanglement`) that records the config/storage entanglement
without gating. If you author new fixtures, do not reach for `harbor-storage-is-sqlite` (or
any similarly generic hub) as a hard negative.

## 2. What makes a good hard negative here (design principle)

A good hard negative is **plausibly-near-but-decisively-wrong**: close enough that a
degraded/drifted embedding pulls it into top-k *first*, wrong enough that a healthy ranker
has no defense for including it. It should sit **just outside top-k on the healthy
baseline** — that's the earliest-firing regression tripwire.

- A *maximally-unrelated* slug (e.g. `harbor-webhook-delivery` for a config query) proves
  nothing: it can't catch drift (miles of margin) and can't calibrate anything.
- The config fixture's hard negative is `harbor-structured-logging`: saturated with
  format/JSON vocabulary (the direct rival to YAML) so drift pulls it in, yet log-output
  format is unambiguously not config-file format. Verified outside top-5 on the healthy
  baseline (thin margin — its cluster-mates `harbor-observability-otel` / `harbor-prometheus-metrics`
  sit at ranks 3/5, so it is the true decision boundary).

## 3. The cross-domain "embedding-recall ceiling" hypothesis was refuted (for this pair)

The spec predicted `harbor-no-global-mutable-state` (architecture) and
`harbor-cache-is-process-singleton` (performance) — opposed in substance, low vocabulary
overlap — likely would NOT co-rank. Two paired `measure_only` probes tested it:

| probe | query | recall@5 | MRR |
|---|---|---|---|
| A (`cross-domain-ceiling-A-lexical`) | "...share global **singletons** or pass dependencies..." | 1.00 | 1.00 |
| B (`cross-domain-ceiling-B-semantic`) | "...manage **state shared between concurrent requests**?" | 1.00 | 0.50 |

Probe B deliberately shares no gift-words ("singleton"/"global"/"dependency") with either
decision. The pair **still co-ranked** (recall 1.0), so the co-ranking is genuinely
*semantic*, not a lexical artifact. The MRR drop (1.0→0.5, first relevant hit slips from
rank 1 to rank 2) shows the semantic pull is real but weaker without the lexical bridge.
The single-probe version would have looked like a lexical fluke; the A/B contrast is what
made the result interpretable. Keep both probes.

## 4. Precision@5 is not a quality signal in this set

Every retrieval fixture has exactly 2 relevant slugs, so precision@5 is capped at 0.4
(2/5) by construction even on a perfect run. Read **recall@k, MRR, and hard-negative FP**
for quality; precision is reported for completeness, not gated.
