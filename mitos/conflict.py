"""The Conflict sensor's core — constants first, pipeline (Phase 2a+) to follow.

This module is the seed of the sync-time Conflict sensor: a safety net inside
``mitos sync`` that judges each parsed decision entry against its undeclared close
neighbours and, at high confidence, surfaces the tension at the accept prompt. The
sensor is advisory — it applies no verb, mutates nothing, and never blocks a commit.

Phase 1a lands only the numeric dials below (the §8 catalog). Later phases fill this
file with the candidate-gathering pipeline (2a) and the Anthropic judgment call (3b).

**Tier-1 leaf, permanently.** This module must never import a higher-tier ``mitos``
module or a heavy dependency (``anthropic``, the Qdrant/genai clients) at module
scope — ``from mitos.conflict import CONFLICT_TOP_K`` must stay cheap forever. When
2a/3b need a client, inject it as a parameter and guard the type annotation behind
``if TYPE_CHECKING:`` (the ``importer.py`` shape). The dep-free import test pins this.
"""

# The §8 constants catalog — the sensor's honesty made numeric. Each value is the
# dial one later phase reads instead of a magic number buried in prose.

CONFLICT_SURFACE_THRESHOLD = 0.85       # CONF-D4 — surface a not-tenable finding only at ≥ this confidence (high precision over recall; a sensor that cries wolf gets muted).
CONFLICT_TOP_K = 5                      # CONF-D2/D7 — cap on the FINAL post-filter batch the LLM judge sees.
CONFLICT_JUDGMENT_TEMPERATURE = 0.3     # CONF-D5 — nuance task; temp-0 over-literalizes the contradiction judgment.
CONFLICT_LLM_TIMEOUT_S = 15             # CONF-D5/D10 — hard cap on the judgment call, 3× the P95 budget ("slow AI is failed AI", P14).
CONFLICT_SIMILARITY_FLOOR = 0.55        # ⚠️ PROVISIONAL — corpus-empirical; calibrated against the §6.3 golden fixtures in Phase 4b (CONF-D2). NOT first-principles-derivable — recall-first, so err low. Do not treat this number as final.
