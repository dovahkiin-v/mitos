# Harbor — Reference Decisions (golden test corpus)

<!-- A synthetic, hand-authored corpus for the mitos golden-dataset harness.
     "Harbor" is a fictional file-sync/storage service. Every entry exists to
     exercise a named graph behaviour; see oracle.reference.json for the intent
     and expected outcomes. Frozen — edit only deliberately, then re-verify. -->
<!-- DO NOT MODIFY ABOVE THIS LINE -->

<!-- BEGIN ENTRIES — newest first -->

### harbor-duomenys-gali-buti-es
**Decided:** Harbor gali saugoti naudotojų duomenis bet kurioje Europos Sąjungos šalyje esančiuose serveriuose.
**Rejected:** Tik Lietuva — brangesnė infrastruktūra ir prastesnis vėlavimas kitų ES šalių naudotojams.
**Scope:** compliance

### harbor-duomenu-saugojimas-lietuvoje
**Decided:** Harbor saugo Lietuvos naudotojų duomenis tik Lietuvoje esančiuose serveriuose.
**Rejected:** Saugojimas bet kur ES — neatitinka nacionalinio duomenų suvereniteto reikalavimo.
**Scope:** compliance

### harbor-cache-is-process-singleton
**Decided:** Harbor's metadata cache is a single process-wide singleton shared across all request handlers.
**Rejected:** A per-request cache — cold on every call, defeating the point of caching hot metadata.
**Scope:** performance

### harbor-no-global-mutable-state
**Decided:** Harbor components hold no global mutable state; every dependency is passed in explicitly.
**Rejected:** Module-level singletons for convenience — they make request isolation and testing unreliable.
**Scope:** architecture

### harbor-config-yaml-strict-schema
**Decided:** Harbor validates its YAML config against a strict schema and refuses to boot on an unknown key.
**Rejected:** Ignoring unknown keys — silently drops a misspelled setting and boots misconfigured.
**Scope:** api

### harbor-config-yaml
**Decided:** Harbor reads its runtime configuration from a single YAML file at a fixed path.
**Rejected:** Environment variables only — unwieldy for the nested config Harbor needs.
**Scope:** api

### harbor-delete-is-immediate-hard
**Decided:** Harbor deletes are immediate and irreversible — the blob and its metadata are purged at once.
**Rejected:** A grace period — regulated tenants require provable immediate erasure on request.
**Scope:** storage

### harbor-delete-is-soft-30d
**Decided:** Harbor deletes are soft: a deleted file is recoverable for 30 days before purge.
**Rejected:** Immediate hard delete — one fat-fingered call loses a customer's data with no recourse.
**Scope:** storage

### harbor-sync-crdt-merge
**Decided:** Harbor merges concurrent edits to the same file with a CRDT so no write is lost.
**Rejected:** Last-write-wins — silently discards one side of a genuine concurrent edit.
**Scope:** sync
**Mechanisms:** crdt
**Contradicts:** harbor-sync-last-write-wins

### harbor-sync-last-write-wins
**Decided:** Harbor resolves concurrent edits to the same file by keeping the last write to arrive.
**Rejected:** CRDT merge — more machinery than the pilot's single-writer-per-file workload needs.
**Scope:** sync

### harbor-observability-otel
**Decided:** Harbor emits logs and metrics through a single OpenTelemetry pipeline.
**Rejected:** Separate ad-hoc exporters per signal — divergent config and no shared trace context.
**Scope:** observability
**Mechanisms:** otel
**Amends:** harbor-structured-logging, harbor-prometheus-metrics

### harbor-prometheus-metrics
**Decided:** Harbor exposes runtime metrics in Prometheus exposition format on a /metrics endpoint.
**Rejected:** Push-based metrics — needs a gateway Harbor's single-node deployment doesn't run.
**Scope:** observability
**Mechanisms:** prometheus

### harbor-structured-logging
**Decided:** Harbor writes structured JSON logs, one object per line.
**Rejected:** Free-text logs — unparseable for the aggregation the ops team needs.
**Scope:** observability

### harbor-webhook-delivery
**Decided:** Harbor delivers change notifications to tenant webhooks with at-least-once retry.
**Rejected:** Fire-and-forget — a transient tenant outage silently drops the notification.
**Scope:** api
**Cites:** harbor-api-versioning

### harbor-api-versioning
**Decided:** Harbor versions its HTTP API with a leading path segment (/v1, /v2).
**Rejected:** Header-based version negotiation — invisible in logs and harder for clients to pin.
**Scope:** api
**Cites:** harbor-storage-is-sqlite
**Depends-On:** harbor-auth-sessions-v3

### harbor-health-endpoint-public
**Decided:** Harbor's /health endpoint is reachable without authentication.
**Rejected:** Authenticating /health — the load balancer's probe has no credential to present.
**Scope:** api
**Narrows:** harbor-all-endpoints-authenticated

### harbor-all-endpoints-authenticated
**Decided:** Every Harbor API endpoint requires an authenticated session.
**Rejected:** Public read endpoints — widens the unauthenticated surface for no pilot need.

### harbor-drop-ftp-gateway
**Decided:** Harbor removes the FTP ingress gateway entirely; all uploads go through the HTTPS API.
**Rejected:** Keeping FTP behind a feature flag — an unauthenticated plaintext surface we would still have to patch.
**Scope:** api
**Corrects:** harbor-legacy-ftp-gateway

### harbor-legacy-ftp-gateway
**Decided:** Harbor accepts uploads over an FTP gateway for legacy clients.
**Rejected:** HTTPS-only from day one — the pilot customers' scanners still spoke FTP.
**Scope:** api

### harbor-premium-exempt-rate-limit
**Decided:** Premium tenants are exempt from the per-account API rate limit.
**Rejected:** A higher fixed ceiling for everyone — still throttles the paying tenants we most want to keep responsive.
**Scope:** api
**Narrows:** harbor-api-rate-limit

### harbor-api-rate-limit
**Decided:** Harbor caps each account at 600 API requests per minute.
**Rejected:** No limit — a single client can starve the shared node.
**Scope:** api

### harbor-blob-key-rotation-quarterly
**Decided:** Harbor rotates blob encryption keys every quarter, re-wrapping data keys under a fresh master key.
**Rejected:** Never rotating — a leaked master key would expose the whole corpus indefinitely.
**Scope:** storage
**Mechanisms:** kms
**Amends:** harbor-blob-encryption-at-rest

### harbor-blob-encryption-at-rest
**Decided:** Harbor encrypts every stored blob at rest with per-blob data keys.
**Rejected:** Relying on disk-level encryption only — leaves blobs readable to any process on the host.
**Scope:** storage
**Mechanisms:** aes-gcm

### harbor-auth-sessions-v3
**Decided:** Harbor authenticates API calls with server-side session tokens looked up per request.
**Rejected:** Sticking with stateless JWTs — no way to revoke a compromised token before expiry.
**Scope:** auth
**Mechanisms:** session-tokens
**Supersedes:** harbor-auth-jwt-v2

### harbor-auth-jwt-v2
**Decided:** Harbor authenticates API calls with short-lived JWTs carrying a tenant claim.
**Rejected:** Long-lived JWTs — widen the blast radius of a leaked token.
**Scope:** auth
**Mechanisms:** jwt
**Supersedes:** harbor-auth-jwt-v1

### harbor-auth-jwt-v1
**Decided:** Harbor authenticates API calls with JWTs.
**Rejected:** Basic auth — credentials on every request, no expiry.
**Scope:** auth
**Mechanisms:** jwt

### harbor-blobs-on-s3
**Decided:** Harbor stores blob payloads in an S3-compatible object store.
**Rejected:** Blobs in SQLite — bloats the metadata DB and wrecks its cache locality.
**Scope:** storage
**Mechanisms:** s3

### harbor-storage-is-sqlite
**Decided:** Harbor keeps file and account metadata in a single embedded SQLite database.
**Rejected:** Postgres — operational weight unjustified for a single-node service.
**Scope:** storage
**Mechanisms:** sqlite
