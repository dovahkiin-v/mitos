# Harbor — Reference Decisions (golden test corpus)

<!-- A synthetic, hand-authored corpus for the mitos golden-dataset harness.
     "Harbor" is a fictional file-sync/storage service. Every entry exists to
     exercise a named graph behaviour; see oracle.reference.json for the intent
     and expected outcomes. Frozen — edit only deliberately, then re-verify. -->
<!-- DO NOT MODIFY ABOVE THIS LINE -->

<!-- BEGIN ENTRIES — newest first -->

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
