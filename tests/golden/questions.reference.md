# Harbor — Open Questions (golden test corpus)

<!-- Open-question half of the reference corpus. Committed before the decisions so a
     decision's `resolves` edge (decision→open_question) finds its target. See
     oracle.reference.json (oq_state) for expected Stage-2 parked/resolved state. -->
<!-- DO NOT MODIFY ABOVE THIS LINE -->

<!-- BEGIN ENTRIES — newest first -->

### oq-harbor-multiregion
**Topic:** Whether Harbor should support multi-region replication.
**Questions:** Do pilot tenants need cross-region failover? What consistency model would it use?
**Scope:** storage

### oq-harbor-backup-cadence
**Topic:** How often Harbor should back up the metadata database, and with what retention.
**Questions:** Nightly or continuous? How many days of history are retained?
**Scope:** storage
