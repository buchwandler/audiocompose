---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0003
release_version: v0.3.11
kind: fixed
summary: Fixed recomposition of stale handoff failures after VoiceRender rerender
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - audiocompose/variants.py
  - audiocompose/bundle_batch.py
  - audiocompose/cli.py
  - tests/test_bundle_batch_compose.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 3
---

Batch composition now exposes rerender and recompose actions, skips incompatible inputs without invoking VoiceRender, and retries stale handoff failures without force after current artifacts validate. Unrelated and unknown failures remain fail-closed.
