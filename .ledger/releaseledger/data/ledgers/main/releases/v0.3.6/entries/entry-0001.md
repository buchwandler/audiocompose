---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.3.6
kind: changed
summary: Skipped optional loudness diagnostics by default during bundle composition
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0028
paths:
  - pyproject.toml
  - audiocompose/bundle.py
  - audiocompose/bundle_batch.py
  - audiocompose/cli.py
  - README.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Bundle composition accepts --collect-metrics and --no-collect-metrics. The default now skips diagnostic integrated loudness and sample-peak analysis while retaining the default true-peak ceiling; LUFS-targeted gain still requests integrated loudness. The AudioSig minimum is 0.1.9.
