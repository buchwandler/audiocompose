---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.3.9
kind: added
summary:
  Added trimming of confirmed terminal silence from synthesized RenderPlan
  speech before composition
status: accepted
audience: null
scopes: []
source_refs:
  - git:2b61827dd25cf00fd2c3e5f0fd65163fb404e989
paths:
  - README.md
  - audiocompose/integrations/renderplan.py
  - audiocompose/integrations/renderplan_metadata.py
  - audiocompose/speech_boundaries.py
  - pyproject.toml
  - tests/test_renderplan_materialization.py
  - tests/test_speech_boundaries.py
issues: []
prs: []
sources:
  - git:2b61827dd25cf00fd2c3e5f0fd65163fb404e989
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
