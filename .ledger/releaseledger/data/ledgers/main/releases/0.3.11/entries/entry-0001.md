---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.3.11
kind: changed
summary:
  Changed speech-adjacent pauses to credit up to 150 ms of confirmed retained
  AudioSig tail
status: accepted
audience: null
scopes: []
source_refs:
  - git:585344251340b372ef880df0a2c6767a96070b07
  - tl:task-0032
paths:
  - audiocompose/integrations/renderplan.py
  - audiocompose/integrations/renderplan_metadata.py
  - audiocompose/integrations/voicerender.py
  - audiocompose/speech_boundaries.py
  - README.md
  - docs/architecture.md
  - pyproject.toml
  - tests/test_renderplan_materialization.py
  - tests/test_speech_boundaries.py
  - tests/test_voicerender_materialization.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---
