---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0002
release_version: v0.3.0
kind: changed
summary:
  Changed RenderPlan/VoiceRender floors and validated VoiceRender v0.2 deferred-control
  handoffs
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - pyproject.toml
  - audiocompose/integrations/voicerender.py
  - audiocompose/integrations/renderplan.py
  - audiocompose/book.py
issues: []
prs: []
sources: []
contributors: []
breaking: true
internal: false
order: 2
---

The base dependency now requires RenderPlan 0.1.2. The optional VoiceRender extra requires VoiceRender 0.2.x and no longer supports 0.1.x. Book composition requires a complete manifest and diagnostics sidecar; only explicitly declared deferred capabilities are accepted, while generic RenderPlan materialization remains strict.
