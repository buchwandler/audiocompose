---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 6
entry_id: entry-0002
release_version: v0.3.11
kind: changed
summary:
  Changed VoiceRender to require >=0.2.12,<0.3 and voicerender.preserve-silence.v3,
  preserving silence for AudioCompose
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - audiocompose/integrations/voicerender.py
  - audiocompose/variants.py
  - audiocompose/bundle_batch.py
  - README.md
  - docs/architecture.md
  - docs/changelog.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---

AudioCompose now requires voicerender.preserve-silence.v3 from VoiceRender >=0.2.12,<0.3. VoiceRender retains renderer semantics while preserving generic silence; AudioCompose owns downstream boundary conditioning, the existing 150 ms tail reserve and pause credit, and composition. Variant and batch diagnostics distinguish rerender from recompose recovery, and a genuine rerender makes stale handoff failures recomposable without force.
