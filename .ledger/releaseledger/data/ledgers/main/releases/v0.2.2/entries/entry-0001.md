---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.2.2
kind: added
summary:
  Added optional VoiceRender materialization of validated UtterPlan handoffs
  into generic AudioJob bundles
status: accepted
audience: null
scopes: []
source_refs:
  - git:94c6249e91e2f4b373128982af7e1b6cd6d75319
paths:
  - README.md
  - audiocompose/cli.py
  - audiocompose/errors.py
  - audiocompose/integrations/__init__.py
  - audiocompose/integrations/media.py
  - audiocompose/integrations/voicerender.py
  - docs/architecture.md
  - pyproject.toml
  - tests/test_voicerender_materialization.py
issues: []
prs: []
sources:
  - git:94c6249e91e2f4b373128982af7e1b6cd6d75319
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
