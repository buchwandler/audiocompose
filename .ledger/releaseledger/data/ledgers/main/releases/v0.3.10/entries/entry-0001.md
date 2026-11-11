---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.3.10
kind: changed
summary:
  Changed VoiceRender handoff validation to require version 0.2.11+ and the
  raw-synthesis artifact policy
status: accepted
audience: null
scopes: []
source_refs:
  - git:2c25827863d80a517f09fb3e2c117b153e514173
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - audiocompose/integrations/voicerender.py
  - pyproject.toml
  - tests/_bundle_fixtures.py
  - tests/installed_renderplan_smoke.py
  - tests/test_bundle_batch_compose.py
  - tests/test_compose_book.py
  - tests/test_compose_document.py
  - tests/test_renderplan_materialization.py
  - tests/test_variants.py
  - tests/test_voicerender_materialization.py
issues: []
prs: []
sources:
  - git:2c25827863d80a517f09fb3e2c117b153e514173
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
