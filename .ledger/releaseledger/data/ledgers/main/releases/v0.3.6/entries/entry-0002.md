---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.3.6
kind: changed
summary:
  Raised the shared RenderPlan dependency to 0.2.1 and aligned supported UtterPlan/VoiceRender
  producer floors with the RenderPlan 0.2 contract line
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0029
paths:
  - pyproject.toml
  - README.md
  - docs/architecture.md
  - tests/test_renderplan_materialization.py
  - tests/test_compose_book.py
  - tests/installed_renderplan_smoke.py
  - .github/workflows/tests.yml
  - .github/workflows/python-publish.yml
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---
