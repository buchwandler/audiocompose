---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0001
release_version: v0.3.2
kind: added
summary:
  Added SSMDDoc workspace composition with validated handoffs and reproducible
  artifact reuse
status: accepted
audience: null
scopes: []
source_refs:
  - git:a9f0367e0de4b8d933b32964c43e66a16519e187
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - README.md
  - audiocompose/__init__.py
  - audiocompose/_workspace_policy.py
  - audiocompose/book.py
  - audiocompose/cli.py
  - audiocompose/document.py
  - audiocompose/document_contract.py
  - audiocompose/errors.py
  - docs/architecture.md
  - docs/index.md
  - docs/migration-v0.3.md
  - pyproject.toml
  - tests/installed_renderplan_smoke.py
  - tests/test_compose_book.py
  - tests/test_compose_document.py
  - tests/test_rate_pitch_envelope_quality.py
  - tests/test_renderplan_materialization.py
  - tests/test_schema_resources.py
issues: []
prs: []
sources:
  - git:a9f0367e0de4b8d933b32964c43e66a16519e187
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
