"""Compose either supported completed SSMD bundle kind without synthesizing speech."""

from pathlib import Path

from audiocompose import BundleComposeOptions, compose_bundle

bundle = Path("project.ssmd-bundle")
variant = "narration--525856299523"

result = compose_bundle(bundle, variant=variant, options=BundleComposeOptions(chapters="all"))
print("kind:", result.kind)
if result.kind == "book":
    for chapter in result.value.outcomes:
        print(chapter.source_number, chapter.status, chapter.output)
else:
    print(result.value.status, result.value.output)

if result.exit_code != 0:
    raise SystemExit(result.exit_code)
