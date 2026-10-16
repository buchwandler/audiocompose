"""Compose an existing VoiceRender variant without synthesizing speech."""

from pathlib import Path

from audiocompose import compose_book

book = Path("my-book.ssmdbook")
variant = "narration--525856299523"

# VoiceRender output must already exist and be complete. If RenderPlan emphasis
# should be applied by AudioCompose, VoiceRender must have used
# --emphasis-policy defer-to-compose during preflight and rendering.
result = compose_book(book, variant=variant, chapters="all")

for chapter in result.outcomes:
    print(chapter.source_number, chapter.status, chapter.output)

if result.exit_code != 0:
    raise SystemExit(result.exit_code)
