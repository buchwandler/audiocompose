from __future__ import annotations

import math
import shutil
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TextIO

from .progress import CompositionProgress, ProgressEstimator, ProgressTiming

if TYPE_CHECKING:
    from .book import BookComposeProgress


ProgressMode = Literal["auto", "always", "never"]


class CliProgressRenderer:
    """Private stderr-only progress display for AudioCompose rendering commands."""

    _MIN_REFRESH_SECONDS = 0.25

    def __init__(
        self,
        *,
        mode: ProgressMode = "auto",
        json_mode: bool = False,
        command: str = "compose",
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in ("auto", "always", "never"):
            raise ValueError("progress mode must be 'auto', 'always', or 'never'")
        self._stream = stream if stream is not None else sys.stderr
        try:
            is_tty = bool(self._stream.isatty())
        except (AttributeError, OSError):
            is_tty = False
        self.enabled = mode == "always" or (mode == "auto" and is_tty and not json_mode)
        self._tty = self.enabled and is_tty
        self._clock = clock
        now = self._now()
        self._started_at = now
        self._last_draw_at: float | None = None
        self._line_open = False
        self._command = command
        self._context: str | None = None
        self._phase: str | None = None
        self._composition: CompositionProgress | None = None
        self._composition_timing: ProgressTiming | None = None
        self._estimator = ProgressEstimator(clock=clock)
        self._book_event: BookComposeProgress | None = None
        self._book_timing: ProgressTiming | None = None
        self._book_estimator: ProgressEstimator | None = None
        self._book_chapter_id: str | None = None
        self._book_failed = False
        self._finished = False

    def set_context(self, context: str | None) -> None:
        """Set batch context and reset progress estimates for the next variant."""
        if context == self._context:
            return
        self._context = context
        now = self._now()
        self._started_at = now
        self._last_draw_at = None
        self._phase = None
        self._composition = None
        self._composition_timing = None
        self._estimator = ProgressEstimator(clock=self._clock)
        self._book_event = None
        self._book_timing = None
        self._book_estimator = None
        self._book_chapter_id = None
        self._book_failed = False
        if context is not None:
            self._render(force=True)

    def set_phase(self, phase: str) -> None:
        if not self.enabled:
            return
        changed = phase != self._phase
        self._phase = phase
        if changed:
            self._render(force=True)

    def on_composition_progress(self, event: CompositionProgress) -> None:
        if not self.enabled:
            return
        changed = False
        if event.kind == "compose_started":
            phase = "composing RenderPlan" if self._command == "renderplan" else "composing"
            changed = phase != self._phase
            self._phase = phase
        self._composition = event
        self._composition_timing = self._estimator.update(event)
        if event.kind == "compose_completed":
            changed = self._phase != "writing output"
            self._phase = "writing output"
        self._render(force=changed or event.kind in {"compose_started", "compose_completed"})

    def on_book_progress(self, event: BookComposeProgress) -> None:
        if not self.enabled:
            return
        previous_chapter = self._book_chapter_id
        if event.chapter_id is not None:
            self._book_chapter_id = event.chapter_id
        changed = previous_chapter != self._book_chapter_id
        self._book_event = event
        self._composition = event.composition

        if event.composition is not None:
            if self._book_estimator is None or previous_chapter != event.chapter_id:
                self._book_estimator = ProgressEstimator(clock=self._clock)
            self._book_timing = self._book_estimator.update(event.composition)
            phase = (
                "writing output" if event.composition.kind == "compose_completed" else "composing"
            )
            changed = changed or phase != self._phase
            self._phase = phase
        else:
            self._book_timing = None
            if event.kind == "book_started":
                phase = "preparing book"
            elif event.kind == "chapter_started":
                phase = "preparing handoff"
            elif event.kind == "chapter_materialized":
                phase = "AudioJob persisted"
            elif event.kind == "chapter_compose_started":
                phase = "starting composition"
            elif event.kind == "chapter_output_verification_started":
                phase = "verifying output"
            elif event.outcome is not None:
                phase = {
                    "written": "chapter written",
                    "reused": "reused verified output",
                    "failed": "chapter failed",
                    "skipped": "chapter skipped",
                }[event.outcome.status]
                if event.outcome.status in ("failed", "skipped"):
                    self._book_failed = True
            elif event.kind == "book_completed":
                failed = event.details.get("failed", 0)
                skipped = event.details.get("skipped", 0)
                self._book_failed = self._book_failed or bool(failed or skipped)
                phase = "finalizing report"
            else:
                phase = event.kind.replace("_", " ")
            changed = changed or phase != self._phase
            self._phase = phase

        self._render(
            force=changed
            or event.outcome is not None
            or event.kind in {"book_started", "book_completed", "chapter_compose_started"}
            or (event.composition is not None and event.composition.kind == "compose_completed")
        )

    def finish(self, *, success: bool) -> None:
        if not self.enabled or self._finished:
            return
        self._finished = True
        if success:
            self._phase = "complete"
        else:
            self._phase = "failed"
            self._book_failed = True
        self._render(force=True)
        if self._tty and self._line_open:
            self._stream.write("\n")
            self._stream.flush()
            self._line_open = False

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (OverflowError, TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = self._now()
        if (
            not force
            and self._last_draw_at is not None
            and now - self._last_draw_at < self._MIN_REFRESH_SECONDS
        ):
            return
        line = self._format_line(now)
        if self._tty:
            width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            line = line[:width].ljust(width)
            self._stream.write("\r" + line)
            self._line_open = True
        else:
            self._stream.write(line + "\n")
        self._stream.flush()
        self._last_draw_at = now

    def _format_line(self, now: float) -> str:
        elapsed = max(0.0, now - self._started_at)
        if self._command == "book" and self._book_event is not None:
            return self._format_book_line(elapsed)

        composition = self._composition
        fraction = composition.fraction if composition is not None else None
        phase = self._phase or "preparing"
        if phase == "writing output" and fraction is not None:
            phase = "composition · writing output"
        if self._phase == "complete":
            phase = "complete"
        if self._phase == "failed":
            fraction = None
        pieces: list[str] = [f"[{self._context}]"] if self._context else []
        if fraction is not None:
            pieces.append(f"[{fraction * 100.0:5.1f}%]")
        pieces.append(phase)
        if composition is not None and composition.total_items:
            pieces.append(f"{composition.completed_items}/{composition.total_items} items")
            if composition.item_id:
                pieces.append(composition.item_id)
        pieces.append(f"elapsed {self._format_duration(elapsed)}")
        timing = self._composition_timing
        if (
            timing is not None
            and timing.status == "estimated"
            and timing.estimated_remaining_seconds is not None
            and self._phase not in {"writing output", "complete", "failed"}
        ):
            pieces.append(f"ETA {self._format_duration(timing.estimated_remaining_seconds)}")
        return " · ".join(pieces)

    def _format_book_line(self, elapsed: float) -> str:
        event = self._book_event
        assert event is not None
        fraction = event.fraction
        phase = self._phase or "preparing book"
        outcome = event.outcome
        terminal_failure = outcome is not None and outcome.status in ("failed", "skipped")
        if self._phase == "complete" and not self._book_failed:
            fraction = 1.0 if event.requested_chapters > 0 else None
        elif event.kind == "book_completed":
            fraction = None
            phase = "completed with failures" if self._book_failed else "finalizing report"
        elif event.composition is not None and event.composition.kind == "compose_completed":
            fraction = None
        elif outcome is not None and event.completed_chapters >= event.requested_chapters:
            fraction = None
        elif self._phase == "failed" or terminal_failure:
            fraction = None
        pieces: list[str] = [f"[{self._context}]"] if self._context else []
        if fraction is not None:
            pieces.append(f"[{fraction * 100.0:5.1f}%]")
        if event.chapter_id is not None and event.requested_chapters:
            chapter_number = event.completed_chapters + (0 if outcome is not None else 1)
            chapter_number = min(max(chapter_number, 1), event.requested_chapters)
            pieces.append(f"chapter {chapter_number}/{event.requested_chapters}")
        pieces.append(phase)
        composition = event.composition
        if composition is not None and composition.fraction is not None:
            pieces.append(f"chapter {composition.fraction * 100.0:.1f}%")
        pieces.append(f"elapsed {self._format_duration(elapsed)}")
        timing = self._book_timing
        if (
            timing is not None
            and timing.status == "estimated"
            and timing.estimated_remaining_seconds is not None
            and phase == "composing"
        ):
            pieces.append(f"ETA {self._format_duration(timing.estimated_remaining_seconds)}")
        return " · ".join(pieces)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        whole_seconds = int(round(seconds))
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, remaining = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
        return f"{minutes:02d}:{remaining:02d}"
