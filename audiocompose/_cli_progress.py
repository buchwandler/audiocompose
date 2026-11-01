from __future__ import annotations

import math
import shutil
import sys
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal, TextIO

from .progress import CompositionProgress, ProgressEstimator, ProgressTiming

if TYPE_CHECKING:
    from .book import BookComposeProgress


ProgressMode = Literal["auto", "always", "never"]


def _split_context(context: str | None) -> tuple[str | None, str | None]:
    if not context:
        return None, None
    slot, separator, label = context.partition(" ")
    if not separator:
        return context, None
    return slot, label or None


def _synchronized(method: Callable[..., None]) -> Callable[..., None]:
    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            method(self, *args, **kwargs)

    return wrapped


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
        heartbeat_interval: float | None = 1.0,
    ) -> None:
        if mode not in ("auto", "always", "never"):
            raise ValueError("progress mode must be 'auto', 'always', or 'never'")
        if heartbeat_interval is not None and (
            not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0
        ):
            raise ValueError("heartbeat interval must be a positive finite number or None")
        self._lock = threading.RLock()
        self._stream = stream if stream is not None else sys.stderr
        try:
            is_tty = bool(self._stream.isatty())
        except (AttributeError, OSError):
            is_tty = False
        self.enabled = mode == "always" or (mode == "auto" and is_tty and not json_mode)
        self._tty = self.enabled and is_tty
        self._clock = clock
        self._heartbeat_interval = heartbeat_interval
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
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @_synchronized
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

    @_synchronized
    def set_phase(self, phase: str) -> None:
        if not self.enabled:
            return
        changed = phase != self._phase
        self._phase = phase
        if changed:
            self._render(force=True)

    @_synchronized
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

    @_synchronized
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
        thread: threading.Thread | None = None
        with self._lock:
            if not self.enabled or self._finished:
                return
            self._finished = True
            self._heartbeat_stop.set()
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
            thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            timeout = (self._heartbeat_interval or 0.0) + 0.1
            thread.join(timeout=timeout)

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (OverflowError, TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _ensure_heartbeat(self) -> None:
        if (
            not self._tty
            or self._finished
            or self._heartbeat_interval is None
            or self._heartbeat_thread is not None
        ):
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="audiocompose-progress-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        interval = self._heartbeat_interval
        if interval is None:
            return
        while not self._heartbeat_stop.wait(interval):
            self._heartbeat_redraw()

    @_synchronized
    def _heartbeat_redraw(self) -> None:
        if not self._tty or self._finished:
            return
        self._render(force=True)

    @_synchronized
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
        if self._tty:
            width = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
            line = self._format_interactive_line(now, width)
            line = line[:width].ljust(width)
            self._stream.write("\r" + line)
            self._line_open = True
        else:
            line = self._format_line(now)
            self._stream.write(line + "\n")
        self._stream.flush()
        self._last_draw_at = now
        self._ensure_heartbeat()

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

    def _format_interactive_line(self, now: float, width: int) -> str:
        elapsed = max(0.0, now - self._started_at)
        slot, variant = _split_context(self._context)
        if self._command == "book" and self._book_event is not None:
            return self._format_interactive_book_line(elapsed, width, slot, variant)

        composition = self._composition
        fraction = composition.fraction if composition is not None else None
        phase = self._phase or "preparing"
        if phase == "writing output" and fraction is not None:
            phase = "composition · writing output"
        if self._phase == "complete":
            phase = "complete"
        if self._phase == "failed":
            fraction = None
        primary = [f"[{slot}]"] if slot else []
        if fraction is not None:
            primary.append(f"{fraction * 100.0:.1f}%")
        primary.extend((phase, self._format_duration(elapsed)))

        secondary: list[str] = []
        timing = self._composition_timing
        if (
            timing is not None
            and timing.status == "estimated"
            and timing.estimated_remaining_seconds is not None
            and self._phase not in {"writing output", "complete", "failed"}
        ):
            secondary.append(f"ETA {self._format_duration(timing.estimated_remaining_seconds)}")
        if composition is not None and composition.total_items:
            secondary.append(f"{composition.completed_items}/{composition.total_items} items")
            if composition.item_id:
                secondary.append(composition.item_id)
        return self._fit_interactive_fields(primary, secondary, variant, width)

    def _format_interactive_book_line(
        self, elapsed: float, width: int, slot: str | None, variant: str | None
    ) -> str:
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

        primary = [f"[{slot}]"] if slot else []
        if fraction is not None:
            primary.append(f"{fraction * 100.0:.1f}%")
        primary.extend((phase, self._format_duration(elapsed)))
        secondary: list[str] = []
        timing = self._book_timing
        if (
            timing is not None
            and timing.status == "estimated"
            and timing.estimated_remaining_seconds is not None
            and phase == "composing"
        ):
            secondary.append(f"ETA {self._format_duration(timing.estimated_remaining_seconds)}")

        composition = event.composition
        if event.chapter_id is not None and event.requested_chapters > 1:
            chapter_number = event.completed_chapters + (0 if outcome is not None else 1)
            chapter_number = min(max(chapter_number, 1), event.requested_chapters)
            secondary.append(f"ch {chapter_number}/{event.requested_chapters}")
            if composition is not None and composition.fraction is not None:
                secondary.append(f"ch {composition.fraction * 100.0:.1f}%")
        if composition is not None and composition.total_items:
            secondary.append(f"{composition.completed_items}/{composition.total_items} items")
            if composition.item_id:
                secondary.append(composition.item_id)
        return self._fit_interactive_fields(primary, secondary, variant, width)

    @staticmethod
    def _fit_interactive_fields(
        primary: list[str], secondary: list[str], variant: str | None, width: int
    ) -> str:
        line = " · ".join(primary)
        for field in secondary:
            candidate = f"{line} · {field}" if line else field
            if len(candidate) <= width:
                line = candidate
        if variant:
            available = width - len(line) - 3
            label = CliProgressRenderer._elide_variant(variant, available)
            if label:
                line = f"{line} · {label}" if line else label
        return line

    @staticmethod
    def _elide_variant(label: str, available: int) -> str | None:
        if available <= 0:
            return None
        if len(label) <= available:
            return label
        if available < 8:
            return None
        suffix_length = min(12, max(4, (available - 3) // 2))
        prefix_length = available - suffix_length - 3
        if prefix_length < 1:
            return None
        return f"{label[:prefix_length]}...{label[-suffix_length:]}"

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
