"""Tone bridge for Layer 3.

Maps raw valence/arousal measurements into stylistic hints that can be used by
fillers, modulation engines or other high level orchestrators.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToneProfile:
    style: str
    filler_hint: str
    valence_avg: float
    arousal_avg: float
    last_label: str


class ToneStyleBridge:
    """Smooths tone signals and exposes a high-level style descriptor."""

    def __init__(self, smoothing: float = 0.25) -> None:
        self.smoothing = smoothing
        self._valence_avg = 0.5
        self._arousal_avg = 0.5
        self._last_label = "neutral"
        self._last_style = "neutral_support"
        self._last_hint = "keep-balanced"

    def update(self, label: str, valence: float, arousal: float) -> ToneProfile:
        """Update smoothed metrics with the latest observation."""
        self._valence_avg = (1.0 - self.smoothing) * self._valence_avg + self.smoothing * float(valence)
        self._arousal_avg = (1.0 - self.smoothing) * self._arousal_avg + self.smoothing * float(arousal)
        self._last_label = label

        self._last_style, self._last_hint = self._infer_style()

        return ToneProfile(
            style=self._last_style,
            filler_hint=self._last_hint,
            valence_avg=self._valence_avg,
            arousal_avg=self._arousal_avg,
            last_label=self._last_label,
        )

    def snapshot(self) -> ToneProfile:
        return ToneProfile(
            style=self._last_style,
            filler_hint=self._last_hint,
            valence_avg=self._valence_avg,
            arousal_avg=self._arousal_avg,
            last_label=self._last_label,
        )

    def reset(self) -> None:
        self._valence_avg = 0.5
        self._arousal_avg = 0.5
        self._last_label = "neutral"
        self._last_style = "neutral_support"
        self._last_hint = "keep-balanced"

    def _infer_style(self) -> tuple[str, str]:
        v = self._valence_avg
        a = self._arousal_avg

        if v >= 0.65:
            if a >= 0.6:
                return "energetic_positive", "match_energy_positive"
            return "warm_positive", "calm_positive_fillers"
        if v <= 0.35:
            if a >= 0.6:
                return "urgent_support", "short_assurance_fillers"
            return "soft_support", "soothing_fillers"
        if a >= 0.7:
            return "focused_alert", "steadying_fillers"
        if a <= 0.3:
            return "low_energy", "gentle_engagement"
        return "neutral_support", "neutral_fillers"