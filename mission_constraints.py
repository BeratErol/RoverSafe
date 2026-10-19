"""Görev sınırları ve sabit başlangıç kuralını teknik analizden ayrı tutar."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionConfig:
    mode: str = "select_within_zone"
    fixed_start_pixel: tuple[int, int] | None = None
    exit_pixel: tuple[int, int] | None = None
    rule_context: str = "research_example_not_competition_approval"

    def __post_init__(self):
        if self.mode not in {"select_within_zone", "assess_fixed_start"}:
            raise ValueError("Bilinmeyen görev biçimi.")
        for name in ("fixed_start_pixel", "exit_pixel"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (tuple, list)) or len(value) != 2 or
                                      any(type(item) is not int or item < 0 for item in value)):
                raise ValueError(f"{name} negatif olmayan iki piksel koordinatı olmalı.")
        if self.mode == "assess_fixed_start" and self.fixed_start_pixel is None:
            raise ValueError("Sabit başlangıç değerlendirmesi bir başlangıç noktası gerektirir.")
        if self.mode == "select_within_zone" and self.fixed_start_pixel is not None:
            raise ValueError("Seçim modunda sabit başlangıç verilmez.")
        if not isinstance(self.rule_context, str) or not self.rule_context:
            raise ValueError("Görev bağlamı gerekli.")
