"""Roverın metre cinsinden açıklık ve arazi sınırlarını tanımlar."""

from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class RoverProfile:
    name: str = "example_rover_not_verified"
    width_m: float = 0.8
    length_m: float = 1.0
    margin_m: float = 0.2
    max_slope_deg: float = 10.0
    max_roughness_m: float = 0.03
    max_step_m: float = 0.10
    min_cells_across: int = 8
    source: str = "Illustrative engineering settings; not an ITU/ERC/URC specification."
    verified: bool = False

    def __post_init__(self):
        for key in ("width_m", "length_m", "margin_m", "max_slope_deg", "max_roughness_m", "max_step_m"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"{key} negatif olmayan sonlu bir sayı olmalı.")
        if min(self.width_m, self.length_m, self.max_slope_deg, self.max_roughness_m, self.max_step_m) <= 0 or self.max_slope_deg >= 90:
            raise ValueError("Rover boyutları ve sınırları pozitif; eğim sınırı 90'dan küçük olmalı.")
        if type(self.min_cells_across) is not int or self.min_cells_across < 3:
            raise ValueError("Ayak izi çözünürlüğü en az üç hücre olmalı.")
        if not isinstance(self.name, str) or not self.name or not isinstance(self.source, str) or not self.source:
            raise ValueError("Profil adı ve kaynağı gerekli.")
        if type(self.verified) is not bool:
            raise ValueError("verified boolean olmalı.")

    def footprint_pixels(self, dx, dy):
        if min(dx, dy) <= 0 or not all(isfinite(value) for value in (dx, dy)):
            raise ValueError("Piksel aralıkları pozitif metre olmalı.")
        if min(self.width_m / dx, self.length_m / dy) < self.min_cells_across:
            raise ValueError("Yükseklik haritası bu roverın ayak izini değerlendirmek için fazla kaba.")
        # Tek hücre fazlası, hücre merkezleri arasındaki fiziksel açıklığı da kapsar.
        return ceil((self.width_m + 2 * self.margin_m) / dx) + 1, ceil((self.length_m + 2 * self.margin_m) / dy) + 1
