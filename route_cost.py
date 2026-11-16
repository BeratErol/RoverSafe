"""Arazi ve engel açıklığından rota için metre başına maliyet üretir."""

from dataclasses import dataclass
from math import isfinite, hypot
import cv2
import numpy as np


@dataclass(frozen=True)
class RoutingConfig:
    mode: str = 'shortest'
    terrain_weight: float = 4.0
    clearance_weight: float = 1.0

    def __post_init__(self):
        if self.mode not in ('shortest','terrain_cost'):
            raise ValueError('Rota modu shortest veya terrain_cost olmalı.')
        for key in ('terrain_weight','clearance_weight'):
            value = getattr(self,key)
            if type(value) not in (int,float) or not isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f'{key} 0–100 arasında sonlu bir sayı olmalı.')


def build_route_cost(terrain, profile, motion, clearance, dx, dy, config):
    """Zorunlu maskeyi değiştirmez; yalnız uygun merkezlerin maliyetini hesaplar."""
    risk = np.full(motion.shape,np.nan,dtype='float32')
    cost = np.full(motion.shape,np.nan,dtype='float64')
    if not motion.any():
        return risk,cost
    limits = (profile.max_slope_deg,profile.max_roughness_m,profile.max_step_m)
    severity = sum(weight*np.clip(np.nan_to_num(values/limit,nan=1,posinf=1,neginf=1),0,1)
                   for weight,values,limit in zip((.5,.3,.2),terrain.decision_maps(),limits))
    radius = hypot(profile.width_m,profile.length_m)/2+profile.margin_m
    envelope = radius+hypot(dx,dy)/2
    rx,ry = int(np.ceil(envelope/dx)),int(np.ceil(envelope/dy))
    yy,xx = np.mgrid[-ry:ry+1,-rx:rx+1]
    kernel = (np.hypot(xx*dx,yy*dy) <= envelope).astype('uint8')
    # Araç merkezinin yanı sıra tüm dairesel ayak izi çevresindeki en kötü değeri kullan.
    footprint_risk = cv2.dilate(severity.astype('float32'),kernel)
    risk[motion] = footprint_risk[motion]
    proximity = np.clip(radius/np.maximum(clearance[motion],radius),0,1)
    cost[motion] = 1+config.terrain_weight*risk[motion]+config.clearance_weight*proximity
    return risk,cost


def route_integral(route, cost, dx, dy):
    """Simetrik kenar maliyetlerinin toplamı; fiziksel enerji tahmini değildir."""
    return float(sum(hypot((b[0]-a[0])*dx,(b[1]-a[1])*dy)*(cost[a[1],a[0]]+cost[b[1],b[0]])/2
                     for a,b in zip(route,route[1:])))
