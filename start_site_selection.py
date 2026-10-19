"""Roverın fiziksel ayak izi ve çıkış bağlantısıyla başlangıç noktası seçer."""

from dataclasses import dataclass
import heapq
from math import hypot
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from elevation_analysis import ElevationResult
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from segmentation import SegmentationConfig, find_valid_centers


@dataclass
class StartSiteResult:
    status: str
    best: dict | None
    traversable: np.ndarray
    valid_centers: np.ndarray
    reachable_centers: np.ndarray
    score_map: np.ndarray
    route_pixels: list[tuple[int, int]]
    rejected_pixels: dict
    point_checks: dict


def _mask(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != bool or value.shape != shape:
        raise ValueError(f"{name} görüntüyle aynı boyutta bool maske olmalı.")
    return value


def _inside(point, shape):
    if point is None:
        return
    if point[0] >= shape[1] or point[1] >= shape[0]:
        raise ValueError("Görev noktası harita dışında.")


def find_route(mask, start, goal, dx, dy):
    """Dört komşulu A*: çapraz engel köşelerinden geçmez."""
    start, goal = tuple(start), tuple(goal)
    queue = [(0.0, start)]
    cost, previous = {start: 0.0}, {}
    while queue:
        priority, point = heapq.heappop(queue)
        if point == goal:
            route = [point]
            while route[-1] != start:
                route.append(previous[route[-1]])
            return route[::-1]
        x, y = point
        for nx, ny, step in ((x-1,y,dx),(x+1,y,dx),(x,y-1,dy),(x,y+1,dy)):
            neighbor = (nx, ny)
            if not (0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0] and mask[ny,nx]):
                continue
            candidate = cost[point] + step
            if candidate < cost.get(neighbor, float('inf')):
                cost[neighbor], previous[neighbor] = candidate, point
                estimate = abs(nx-goal[0])*dx + abs(ny-goal[1])*dy
                heapq.heappush(queue, (candidate+estimate, neighbor))
    return []


def select_start_site(terrain: ElevationResult, profile: RoverProfile, mission: MissionConfig,
                      dx, dy, *, ground_mask, allowed_mask, start_zone, obstacle_mask=None):
    shape = terrain.valid.shape
    ground = _mask(ground_mask, shape, 'Zemin')
    allowed = _mask(allowed_mask, shape, 'İzinli hareket alanı')
    zone = _mask(start_zone, shape, 'Başlangıç bölgesi')
    obstacles = np.zeros(shape, bool) if obstacle_mask is None else _mask(obstacle_mask, shape, 'Nesne')
    for point in (mission.fixed_start_pixel, mission.exit_pixel):
        _inside(point, shape)
    width, height = profile.footprint_pixels(dx, dy)
    decision_maps = terrain.decision_maps()
    checks = {
        'unknown_elevation': ~terrain.valid,
        'not_ground': ~ground,
        'outside_allowed_area': ~allowed,
        'detected_obstacle': obstacles,
        'slope_limit': ~(decision_maps[0] <= profile.max_slope_deg),
        'roughness_limit': ~(decision_maps[1] <= profile.max_roughness_m),
        'step_limit': ~(decision_maps[2] <= profile.max_step_m),
    }
    if terrain.quality_invalid is not None:
        checks['unknown_elevation_quality'] = terrain.quality_invalid
    hard = ~np.logical_or.reduce(list(checks.values()))
    # Hücre köşesine kadar olan belirsizliği düşür; boşluk ölçüsü metre cinsinden.
    clearance = distance_transform_edt(np.pad(hard,1), sampling=(dy,dx))[1:-1,1:-1] - hypot(dx,dy)/2
    radius = hypot(profile.width_m, profile.length_m)/2 + profile.margin_m
    motion = hard & (clearance >= radius)
    seg = SegmentationConfig(min_area=1,opening_size=1,rover_width=width,rover_height=height,safety_margin=0)
    zone_centers = find_valid_centers(hard & zone, seg)
    centers = motion & zone_centers
    reachable = np.zeros(shape, dtype=bool)
    scores = np.full(shape, np.nan, dtype=np.float32)
    rejected = {name:int(value.sum()) for name,value in checks.items()}
    def result(status, best=None, route=None):
        details = {}
        for name,point in (('exit',mission.exit_pixel),('fixed_start',mission.fixed_start_pixel)):
            if point is None:
                continue
            x,y = point
            details[name] = {'pixel':list(point),
                'failed_center_constraints':[key for key,mask in checks.items() if mask[y,x]],
                'usable_clearance_m':max(0.0,float(clearance[y,x])),
                'required_motion_radius_m':radius,'motion_clearance_ok':bool(motion[y,x])}
            for label,maps in (('center_observed',(terrain.slope_deg,terrain.roughness_m,terrain.step_m)),
                               ('center_decision',decision_maps)):
                details[name][label] = {key:float(value[y,x]) if np.isfinite(value[y,x]) else None
                    for key,value in zip(('slope_deg','roughness_m','local_step_indicator_m'),maps)}
            details[name]['error_bound_m'] = (float(terrain.error_bound_m[y,x])
                if terrain.error_bound_m is not None and np.isfinite(terrain.error_bound_m[y,x]) else None)
            if name == 'fixed_start':
                details[name]['footprint_on_valid_start_ground'] = bool(zone_centers[y,x])
                details[name]['reachable_start_candidate'] = bool(reachable[y,x])
        return StartSiteResult(status,best,hard,centers,reachable,scores,route or [],rejected,details)
    if mission.exit_pixel is None:
        return result('exit_goal_required')
    gx,gy = mission.exit_pixel
    if not motion[gy,gx]:
        return result('exit_goal_not_traversable')
    _, labels = cv2.connectedComponents(motion.astype(np.uint8),connectivity=4)
    reachable[:] = centers & (labels == labels[gy,gx])
    if mission.mode == 'assess_fixed_start':
        x,y = mission.fixed_start_pixel
        if not reachable[y,x]:
            return result('fixed_start_rejected')
        eligible=np.zeros(shape,dtype=bool)
        eligible[y,x]=True
    else:
        eligible=reachable
    if not eligible.any():
        return result('no_reachable_candidate')
    kernel=np.ones((height,width),np.uint8)
    # Tam paylı ayak izinin en kötü değerleri puanlanır; tek merkez pikseli değil.
    worst=[cv2.dilate(np.nan_to_num(value,nan=np.inf),kernel) for value in decision_maps]
    value=(0.4*np.clip(1-worst[0]/profile.max_slope_deg,0,1)
           +0.3*np.clip(1-worst[1]/profile.max_roughness_m,0,1)
           +0.1*np.clip(1-worst[2]/profile.max_step_m,0,1)
           +0.2*np.clip((clearance-radius)/max(radius,0.1),0,1))
    scores[eligible]=value[eligible]
    peak=eligible & (scores == scores[eligible].max())
    peak &= clearance == clearance[peak].max()
    y,x=map(int,np.argwhere(peak)[0])
    route=find_route(motion,(x,y),mission.exit_pixel,dx,dy)
    if not route:
        return result('route_not_found')
    best={'center_pixel':[x,y], 'score':float(scores[y,x]),
          'terrain_value_kind':'upper_bounds' if terrain.slope_upper_deg is not None else 'nominal',
          'footprint_with_margin_px':[x-width//2,y-height//2,width,height],
          'max_slope_deg':float(worst[0][y,x]),'roughness_m':float(worst[1][y,x]),
          'local_step_indicator_m':float(worst[2][y,x]),'clearance_m':float(clearance[y,x]),
          'route_length_m':float(sum(hypot((b[0]-a[0])*dx,(b[1]-a[1])*dy) for a,b in zip(route,route[1:]))) }
    return result('fixed_start_accepted' if mission.mode == 'assess_fixed_start' else 'candidate_selected',best,route)
