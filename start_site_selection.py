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
from route_cost import RoutingConfig, build_route_cost, route_integral
from decision_diagnostics import footprint_checks


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
    terrain_risk: np.ndarray
    route_cost_map: np.ndarray
    eligible_centers: np.ndarray | None = None


def _mask(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != bool or value.shape != shape:
        raise ValueError(f"{name} görüntüyle aynı boyutta bool maske olmalı.")
    return value


def _inside(point, shape):
    if point is None:
        return
    if point[0] >= shape[1] or point[1] >= shape[0]:
        raise ValueError("Görev noktası harita dışında.")


def find_route(mask, start, goal, dx, dy, cell_cost=None):
    """Dört komşulu A*: çapraz engel köşelerinden geçmez."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != bool:
        raise ValueError('Rota maskesi iki boyutlu bool dizi olmalı.')
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError('Rota çözünürlüğü pozitif ve sonlu olmalı.')
    if cell_cost is not None:
        if (not isinstance(cell_cost,np.ndarray) or cell_cost.shape != mask.shape
                or cell_cost.dtype.kind not in 'iuf' or not np.isfinite(cell_cost[mask]).all()
                or (cell_cost[mask] < 1).any()):
            raise ValueError('Rota maliyeti uygun hücrelerde sonlu ve en az 1 olmalı.')
    for point in (start, goal):
        if (not isinstance(point, (tuple, list)) or len(point) != 2
                or any(type(item) is not int or item < 0 for item in point)):
            raise ValueError('Rota noktaları negatif olmayan iki tam sayı olmalı.')
        _inside(point, mask.shape)
        if not mask[point[1], point[0]]:
            return []
    start, goal = tuple(start), tuple(goal)
    queue = [(0.0, 0.0, start)]
    cost, previous = {start: 0.0}, {}
    while queue:
        priority, queued_cost, point = heapq.heappop(queue)
        if queued_cost > cost[point]:
            continue
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
            multiplier = 1 if cell_cost is None else (cell_cost[y,x]+cell_cost[ny,nx])/2
            candidate = cost[point] + step*multiplier
            if candidate < cost.get(neighbor, float('inf')):
                cost[neighbor], previous[neighbor] = candidate, point
                estimate = abs(nx-goal[0])*dx + abs(ny-goal[1])*dy
                heapq.heappush(queue, (candidate+estimate, candidate, neighbor))
    return []


def select_start_site(terrain: ElevationResult, profile: RoverProfile, mission: MissionConfig,
                      dx, dy, *, ground_mask, allowed_mask, start_zone, obstacle_mask=None, routing=None):
    routing = RoutingConfig() if routing is None else routing
    if not isinstance(routing,RoutingConfig):
        raise ValueError('routing bir RoutingConfig olmalı.')
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
    terrain_risk,route_cost = build_route_cost(terrain,profile,motion,clearance,dx,dy,routing)
    seg = SegmentationConfig(min_area=1,opening_size=1,rover_width=width,rover_height=height,safety_margin=0)
    zone_centers = find_valid_centers(hard & zone, seg)
    centers = motion & zone_centers
    distance_ok=np.ones(shape,dtype=bool)
    if mission.exit_pixel is not None:
        gx,gy=mission.exit_pixel
        yy,xx=np.ogrid[:shape[0],:shape[1]]
        distance_ok=np.hypot((xx-gx)*dx,(yy-gy)*dy)>=mission.min_start_exit_distance_m
    reachable = np.zeros(shape, dtype=bool)
    scores = np.full(shape, np.nan, dtype=np.float32)
    rejected = {name:int(value.sum()) for name,value in checks.items()}
    rejected['start_centers_below_min_exit_distance']=int((centers & ~distance_ok).sum())
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
            details[name].update(footprint_checks(point,checks,zone,width,height,radius,dx,dy))
            if name == 'exit':
                details[name].pop('footprint_outside_start_zone_cells')
            for label,maps in (('center_observed',(terrain.slope_deg,terrain.roughness_m,terrain.step_m)),
                               ('center_decision',decision_maps)):
                details[name][label] = {key:float(value[y,x]) if np.isfinite(value[y,x]) else None
                    for key,value in zip(('slope_deg','roughness_m','local_step_indicator_m'),maps)}
            details[name]['error_bound_m'] = (float(terrain.error_bound_m[y,x])
                if terrain.error_bound_m is not None and np.isfinite(terrain.error_bound_m[y,x]) else None)
            if name == 'fixed_start':
                details[name]['footprint_on_valid_start_ground'] = bool(zone_centers[y,x])
                details[name]['reachable_start_candidate'] = bool(reachable[y,x])
                details[name]['distance_to_exit_m']=(hypot((x-mission.exit_pixel[0])*dx,(y-mission.exit_pixel[1])*dy)
                    if mission.exit_pixel is not None else None)
                details[name]['minimum_exit_distance_ok']=bool(distance_ok[y,x])
        return StartSiteResult(status,best,hard,centers,reachable,scores,route or [],rejected,details,terrain_risk,route_cost,
                               eligible_centers=reachable & distance_ok)
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
        if not distance_ok[y,x]:
            return result('fixed_start_too_close_to_exit')
        eligible=np.zeros(shape,dtype=bool)
        eligible[y,x]=True
    else:
        eligible=reachable & distance_ok
    if not eligible.any():
        return result('no_start_meets_exit_distance' if reachable.any() else 'no_reachable_candidate')
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
    route=find_route(motion,(x,y),mission.exit_pixel,dx,dy,
                     cell_cost=route_cost if routing.mode == 'terrain_cost' else None)
    if not route:
        return result('route_not_found')
    best={'center_pixel':[x,y], 'score':float(scores[y,x]),
          'distance_to_exit_m':hypot((x-gx)*dx,(y-gy)*dy),
          'terrain_value_kind':'upper_bounds' if terrain.slope_upper_deg is not None else 'nominal',
          'footprint_with_margin_px':[x-width//2,y-height//2,width,height],
          'max_slope_deg':float(worst[0][y,x]),'roughness_m':float(worst[1][y,x]),
          'local_step_indicator_m':float(worst[2][y,x]),'clearance_m':float(clearance[y,x]),
          'route_length_m':float(sum(hypot((b[0]-a[0])*dx,(b[1]-a[1])*dy) for a,b in zip(route,route[1:]))) }
    route_x, route_y = np.asarray(route).T
    best['route_min_clearance_m'] = float(clearance[route_y, route_x].min())
    best['required_motion_radius_m'] = radius
    best['route_cost_m_equivalent'] = route_integral(route,route_cost,dx,dy)
    best['route_max_terrain_risk'] = float(terrain_risk[route_y,route_x].max())
    best['route_mean_terrain_risk'] = float(terrain_risk[route_y,route_x].mean())
    return result('fixed_start_accepted' if mission.mode == 'assess_fixed_start' else 'candidate_selected',best,route)
