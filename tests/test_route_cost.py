"""Maliyetli A* için bağımsız Dijkstra karşılaştırması ve arazi tercihi testleri."""

import heapq
import unittest
import numpy as np
from elevation_analysis import ElevationResult
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from route_cost import RoutingConfig, route_integral
from start_site_selection import find_route, select_start_site


class RouteCostTests(unittest.TestCase):
    def test_invalid_settings_and_costmaps(self):
        for args in ({'mode':'unsafe'},{'terrain_weight':True},{'terrain_weight':-1},
                     {'clearance_weight':float('nan')},{'terrain_weight':101}):
            with self.assertRaises(ValueError):
                RoutingConfig(**args)
        mask = np.ones((10,12),bool)
        for cost in (np.ones((3,3)),np.full(mask.shape,.9),np.full(mask.shape,np.inf),mask):
            with self.assertRaises(ValueError):
                find_route(mask,(1,5),(10,5),.1,.2,cell_cost=cost)

    def test_costly_shortcut_is_avoided_without_crossing_blocked_cells(self):
        mask = np.ones((15,24),bool)
        mask[0,0] = False
        cost = np.ones(mask.shape)
        cost[5:10,9:15] = 12
        shortest = find_route(mask,(3,7),(20,7),.1,.1)
        preferred = find_route(mask,(3,7),(20,7),.1,.1,cell_cost=cost)
        self.assertGreater(len(preferred),len(shortest))
        self.assertLess(route_integral(preferred,cost,.1,.1),route_integral(shortest,cost,.1,.1))
        self.assertTrue(all(mask[y,x] for x,y in preferred))
        mask[:,12] = False
        self.assertEqual(find_route(mask,(3,7),(20,7),.1,.1,cell_cost=cost),[])

    def test_weighted_astar_matches_independent_dijkstra(self):
        rng = np.random.default_rng(42)
        for _ in range(5):
            mask = rng.random((9,11)) > .15
            start,goal = (0,0),(10,8)
            mask[0,0] = mask[-1,-1] = True
            values = rng.uniform(1,8,mask.shape)
            pending = [(0.0,start)]
            distance = {start:0.0}
            while pending:
                old,(x,y) = heapq.heappop(pending)
                if old != distance[(x,y)]:
                    continue
                for nx,ny,step in ((x-1,y,.1),(x+1,y,.1),(x,y-1,.2),(x,y+1,.2)):
                    if 0 <= nx < 11 and 0 <= ny < 9 and mask[ny,nx]:
                        new = old+step*(values[y,x]+values[ny,nx])/2
                        if new < distance.get((nx,ny),float('inf')):
                            distance[(nx,ny)] = new
                            heapq.heappush(pending,(new,(nx,ny)))
            route = find_route(mask,start,goal,.1,.2,cell_cost=values)
            if goal not in distance:
                self.assertEqual(route,[])
            else:
                self.assertAlmostEqual(route_integral(route,values,.1,.2),distance[goal],places=10)

    def test_terrain_routing_preserves_hard_constraints_and_fixed_start(self):
        shape = (120,160)
        zero = np.zeros(shape,'float32')
        roughness = zero.copy(); roughness[40:80,65:95] = .028
        terrain = ElevationResult(zero,roughness,zero,np.ones(shape,bool),5)
        mask = np.ones(shape,bool)
        mission = MissionConfig(mode='assess_fixed_start',fixed_start_pixel=(35,60),exit_pixel=(130,60))
        options = dict(ground_mask=mask,allowed_mask=mask,start_zone=mask)
        shortest = select_start_site(terrain,RoverProfile(),mission,.1,.1,**options,
                                    routing=RoutingConfig(mode='shortest',terrain_weight=8,clearance_weight=0))
        weighted = select_start_site(terrain,RoverProfile(),mission,.1,.1,**options,
                                    routing=RoutingConfig(mode='terrain_cost',terrain_weight=8,clearance_weight=0))
        self.assertEqual(weighted.best['center_pixel'],[35,60])
        np.testing.assert_array_equal(shortest.traversable,weighted.traversable)
        np.testing.assert_array_equal(shortest.valid_centers,weighted.valid_centers)
        self.assertGreater(weighted.best['route_length_m'],shortest.best['route_length_m'])
        self.assertLess(weighted.best['route_cost_m_equivalent'],shortest.best['route_cost_m_equivalent'])
        self.assertLess(weighted.best['route_max_terrain_risk'],shortest.best['route_max_terrain_risk'])

    def test_single_point_route_and_uniform_cost(self):
        mask = np.ones((10,12),bool)
        cost = np.ones(mask.shape)
        route = find_route(mask,(2,5),(10,5),.1,.2,cell_cost=cost)
        self.assertAlmostEqual(route_integral(route,cost,.1,.2),.8)
        self.assertEqual(find_route(mask,(2,5),(2,5),.1,.2,cell_cost=cost),[(2,5)])
        self.assertEqual(route_integral([(2,5)],cost,.1,.2),0)


if __name__ == '__main__':
    unittest.main()
