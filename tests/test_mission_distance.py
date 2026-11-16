"""Görev mesafesinin arazi koşullarını gevşetmeden uygulanmasını doğrular."""

import unittest
import numpy as np
from elevation_analysis import analyze_elevation
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from start_site_selection import select_start_site


class MissionDistanceTests(unittest.TestCase):
    def setUp(self):
        self.terrain=analyze_elevation(np.zeros((120,160)),.1,.1)
        self.ground=np.ones((120,160),dtype=bool)
    def select(self,mission):
        return select_start_site(self.terrain,RoverProfile(),mission,.1,.1,
            ground_mask=self.ground,allowed_mask=np.ones_like(self.ground),start_zone=np.ones_like(self.ground))

    def test_distance_configuration_rejects_invalid_or_unanchored_values(self):
        for value in (-1,float('nan'),float('inf'),True,'2'):
            with self.assertRaises(ValueError): MissionConfig(exit_pixel=(80,60),min_start_exit_distance_m=value)
        with self.assertRaises(ValueError): MissionConfig(min_start_exit_distance_m=1)

    def test_selection_respects_metric_distance_without_changing_motion_map(self):
        baseline=self.select(MissionConfig(exit_pixel=(80,60)))
        result=self.select(MissionConfig(exit_pixel=(80,60),min_start_exit_distance_m=4))
        self.assertEqual(result.status,'candidate_selected')
        self.assertGreaterEqual(result.best['distance_to_exit_m'],4)
        self.assertGreaterEqual(result.best['route_length_m'],4)
        np.testing.assert_array_equal(result.traversable,baseline.traversable)
        np.testing.assert_array_equal(result.reachable_centers,baseline.reachable_centers)
        self.assertFalse(result.eligible_centers[60,80])

    def test_impossible_distance_has_no_short_route_fallback(self):
        result=self.select(MissionConfig(exit_pixel=(80,60),min_start_exit_distance_m=20))
        self.assertEqual(result.status,'no_start_meets_exit_distance')
        self.assertIsNone(result.best)
        self.assertEqual(result.route_pixels,[])
        self.assertTrue(result.reachable_centers.any())
        self.assertFalse(result.eligible_centers.any())

    def test_non_square_cells_use_physical_distance_in_both_axes(self):
        terrain=analyze_elevation(np.zeros((120,160)),.05,.1)
        result=select_start_site(terrain,RoverProfile(),MissionConfig(exit_pixel=(80,60),min_start_exit_distance_m=4),
            .05,.1,ground_mask=self.ground,allowed_mask=self.ground,start_zone=self.ground)
        x,y=result.best['center_pixel']
        self.assertAlmostEqual(result.best['distance_to_exit_m'],np.hypot((x-80)*.05,(y-60)*.1))
        self.assertGreaterEqual(result.best['distance_to_exit_m'],4)

    def test_fixed_start_is_not_relocated_when_too_close(self):
        result=self.select(MissionConfig(mode='assess_fixed_start',fixed_start_pixel=(80,60),
            exit_pixel=(85,60),min_start_exit_distance_m=2))
        self.assertEqual(result.status,'fixed_start_too_close_to_exit')
        self.assertIsNone(result.best)
        self.assertTrue(result.point_checks['fixed_start']['motion_clearance_ok'])
        self.assertFalse(result.point_checks['fixed_start']['minimum_exit_distance_ok'])

    def test_route_detour_does_not_satisfy_straight_line_separation(self):
        self.ground[:90,80]=False
        fixed=dict(mode='assess_fixed_start',fixed_start_pixel=(65,60),exit_pixel=(95,60))
        baseline=self.select(MissionConfig(**fixed))
        self.assertGreater(baseline.best['route_length_m'],4)
        result=self.select(MissionConfig(**fixed,min_start_exit_distance_m=4))
        self.assertEqual(result.status,'fixed_start_too_close_to_exit')

    def test_neighboring_obstacle_explains_valid_center_rejection(self):
        self.ground[60,40]=False
        result=self.select(MissionConfig(mode='assess_fixed_start',fixed_start_pixel=(35,60),exit_pixel=(130,60)))
        point=result.point_checks['fixed_start']
        self.assertEqual(result.status,'fixed_start_rejected')
        self.assertEqual(point['failed_center_constraints'],[])
        self.assertEqual(point['failed_footprint_constraints']['not_ground'],1)
        self.assertEqual(point['failed_motion_envelope_constraints']['not_ground'],1)

    def test_motion_envelope_can_fail_outside_axis_aligned_start_footprint(self):
        self.ground[60,43]=False
        result=self.select(MissionConfig(mode='assess_fixed_start',fixed_start_pixel=(35,60),exit_pixel=(130,60)))
        point=result.point_checks['fixed_start']
        self.assertEqual(point['failed_footprint_constraints'],{})
        self.assertEqual(point['failed_motion_envelope_constraints']['not_ground'],1)
        self.assertFalse(point['motion_clearance_ok'])


if __name__=='__main__': unittest.main()
