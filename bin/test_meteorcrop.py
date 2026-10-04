import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import meteorcrop


class VideoTrackTests(unittest.TestCase):
    def frames(self, start, end, count=8, trail=False):
        frames = np.full((30, 96, 128), 20, dtype=np.uint8)
        for frame in frames:
            cv2.circle(frame, (100, 20), 2, 180, -1)
        for i, point in enumerate(np.linspace(start, end, count)):
            xy = tuple(np.round(point).astype(int))
            cv2.circle(frames[10 + i], xy, 3, 220, -1)
            if trail:
                for j in range(10 + i + 1, 30):
                    cv2.circle(frames[j], xy, 3, 70, -1)
        return frames

    def test_both_temporal_directions(self):
        for start, end in [((20, 30), (85, 65)), ((85, 65), (20, 30))]:
            with self.subTest(start=start):
                result = meteorcrop._fit_video_track(self.frames(start, end))
                self.assertIsNotNone(result)
                np.testing.assert_allclose(result, [start, end], atol=3)

    def test_persistent_trail_does_not_reverse_motion(self):
        result = meteorcrop._fit_video_track(self.frames((85, 65), (20, 30), trail=True))
        self.assertIsNotNone(result)
        self.assertLess(result[1][0], result[0][0])

    def test_static_and_short_flashes_are_ambiguous(self):
        for frames in [self.frames((40, 40), (40, 40)),
                       self.frames((40, 40), (60, 60), count=2),
                       np.full((30, 96, 128), 20, dtype=np.uint8)]:
            self.assertIsNone(meteorcrop._fit_video_track(frames))

    def test_global_brightness_changes_are_not_motion(self):
        frames = np.full((30, 96, 128), 20, dtype=np.uint8)
        frames[10:18] = np.arange(40, 200, 20, dtype=np.uint8)[:, None, None]
        self.assertIsNone(meteorcrop._fit_video_track(frames))

    def test_direction_uses_video_not_spatial_order(self):
        start, end = [20., 30.], [85., 65.]
        with patch.object(meteorcrop, '_get_video_track', return_value=(end, start)):
            self.assertEqual(meteorcrop.refine_video_track(None, None, start, end), (end, start))

    def test_good_axis_keeps_refined_extent(self):
        start, end = [20., 30.], [85., 65.]
        with patch.object(meteorcrop, '_get_video_track', return_value=([25., 33.], [80., 63.])):
            self.assertEqual(meteorcrop.refine_video_track(None, None, start, end), (start, end))

    def test_bad_axis_is_replaced_by_measured_motion(self):
        measured = ([95., 10.], [45., 90.])
        with patch.object(meteorcrop, '_get_video_track', return_value=measured):
            self.assertEqual(meteorcrop.refine_video_track(None, None, [80., 50.], [110., 90.]), measured)

    def test_unrelated_motion_away_from_trail_is_rejected(self):
        start, end = [20., 30.], [85., 65.]
        with patch.object(meteorcrop, '_get_video_track', return_value=([150., 40.], [215., 75.])):
            self.assertEqual(meteorcrop.refine_video_track(None, None, start, end), (start, end))

    def test_ambiguous_motion_keeps_original_geometry(self):
        start, end = [20., 30.], [85., 65.]
        with patch.object(meteorcrop, '_get_video_track', return_value=None):
            self.assertEqual(meteorcrop.refine_video_track(None, None, start, end), (start, end))

    def test_video_resolution_is_scaled_to_pto_coordinates(self):
        frames = self.frames((85, 65), (20, 30))
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_COUNT: len(frames),
            cv2.CAP_PROP_FRAME_WIDTH: 128,
            cv2.CAP_PROP_FRAME_HEIGHT: 96,
        }[key]
        cap.read.side_effect = [(True, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)) for frame in frames] + [(False, None)]
        pto = ({}, [{'w': 256, 'h': 192}])
        with patch.object(meteorcrop.cv2, 'VideoCapture', return_value=cap), \
             patch.object(meteorcrop.pto_mapper, 'parse_pto_file', return_value=pto), \
             patch.object(meteorcrop.pto_mapper, 'map_pano_to_image', side_effect=lambda data, x, y: (0, x, y)), \
             patch.object(meteorcrop.pto_mapper, 'map_image_to_pano', side_effect=lambda data, index, x, y: (x, y)):
            result = meteorcrop._get_video_track('source.mp4', 'base.pto', [40, 60], [170, 130])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [[170, 130], [40, 60]], atol=6)
        cap.release.assert_called_once()

    def test_inconsistent_flashes_are_rejected(self):
        rng = np.random.default_rng(23)
        frames = np.full((30, 96, 128), 20, dtype=np.uint8)
        for i in range(8):
            cv2.circle(frames[10 + i], tuple(rng.integers([10, 10], [118, 86])), 3, 220, -1)
        self.assertIsNone(meteorcrop._fit_video_track(frames))

    def test_crop_roll_maps_temporal_start_to_left_in_all_quadrants(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base.pto'
            output = Path(directory) / 'fireball.pto'
            data = ({'f': 0, 'w': 1920, 'h': 2560, 'v': 45},
                    [{'f': 0, 'w': 1920, 'h': 2560, 'v': 45, 'y': 0, 'p': 0, 'r': 0}])
            meteorcrop.pto_mapper.write_pto_file(data, str(base))
            for angle in range(-180, 180, 30):
                delta = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))]) * 80
                start = np.array([1050., 1400.]) - delta
                end = np.array([1050., 1400.]) + delta
                width, height = meteorcrop.create_fireball_pto(base, output, start, end)
                crop = meteorcrop.pto_mapper.parse_pto_file(str(output))
                mapped = np.array([meteorcrop.pto_mapper.map_image_to_pano(crop, 0, *point)
                                   for point in (start, end)])
                left, right, top, bottom = crop[0]['S']
                with self.subTest(angle=angle):
                    self.assertGreater(mapped[1, 0], mapped[0, 0])
                    self.assertAlmostEqual(mapped[1, 1], mapped[0, 1], places=3)
                    self.assertTrue(np.all((mapped[:, 0] >= left) & (mapped[:, 0] <= right)))
                    self.assertTrue(np.all((mapped[:, 1] >= top) & (mapped[:, 1] <= bottom)))
                    self.assertEqual(right - left, width)
                    self.assertEqual(bottom - top, height)


if __name__ == '__main__':
    unittest.main()
