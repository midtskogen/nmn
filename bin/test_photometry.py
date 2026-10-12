import itertools

import numpy as np
import pytest

import photometry
from photometry import photometric_mass


@pytest.mark.parametrize('magnitude,range_km', [(0.0, 100.0), (-2.5, 200.0), (1.0, 0.5)])
@pytest.mark.parametrize('speed_ms', [1000.0, 20000.0, 80000.0, 300000.0])
def test_constant_curve_matches_analytic_mass(magnitude, range_km, speed_ms):
    duration = 3.0
    tau = min(0.20, max(0.001, 0.0007 * speed_ms / 1000.0))
    luminosity = 1500.0 * (range_km / 100.0) ** 2 * 10 ** (-0.4 * magnitude)
    expected = 2.0 * luminosity * duration / (tau * speed_ms ** 2)

    actual = photometric_mass([magnitude] * 3, [range_km] * 3,
                              [10.0, 11.0, 13.0], [speed_ms] * 3)

    assert actual == pytest.approx(expected)


def test_doubling_duration_doubles_mass():
    args = ([0.0] * 3, [100.0] * 3)
    mass = photometric_mass(*args, [0.0, 1.0, 2.0], [20000.0] * 3)
    doubled = photometric_mass(*args, [0.0, 2.0, 4.0], [20000.0] * 3)

    assert doubled == pytest.approx(2.0 * mass)


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_variable_speed_integrates_pointwise_mass_loss_in_time_order(order):
    times = [100.0, 101.0, 104.0]
    magnitudes = [0.0, -2.5, 0.0]
    ranges = [100.0, 100.0, 200.0]
    speeds = [10000.0, 20000.0, 40000.0]
    rates = [2.0 * 1500.0 / (0.007 * 10000.0 ** 2),
             2.0 * 15000.0 / (0.014 * 20000.0 ** 2),
             2.0 * 6000.0 / (0.028 * 40000.0 ** 2)]
    expected = (rates[0] + rates[1]) / 2.0 + 3.0 * (rates[1] + rates[2]) / 2.0

    actual = photometric_mass([magnitudes[i] for i in order],
                              [ranges[i] for i in order],
                              [times[i] for i in order],
                              [speeds[i] for i in order])

    assert actual == pytest.approx(expected)
    assert actual > 0


@pytest.mark.parametrize('column', range(4))
@pytest.mark.parametrize('invalid', [None, float('nan'), float('inf'), -float('inf')])
def test_nonfinite_or_missing_samples_are_filtered(column, invalid):
    inputs = [[0.0] * 3, [100.0] * 3, [0.0, 1.0, 2.0], [20000.0] * 3]
    inputs[column][1] = invalid
    expected = photometric_mass([0.0] * 2, [100.0] * 2,
                                [0.0, 2.0], [20000.0] * 2)

    assert photometric_mass(*inputs) == pytest.approx(expected)

    inputs[column][0] = invalid
    assert photometric_mass(*inputs) is None


@pytest.mark.parametrize('column', [1, 3])
@pytest.mark.parametrize('invalid', [0.0, -1.0])
def test_nonpositive_range_and_speed_are_filtered(column, invalid):
    inputs = [[0.0] * 3, [100.0] * 3, [0.0, 1.0, 2.0], [20000.0] * 3]
    inputs[column][1] = invalid
    expected = photometric_mass([0.0] * 2, [100.0] * 2,
                                [0.0, 2.0], [20000.0] * 2)

    assert photometric_mass(*inputs) == pytest.approx(expected)

    inputs[column][0] = invalid
    assert photometric_mass(*inputs) is None


@pytest.mark.parametrize('times', [[], [1.0], [1.0, 1.0], [2.0, 2.0, 2.0],
                                   [0.0, 61.0], [61.0, 0.0]])
def test_insufficient_or_invalid_duration_returns_none(times):
    assert photometric_mass([0.0] * len(times), [100.0] * len(times),
                            times, [20000.0] * len(times)) is None


def test_duplicate_times_do_not_add_duration():
    actual = photometric_mass([0.0] * 4, [100.0] * 4,
                              [2.0, 0.0, 1.0, 1.0], [20000.0] * 4)
    expected = photometric_mass([0.0] * 2, [100.0] * 2,
                                [0.0, 2.0], [20000.0] * 2)

    assert actual == pytest.approx(expected)


# --- range_to_track -------------------------------------------------------


def _geo_obs():
    return 10.0, 60.0, 0.0          # lon, lat, elev


def test_range_to_track_clamps_beyond_segment_end():
    lon, lat, elev = _geo_obs()
    r_start = photometry.llh2ecef(10.0, 60.5, 100000.0)
    r_end = photometry.llh2ecef(10.0, 60.55, 95000.0)
    # LOS points south, away from the segment entirely
    rng, frac = photometry.range_to_track(lon, lat, elev, 180.0, 0.0,
                                          r_start, r_end)
    r_obs = photometry.llh2ecef(lon, lat, elev)
    assert frac > 1.0
    assert rng == pytest.approx(
        min(np.linalg.norm(r_start - r_obs),
            np.linalg.norm(r_end - r_obs)) / 1000.0, rel=1e-3)


def test_range_to_track_in_segment_unchanged():
    lon, lat, elev = _geo_obs()
    r_start = photometry.llh2ecef(10.0, 60.5, 100000.0)
    r_end = photometry.llh2ecef(10.0, 60.55, 95000.0)
    rng, frac = photometry.range_to_track(lon, lat, elev, 0.0, 60.0,
                                          r_start, r_end)
    assert 0.0 <= frac <= 1.0
    assert 100.0 < rng < 130.0


# --- video photometry -----------------------------------------------------


def _blob_frames(n=12, bg=20.0, start=(10.0, 32.0), step=5.0, flux=5000.0):
    frames = []
    for i in range(n):
        g = np.full((64, 96), bg, dtype=np.float32)
        x, y = start[0] + step * i, start[1]
        yy, xx = np.mgrid[0:64, 0:96]
        g += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 6.0)
        frames.append(g)
    return frames


def test_video_detections_finds_moving_blob():
    frames = _blob_frames()
    det, mean = photometry._video_detections(iter(frames),
                                             (0, 0), (96, 64), thresh=30.0)
    assert mean is not None
    assert np.median(mean) == pytest.approx(20.0, abs=5.0)
    assert len(det) >= 8
    # centroid tracks the moving blob
    xs = [d[1] for d in det]
    assert xs[-1] - xs[0] > 20


def test_video_photometry_matches_trail_points(monkeypatch, tmp_path):
    frames = _blob_frames()
    monkeypatch.setattr(photometry, '_find_event_video',
                        lambda d: tmp_path / 'v-orig.mp4')
    monkeypatch.setattr(photometry, '_read_gray_frames',
                        lambda v: iter(frames))
    monkeypatch.setattr(photometry, '_video_zeropoint',
                        lambda *a, **k: (20.0, 3, 'transfer'))
    positions = np.array([[10.0 + 5.0 * i, 32.0] for i in range(10)])
    vid = photometry._video_photometry(tmp_path, positions, None,
                                       60.0, 10.0, 0.0, None, [])
    assert vid is not None
    assert vid['n_matched'] == len(positions)
    assert all(f and f > 0 for f in vid['flux'])


def test_video_photometry_rejects_foreign_positions(monkeypatch, tmp_path):
    frames = _blob_frames()
    monkeypatch.setattr(photometry, '_find_event_video',
                        lambda d: tmp_path / 'v-orig.mp4')
    monkeypatch.setattr(photometry, '_read_gray_frames',
                        lambda v: iter(frames))
    # positions in a different image space (gnomonic): far from the blob
    positions = np.array([[900.0 + i, 800.0 + i] for i in range(10)])
    vid = photometry._video_photometry(tmp_path, positions, None,
                                       60.0, 10.0, 0.0, None, [])
    assert vid is None


def test_video_photometry_flags_saturation(monkeypatch, tmp_path):
    frames = _blob_frames(flux=90000.0)      # drives peak to 255 clip
    for g in frames:
        np.clip(g, 0, 255, out=g)
    monkeypatch.setattr(photometry, '_find_event_video',
                        lambda d: tmp_path / 'v-orig.mp4')
    monkeypatch.setattr(photometry, '_read_gray_frames',
                        lambda v: iter(frames))
    monkeypatch.setattr(photometry, '_video_zeropoint',
                        lambda *a, **k: (20.0, 3, 'transfer'))
    positions = np.array([[10.0 + 5.0 * i, 32.0] for i in range(10)])
    vid = photometry._video_photometry(tmp_path, positions, None,
                                       60.0, 10.0, 0.0, None, [])
    assert vid is not None
    assert any(vid['sat'])


# --- camera_photometry zero-signal guard ----------------------------------


def _make_cam_dir(tmp_path):
    from PIL import Image
    cam = tmp_path / 'cam1'
    cam.mkdir()
    (cam / 'event.txt').write_text(
        '[trail]\n'
        'positions = 10,10 20,20 30,30\n'
        'timestamps = 1000.0 1000.1 1000.2\n'
        'coordinates = 10.0,20.0 10.1,19.9 10.2,19.8\n'
        'brightness = 0.0 0.0 0.0\n'
        '[summary]\nlatitude = 60.0\nlongitude = 10.0\nelevation = 0\n')
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(
        cam / 'x-gnomonic.jpg')
    (cam / 'gnomonic_corr_grid.pto').write_text('p f2 w36000 h18000 v360\n')
    return cam


def test_camera_photometry_zero_brightness_yields_no_signal(
        monkeypatch, tmp_path):
    cam = _make_cam_dir(tmp_path)
    monkeypatch.setattr(photometry, '_star_zeropoint',
                        lambda *a, **k: (10.0, 3, 0.1))
    monkeypatch.setattr(photometry, '_catalog_stars', lambda *a, **k: [])
    monkeypatch.setattr(photometry, '_video_photometry',
                        lambda *a, **k: None)
    r = photometry.camera_photometry(cam, use_video=True)
    assert r is not None
    assert all(m is None for m in r['m_app'])
    assert r['n_signal'] == 0
    assert r['confidence'] == 'none'
    assert 'no brightness signal' in r['warnings']


def test_camera_photometry_video_source(monkeypatch, tmp_path):
    cam = _make_cam_dir(tmp_path)
    monkeypatch.setattr(photometry, '_star_zeropoint',
                        lambda *a, **k: (10.0, 3, 0.1))
    monkeypatch.setattr(photometry, '_catalog_stars', lambda *a, **k: [])
    monkeypatch.setattr(
        photometry, '_video_photometry',
        lambda *a, **k: {'flux': [100.0, 200.0, 400.0],
                         'peak': [100, 200, 230], 'sat': [False] * 3,
                         'n_matched': 3, 'n_det': 3, 'zp': 15.0,
                         'zp_stars': 4, 'zp_source': 'transfer',
                         'video': 'x-orig.mp4'})
    r = photometry.camera_photometry(cam, use_video=True)
    ext = [photometry.extinction_mag(a) for a in (20.0, 19.9, 19.8)]
    assert r['video']['m_app'] == pytest.approx(
        [15.0 - 2.5 * np.log10(f) - e
         for f, e in zip((100.0, 200.0, 400.0), ext)])
    # extinction-free magnitudes are brighter than observed
    assert all(m < 15.0 - 2.5 * np.log10(f)
               for m, f in zip(r['video']['m_app'], (100., 200., 400.)))
