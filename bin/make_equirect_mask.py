#!/usr/bin/env python3
"""
Create a binary sky mask for an equirectangular timelapse video.

The resulting mask is a single-channel PNG of the same width/height as the
input video.  White pixels (255) mark the sky; black pixels (0) mark
everything that should be ignored (ground, trees, buildings, masts, the
timestamp overlay, etc.).

Algorithm
---------
The foreground is completely static; the sky never is (clouds, sun, moon,
light changes).  The tool exploits this with whole-day statistics computed
over the daytime frames only (frames are classified day/night by mean
brightness with an Otsu threshold):

1. ``day_mean``  -- per-pixel mean brightness.  The ground band, forest
   interior and buildings are absolutely dark; an Otsu threshold (clamped)
   gives the absolute-darkness seed.
2. ``dark_frac`` -- per-pixel fraction of day frames in which the pixel is
   darker than its local background (a very wide Gaussian blur) by a fixed
   margin.  Trees, the distant horizon ridge and masts are *persistently*
   darker than the sky behind them; clouds and the sun are not, because
   they move.  This gives the persistence seed.
3. ``vgrad``     -- per-pixel mean absolute vertical gradient over the day.
   Static horizontal edges (the horizon ridge) and high-frequency canopy
   texture accumulate; moving cloud edges average out.  This both guides
   the horizon seam and seeds bright sunlit canopy tops that the two
   brightness cues miss.

The mask is then assembled structurally:

* A horizon seam h(x) is traced by cyclic 1D dynamic programming across the
  lower image band.  Each column votes with a Gaussian well at the topmost
  row whose edge reward (vgrad + dark_frac) is at least 60% of the column
  maximum; rewards below a noise floor are ignored so haze gradients cannot
  pull the seam, and columns with no real edge are bridged by smoothness
  (hard max-slope + quadratic penalty).  Everything below the seam --
  ground, lake, the timestamp overlay -- is foreground.
* Above the seam, seed components (trees, mast) that attach to the
  below-seam foreground are added, and enclosed sky pockets inside the
  canopy are filled (sky must remain connected to the top edge).
* Thin static structures too fine to trigger the persistence seed
  (antennas, poles, wires) are found with vertical/horizontal blackhat
  transforms of ``day_mean``; components hanging just above the foreground
  are attached and completed with a short vertical drop-line.

All pixel-space parameters scale with the video resolution relative to the
1280px-wide reference geometry.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False

# Reference resolution for pixel-space parameters (SD timelapse width).
REF_W = 1280.0

# The local background is estimated on a frame downscaled by this factor
# (blur kernel scales accordingly); identical result to a full-resolution
# wide Gaussian to within ~1-2 gray levels, but hundreds of times faster.
BG_DOWNSCALE = 8


def _odd(k):
    k = max(1, int(round(k)))
    return k if k % 2 == 1 else k + 1


def _local_background(gray, blur_kernel):
    """Wide local-background estimate via downscale -> blur -> upscale.

    A full-resolution Gaussian with a ~1/8-width kernel is the dominant
    cost of the statistics pass; doing it on a 1/8-resolution image with a
    correspondingly smaller kernel is equivalent to within ~1-2 gray
    levels (the margin used downstream is 22).
    """
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // BG_DOWNSCALE),
                              max(1, h // BG_DOWNSCALE)),
                       interpolation=cv2.INTER_AREA)
    k = _odd(blur_kernel / BG_DOWNSCALE)
    small = cv2.GaussianBlur(small, (k, k), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def decode_video_stats(video_path, max_frames=0, sample_interval=2,
                       blur_kernel=0, margin=22.0):
    """Decode the video and compute per-pixel daytime statistics.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    max_frames : int
        Maximum number of frames to analyse in the statistics pass
        (0 = all).
    sample_interval : int
        Use every Nth frame in the statistics pass.
    blur_kernel : int
        Size of the local-background blur kernel (0 = auto from width).
    margin : float
        Gray levels by which a pixel must be darker than its local
        background to count as "dark" for the persistence statistic.

    Returns
    -------
    dict with day_mean, dark_frac, vgrad (all HxW float32) and meta.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if blur_kernel <= 0:
        blur_kernel = _odd(151 * width / REF_W)

    # --- Pass 1: day/night classification from subsampled frame means ------
    means = []
    for i in range(0, total, 8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        means.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    means = np.array(means)
    if len(means) == 0:
        cap.release()
        raise RuntimeError("No frames were decoded")

    if means.max() - means.min() < 10:
        # No meaningful brightness variation (e.g. polar night/day): use all.
        day_lo = means.min() - 1.0
    else:
        m8 = cv2.normalize(means.astype(np.float32), None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
        t, _ = cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        day_lo = means.min() + (t / 255.0) * (means.max() - means.min())
    cap.release()

    # --- Pass 2: accumulate per-pixel stats over the day frames ------------
    cap = cv2.VideoCapture(str(video_path))
    day_sum = np.zeros((height, width), np.float64)
    day_dark = np.zeros((height, width), np.int32)
    day_grad = np.zeros((height, width), np.float64)
    n_day = 0
    used = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if gray.mean() >= day_lo:
                n_day += 1
                used += 1
                day_sum += gray
                bg = _local_background(gray, blur_kernel)
                day_dark += (gray < bg - margin)
                day_grad += np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        frame_idx += 1
        if max_frames > 0 and used >= max_frames:
            break
    cap.release()

    if n_day < 10:
        # Very few day frames (short polar day, bad weather): redo the pass
        # using the brightest 40% of frames instead.
        day_lo = np.percentile(means, 60)
        cap = cv2.VideoCapture(str(video_path))
        day_sum[:] = 0
        day_dark[:] = 0
        day_grad[:] = 0
        n_day = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if gray.mean() >= day_lo:
                    n_day += 1
                    day_sum += gray
                    bg = _local_background(gray, blur_kernel)
                    day_dark += (gray < bg - margin)
                    day_grad += np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            frame_idx += 1
        cap.release()
        if n_day == 0:
            raise RuntimeError("No usable frames were decoded")

    meta = {
        "total": total,
        "used": n_day,
        "width": width,
        "height": height,
        "fps": fps,
        "day_threshold": float(day_lo),
        "blur_kernel": blur_kernel,
    }
    return {
        "day_mean": (day_sum / n_day).astype(np.float32),
        "dark_frac": day_dark.astype(np.float32) / n_day,
        "vgrad": (day_grad / n_day).astype(np.float32),
        "meta": meta,
    }


if _HAVE_NUMBA:
    @njit(cache=True)
    def _dp_run_numba(data2, max_step, smooth):
        """Banded forward DP over columns; returns (final costs, backptrs)."""
        R, W2 = data2.shape
        back = np.empty((R, W2), np.int32)
        dp = data2[:, 0].copy()
        for x in range(1, W2):
            new_dp = np.empty(R)
            for j in range(R):
                lo = j - max_step
                if lo < 0:
                    lo = 0
                hi = j + max_step
                if hi > R - 1:
                    hi = R - 1
                best_v = 1e18
                best_p = lo
                for p in range(lo, hi + 1):
                    v = dp[p] + smooth * (j - p) * (j - p)
                    if v < best_v:
                        best_v = v
                        best_p = p
                back[j, x] = best_p
                new_dp[j] = best_v + data2[j, x]
            dp = new_dp
        return dp, back


def horizon_seam(vgrad, dark_frac, y_lo_frac=0.70, max_step=3,
                 smooth=0.008, floor=0.30, top_ratio=0.6,
                 grad_scale=220.0, sigma=1.5):
    """Trace the horizon as a cyclic seam h(x) via dynamic programming.

    In each column the strongest static-edge rows form a Gaussian cost well
    at the *topmost* row reaching ``top_ratio`` of the column maximum; the
    well depth is the column maximum minus a noise ``floor`` (weak haze
    gradients cannot pull the seam).  Columns without a confident edge are
    flat and get interpolated by the smoothness term (hard ``max_step``
    slope limit plus quadratic penalty).  The DP runs over the doubled
    width so the solution is consistent across the 360-degree wrap.

    Returns h as an int64 array of length W: the first foreground row per
    column.
    """
    H, W = vgrad.shape
    y_lo = int(y_lo_frac * H)
    R = H - y_lo
    reward = (0.5 * np.minimum(vgrad[y_lo:, :] / grad_scale, 1.0)
              + 0.5 * dark_frac[y_lo:, :])

    colmax = reward.max(axis=0)
    yc = np.full(W, -1, np.int64)
    depth = np.zeros(W)
    for x in range(W):
        cm = colmax[x]
        if cm <= floor:
            continue
        yc[x] = np.where(reward[:, x] >= top_ratio * cm)[0][0]
        depth[x] = cm - floor
    jj = np.arange(R, dtype=np.float64)[:, None]
    data = np.zeros((R, W))
    valid = yc >= 0
    data[:, valid] = -depth[valid][None, :] * np.exp(
        -((jj - yc[valid][None, :]) ** 2) / (2 * sigma * sigma))

    data2 = np.hstack([data, data])
    if _HAVE_NUMBA:
        dp, back = _dp_run_numba(data2, max_step, smooth)
    else:
        j = np.arange(R)
        diff = j[None, :] - j[:, None]
        trans = np.where(np.abs(diff) <= max_step, smooth * diff * diff, 1e9)
        dp = data2[:, 0].copy()
        back = np.zeros((R, 2 * W), np.int16)
        for x in range(1, 2 * W):
            cost = dp[:, None] + trans
            best = np.argmin(cost, axis=0)
            dp = cost[best, np.arange(R)] + data2[:, x]
            back[:, x] = best
    h2 = np.zeros(2 * W, np.int64)
    h2[-1] = int(np.argmin(dp))
    for x in range(2 * W - 1, 0, -1):
        h2[x - 1] = back[h2[x], x]
    h = h2[W:] + y_lo

    # Gentle circular median to kill residual jitter.
    k = max(3, _odd(21 * W / REF_W) // 2 * 2 + 1)
    half = k // 2
    hpad = np.concatenate([h[-half:], h, h[:half]])
    h = np.array([np.median(hpad[i:i + k]) for i in range(W)]).astype(np.int64)
    return h


def attach_components(fg, cand, min_area, anchor_ksize):
    """Add components of ``cand`` that touch (dilated) ``fg``.

    Whole components are kept, not just the overlapping part, so treetops
    connected to the ground by a thin trunk are preserved.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand,
                                                             connectivity=8)
    anchor = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      anchor_ksize))
    touching = np.unique(labels[anchor > 0])
    ok = [i for i in touching
          if i != 0 and stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not ok:
        return fg
    keep = np.isin(labels, ok).astype(np.uint8) * 255
    return cv2.bitwise_or(fg, keep)


def fill_enclosed_sky(fg):
    """Any sky not connected to the top edge is enclosed -> foreground."""
    sky = cv2.bitwise_not(fg)
    num, labels, _, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    top_labels = set(np.unique(labels[0, :]).tolist()) - {0}
    real_sky = np.isin(labels, list(top_labels)).astype(np.uint8) * 255
    return cv2.bitwise_not(real_sky)


def build_sky_mask(video_path, output_path, max_frames=0, sample_interval=2,
                   blur_kernel=0, margin=22.0, dark_frac_thresh=0.5,
                   texture_thresh=70.0, abs_clamp=110.0, seam_y_frac=0.70,
                   seam_max_step=3, seam_smooth=0.008, seam_floor=0.30,
                   seam_top_ratio=0.6, grad_scale=220.0, preview_path=None,
                   invert_output=False):
    """Build and save a sky mask for the given equirectangular video."""
    stats = decode_video_stats(video_path, max_frames=max_frames,
                               sample_interval=sample_interval,
                               blur_kernel=blur_kernel, margin=margin)
    day_mean = stats["day_mean"]
    dark_frac = stats["dark_frac"]
    vgrad = stats["vgrad"]
    meta = stats["meta"]
    height, width = meta["height"], meta["width"]
    s = width / REF_W  # pixel-space scale factor relative to 1280px reference

    # --- Seeds: absolute darkness + persistent local contrast --------------
    dm8 = cv2.normalize(day_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t8, _ = cv2.threshold(dm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = float(day_mean.min()), float(day_mean.max())
    t_abs = min(lo + (t8 / 255.0) * (hi - lo), abs_clamp)

    seed = ((day_mean < t_abs) | (dark_frac > dark_frac_thresh)
            | (vgrad > texture_thresh)).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(3 * s), _odd(3 * s))))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(7 * s), _odd(7 * s))))

    # --- Horizon seam: everything below it is foreground -------------------
    h = horizon_seam(vgrad, dark_frac, y_lo_frac=seam_y_frac,
                     max_step=max(1, int(round(seam_max_step * s))),
                     smooth=seam_smooth, floor=seam_floor,
                     top_ratio=seam_top_ratio, grad_scale=grad_scale,
                     sigma=1.5 * s)
    rows = np.arange(height)[:, None]
    fg = (rows >= h[None, :]).astype(np.uint8) * 255

    # --- Tier 1: above-seam seed components attached to the foreground -----
    above = (rows < h[None, :]).astype(np.uint8) * 255
    cand = cv2.bitwise_and(seed, above)
    fg = attach_components(fg, cand, max(4, int(round(8 * s * s))),
                           (_odd(7 * s), _odd(7 * s)))
    fg = fill_enclosed_sky(fg)

    # --- Tier 2: thin static structures (antennas, poles, wires) -----------
    # Too thin to trigger the persistence seed: vertical/horizontal blackhat
    # of the day mean finds dark static lines.  Keep only genuinely thin
    # components floating in the sky (antennas, wires) -- treeline-edge
    # blackhat responses touch the seed mass and fuzzy needle-tip clumps
    # above crowns are not thin; both are already handled by tier 1.
    bh_v = cv2.morphologyEx(day_mean, cv2.MORPH_BLACKHAT,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (_odd(3 * s), _odd(15 * s))))
    bh_h = cv2.morphologyEx(day_mean, cv2.MORPH_BLACKHAT,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (_odd(15 * s), _odd(3 * s))))
    bh = np.maximum(bh_v, bh_h)
    cand2 = cv2.bitwise_and(((bh > 10.0).astype(np.uint8) * 255), above)
    cand2 = cv2.morphologyEx(cand2, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,
                                                       (1, _odd(9 * s))))
    near_seed = cv2.dilate(seed, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(9 * s), _odd(9 * s))))
    cand2 = cv2.bitwise_and(cand2, cv2.bitwise_not(near_seed))
    num_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(cand2,
                                                             connectivity=8)
    thin_max = max(4, int(round(3.5 * s)))
    ok = [i for i in range(1, num_c)
          if min(st_c[i, cv2.CC_STAT_WIDTH], st_c[i, cv2.CC_STAT_HEIGHT])
          <= thin_max]
    cand2 = np.isin(lab_c, ok).astype(np.uint8) if ok else np.zeros_like(cand2)
    # Attach components hanging just above the foreground, then complete each
    # line down to the foreground (the base is often invisible where it
    # crosses a bright canopy edge).
    fg2 = attach_components(fg, cand2, max(3, int(round(6 * s * s))),
                            (_odd(5 * s), _odd(45 * s)))
    added = cv2.bitwise_and(fg2, cv2.bitwise_not(fg))
    drop = int(round(31 * s))
    for x in range(width):
        idx = np.where(added[:, x] > 0)[0]
        if not len(idx):
            continue
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            r1 = run[-1]
            below = np.where(fg2[r1 + 1:r1 + drop, x] > 0)[0]
            if len(below):
                fg2[r1:r1 + 2 + below[0], x] = 255
    fg = fg2

    # --- Final: sky must stay connected to the top edge ---------------------
    fg = fill_enclosed_sky(fg)
    sky_mask = cv2.bitwise_not(fg)

    # Default output: white = sky (intuitive image mask).  With --invert,
    # white = non-sky, which matches the convention used by scan_stack.py.
    output_mask = sky_mask if not invert_output else cv2.bitwise_not(sky_mask)
    cv2.imwrite(str(output_path), output_mask)

    if preview_path:
        cap = cv2.VideoCapture(str(video_path))
        mid_frame = int(meta["total"] / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Preview always shows the sky mask boundary as a red contour
            # over a mid-day frame so it is easy to review.
            contour = cv2.Canny(sky_mask, 100, 200)
            preview = frame.copy()
            preview[contour > 0] = (0, 0, 255)
            cv2.imwrite(str(preview_path), preview)

    sky_pixels = int(np.count_nonzero(sky_mask == 255))
    print(f"Mask saved: {output_path}")
    print(f"  Convention: {'white=sky' if not invert_output else 'white=non-sky (scan_stack compatible)'}")
    print(f"  Video: {video_path} ({meta['total']} frames, "
          f"{width}x{height}, {meta['fps']:.2f} fps)")
    print(f"  Day frames analysed: {meta['used']} "
          f"(frame-mean day threshold {meta['day_threshold']:.1f}, "
          f"blur kernel {meta['blur_kernel']})")
    print(f"  abs-dark threshold: {t_abs:.1f}")
    print(f"  Sky pixels: {sky_pixels} / {width * height} "
          f"({100 * sky_pixels / (width * height):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a sky mask for an equirectangular timelapse video.")
    parser.add_argument("video", help="Path to the equirectangular timelapse MP4")
    parser.add_argument("output", help="Output PNG mask path")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum day frames to analyse (0 = all, default 0)")
    parser.add_argument("--sample-interval", type=int, default=2,
                        help="Analyse every Nth frame (default 2)")
    parser.add_argument("--blur-kernel", type=int, default=0,
                        help="Local-background blur kernel (0 = auto, ~12%% of width)")
    parser.add_argument("--margin", type=float, default=22.0,
                        help="Gray levels darker than the local background that "
                             "count as persistent contrast (default 22)")
    parser.add_argument("--dark-frac", type=float, default=0.5,
                        help="Fraction of day frames a pixel must be persistently "
                             "dark to seed foreground (default 0.5)")
    parser.add_argument("--texture-thresh", type=float, default=70.0,
                        help="Persistent vertical-gradient level that seeds "
                             "foreground (catches bright sunlit canopy tops; "
                             "default 70)")
    parser.add_argument("--abs-clamp", type=float, default=110.0,
                        help="Upper clamp for the absolute-darkness Otsu "
                             "threshold (default 110)")
    parser.add_argument("--seam-y-frac", type=float, default=0.70,
                        help="The horizon seam is searched in the image band "
                             "below this fraction of the height (default 0.70)")
    parser.add_argument("--seam-max-step", type=int, default=3,
                        help="Max seam slope, pixels per column at reference "
                             "resolution (default 3)")
    parser.add_argument("--seam-smooth", type=float, default=0.008,
                        help="Quadratic smoothness weight of the seam DP "
                             "(default 0.008)")
    parser.add_argument("--seam-floor", type=float, default=0.30,
                        help="Edge-reward noise floor; weaker (e.g. haze) edges "
                             "cannot pull the seam (default 0.30)")
    parser.add_argument("--seam-top-ratio", type=float, default=0.6,
                        help="The seam well sits at the topmost row reaching "
                             "this fraction of the column's max edge reward "
                             "(default 0.6)")
    parser.add_argument("--grad-scale", type=float, default=220.0,
                        help="Persistent vertical gradient that maps to full "
                             "edge reward (default 220)")
    parser.add_argument("--preview", action="store_true",
                        help="Also write a boundary overlay next to the mask")
    parser.add_argument("--invert", action="store_true",
                        help="Output white=non-sky (compatible with scan_stack.py)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview = None
    if args.preview:
        preview = output_path.with_stem(output_path.stem + "_preview").with_suffix(".jpg")

    build_sky_mask(args.video, output_path, max_frames=args.max_frames,
                   sample_interval=args.sample_interval,
                   blur_kernel=args.blur_kernel, margin=args.margin,
                   dark_frac_thresh=args.dark_frac, abs_clamp=args.abs_clamp,
                   seam_y_frac=args.seam_y_frac, seam_max_step=args.seam_max_step,
                   seam_smooth=args.seam_smooth, seam_floor=args.seam_floor,
                   seam_top_ratio=args.seam_top_ratio, grad_scale=args.grad_scale,
                   texture_thresh=args.texture_thresh,
                   preview_path=preview, invert_output=args.invert)


if __name__ == "__main__":
    main()
