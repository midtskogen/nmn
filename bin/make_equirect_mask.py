#!/usr/bin/env python3
"""
Create a binary sky mask for an equirectangular timelapse video.

The resulting mask is a single-channel PNG of the same width/height as the
input video.  White pixels (255) mark the sky; black pixels (0) mark
everything that should be ignored (ground, trees, timestamp overlay,
bright static lights, etc.).

The tool uses three whole-day statistics:

1. Temporal mean / variance: the sky is brighter and more dynamic than the
   ground, so a combined score separates them.
2. Persistent gradient edges: edges that are present throughout the day form
   the static horizon and the high-contrast outlines of trees.  This avoids
   treating moving clouds as foreground.
3. High-gradient + high-variance components attached to the tree mask: these
   catch moving twigs and branches that stick up above the main tree mass.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2


def decode_video_stats(video_path, max_frames=0, sample_interval=1):
    """Decode video frames and return running mean, variance and mean gradient.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    max_frames : int
        Maximum number of frames to analyse.  0 means "analyse all frames".
    sample_interval : int
        Use every Nth frame (1 = every frame).

    Returns
    -------
    mean, variance, mean_grad, shape, meta
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames > 0 and sample_interval == 1:
        sample_interval = max(1, total // max_frames)

    mean = np.zeros((height, width), dtype=np.float64)
    m2 = np.zeros((height, width), dtype=np.float64)
    mean_grad = np.zeros((height, width), dtype=np.float64)
    count = 0

    frame_idx = 0
    used = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        count += 1
        used += 1
        delta = gray - mean
        mean += delta / count
        delta2 = gray - mean
        m2 += delta * delta2

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(sx * sx + sy * sy)
        delta_grad = grad - mean_grad
        mean_grad += delta_grad / count

        frame_idx += 1
        if max_frames > 0 and used >= max_frames:
            break

    cap.release()

    if count == 0:
        raise RuntimeError("No frames were decoded")

    variance = m2 / count
    meta = {
        "total": total,
        "used": used,
        "width": width,
        "height": height,
        "fps": fps,
    }
    return mean, variance, mean_grad, (height, width), meta


def fill_gaps(non_sky, max_gap_area=8000, tendril_kernel=7):
    """Fill small sky-through-canopy gaps enclosed by non-sky pixels.

    High-resolution timelapses resolve individual gaps between leaves and
    twigs.  These slivers of sky flicker as branches move in the wind and can
    trigger false meteor detections.  A gap is only filled if it is small
    (area at or below ``max_gap_area``) and not part of the main sky region.

    Many such gaps are joined to the main sky by a thin tendril only a few
    pixels wide (a narrow channel between two twigs), which would otherwise
    make connected-component analysis treat them as part of the real sky.
    A morphological opening breaks these thin tendrils first so the isolated
    pockets can be correctly identified, while the main sky region -- which
    is much wider than the opening kernel -- survives intact.
    """
    sky = cv2.bitwise_not(non_sky)

    if tendril_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (tendril_kernel, tendril_kernel))
        core = cv2.morphologyEx(sky, cv2.MORPH_OPEN, k)
    else:
        k = None
        core = sky

    num, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    top_labels = set(np.unique(labels[0, :]).tolist()) - {0}
    main_sky = np.isin(labels, list(top_labels)).astype(np.uint8) * 255
    if k is not None:
        main_sky = cv2.dilate(main_sky, k)
    main_sky = cv2.bitwise_and(main_sky, sky)

    # Everything else that is still "sky" but not part of the main region is
    # an isolated pocket -- fill it if it is small enough.
    isolated = cv2.bitwise_and(sky, cv2.bitwise_not(main_sky))
    num2, labels2, stats2, _ = cv2.connectedComponentsWithStats(isolated,
                                                                 connectivity=8)
    filled = non_sky.copy()
    for i in range(1, num2):
        if stats2[i, cv2.CC_STAT_AREA] <= max_gap_area:
            filled[labels2 == i] = 255
    return filled


def build_sky_mask(video_path, output_path, max_frames=0, close_kernel=15,
                   open_kernel=3, bottom_band=45, preview_path=None,
                   mean_weight=0.7, var_weight=0.3, invert_output=False,
                   fill_gaps_enabled=False, gap_max_area=8000):
    """Build and save a sky mask for the given equirectangular video."""
    mean, variance, mean_grad, _, meta = decode_video_stats(video_path,
                                                            max_frames=max_frames)

    height, width = meta["height"], meta["width"]

    # Normalise statistics to 0..255.
    mean_u8 = cv2.normalize(mean.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    var_u8 = cv2.normalize(variance.astype(np.float32), None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    grad_u8 = cv2.normalize(mean_grad.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    # Initial sky mask from mean brightness + temporal variance.
    score = cv2.addWeighted(mean_u8, mean_weight, var_u8, var_weight, 0)
    _, sky_mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_kernel > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (close_kernel, close_kernel))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, k_close)
    if open_kernel > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (open_kernel, open_kernel))
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, k_open)

    # Make the sky region contiguous from the top edge.
    for x in range(width):
        col = sky_mask[:, x]
        sky_indices = np.where(col > 0)[0]
        if len(sky_indices) > 0:
            y_first = sky_indices[0]
            col[:y_first + open_kernel] = 255

    # Mask out the bottom band: the timestamp overlay lives here, and in an
    # equirect projection of all-sky cameras the ground is below the horizon.
    # A flat band is more stable here than gradient-based horizon detection,
    # which can produce wavy artefacts on low, treeless terrain.
    if bottom_band > 0:
        h_band = min(bottom_band, height)
        sky_mask[-h_band:, :] = 0

    non_sky = cv2.bitwise_not(sky_mask)

    # --- Twig / branch suppression -------------------------------------------
    # Twigs are darker than the sky but move in the wind.  They are high
    # contrast (gradient) AND high variance.  Clouds are mostly lower contrast
    # or not attached to the tree mask, so requiring both gradient and variance
    # keeps the false positives low.
    var_thresh = np.percentile(var_u8, 50)
    grad_thresh_low = np.percentile(grad_u8, 60)
    dynamic = ((var_u8 > var_thresh) & (grad_u8 > grad_thresh_low)).astype(np.uint8) * 255

    # Keep components larger than a single star but small enough to be twigs.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dynamic, connectivity=8)
    candidates = np.zeros_like(dynamic)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            candidates[labels == i] = 255

    # Dilate slightly to connect neighbouring twigs into a single branch mask.
    candidates = cv2.dilate(candidates,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Only keep dynamic candidates that are connected to the existing non-sky
    # mask (ground or main tree mass).
    non_sky_dilated = cv2.dilate(non_sky,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    twig_mask = cv2.bitwise_and(candidates, non_sky_dilated)

    non_sky = cv2.bitwise_or(non_sky, twig_mask)
    # Fill small holes that appeared inside the tree mask after this step.
    non_sky = cv2.morphologyEx(non_sky, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    # --- Gap filling (optional) -----------------------------------------------
    # High-resolution video resolves individual gaps between leaves/twigs that
    # flicker with the wind.  Close small isolated sky slivers within the tree
    # mask while leaving the main sky region and larger genuine openings alone.
    if fill_gaps_enabled:
        non_sky = fill_gaps(non_sky, max_gap_area=gap_max_area)

    sky_mask = cv2.bitwise_not(non_sky)

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Preview always shows sky as white so it is easy to review.
            preview = cv2.addWeighted(gray, 0.5, sky_mask, 0.5, 0)
            cv2.imwrite(str(preview_path), preview)

    sky_pixels = int(np.count_nonzero(sky_mask == 255))
    print(f"Mask saved: {output_path}")
    print(f"  Convention: {'white=sky' if not invert_output else 'white=non-sky (scan_stack compatible)'}")
    print(f"  Video: {video_path} ({meta['total']} frames, "
          f"{width}x{height}, {meta['fps']:.2f} fps)")
    print(f"  Analysed {meta['used']} frames")
    print(f"  Sky pixels: {sky_pixels} / {width * height} "
          f"({100 * sky_pixels / (width * height):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a sky mask for an equirectangular timelapse video.")
    parser.add_argument("video", help="Path to the equirectangular timelapse MP4")
    parser.add_argument("output", help="Output PNG mask path")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to analyse (0 = all, default 0)")
    parser.add_argument("--close-kernel", type=int, default=15,
                        help="Morphological close kernel size (default 15)")
    parser.add_argument("--open-kernel", type=int, default=3,
                        help="Morphological open kernel size (default 3)")
    parser.add_argument("--bottom-band", type=int, default=45,
                        help="Height of bottom band to mask out (default 45)")
    parser.add_argument("--preview", action="store_true",
                        help="Also write a preview overlay next to the mask")
    parser.add_argument("--mean-weight", type=float, default=0.7,
                        help="Weight of mean brightness in the sky score (default 0.7)")
    parser.add_argument("--var-weight", type=float, default=0.3,
                        help="Weight of temporal variance in the sky score (default 0.3)")
    parser.add_argument("--invert", action="store_true",
                        help="Output white=non-sky (compatible with scan_stack.py)")
    parser.add_argument("--fill-gaps", action="store_true",
                        help="Fill small sky-through-canopy gaps (useful for hires video)")
    parser.add_argument("--gap-max-area", type=int, default=8000,
                        help="Max area in pixels of a gap to fill (default 8000)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview = None
    if args.preview:
        preview = output_path.with_stem(output_path.stem + "_preview").with_suffix(".jpg")

    build_sky_mask(args.video, output_path, max_frames=args.max_frames,
                   close_kernel=args.close_kernel, open_kernel=args.open_kernel,
                   bottom_band=args.bottom_band, preview_path=preview,
                   mean_weight=args.mean_weight, var_weight=args.var_weight,
                   invert_output=args.invert, fill_gaps_enabled=args.fill_gaps,
                   gap_max_area=args.gap_max_area)


if __name__ == "__main__":
    main()
