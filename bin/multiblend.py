#!/usr/bin/env python3
"""multiblend.py – self-contained Laplacian pyramid image blender.

Based on Davi1d Horman's multiblend

Usage:
    ./multiblend.py -o output.tif left.tif right.tif [...]
    python multiblend.py --help

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.
"""

# ============================================================
#  Standard imports
# ============================================================
import sys
import os
import time
import ctypes
# Disable scipy/numpy thread pool: for our banded array sizes the 48-thread
# overhead (futex + context switches) costs ~3.6s sys and ~5s user with no
# real-time benefit.  Users can override by setting OMP_NUM_THREADS in env.
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, convolve1d
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable

__version__ = '0.6.2'

# ---------------------------------------------------------------------------
# Keep freed large buffers in the process heap (no munmap).
# This matches C++ allocator behaviour: glibc malloc reuses the heap slab
# instead of calling mmap/munmap for every >128KB allocation.
# M_MMAP_THRESHOLD (-3): raise the threshold so all our arrays use brk()
# M_TRIM_THRESHOLD (-1): don't shrink the heap between image iterations
# ---------------------------------------------------------------------------
try:
    _libc = ctypes.CDLL(None)
    _libc.mallopt(-3, 1 << 30)  # M_MMAP_THRESHOLD = 1 GiB
    _libc.mallopt(-1, 1 << 30)  # M_TRIM_THRESHOLD = 1 GiB
except Exception:
    pass

# ============================================================
#  Data structures
# ============================================================

@dataclass
class GeoTIFFInfo:
    x_geo_ref: float = 0.0
    y_geo_ref: float = 0.0
    x_cell_res: float = 1.0
    y_cell_res: float = 1.0
    nodata: int = 0
    set: bool = False


@dataclass
class ImageInfo:
    filename: str
    bpp: int
    width: int
    height: int
    xpos: int
    ypos: int
    channels: List[np.ndarray]
    mask: np.ndarray
    xres: float = -1.0
    yres: float = -1.0
    geotiff: Optional[GeoTIFFInfo] = None


# ============================================================
#  Image loading (_load)
# ============================================================

def _rational_to_float(v) -> float:
    if v is None:
        return -1.0
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return float(v[0]) / float(v[1]) if v[1] else 0.0
    return float(v)


def _strip_row_hint(page) -> tuple:
    """Return (row_lo, row_hi) bounding the content rows by inspecting TIFF strip bytecounts.

    Fully-transparent strips always compress to the same (small) byte count; content strips
    are larger.  The mode of the lower-half byte counts identifies the transparent floor.
    Reading bytecounts is instant (already in the IFD header) — no pixel data is decoded.
    """
    h   = int(page.imagelength)
    rps = int(getattr(page, 'rowsperstrip', h))
    bc  = page.databytecounts
    n   = len(bc)
    if n <= 2:
        return 0, h
    bc_arr = np.asarray(bc, dtype=np.int64)
    # Find the transparent-strip bytecount: the most common value in the lower 60%.
    # All-zero strips always compress identically (same zlib output) so their counts cluster.
    lower = np.sort(bc_arr)[:max(n * 6 // 10, 3)]
    unique, counts = np.unique(lower, return_counts=True)
    mode_bc = int(unique[np.argmax(counts)])
    if int(counts[np.argmax(counts)]) < 3:
        return 0, h   # no clear transparent floor — fall back to full scan
    content = np.where(bc_arr > mode_bc)[0]
    if len(content) == 0:
        return 0, h   # all transparent
    row_lo = max(0, int(content[0]) * rps)
    row_hi = min(h, (int(content[-1]) + 1) * rps)
    return row_lo, row_hi


def _trim(data: np.ndarray) -> tuple:
    """Return (top, left, bottom, right) bounding box of opaque pixels."""
    if data.dtype == np.uint8:
        # View each RGBA pixel as a contiguous uint32 (LE: A in high byte).
        # Avoids stride-4 uint8 access; alpha>0xFE ↔ pixel>=0xFF000000.
        solid = data.view(np.uint32).reshape(data.shape[0], data.shape[1]) >= np.uint32(0xFF000000)
    else:
        solid = data[..., 3] == 0xffff

    rows = np.any(solid, axis=1)
    cols = np.any(solid, axis=0)

    nz_rows = np.flatnonzero(rows)
    if len(nz_rows) == 0:
        h, w = data.shape[:2]
        return 0, 0, h - 1, w - 1

    nz_cols = np.flatnonzero(cols)
    return int(nz_rows[0]), int(nz_cols[0]), int(nz_rows[-1]), int(nz_cols[-1])


def _extract_and_inpaint(data: np.ndarray, bpp: int) -> tuple:
    """Extract [R,G,B] channels + bool mask; inpaint transparent pixels."""
    if bpp == 8:
        solid = data[..., 3] > 0xfe
    else:
        solid = data[..., 3] == 0xffff

    if solid.all():
        # Contiguous per-channel copies; lets caller free the 4-ch trimmed array
        # immediately and gives blend 100% cache-line efficiency (no stride-4 reads).
        return [np.ascontiguousarray(data[..., 0]),
                np.ascontiguousarray(data[..., 1]),
                np.ascontiguousarray(data[..., 2])], solid

    # Inpaint transparent pixels with nearest solid pixel value.
    # Run EDT at 8× downscale for ~64× speedup; ±8 px accuracy is fine for blending.
    r = data[..., 0].copy()
    g = data[..., 1].copy()
    b = data[..., 2].copy()
    mask2d = ~solid
    ds = 8
    H, W = solid.shape
    ph = ((H + ds - 1) // ds) * ds
    pw = ((W + ds - 1) // ds) * ds
    solid_pad = np.zeros((ph, pw), dtype=bool)
    solid_pad[:H, :W] = solid
    solid_ds   = solid_pad[::ds, ::ds]
    ri_ds, ci_ds = distance_transform_edt(~solid_ds, return_distances=False, return_indices=True)
    if _NUMBA_OK:
        _nb_inpaint(r, g, b, mask2d, ri_ds, ci_ds, ds)
    else:
        ri = np.repeat(np.repeat(ri_ds * ds, ds, axis=0), ds, axis=1)[:H, :W]
        ci = np.repeat(np.repeat(ci_ds * ds, ds, axis=0), ds, axis=1)[:H, :W]
        r[mask2d] = r[ri[mask2d], ci[mask2d]]
        g[mask2d] = g[ri[mask2d], ci[mask2d]]
        b[mask2d] = b[ri[mask2d], ci[mask2d]]
    return [r, g, b], solid


def _geotiff_read(tags: dict) -> Optional[GeoTIFFInfo]:
    """Read GeoTIFF positioning tags from a tifffile tag-code dict."""
    scale     = tags.get(33550)
    tiepoints = tags.get(33922)
    if scale is None or tiepoints is None:
        return None

    scale_arr = np.asarray(scale,     dtype=np.float64).ravel()
    tie_arr   = np.asarray(tiepoints, dtype=np.float64).ravel()
    if len(scale_arr) < 2 or len(tie_arr) < 6:
        return None

    geo = GeoTIFFInfo()
    geo.x_cell_res = float(scale_arr[0])
    geo.y_cell_res = float(scale_arr[1])
    geo.x_geo_ref  = float(tie_arr[3]) - float(tie_arr[0]) * geo.x_cell_res
    geo.y_geo_ref  = float(tie_arr[4]) - float(tie_arr[1]) * geo.y_cell_res

    nodata_tag = tags.get(42113)
    if nodata_tag is not None:
        try:
            geo.nodata = int(str(nodata_tag).rstrip('\x00'))
        except (ValueError, TypeError):
            geo.nodata = 0

    geo.set = True
    return geo


def load_images(filenames: List[str], verbosity: int = 1,
                print_func: Callable = print) -> tuple:
    """Load TIFF images; returns (images, workbpp, xres, yres, geotiff, workwidth, workheight)."""
    images: List[ImageInfo] = []
    workbpp = None
    xres = yres = -1.0
    first_geotiff: Optional[GeoTIFFInfo] = None
    workwidth = workheight = 0

    for filename in filenames:
        if verbosity >= 1:
            print_func(f"  processing {filename}...")

        with tifffile.TiffFile(filename) as tif:
            page = tif.pages[0]
            bpp  = int(page.bitspersample)
            spp  = int(page.samplesperpixel)
            tiff_w = int(page.imagewidth)
            tiff_h = int(page.imagelength)

            if bpp not in (8, 16):
                raise ValueError(f"{bpp}bpp not valid!")
            if spp != 4:
                raise ValueError(f"Cannot handle {spp} samples per pixel (need 4 RGBA)!")
            if workbpp is None:
                workbpp = bpp
            elif bpp != workbpp:
                raise ValueError("Cannot mix 8bpp and 16bpp images!")

            tags_raw = {tag.code: tag.value for tag in page.tags.values()}

            tiff_xres = _rational_to_float(tags_raw.get(282))
            tiff_yres = _rational_to_float(tags_raw.get(283))
            tiff_xpos = _rational_to_float(tags_raw.get(286))
            tiff_ypos = _rational_to_float(tags_raw.get(287))

            xpos = ypos = 0
            geotiff = _geotiff_read(tags_raw)

            if tiff_xpos < 0 and tiff_ypos < 0:
                if geotiff is not None and first_geotiff is not None:
                    xpos = int(geotiff.x_geo_ref / first_geotiff.x_cell_res)
                    ypos = int(geotiff.y_geo_ref / first_geotiff.y_cell_res)
                elif geotiff is not None:
                    first_geotiff = geotiff
                    xpos = ypos = 0
            else:
                if tiff_xpos >= 0 and tiff_xres > 0:
                    xpos = int(tiff_xpos * tiff_xres + 0.5)
                if tiff_ypos >= 0 and tiff_yres > 0:
                    ypos = int(tiff_ypos * tiff_yres + 0.5)
                if xres < 0 and tiff_xres > 0:
                    xres, yres = tiff_xres, tiff_yres
                elif tiff_xres > 0 and (tiff_xres != xres or tiff_yres != yres):
                    if verbosity >= 0:
                        print_func(f"  WARNING: resolution mismatch ({xres}/{tiff_xres} dpi)")

            row_lo, row_hi = _strip_row_hint(page)
            data = tif.asarray()

        if data.ndim == 2 or data.shape[2] != 4:
            raise ValueError(f"Expected RGBA data in {filename}, got shape {data.shape}")

        # Trim: scan only the hinted content rows — much faster for sparse images
        top_rel, left, bottom_rel, right = _trim(data[row_lo:row_hi])
        top    = row_lo + top_rel
        bottom = row_lo + bottom_rel
        trimmed = data[top:bottom + 1, left:right + 1].copy()
        del data

        img_xpos = xpos + left
        img_ypos = ypos + top
        img_w    = right - left + 1
        img_h    = bottom - top + 1

        workwidth  = max(workwidth,  xpos + tiff_w)
        workheight = max(workheight, ypos + tiff_h)

        channels, mask = _extract_and_inpaint(trimmed, bpp)
        del trimmed

        images.append(ImageInfo(
            filename=filename, bpp=bpp,
            width=img_w, height=img_h,
            xpos=img_xpos, ypos=img_ypos,
            channels=channels, mask=mask,
            xres=xres, yres=yres, geotiff=geotiff,
        ))

    return images, workbpp or 8, xres, yres, first_geotiff, workwidth, workheight


def tighten(images: List[ImageInfo]) -> tuple:
    """Crop workspace to minimum bounding box; returns (min_left, min_top, new_w, new_h)."""
    min_left = min(img.xpos for img in images)
    min_top  = min(img.ypos for img in images)
    for img in images:
        img.xpos -= min_left
        img.ypos -= min_top
    max_right  = max(img.xpos + img.width  for img in images)
    max_bottom = max(img.ypos + img.height for img in images)
    return min_left, min_top, max_right, max_bottom


def compute_levels(images: List[ImageInfo], workwidth: int, workheight: int,
                   wideblend: bool, max_levels: int, sub_levels: int) -> int:
    if not wideblend:
        blend_wh = int(sum(img.width + img.height for img in images) * 0.5 / len(images))
    else:
        blend_wh = min(workwidth, workheight)
    levels = 0
    while blend_wh > 4:
        blend_wh = (blend_wh + 1) >> 1
        levels += 1
    return max(0, min(max_levels, levels) - sub_levels)


# ============================================================
#  Seaming (_seam)
# ============================================================

def _build_workspace_masks(images: List[ImageInfo], workwidth: int, workheight: int) -> List[np.ndarray]:
    masks = []
    for img in images:
        m = np.zeros((workheight, workwidth), dtype=bool)
        m[img.ypos:img.ypos + img.height, img.xpos:img.xpos + img.width] = img.mask
        masks.append(m)
    return masks


def _voronoi_seam(images: list, workwidth: int, workheight: int,
                  reverse: bool, ds: int = 4) -> np.ndarray:
    """Voronoi seam.

    When numba is available: full-resolution unified Chamfer (3,4) EDT
    (one pass over all exclusive zones simultaneously), matching the C++
    leftupxy+rightdownxy algorithm exactly.

    Fallback (no numba): N-pass Euclidean EDT via scipy at ds× downscale.
    """
    n = len(images)

    if _NUMBA_OK and n <= 64:
        # ------------------------------------------------------------------
        # Unified Chamfer (3,4) EDT at 2× downscale with VALMASKED coverage masking.
        # Running at half-resolution (¼ the pixels) yields ~4× speedup; the seam
        # is upsampled back to full res via nearest-neighbour.
        # ------------------------------------------------------------------
        SEAM_DS  = 2
        wh_sm    = (workheight + SEAM_DS - 1) // SEAM_DS
        ww_sm    = (workwidth  + SEAM_DS - 1) // SEAM_DS
        EDT_MAX  = np.uint32(0xfffffbff)
        edt_d_sm    = np.full((wh_sm, ww_sm), EDT_MAX, dtype=np.uint32)
        edt_i_sm    = np.full((wh_sm, ww_sm), np.uint8(255), dtype=np.uint8)
        cov_bits_sm = np.zeros((wh_sm, ww_sm), dtype=np.uint64)
        coverage_sm = np.zeros((wh_sm, ww_sm), dtype=np.uint8)

        # Build workspace-aligned downscaled masks once, then reuse for cov + seed.
        masks_sm = []
        for img in images:
            r0_sm = img.ypos // SEAM_DS
            c0_sm = img.xpos // SEAM_DS
            r1_sm = (img.ypos + img.height + SEAM_DS - 1) // SEAM_DS
            c1_sm = (img.xpos + img.width  + SEAM_DS - 1) // SEAM_DS
            msk_sm = np.empty((r1_sm - r0_sm, c1_sm - c0_sm), dtype=np.uint8)
            _nb_build_mask_sm(img.mask.view(np.uint8), msk_sm,
                              img.ypos, img.xpos, r0_sm, c0_sm, SEAM_DS)
            masks_sm.append((msk_sm, r0_sm, c0_sm))

        for i, (img, (msk_sm, r0_sm, c0_sm)) in enumerate(zip(images, masks_sm)):
            _nb_add_cov(coverage_sm, cov_bits_sm, msk_sm, i, r0_sm, c0_sm)

        # Seed exclusive zones: dist=0, image=i (smaller index wins ties)
        for idx in (reversed(range(n)) if reverse else range(n)):
            msk_sm, r0_sm, c0_sm = masks_sm[idx]
            _nb_seed_excl(edt_d_sm, edt_i_sm, msk_sm, coverage_sm, idx, r0_sm, c0_sm)

        del coverage_sm
        _chamfer_voronoi(edt_d_sm, edt_i_sm, cov_bits_sm)
        del edt_d_sm, cov_bits_sm

        # Upsample assignment (nearest-neighbour) to full workspace size
        assignment = np.repeat(np.repeat(edt_i_sm, SEAM_DS, axis=0),
                               SEAM_DS, axis=1)[:workheight, :workwidth].copy()

        # Fill any remaining unassigned (255) pixels at full resolution
        for i, img in enumerate(images):
            _nb_fill_unassigned(assignment, img.mask.view(np.uint8),
                                i, img.ypos, img.xpos)

        return assignment

    # ------------------------------------------------------------------
    # Fallback: N-pass Euclidean EDT (scipy) at ds× downscale
    # ------------------------------------------------------------------
    ph = ((workheight + ds - 1) // ds) * ds
    pw = ((workwidth  + ds - 1) // ds) * ds
    dsh, dsw = ph // ds, pw // ds

    ds_masks = []
    mp = np.empty((ph, pw), dtype=bool)
    for img in images:
        mp.fill(False)
        mp[img.ypos:img.ypos + img.height, img.xpos:img.xpos + img.width] = img.mask
        ds_masks.append(mp.reshape(dsh, ds, pw).any(axis=1).reshape(dsh, dsw, ds).any(axis=2))

    coverage_ds = np.zeros((dsh, dsw), dtype=np.uint16)
    for dm in ds_masks:
        coverage_ds += dm
    cov1_ds = coverage_ds == 1
    del coverage_ds

    excl_inv_ds   = np.empty((dsh, dsw), dtype=bool)
    edt_buf_ds    = np.empty((dsh, dsw), dtype=np.float64)
    update_ds     = np.empty((dsh, dsw), dtype=bool)
    assignment_ds = np.full((dsh, dsw), 255, dtype=np.uint8)
    min_dist_ds   = np.full((dsh, dsw), np.finfo(np.float64).max, dtype=np.float64)

    for i in (reversed(range(n)) if reverse else range(n)):
        np.logical_and(ds_masks[i], cov1_ds, out=excl_inv_ds)
        np.logical_not(excl_inv_ds, out=excl_inv_ds)
        distance_transform_edt(excl_inv_ds, distances=edt_buf_ds)
        np.less(edt_buf_ds, min_dist_ds, out=update_ds)
        np.minimum(min_dist_ds, edt_buf_ds, out=min_dist_ds)
        assignment_ds[update_ds] = i

    del edt_buf_ds, cov1_ds, min_dist_ds, ds_masks

    assignment = np.repeat(np.repeat(assignment_ds, ds, axis=0), ds, axis=1)
    assignment = assignment[:workheight, :workwidth]

    for i, img in enumerate(images):
        r0 = img.ypos; r1 = r0 + img.height
        c0 = img.xpos; c1 = c0 + img.width
        band = assignment[r0:r1, c0:c1]
        unassigned = (band == 255) & img.mask
        if np.any(unassigned):
            band[unassigned] = np.uint8(i)

    return assignment


def _simple_seam(images: List[ImageInfo], workwidth: int, workheight: int) -> np.ndarray:
    assignment = np.full((workheight, workwidth), 255, dtype=np.uint8)
    yy, xx     = np.mgrid[0:workheight, 0:workwidth]
    min_dist2  = np.full((workheight, workwidth), np.finfo(np.float64).max)
    for i, img in enumerate(images):
        d2     = (yy - (img.ypos + img.height * 0.5)) ** 2 + (xx - (img.xpos + img.width * 0.5)) ** 2
        update = d2 < min_dist2
        min_dist2[update]  = d2[update]
        assignment[update] = i
    return assignment


def _dp_min_seam_v(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost vertical seam (top→bottom) through cost[H, W].

    At each row the column moves by at most ±1 from the previous row.
    Returns seam[row] = column index of the optimal path.
    """
    if not cost.flags['C_CONTIGUOUS']:
        cost = np.ascontiguousarray(cost)
    H, W = cost.shape
    dp = np.empty((H, W), dtype=np.float32)
    seam = np.empty(H, dtype=np.int32)
    if _NUMBA_OK:
        _nb_dp_fwd(cost, dp)
        _nb_dp_backtrack(dp, seam)
    else:
        dp[0] = cost[0]
        ix  = np.arange(W, dtype=np.int32)
        lo  = np.maximum(ix - 1, 0)
        hi  = np.minimum(ix + 1, W - 1)
        for r in range(1, H):
            prev = np.minimum(np.minimum(dp[r - 1, lo], dp[r - 1, ix]), dp[r - 1, hi])
            dp[r] = cost[r] + prev
        seam[-1] = int(np.argmin(dp[-1]))
        for r in range(H - 2, -1, -1):
            c       = seam[r + 1]
            c0, c1  = max(0, c - 1), min(W, c + 2)
            seam[r] = c0 + int(np.argmin(dp[r, c0:c1]))
    return seam


def _content_aware_seam(images: List[ImageInfo],
                         assignment: np.ndarray,
                         workwidth: int, workheight: int) -> np.ndarray:
    """Refine seam placement by routing each boundary through the minimum
    colour-difference path in the overlap region.

    For each adjacent pair (i, j) that share a seam boundary a dynamic-
    programming search finds the connected path with lowest summed squared
    RGB difference.  Pixels on each side of the path are assigned to their
    respective image.  Transparent pixels are given a very high cost so the
    path stays within fully-solid regions of both images.
    """
    result = assignment.copy()
    n = len(images)
    _tmp = None  # non-Numba scratch buffer; lazily allocated, reused across pairs

    for i in range(n):
        img_i = images[i]
        r0_i, r1_i = img_i.ypos, img_i.ypos + img_i.height
        c0_i, c1_i = img_i.xpos, img_i.xpos + img_i.width

        for j in range(i + 1, n):
            img_j = images[j]
            r0_j, r1_j = img_j.ypos, img_j.ypos + img_j.height
            c0_j, c1_j = img_j.xpos, img_j.xpos + img_j.width

            r0 = max(r0_i, r0_j); r1 = min(r1_i, r1_j)
            c0 = max(c0_i, c0_j); c1 = min(c1_i, c1_j)
            if r1 <= r0 or c1 <= c0:
                continue

            H, W = r1 - r0, c1 - c0
            ri0 = r0 - r0_i;  ri1 = ri0 + H
            ci0 = c0 - c0_i;  ci1 = ci0 + W
            rj0 = r0 - r0_j;  rj1 = rj0 + H
            cj0 = c0 - c0_j;  cj1 = cj0 + W

            # Only refine an existing Voronoi boundary: if the Voronoi assigned
            # the entire overlap to one image, that assignment was deliberate
            # (its exclusive zone is the nearest) and we must not override it.
            band = assignment[r0:r1, c0:c1]
            if not (np.any(band == i) and np.any(band == j)):
                continue
            # Also require both images to have physically valid pixels here.
            valid = img_i.mask[ri0:ri1, ci0:ci1] & img_j.mask[rj0:rj1, cj0:cj1]
            if not valid.any():
                continue

            # Squared RGB difference cost.
            cost = np.empty((H, W), dtype=np.float32)
            if _NUMBA_OK:
                _nb_seam_cost(
                    img_i.channels[0][ri0:ri1, ci0:ci1],
                    img_j.channels[0][rj0:rj1, cj0:cj1],
                    img_i.channels[1][ri0:ri1, ci0:ci1],
                    img_j.channels[1][rj0:rj1, cj0:cj1],
                    img_i.channels[2][ri0:ri1, ci0:ci1],
                    img_j.channels[2][rj0:rj1, cj0:cj1],
                    cost)
            else:
                cost[:] = 0
                if _tmp is None or _tmp.shape != (H, W):
                    _tmp = np.empty((H, W), dtype=np.float32)
                for ch in range(3):
                    pi = img_i.channels[ch][ri0:ri1, ci0:ci1].astype(np.float32)
                    pj = img_j.channels[ch][rj0:rj1, cj0:cj1].astype(np.float32)
                    d = pi - pj;  d *= d;  cost += d
                    for g in np.gradient(pi):  g *= g;  cost += g
                    for g in np.gradient(pj):  g *= g;  cost += g

            cost[~valid] = np.float32(1e9)

            dx = abs(img_i.xpos - img_j.xpos)
            dy = abs(img_i.ypos - img_j.ypos)
            if dx >= dy:
                # Images are primarily left-right neighbours → vertical seam.
                seam    = _dp_min_seam_v(cost)
                left_i  = i if img_i.xpos <= img_j.xpos else j
                right_i = j if left_i == i else i
                cols     = np.arange(W, dtype=np.int32)
                mask_l   = cols[np.newaxis, :] < seam[:, np.newaxis]  # (H, W)
                new_asgn = np.where(mask_l, np.uint8(left_i), np.uint8(right_i))
            else:
                # Images are primarily top-bottom neighbours → horizontal seam.
                seam    = _dp_min_seam_v(cost.T)
                top_i   = i if img_i.ypos <= img_j.ypos else j
                bot_i   = j if top_i == i else i
                rows    = np.arange(H, dtype=np.int32)
                mask_t  = rows[:, np.newaxis] < seam[np.newaxis, :]   # (H, W)
                new_asgn = np.where(mask_t, np.uint8(top_i), np.uint8(bot_i))
            # Only overwrite pixels where: (1) both images are valid, AND
            # (2) the pixel is currently assigned to i or j.  This prevents a
            # later pair from clobbering a correct assignment made by an earlier
            # pair in 3-way (or higher) overlap zones.
            current = result[r0:r1, c0:c1]
            modifiable = valid & ((current == np.uint8(i)) | (current == np.uint8(j)))
            result[r0:r1, c0:c1] = np.where(modifiable, new_asgn, current)

    return result


def _seam_palette():
    palette = []
    base = 2.0
    for _ in range(255):
        rad = base
        r = max(0.0, min(1.0, min(rad, 4.0 - rad)))
        rad2 = (rad + 2.0) % 6.0
        g = max(0.0, min(1.0, min(rad2, 4.0 - rad2)))
        rad3 = (rad2 + 2.0) % 6.0
        b = max(0.0, min(1.0, min(rad3, 4.0 - rad3)))
        base = (base + 6.0 * 0.618033988749895) % 6.0
        palette.append((int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5)))
    palette.append((0, 0, 0))
    return palette


def _save_seams_png(filename: str, assignment: np.ndarray, workwidth: int, workheight: int,
                    verbosity: int, print_func: Callable = print):
    if verbosity >= 1:
        print_func(f"  saving seams to {filename}...")
    pal = _seam_palette()
    try:
        import png as _png
        p = _png.Writer(width=workwidth, height=workheight, palette=pal, bitdepth=8)
        with open(filename, 'wb') as f:
            p.write(f, assignment.tolist())
        return
    except ImportError:
        pass
    try:
        from PIL import Image
        pal_flat = [v for rgb in pal for v in rgb]
        img = Image.fromarray(assignment, mode='P')
        img.putpalette(pal_flat)
        img.save(filename)
    except ImportError:
        print_func("WARNING: pypng and Pillow unavailable; cannot save seams PNG")


def _load_seams_png(filename: str, workwidth: int, workheight: int,
                    verbosity: int, print_func: Callable = print) -> np.ndarray:
    if verbosity >= 1:
        print_func(f"  loading seams from {filename}...")
    try:
        from PIL import Image
        return np.array(Image.open(filename).convert('P'), dtype=np.uint8)
    except Exception:
        pass
    try:
        import png as _png
        _, _, rows, _ = _png.Reader(filename=filename).read()
        return np.array(list(rows), dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Cannot load seams PNG {filename}: {e}") from e


def compute_seams(images: List[ImageInfo], workwidth: int, workheight: int,
                  reverse: bool = False, simple_seam: bool = False,
                  content_seam: bool = False,
                  seam_load_filename: Optional[str] = None,
                  seam_save_filename: Optional[str] = None,
                  xor_filename: Optional[str] = None,
                  verbosity: int = 1,
                  print_func: Callable = print) -> Tuple[np.ndarray, List[bool]]:
    n = len(images)

    if seam_load_filename:
        assignment = _load_seams_png(seam_load_filename, workwidth, workheight, verbosity, print_func)
    else:
        if verbosity >= 1:
            print_func("  seaming...")
        if simple_seam:
            assignment = _simple_seam(images, workwidth, workheight)
        else:
            assignment = _voronoi_seam(images, workwidth, workheight, reverse)
        if content_seam:
            assignment = _content_aware_seam(images, assignment, workwidth, workheight)

    counts = np.bincount(assignment.ravel(), minlength=n + 1)
    seam_present = [bool(counts[i]) for i in range(n)]
    if not all(seam_present):
        print_func("WARNING: some images are completely overlapped")

    if seam_save_filename:
        _save_seams_png(seam_save_filename, assignment, workwidth, workheight, verbosity, print_func)

    if xor_filename:
        ws_masks2 = _build_workspace_masks(images, workwidth, workheight)
        coverage  = sum(m.astype(np.uint16) for m in ws_masks2)
        xor_map   = np.full((workheight, workwidth), 255, dtype=np.uint8)
        for i, m in enumerate(ws_masks2):
            xor_map[m & (coverage == 1)] = i
        _save_seams_png(xor_filename, xor_map, workwidth, workheight, verbosity, print_func)

    return assignment, seam_present


# ============================================================
#  Blending (_blend)
# ============================================================

_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float32)

# ---------------------------------------------------------------------------
# Optional numba-accelerated pyramid kernels (70x faster downsample vs scipy,
# 24x faster upsample vs numpy).  Detected at import time; blend() uses the
# scipy/numpy fallback path when numba is unavailable.
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit, prange as _prange
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

if _NUMBA_OK:
    @_njit(parallel=True, cache=True, fastmath=True)
    def _nb_ds3(src, dst):
        """Full 2x downsample: [0.25,0.5,0.25] separable kernel, (C,H,W) float32."""
        C = src.shape[0]; H = src.shape[1]; W = src.shape[2]
        dH = dst.shape[1]; dW = dst.shape[2]
        for c_di in _prange(C * dH):
            c  = c_di // dH
            di = c_di %  dH
            si   = di * 2
            rim1 = si - 1 if si > 0     else 0
            rip1 = si + 1 if si + 1 < H else H - 1
            for dj in range(dW):
                sj   = dj * 2
                cjm1 = sj - 1 if sj > 0     else 0
                cjp1 = sj + 1 if sj + 1 < W else W - 1
                dst[c, di, dj] = (
                    np.float32(0.0625) * (src[c, rim1, cjm1] + src[c, rim1, cjp1] +
                                          src[c, rip1, cjm1] + src[c, rip1, cjp1]) +
                    np.float32(0.125)  * (src[c, rim1, sj]   + src[c, si,   cjm1] +
                                          src[c, si,   cjp1] + src[c, rip1, sj]) +
                    np.float32(0.25)   *  src[c, si,   sj]
                )

    @_njit(parallel=True, cache=True, fastmath=True)
    def _nb_ds2(src, dst):
        """Full 2x downsample: [0.25,0.5,0.25] separable kernel, (H,W) float32."""
        H = src.shape[0]; W = src.shape[1]
        dH = dst.shape[0]; dW = dst.shape[1]
        for di in _prange(dH):
            si   = di * 2
            rim1 = si - 1 if si > 0     else 0
            rip1 = si + 1 if si + 1 < H else H - 1
            for dj in range(dW):
                sj   = dj * 2
                cjm1 = sj - 1 if sj > 0     else 0
                cjp1 = sj + 1 if sj + 1 < W else W - 1
                dst[di, dj] = (
                    np.float32(0.0625) * (src[rim1, cjm1] + src[rim1, cjp1] +
                                          src[rip1, cjm1] + src[rip1, cjp1]) +
                    np.float32(0.125)  * (src[rim1, sj]   + src[si,   cjm1] +
                                          src[si,   cjp1] + src[rip1, sj]) +
                    np.float32(0.25)   *  src[si,   sj]
                )

    @_njit(parallel=True, cache=True, fastmath=True)
    def _nb_us(src, dst, dr0, dr1, dc0, dc1):
        """Banded 2x bilinear upsample, (C,h,w) → dst (C,H,W) band [dr0:dr1, dc0:dc1]."""
        C = src.shape[0]; h = src.shape[1]; w = src.shape[2]
        H = dst.shape[1]; W = dst.shape[2]
        si_lo = dr0 >> 1
        si_hi = (dr1 + 1) >> 1
        sj_lo = dc0 >> 1
        sj_hi = (dc1 + 1) >> 1
        n_si = si_hi - si_lo
        for c_si in _prange(C * n_si):
            c  = c_si // n_si
            si = si_lo + c_si % n_si
            di0  = si * 2
            di1  = di0 + 1
            si_n = si + 1 if si + 1 < h else h - 1
            for sj in range(sj_lo, sj_hi):
                dj0  = sj * 2
                dj1  = dj0 + 1
                sj_n = sj + 1 if sj + 1 < w else w - 1
                v00 = src[c, si,   sj]
                v01 = src[c, si,   sj_n]
                v10 = src[c, si_n, sj]
                v11 = src[c, si_n, sj_n]
                if dj0 < W:
                    if di0 < H: dst[c, di0, dj0] = v00
                    if di1 < H: dst[c, di1, dj0] = (v00 + v10) * np.float32(0.5)
                if dj1 < W:
                    if di0 < H: dst[c, di0, dj1] = (v00 + v01) * np.float32(0.5)
                    if di1 < H: dst[c, di1, dj1] = (v00 + v01 + v10 + v11) * np.float32(0.25)

    @_njit(parallel=True, cache=True, fastmath=True)
    def _nb_accum(gauss_l, lapl_l, mask_l, out_l, r0, r1, c0, c1):
        """Fused: lapl_band = gauss_band - lapl_band; lapl_band *= mask; out_l += lapl_band.
        lapl_l[..., r0:r1, c0:c1] holds the upsample result on entry; gauss_l is the
        fine Gaussian level.  Replaces 3 separate numpy ufunc calls."""
        C = gauss_l.shape[0]
        n_r = r1 - r0
        for ch_r in _prange(C * n_r):
            ch = ch_r // n_r
            r  = r0 + ch_r % n_r
            for c in range(c0, c1):
                v = (gauss_l[ch, r, c] - lapl_l[ch, r, c]) * mask_l[r, c]
                lapl_l[ch, r, c] = v
                out_l[ch, r, c] += v


    @_njit(cache=True)
    def _chamfer_voronoi(edt_d, edt_i, cov_bits):
        """Two-pass Chamfer (H=3, D=4) unified Voronoi EDT with VALMASKED coverage masking.
        Matches C++ leftupxy+rightdownxy+VALMASKED algorithm.
        edt_d:    uint32  Chamfer distance (0=exclusive zone, 0xfffffbff=unset)
        edt_i:    uint8   nearest image index (0..n-1 or 255=unset)
        cov_bits: uint64  per-pixel bitmask: bit i=1 if image i covers this pixel.
        Coverage masking: in overlap zones (cov_bits has ≥2 bits set), only propagate
        from images that cover the current pixel.  Background (cov=0) and exclusive
        (single-bit cov) pixels allow unrestricted propagation — this matches C++
        VALMASKED which adds MASKOFF (not block) for uncovered pixels, ensuring
        background/exclusive pixels always get a valid assignment."""
        H = np.uint32(3)
        D = np.uint32(4)
        ONE64 = np.uint64(1)
        ZERO64 = np.uint64(0)
        rows = edt_d.shape[0]
        cols = edt_d.shape[1]
        # Forward pass: top-left → bottom-right (considers top/left neighbors)
        for r in range(rows):
            for c in range(cols):
                d = edt_d[r, c]
                i = edt_i[r, c]
                cov = cov_bits[r, c]
                # is_overlap: cov has ≥2 bits set  ↔  cov & (cov-1) != 0
                is_overlap = (cov & (cov - ONE64)) != ZERO64
                if r > 0:
                    if c > 0:
                        ni = edt_i[r-1, c-1]
                        if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                            nd = edt_d[r-1, c-1] + D
                            if nd < d or (nd == d and ni < i): d = nd; i = ni
                    ni = edt_i[r-1, c]
                    if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                        nd = edt_d[r-1, c] + H
                        if nd < d or (nd == d and ni < i): d = nd; i = ni
                    if c + 1 < cols:
                        ni = edt_i[r-1, c+1]
                        if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                            nd = edt_d[r-1, c+1] + D
                            if nd < d or (nd == d and ni < i): d = nd; i = ni
                if c > 0:
                    ni = edt_i[r, c-1]
                    if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                        nd = edt_d[r, c-1] + H
                        if nd < d or (nd == d and ni < i): d = nd; i = ni
                edt_d[r, c] = d; edt_i[r, c] = i
        # Backward pass: bottom-right → top-left (considers bottom/right neighbors)
        for r in range(rows - 1, -1, -1):
            for c in range(cols - 1, -1, -1):
                d = edt_d[r, c]
                i = edt_i[r, c]
                cov = cov_bits[r, c]
                is_overlap = (cov & (cov - ONE64)) != ZERO64
                if r + 1 < rows:
                    if c + 1 < cols:
                        ni = edt_i[r+1, c+1]
                        if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                            nd = edt_d[r+1, c+1] + D
                            if nd < d or (nd == d and ni < i): d = nd; i = ni
                    ni = edt_i[r+1, c]
                    if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                        nd = edt_d[r+1, c] + H
                        if nd < d or (nd == d and ni < i): d = nd; i = ni
                    if c > 0:
                        ni = edt_i[r+1, c-1]
                        if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                            nd = edt_d[r+1, c-1] + D
                            if nd < d or (nd == d and ni < i): d = nd; i = ni
                if c + 1 < cols:
                    ni = edt_i[r, c+1]
                    if ni != np.uint8(255) and (not is_overlap or (cov >> np.uint64(ni)) & ONE64):
                        nd = edt_d[r, c+1] + H
                        if nd < d or (nd == d and ni < i): d = nd; i = ni
                edt_d[r, c] = d; edt_i[r, c] = i

    @_njit(parallel=True, cache=True)
    def _nb_inpaint(r, g, b, mask2d, ri_ds, ci_ds, ds):
        """Inpaint transparent pixels via downscaled EDT lookup; parallel over rows."""
        H = mask2d.shape[0]
        W = mask2d.shape[1]
        for row in _prange(H):
            for col in range(W):
                if mask2d[row, col]:
                    src_r = ri_ds[row // ds, col // ds] * ds
                    src_c = ci_ds[row // ds, col // ds] * ds
                    if src_r >= H: src_r = H - 1
                    if src_c >= W: src_c = W - 1
                    r[row, col] = r[src_r, src_c]
                    g[row, col] = g[src_r, src_c]
                    b[row, col] = b[src_r, src_c]

    @_njit(parallel=True, cache=True)
    def _nb_assign_mask(assignment, i_val, mask):
        """Fill mask[r,c] = 1.0 where assignment[r,c]==i_val, else 0.0 (full workspace)."""
        H = assignment.shape[0]; W = assignment.shape[1]
        for r in _prange(H):
            for c in range(W):
                mask[r, c] = np.float32(1.0) if assignment[r, c] == i_val else np.float32(0.0)

    @_njit(parallel=True, cache=True)
    def _nb_fill_border(ws, r0, r1, c0, c1):
        """Fill gauss[0] (C,H,W) outside [r0:r1, c0:c1] by mirror (symmetric) padding.
        Mirror: row r0-1 → r0, r0-2 → r0+1, … so the extension mirrors image content.
        This prevents constant edge pixels from biasing the coarse Gaussian levels,
        which causes visible brightness gradients (vignetting) at image boundaries.
        Order: top/bottom rows first (image-col band only), then left/right cols (full
        height including already-mirrored top/bottom) so corners get double-mirror."""
        C = ws.shape[0]; H = ws.shape[1]; W = ws.shape[2]
        if r0 > 0:
            for c_ri in _prange(C * r0):
                ch = c_ri // r0
                ri = c_ri %  r0
                src_r = 2 * r0 - 1 - ri        # mirror: ri=r0-1 → r0, ri=0 → 2r0-1
                if src_r >= r1: src_r = r1 - 1  # clamp to last image row
                for col in range(c0, c1):
                    ws[ch, ri, col] = ws[ch, src_r, col]
        n_bot = H - r1
        if n_bot > 0:
            for c_ri in _prange(C * n_bot):
                ch = c_ri // n_bot
                ri = c_ri % n_bot
                src_r = r1 - 1 - ri             # mirror: ri=0 → r1-1, ri=1 → r1-2
                if src_r < r0: src_r = r0       # clamp to first image row
                for col in range(c0, c1):
                    ws[ch, r1 + ri, col] = ws[ch, src_r, col]
        if c0 > 0:
            for c_ri in _prange(C * H):
                ch = c_ri // H
                ri = c_ri %  H
                for col in range(c0):
                    src_c = 2 * c0 - 1 - col    # mirror: col=c0-1 → c0, col=0 → 2c0-1
                    if src_c >= c1: src_c = c1 - 1
                    ws[ch, ri, col] = ws[ch, ri, src_c]
        if c1 < W:
            for c_ri in _prange(C * H):
                ch = c_ri // H
                ri = c_ri %  H
                for col in range(c1, W):
                    src_c = 2 * c1 - 1 - col    # mirror: col=c1 → c1-1, col=c1+1 → c1-2
                    if src_c < c0: src_c = c0
                    ws[ch, ri, col] = ws[ch, ri, src_c]

    @_njit(parallel=True, cache=True)
    def _nb_gamma_u8(src, gamma, dst):
        """Apply power-curve (gamma) correction to uint8.
        dst = round((src/255)^gamma * 255).  Output always in [0,255] – never clips."""
        H = src.shape[0]; W = src.shape[1]
        for row in _prange(H):
            for col in range(W):
                v = np.float32(src[row, col]) * np.float32(1.0 / 255.0)
                if v > np.float32(0.0):
                    v = v ** gamma
                dst[row, col] = np.uint8(v * np.float32(255.0) + np.float32(0.5))

    @_njit(parallel=True, cache=True)
    def _nb_gamma_u16(src, gamma, dst):
        """Apply power-curve (gamma) correction to uint16.
        dst = round((src/65535)^gamma * 65535).  Output always in [0,65535] – never clips."""
        H = src.shape[0]; W = src.shape[1]
        for row in _prange(H):
            for col in range(W):
                v = np.float32(src[row, col]) * np.float32(1.0 / 65535.0)
                if v > np.float32(0.0):
                    v = v ** gamma
                dst[row, col] = np.uint16(v * np.float32(65535.0) + np.float32(0.5))

    @_njit(parallel=True, cache=True)
    def _nb_sat_scale_u8(ch0, ch1, ch2, scale):
        """Scale colour saturation in-place for uint8.
        corrected = grey + (ch - grey) * scale,  grey = (ch0+ch1+ch2)/3."""
        H = ch0.shape[0]; W = ch0.shape[1]
        for row in _prange(H):
            for col in range(W):
                v0 = np.float32(ch0[row, col])
                v1 = np.float32(ch1[row, col])
                v2 = np.float32(ch2[row, col])
                g  = (v0 + v1 + v2) * np.float32(1.0 / 3.0)
                v0 = g + (v0 - g) * scale
                v1 = g + (v1 - g) * scale
                v2 = g + (v2 - g) * scale
                if v0 <   0.0: v0 =   0.0
                elif v0 > 255.0: v0 = 255.0
                if v1 <   0.0: v1 =   0.0
                elif v1 > 255.0: v1 = 255.0
                if v2 <   0.0: v2 =   0.0
                elif v2 > 255.0: v2 = 255.0
                ch0[row, col] = np.uint8(v0 + np.float32(0.5))
                ch1[row, col] = np.uint8(v1 + np.float32(0.5))
                ch2[row, col] = np.uint8(v2 + np.float32(0.5))

    @_njit(parallel=True, cache=True)
    def _nb_sat_scale_u16(ch0, ch1, ch2, scale):
        """Scale colour saturation in-place for uint16."""
        H = ch0.shape[0]; W = ch0.shape[1]
        for row in _prange(H):
            for col in range(W):
                v0 = np.float32(ch0[row, col])
                v1 = np.float32(ch1[row, col])
                v2 = np.float32(ch2[row, col])
                g  = (v0 + v1 + v2) * np.float32(1.0 / 3.0)
                v0 = g + (v0 - g) * scale
                v1 = g + (v1 - g) * scale
                v2 = g + (v2 - g) * scale
                if v0 <     0.0: v0 =     0.0
                elif v0 > 65535.0: v0 = 65535.0
                if v1 <     0.0: v1 =     0.0
                elif v1 > 65535.0: v1 = 65535.0
                if v2 <     0.0: v2 =     0.0
                elif v2 > 65535.0: v2 = 65535.0
                ch0[row, col] = np.uint16(v0 + np.float32(0.5))
                ch1[row, col] = np.uint16(v1 + np.float32(0.5))
                ch2[row, col] = np.uint16(v2 + np.float32(0.5))

    @_njit(parallel=True, cache=True)
    def _nb_dither_clip_u8(src, dither, out):
        """Add tiled dither, clip [0,255], cast to uint8.  src/dither: (C,H,W) float32."""
        C = src.shape[0]; H = src.shape[1]; W = src.shape[2]
        dH = dither.shape[1]; dW = dither.shape[2]
        for ch_row in _prange(C * H):
            ch  = ch_row // H
            row = ch_row %  H
            dr  = row % dH
            for col in range(W):
                dc = col % dW
                v = src[ch, row, col] + dither[ch, dr, dc]
                if   v <   0.0: v =   0.0
                elif v > 255.0: v = 255.0
                out[ch, row, col] = np.uint8(v)

    @_njit(parallel=True, cache=True)
    def _nb_dither_clip_u16(src, dither, out):
        """Add tiled dither, clip [0,65535], cast to uint16.  src/dither: (C,H,W) float32."""
        C = src.shape[0]; H = src.shape[1]; W = src.shape[2]
        dH = dither.shape[1]; dW = dither.shape[2]
        for ch_row in _prange(C * H):
            ch  = ch_row // H
            row = ch_row %  H
            dr  = row % dH
            for col in range(W):
                dc = col % dW
                v = src[ch, row, col] + dither[ch, dr, dc]
                if   v <     0.0: v =     0.0
                elif v > 65535.0: v = 65535.0
                out[ch, row, col] = np.uint16(v)

    @_njit(parallel=True, cache=True)
    def _nb_gain_clip_u8(src, gain, dst):
        """Multiply uint8 src by scalar gain, clip [0,255], write to dst (may alias src)."""
        H = src.shape[0]; W = src.shape[1]
        for row in _prange(H):
            for col in range(W):
                v = src[row, col] * gain
                if   v <   0.0: v =   0.0
                elif v > 255.0: v = 255.0
                dst[row, col] = np.uint8(v)

    @_njit(parallel=True, cache=True)
    def _nb_gain_clip_u16(src, gain, dst):
        """Multiply uint16 src by scalar gain, clip [0,65535], write to dst (may alias src)."""
        H = src.shape[0]; W = src.shape[1]
        for row in _prange(H):
            for col in range(W):
                v = src[row, col] * gain
                if   v <     0.0: v =     0.0
                elif v > 65535.0: v = 65535.0
                dst[row, col] = np.uint16(v)

    @_njit(parallel=True, cache=True)
    def _nb_seam_cost(ch0_i, ch0_j, ch1_i, ch1_j, ch2_i, ch2_j, cost):
        """Cost for content-aware seam: squared RGB diff + gradient-magnitude²
        from BOTH images (central differences, boundary-clamped).

        The gradient term ensures the seam routes AROUND visually complex
        regions (e.g. timestamps) in EITHER image, not just where the two
        images differ.  Because the seam avoids the high-gradient area, the
        assignment naturally puts those pixels on the side of the OTHER image,
        which has smooth content there.
        """
        H = cost.shape[0]; W = cost.shape[1]
        for row in _prange(H):
            rp = row - 1 if row > 0     else 0
            rn = row + 1 if row < H - 1 else H - 1
            for col in range(W):
                cp = col - 1 if col > 0     else 0
                cn = col + 1 if col < W - 1 else W - 1
                # Squared RGB colour difference.
                d0 = np.float32(ch0_i[row, col]) - np.float32(ch0_j[row, col])
                d1 = np.float32(ch1_i[row, col]) - np.float32(ch1_j[row, col])
                d2 = np.float32(ch2_i[row, col]) - np.float32(ch2_j[row, col])
                diff_sq = d0*d0 + d1*d1 + d2*d2
                # Gradient magnitude² of image i.
                iy0 = np.float32(ch0_i[rn, col]) - np.float32(ch0_i[rp, col])
                iy1 = np.float32(ch1_i[rn, col]) - np.float32(ch1_i[rp, col])
                iy2 = np.float32(ch2_i[rn, col]) - np.float32(ch2_i[rp, col])
                ix0 = np.float32(ch0_i[row, cn]) - np.float32(ch0_i[row, cp])
                ix1 = np.float32(ch1_i[row, cn]) - np.float32(ch1_i[row, cp])
                ix2 = np.float32(ch2_i[row, cn]) - np.float32(ch2_i[row, cp])
                grad_i = iy0*iy0 + iy1*iy1 + iy2*iy2 + ix0*ix0 + ix1*ix1 + ix2*ix2
                # Gradient magnitude² of image j.
                jy0 = np.float32(ch0_j[rn, col]) - np.float32(ch0_j[rp, col])
                jy1 = np.float32(ch1_j[rn, col]) - np.float32(ch1_j[rp, col])
                jy2 = np.float32(ch2_j[rn, col]) - np.float32(ch2_j[rp, col])
                jx0 = np.float32(ch0_j[row, cn]) - np.float32(ch0_j[row, cp])
                jx1 = np.float32(ch1_j[row, cn]) - np.float32(ch1_j[row, cp])
                jx2 = np.float32(ch2_j[row, cn]) - np.float32(ch2_j[row, cp])
                grad_j = jy0*jy0 + jy1*jy1 + jy2*jy2 + jx0*jx0 + jx1*jx1 + jx2*jx2
                cost[row, col] = diff_sq + grad_i + grad_j

    @_njit(cache=True)
    def _nb_dp_fwd(cost, dp):
        """Sequential DP forward pass for min-cost seam.  Rows are inherently
        sequential (row r depends on r-1); JIT compilation removes Python loop
        overhead (~20 M iterations for a 10240x2000 overlap).  No prange: W is
        too small to recover the per-row thread-launch cost across 48 cores."""
        H = cost.shape[0]; W = cost.shape[1]
        for c in range(W):
            dp[0, c] = cost[0, c]
        for r in range(1, H):
            for c in range(W):
                lo = c - 1 if c > 0     else 0
                hi = c + 1 if c < W - 1 else W - 1
                dp[r, c] = cost[r, c] + min(dp[r - 1, lo], dp[r - 1, c], dp[r - 1, hi])

    @_njit(cache=True)
    def _nb_dp_backtrack(dp, seam):
        """Backtrack through the DP table to recover the min-cost seam path."""
        H = dp.shape[0]; W = dp.shape[1]
        best_c = 0
        best_v = dp[H - 1, 0]
        for c in range(1, W):
            v = dp[H - 1, c]
            if v < best_v:
                best_v = v; best_c = c
        seam[H - 1] = best_c
        for r in range(H - 2, -1, -1):
            c  = seam[r + 1]
            c0 = c - 1 if c > 0     else 0
            c1 = c + 2 if c < W - 1 else W
            bv = dp[r, c0]; bc = c0
            for cc in range(c0 + 1, c1):
                v = dp[r, cc]
                if v < bv:
                    bv = v; bc = cc
            seam[r] = bc

    @_njit(parallel=True, cache=True)
    def _nb_build_mask_sm(mask_full, mask_sm, ypos, xpos, r0_sm, c0_sm, ds):
        """Build workspace-aligned downscaled mask (uint8).
        mask_sm[r,c] = 1 iff any pixel in workspace block
        [(r0_sm+r)*ds:(r0_sm+r+1)*ds, (c0_sm+c)*ds:(c0_sm+c+1)*ds]
        is covered by the image (mask_full local coords)."""
        H = mask_full.shape[0]; W = mask_full.shape[1]
        Hr = mask_sm.shape[0]; Wr = mask_sm.shape[1]
        for r in _prange(Hr):
            ry0 = (r0_sm + r) * ds - ypos
            ry1 = (r0_sm + r + 1) * ds - ypos
            if ry0 < 0: ry0 = 0
            if ry1 > H: ry1 = H
            for c in range(Wr):
                cx0 = (c0_sm + c) * ds - xpos
                cx1 = (c0_sm + c + 1) * ds - xpos
                if cx0 < 0: cx0 = 0
                if cx1 > W: cx1 = W
                val = np.uint8(0)
                for iy in range(ry0, ry1):
                    for ix in range(cx0, cx1):
                        if mask_full[iy, ix]:
                            val = np.uint8(1)
                            break
                    if val: break
                mask_sm[r, c] = val

    @_njit(parallel=True, cache=True)
    def _nb_add_cov(coverage, cov_bits, mask_u8, i, r0, c0):
        """Fused coverage+=mask, cov_bits|=(1<<i); avoids large uint64 temporaries."""
        bit = np.uint64(1) << np.uint64(i)
        nrows = mask_u8.shape[0]; ncols = mask_u8.shape[1]
        for row in _prange(nrows):
            for col in range(ncols):
                if mask_u8[row, col]:
                    coverage[r0 + row, c0 + col] += np.uint8(1)
                    cov_bits[r0 + row, c0 + col] |= bit

    @_njit(parallel=True, cache=True)
    def _nb_seed_excl(edt_d, edt_i, mask_u8, coverage, i, r0, c0):
        """Seed EDT: exclusive pixels (mask & coverage==1) → dist=0, idx=i."""
        bit_i = np.uint8(i)
        nrows = mask_u8.shape[0]; ncols = mask_u8.shape[1]
        for row in _prange(nrows):
            for col in range(ncols):
                if mask_u8[row, col] and coverage[r0 + row, c0 + col] == np.uint8(1):
                    edt_d[r0 + row, c0 + col] = np.uint32(0)
                    edt_i[r0 + row, c0 + col] = bit_i

    @_njit(parallel=True, cache=True)
    def _nb_fill_unassigned(edt_i, mask_u8, i, r0, c0):
        """Fill any remaining unassigned (255) pixels inside mask with i."""
        bit_i = np.uint8(i)
        nrows = mask_u8.shape[0]; ncols = mask_u8.shape[1]
        for row in _prange(nrows):
            for col in range(ncols):
                if mask_u8[row, col] and edt_i[r0 + row, c0 + col] == np.uint8(255):
                    edt_i[r0 + row, c0 + col] = bit_i


def pyramid_sizes(workwidth: int, workheight: int, levels: int) -> List[tuple]:
    sizes = [(workheight, workwidth)]
    w, h = workwidth, workheight
    for _ in range(levels - 1):
        w = (w + 1) >> 1
        h = (h + 1) >> 1
        sizes.append((h, w))
    return sizes


def _downsample_into(src: np.ndarray, tmp: np.ndarray, dst: np.ndarray,
                     r_lo: int = 0, r_hi: int = -1,
                     c_lo: int = 0, c_hi: int = -1) -> None:
    """Downsample src into dst using tmp as scratch (all pre-allocated).

    r_lo/r_hi, c_lo/c_hi: row/col extent of the non-constant image region.
    Only the 2-D band around the image is convolved; the constant edge-extended
    rows/cols outside are simply copied, skipping up to 75% of work.
    """
    H, W = src.shape[-2], src.shape[-1]
    if r_hi < 0: r_hi = H
    if c_hi < 0: c_hi = W
    lr = max(0, r_lo - 1);  rr = min(H, r_hi + 1)
    lc = max(0, c_lo - 1);  rc = min(W, c_hi + 1)

    if lr == 0 and rr == H and lc == 0 and rc == W:
        convolve1d(src, _KERNEL, axis=-1, mode='reflect', output=tmp)
        convolve1d(tmp, _KERNEL, axis=-2, mode='reflect', output=tmp)
    else:
        convolve1d(src[..., lr:rr, lc:rc], _KERNEL, axis=-1, mode='nearest',
                   output=tmp[..., lr:rr, lc:rc])
        if lr > 0:  tmp[..., :lr, :]   = src[..., :lr, :]
        if rr < H:  tmp[..., rr:, :]   = src[..., rr:, :]
        if lc > 0:  tmp[..., lr:rr, :lc] = src[..., lr:rr, :lc]
        if rc < W:  tmp[..., lr:rr, rc:] = src[..., lr:rr, rc:]
        lr2 = max(0, lr - 1);  rr2 = min(H, rr + 1)
        lc2 = max(0, lc - 1);  rc2 = min(W, rc + 1)
        convolve1d(tmp[..., lr2:rr2, lc2:rc2], _KERNEL, axis=-2, mode='nearest',
                   output=tmp[..., lr2:rr2, lc2:rc2])

    np.copyto(dst, tmp[..., ::2, ::2])


def _upsample_into(src: np.ndarray, tmp: np.ndarray, dst: np.ndarray,
                   dr0: int = 0, dr1: int = -1,
                   dc0: int = 0, dc1: int = -1) -> None:
    """2x linear upsample src into dst using tmp as scratch (all pre-allocated).
    tmp.shape must be (*src.shape[:-1], dst.shape[-1]).
    dr0/dr1, dc0/dc1: optional dst row/col band; only that region of dst is written."""
    h, w = src.shape[-2], src.shape[-1]
    H, W = dst.shape[-2], dst.shape[-1]
    if dr1 < 0: dr1 = H
    if dc1 < 0: dc1 = W
    sr0 = dr0 >> 1;  sr1 = min(h, (dr1 + 2) >> 1)
    nor = H // 2
    if dc0 == 0 and dc1 == W:
        ic = (W + 1) // 2
        tmp[..., sr0:sr1, 0::2] = src[..., sr0:sr1, :ic]
        no = W // 2
        if no > 0:
            np2 = min(no, w - 1)
            if np2 > 0:
                tmp[..., sr0:sr1, 1:1 + 2 * np2:2] = (
                    src[..., sr0:sr1, :np2] + src[..., sr0:sr1, 1:np2 + 1]) * 0.5
            if no > np2:
                tmp[..., sr0:sr1, 1 + 2 * np2::2] = src[..., sr0:sr1, w - 1:w]
    else:
        ec0 = dc0 + (dc0 & 1)
        nec = max(0, (dc1 - ec0 + 1) // 2)
        if nec > 0:
            sk0c = ec0 >> 1;  sk1c = min((W + 1) // 2, sk0c + nec);  nc = sk1c - sk0c
            tmp[..., sr0:sr1, ec0:ec0 + 2 * nc:2] = src[..., sr0:sr1, sk0c:sk1c]
        oc0 = dc0 + (1 - (dc0 & 1))
        noc = max(0, (dc1 - oc0 + 1) // 2)
        if noc > 0:
            sk0c = (oc0 - 1) >> 1
            np2c = min(w - 1, sk0c + noc) - sk0c
            if np2c > 0:
                tmp[..., sr0:sr1, oc0:oc0 + 2 * np2c:2] = (
                    src[..., sr0:sr1, sk0c:sk0c + np2c] +
                    src[..., sr0:sr1, sk0c + 1:sk0c + np2c + 1]) * 0.5
            if np2c < noc:
                oc_s = oc0 + 2 * np2c
                if oc_s < dc1:
                    tmp[..., sr0:sr1, oc_s:dc1:2] = src[..., sr0:sr1, w - 1:w]
    ir = (H + 1) // 2
    if dr0 == 0 and dr1 == H:
        dst[..., 0::2, dc0:dc1] = tmp[..., :ir, dc0:dc1]
        if nor > 0:
            npr = min(nor, h - 1)
            if npr > 0:
                dst[..., 1:1 + 2 * npr:2, dc0:dc1] = (
                    tmp[..., :npr, dc0:dc1] + tmp[..., 1:npr + 1, dc0:dc1]) * 0.5
            if nor > npr:
                dst[..., 1 + 2 * npr::2, dc0:dc1] = tmp[..., h - 1:h, dc0:dc1]
    else:
        e0 = dr0 + (dr0 & 1)
        n_even = max(0, (dr1 - e0 + 1) // 2)
        if n_even > 0:
            sk0 = e0 >> 1;  sk1 = min(ir, sk0 + n_even);  n = sk1 - sk0
            dst[..., e0:e0 + 2 * n:2, dc0:dc1] = tmp[..., sk0:sk1, dc0:dc1]
        if nor > 0:
            o0 = dr0 + (1 - (dr0 & 1))
            n_odd = max(0, (dr1 - o0 + 1) // 2)
            if n_odd > 0:
                sk0 = (o0 - 1) >> 1
                npr = min(h - 1, sk0 + n_odd) - sk0
                if npr > 0:
                    dst[..., o0:o0 + 2 * npr:2, dc0:dc1] = (
                        tmp[..., sk0:sk0 + npr, dc0:dc1] +
                        tmp[..., sk0 + 1:sk0 + npr + 1, dc0:dc1]) * 0.5
                if npr < n_odd:
                    o_start = o0 + 2 * npr
                    if o_start < dr1:
                        dst[..., o_start:dr1:2, dc0:dc1] = tmp[..., h - 1:h, dc0:dc1]


def _place_channels_3d(img: ImageInfo, workheight: int, workwidth: int, bgr: bool) -> np.ndarray:
    ws = np.empty((3, workheight, workwidth), dtype=np.float32)
    r0, r1 = img.ypos, img.ypos + img.height
    c0, c1 = img.xpos, img.xpos + img.width
    for c in range(3):
        ci = (2 - c) if bgr else c
        ws[c, r0:r1, c0:c1] = img.channels[ci]
        if r0 > 0:
            src_r = np.clip(np.arange(2 * r0 - 1, r0 - 1, -1), r0, r1 - 1)
            ws[c, :r0, c0:c1] = ws[c][src_r, c0:c1]
        if r1 < workheight:
            n_bot = workheight - r1
            src_r = np.clip(np.arange(r1 - 1, r1 - 1 - n_bot, -1), r0, r1 - 1)
            ws[c, r1:, c0:c1] = ws[c][src_r, c0:c1]
        if c0 > 0:
            src_c = np.clip(np.arange(2 * c0 - 1, c0 - 1, -1), c0, c1 - 1)
            ws[c, :, :c0] = ws[c][:, src_c]
        if c1 < workwidth:
            n_right = workwidth - c1
            src_c = np.clip(np.arange(c1 - 1, c1 - 1 - n_right, -1), c0, c1 - 1)
            ws[c, :, c1:] = ws[c][:, src_c]
    return ws


def _exposure_correct(images: List[ImageInfo], verbosity: int = 1,
                      print_func: Callable = print) -> List[dict]:
    """Per-channel multiplicative gain correction to normalise overlap-zone means.

    Image 0 is the reference (gain=1).  For every spatially overlapping pair
    (i, j) the per-channel mean in the intersection region is compared, then a
    log-space least-squares solve finds global gains that minimise the squared
    log-ratio across all pairs.  Gains are applied in-place to img.channels.
    """
    n = len(images)
    if n < 2:
        return [{'gains': [1.0, 1.0, 1.0], 'gammas': [1.0, 1.0, 1.0],
                 'method': ['none', 'none', 'none'], 'capped': False} for _ in images]

    edges = []  # (i, j, ch, log_ratio)  where log_ratio = log(mean_i / mean_j)
    for i in range(n):
        img_i = images[i]
        r0_i, r1_i = img_i.ypos, img_i.ypos + img_i.height
        c0_i, c1_i = img_i.xpos, img_i.xpos + img_i.width
        for j in range(i + 1, n):
            img_j = images[j]
            r0_j, r1_j = img_j.ypos, img_j.ypos + img_j.height
            c0_j, c1_j = img_j.xpos, img_j.xpos + img_j.width
            r0 = max(r0_i, r0_j); r1 = min(r1_i, r1_j)
            c0 = max(c0_i, c0_j); c1 = min(c1_i, c1_j)
            if r1 <= r0 or c1 <= c0:
                continue
            mask_i = images[i].mask[r0 - r0_i:r1 - r0_i, c0 - c0_i:c1 - c0_i]
            mask_j = images[j].mask[r0 - r0_j:r1 - r0_j, c0 - c0_j:c1 - c0_j]
            overlap_mask = mask_i & mask_j
            if not overlap_mask.any():
                continue
            for ch in range(3):
                pi = images[i].channels[ch][r0 - r0_i:r1 - r0_i, c0 - c0_i:c1 - c0_i]
                pj = images[j].channels[ch][r0 - r0_j:r1 - r0_j, c0 - c0_j:c1 - c0_j]
                mi = float(np.mean(pi[overlap_mask].astype(np.float64)))
                mj = float(np.mean(pj[overlap_mask].astype(np.float64)))
                if mi > 0 and mj > 0:
                    edges.append((i, j, ch, np.log(mi) - np.log(mj)))

    if not edges:
        return

    # Solve  x_i - x_j = log(mean_i/mean_j)  in log space, anchor x_0 = 0.
    # Free variables: x_1 .. x_{n-1}.  Overdetermined system → lstsq.
    gains_log = np.zeros((n, 3), dtype=np.float64)
    for ch in range(3):
        ch_edges = [(i, j, lr) for i, j, c, lr in edges if c == ch]
        if len(ch_edges) == 0:
            continue
        A = np.zeros((len(ch_edges), n - 1), dtype=np.float64)
        b = np.zeros(len(ch_edges), dtype=np.float64)
        for k, (i, j, lr) in enumerate(ch_edges):
            if i > 0: A[k, i - 1] =  1.0
            if j > 0: A[k, j - 1] = -1.0
            b[k] = -lr
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        gains_log[1:, ch] = x

    # Cap gains to ±4 stops (factor 16) to handle large exposure differences.
    _MAX_STOPS = np.log(16)
    capped = gains_log.copy()
    gains_log = np.clip(gains_log, -_MAX_STOPS, _MAX_STOPS)
    info: List[dict] = []
    for i, img in enumerate(images):
        dtype = img.channels[0].dtype
        maxv_f = float(0xffff if dtype == np.uint16 else 0xff)

        # Per-channel correction strategy (hybrid):
        #   gain <= 1 (darkening): apply linear gain — exact, never clips when darkening.
        #   gain >  1 (brightening): apply gamma power curve — prevents highlight clipping.
        #     gamma = log(target_mean) / log(source_mean)  so source_mean^gamma = target_mean.
        #
        # Using gamma for darkening channels too causes opposite-direction power curves on
        # different channels (e.g. B gamma>1 while R/G gamma<1), which creates hue flips in
        # the shadows that appear as increased colour saturation artefacts.
        lin_gains = np.exp(gains_log[i])   # shape (3,) — used directly for gain<=1
        gammas    = np.ones(3, dtype=np.float64)
        use_gamma = np.zeros(3, dtype=bool)
        for ch in range(3):
            if gains_log[i, ch] > 1e-7:   # brightening: use gamma
                mean_norm = float(np.mean(img.channels[ch].astype(np.float64))) / maxv_f
                if 1e-4 < mean_norm < 1.0 - 1e-4:
                    target = float(np.clip(mean_norm * lin_gains[ch], 1e-4, 1.0 - 1e-4))
                    gammas[ch] = np.clip(np.log(target) / np.log(mean_norm), 0.1, 10.0)
                    use_gamma[ch] = True

        if verbosity >= 2:
            raw = np.exp(capped[i])
            cap_tag = " (CAPPED)" if np.any(np.abs(capped[i]) > _MAX_STOPS) else ""
            ch_strs = []
            for ch, name in enumerate('RGB'):
                if use_gamma[ch]:
                    ch_strs.append(f"{name}=gamma({gammas[ch]:.3f})")
                elif abs(lin_gains[ch] - 1.0) > 1e-5:
                    ch_strs.append(f"{name}=linear({lin_gains[ch]:.3f})")
            print_func(f"    exposure img {i}: {' '.join(ch_strs)}{cap_tag}")

        method = []
        for ch in range(3):
            if use_gamma[ch]:
                method.append('gamma')
            elif abs(lin_gains[ch] - 1.0) > 1e-5:
                method.append('linear')
            else:
                method.append('none')
        info.append({
            'gains':     lin_gains.tolist(),
            'gammas':    gammas.tolist(),
            'method':    method,
            'capped':    bool(np.any(np.abs(capped[i]) > _MAX_STOPS)),
        })

        any_change = (np.any(use_gamma) or
                      np.any(np.abs(gains_log[i]) > 1e-7))
        if not any_change:
            continue

        if _NUMBA_OK:
            if dtype == np.uint8:
                for ch in range(3):
                    if use_gamma[ch] and abs(gammas[ch] - 1.0) > 1e-5:
                        _nb_gamma_u8(img.channels[ch], np.float32(gammas[ch]), img.channels[ch])
                    elif not use_gamma[ch] and abs(lin_gains[ch] - 1.0) > 1e-5:
                        _nb_gain_clip_u8(img.channels[ch], np.float32(lin_gains[ch]), img.channels[ch])
            else:
                for ch in range(3):
                    if use_gamma[ch] and abs(gammas[ch] - 1.0) > 1e-5:
                        _nb_gamma_u16(img.channels[ch], np.float32(gammas[ch]), img.channels[ch])
                    elif not use_gamma[ch] and abs(lin_gains[ch] - 1.0) > 1e-5:
                        _nb_gain_clip_u16(img.channels[ch], np.float32(lin_gains[ch]), img.channels[ch])
        else:
            maxv_np = np.float32(maxv_f)
            for ch in range(3):
                if use_gamma[ch] and abs(gammas[ch] - 1.0) > 1e-5:
                    arr = (img.channels[ch].astype(np.float64) / maxv_f) ** gammas[ch]
                    img.channels[ch] = np.round(arr * maxv_f).astype(dtype)
                elif not use_gamma[ch] and abs(lin_gains[ch] - 1.0) > 1e-5:
                    arr = img.channels[ch].astype(np.float32) * np.float32(lin_gains[ch])
                    img.channels[ch] = np.clip(arr, 0, maxv_np).astype(dtype)

    return info


def _saturation_correct(images: List[ImageInfo], verbosity: int = 1,
                        print_func: Callable = print) -> List[float]:
    """Per-image saturation scale correction via overlap-zone RMS chroma ratios.

    For each overlapping pair the RMS chroma (per-pixel deviation from grey,
    grey = (R+G+B)/3) is compared.  A log-space least-squares solve finds one
    saturation scale per image (image 0 = reference, scale=1).
    Correction applied in-place: corrected = grey + (ch - grey) * scale.
    """
    n = len(images)
    if n < 2:
        return [1.0] * len(images)

    edges = []  # (i, j, log_ratio)  log_ratio = log(chroma_i / chroma_j)
    for i in range(n):
        img_i = images[i]
        r0_i, r1_i = img_i.ypos, img_i.ypos + img_i.height
        c0_i, c1_i = img_i.xpos, img_i.xpos + img_i.width
        for j in range(i + 1, n):
            img_j = images[j]
            r0_j, r1_j = img_j.ypos, img_j.ypos + img_j.height
            c0_j, c1_j = img_j.xpos, img_j.xpos + img_j.width
            r0 = max(r0_i, r0_j); r1 = min(r1_i, r1_j)
            c0 = max(c0_i, c0_j); c1 = min(c1_i, c1_j)
            if r1 <= r0 or c1 <= c0:
                continue
            chs_i = [images[i].channels[ch][r0 - r0_i:r1 - r0_i, c0 - c0_i:c1 - c0_i]
                     .astype(np.float64) for ch in range(3)]
            chs_j = [images[j].channels[ch][r0 - r0_j:r1 - r0_j, c0 - c0_j:c1 - c0_j]
                     .astype(np.float64) for ch in range(3)]
            grey_i = (chs_i[0] + chs_i[1] + chs_i[2]) / 3.0
            grey_j = (chs_j[0] + chs_j[1] + chs_j[2]) / 3.0
            # Normalise chroma by mean luminance so the ratio is brightness-independent.
            # Without this, a brighter image has larger absolute deviations from grey even
            # at the same relative saturation, causing incorrect corrections.
            lum_i = max(float(np.mean(grey_i)), 1.0)
            lum_j = max(float(np.mean(grey_j)), 1.0)
            chroma_i = float(np.sqrt(np.mean(
                (chs_i[0] - grey_i)**2 + (chs_i[1] - grey_i)**2 + (chs_i[2] - grey_i)**2
            ))) / lum_i
            chroma_j = float(np.sqrt(np.mean(
                (chs_j[0] - grey_j)**2 + (chs_j[1] - grey_j)**2 + (chs_j[2] - grey_j)**2
            ))) / lum_j
            if chroma_i > 1e-6 and chroma_j > 1e-6:
                edges.append((i, j, np.log(chroma_i) - np.log(chroma_j)))

    if not edges:
        return [1.0] * n

    scales_log = np.zeros(n, dtype=np.float64)
    A = np.zeros((len(edges), n - 1), dtype=np.float64)
    b = np.zeros(len(edges), dtype=np.float64)
    for k, (i, j, lr) in enumerate(edges):
        if i > 0: A[k, i - 1] =  1.0
        if j > 0: A[k, j - 1] = -1.0
        b[k] = -lr
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scales_log[1:] = x

    _MAX_SAT = np.log(4.0)  # cap at ±2 stops of saturation
    scales_log = np.clip(scales_log, -_MAX_SAT, _MAX_SAT)

    if verbosity >= 2:
        print_func("    saturation scales: " +
                   "  ".join(f"img{i}={np.exp(scales_log[i]):.3f}" for i in range(n)))

    for i, img in enumerate(images):
        s = float(np.exp(scales_log[i]))
        if abs(s - 1.0) < 1e-5:
            continue
        dtype = img.channels[0].dtype
        if _NUMBA_OK:
            if dtype == np.uint8:
                _nb_sat_scale_u8(img.channels[0], img.channels[1], img.channels[2],
                                 np.float32(s))
            else:
                _nb_sat_scale_u16(img.channels[0], img.channels[1], img.channels[2],
                                  np.float32(s))
        else:
            maxv = float(0xffff if dtype == np.uint16 else 0xff)
            grey = (img.channels[0].astype(np.float64) +
                    img.channels[1].astype(np.float64) +
                    img.channels[2].astype(np.float64)) / 3.0
            for ch in range(3):
                arr = grey + (img.channels[ch].astype(np.float64) - grey) * s
                img.channels[ch] = np.clip(np.round(arr), 0, maxv).astype(dtype)

    return [float(np.exp(scales_log[i])) for i in range(n)]


def blend(images: List[ImageInfo], assignment: np.ndarray,
          workwidth: int, workheight: int, levels: int, workbpp: int,
          bgr: bool = False, verbosity: int = 1,
          exposure_correct: bool = False,
          saturation_correct: bool = False,
          out_info: Optional[dict] = None,
          print_func: Callable = print) -> List[np.ndarray]:
    if verbosity >= 1:
        print_func("  blending...")

    if exposure_correct:
        exp_info = _exposure_correct(images, verbosity=verbosity, print_func=print_func)
        if out_info is not None:
            out_info['exposure'] = exp_info
    if saturation_correct:
        sat_info = _saturation_correct(images, verbosity=verbosity, print_func=print_func)
        if out_info is not None:
            out_info['saturation'] = sat_info

    if levels == 0:
        dtype      = np.uint8 if workbpp == 8 else np.uint16
        out_chs    = [np.zeros((workheight, workwidth), dtype=dtype) for _ in range(3)]
        for i, img in enumerate(images):
            sel = assignment == i
            r0, r1 = img.ypos, img.ypos + img.height
            c0, c1 = img.xpos, img.xpos + img.width
            sel_img = sel[r0:r1, c0:c1]
            for c in range(3):
                ci = (2 - c) if bgr else c
                out_chs[c][r0:r1, c0:c1][sel_img] = img.channels[ci][sel_img]
        return out_chs

    pyr_shapes = pyramid_sizes(workwidth, workheight, levels)

    # out_pyr is the accumulator — must be zero; main-thread np.zeros ensures NUMA-local
    # allocation on node 0, which is optimal for the subsequent per-image writes.
    # gauss/lapl/mask_g use np.empty; they are naturally first-touched NUMA-locally in the
    # per-image loop (mostly main-thread writes before parallel kernels read them).
    out_pyr  = [np.zeros((3,) + s, dtype=np.float32) for s in pyr_shapes]
    gauss    = [np.empty((3,) + s, dtype=np.float32) for s in pyr_shapes]
    lapl     = [np.empty((3,) + s, dtype=np.float32) for s in pyr_shapes]
    mask_g   = [np.empty(s, dtype=np.float32) for s in pyr_shapes]
    mask_c   = [np.empty(s, dtype=np.float32) for s in pyr_shapes]
    if not _NUMBA_OK:
        # scipy/numpy path needs these tmp buffers; numba kernels are tmp-free
        uph_t  = [np.empty((3, pyr_shapes[l + 1][0], pyr_shapes[l][1]), dtype=np.float32)
                  for l in range(levels - 1)]

    for i, img in enumerate(images):
        r0, r1 = img.ypos, img.ypos + img.height
        c0, c1 = img.xpos, img.xpos + img.width

        # Mask pyramid: use the full workspace so the Gaussian downsample does not
        # taper near the image bounding-box edge.  A banded init (±1 row/col) leaves
        # zeroes outside the band; _nb_ds2 propagates those zeroes inward, causing a
        # brightness gradient up to 2^(levels-1) pixels wide near any image edge that
        # doesn't reach the workspace boundary.
        if _NUMBA_OK:
            _nb_assign_mask(assignment, np.uint8(i), mask_g[0])
        else:
            mask_g[0][:] = (assignment == np.uint8(i))
        for l in range(levels - 1):
            if _NUMBA_OK:
                _nb_ds2(mask_g[l], mask_g[l + 1])
            else:
                _downsample_into(mask_g[l], mask_c[l], mask_g[l + 1])

        # Place all 3 channels into gauss[0] in-place
        for c in range(3):
            ci = (2 - c) if bgr else c
            gauss[0][c, r0:r1, c0:c1] = img.channels[ci]
        if _NUMBA_OK:
            _nb_fill_border(gauss[0], r0, r1, c0, c1)
        else:
            for c in range(3):
                if r0 > 0:          gauss[0][c, :r0, c0:c1]  = gauss[0][c, r0, c0:c1]
                if r1 < workheight: gauss[0][c, r1:, c0:c1]  = gauss[0][c, r1 - 1, c0:c1]
                if c0 > 0:          gauss[0][c, :, :c0]      = gauss[0][c, :, c0:c0 + 1]
                if c1 < workwidth:  gauss[0][c, :, c1:]      = gauss[0][c, :, c1 - 1:c1]

        # Laplacian pyramid + banded accumulate (merged loop)
        for l in range(levels - 1):
            rl0 = r0 >> l;  rl1 = min((r1 + (1 << l) - 1) >> l, pyr_shapes[l][0])
            cl0 = c0 >> l;  cl1 = min((c1 + (1 << l) - 1) >> l, pyr_shapes[l][1])
            if _NUMBA_OK:
                _nb_ds3(gauss[l], gauss[l + 1])
                _nb_us(gauss[l + 1], lapl[l], rl0, rl1, cl0, cl1)
                _nb_accum(gauss[l], lapl[l], mask_g[l], out_pyr[l], rl0, rl1, cl0, cl1)
            else:
                _downsample_into(gauss[l], lapl[l], gauss[l + 1], rl0, rl1, cl0, cl1)
                _upsample_into(gauss[l + 1], uph_t[l], lapl[l], rl0, rl1, cl0, cl1)
                lb = lapl[l][..., rl0:rl1, cl0:cl1]
                np.subtract(gauss[l][..., rl0:rl1, cl0:cl1], lb, out=lb)
                np.multiply(lb, mask_g[l][rl0:rl1, cl0:cl1], out=lb)
                out_pyr[l][..., rl0:rl1, cl0:cl1] += lb
        # Coarsest level is tiny – full-array ops are negligible
        np.copyto(lapl[-1], gauss[-1])
        np.multiply(lapl[-1], mask_g[-1], out=lapl[-1])
        out_pyr[-1] += lapl[-1]

    # Collapse in-place reusing gauss buffers
    np.copyto(gauss[-1], out_pyr[-1])
    for l in range(levels - 2, -1, -1):
        if _NUMBA_OK:
            _nb_us(gauss[l + 1], gauss[l], 0, pyr_shapes[l][0], 0, pyr_shapes[l][1])
        else:
            _upsample_into(gauss[l + 1], uph_t[l], gauss[l])
        gauss[l] += out_pyr[l]

    result = gauss[0]  # (3, H, W), no copy

    _DTILE = 512
    _tile = np.random.default_rng().random((3, _DTILE, _DTILE), dtype=np.float32)
    _tile -= np.float32(0.5)
    _, _H, _W = result.shape
    if _NUMBA_OK:
        if workbpp == 8:
            _out = np.empty((3, _H, _W), dtype=np.uint8)
            _nb_dither_clip_u8(result, _tile, _out)
        else:
            _out = np.empty((3, _H, _W), dtype=np.uint16)
            _nb_dither_clip_u16(result, _tile, _out)
        return [_out[c] for c in range(3)]
    # Tiled dither fallback: small tile avoids 168 MB alloc + page-fault cost.
    for _r in range(0, _H, _DTILE):
        _r1 = min(_r + _DTILE, _H)
        for _c in range(0, _W, _DTILE):
            _c1 = min(_c + _DTILE, _W)
            result[:, _r:_r1, _c:_c1] += _tile[:, :_r1 - _r, :_c1 - _c]
    if workbpp == 8:
        np.clip(result, 0, 255, out=result)
        return [result[c].astype(np.uint8) for c in range(3)]
    np.clip(result, 0, 65535, out=result)
    return [result[c].astype(np.uint16) for c in range(3)]


def pseudowrap_levels(workwidth: int) -> int:
    split  = workwidth >> 1
    levels = 0
    while (2 << levels) - 2 <= split:
        levels += 1
    return levels


# ============================================================
#  Output writing (_write)
# ============================================================

_COMPRESSION_MAP = {'none': 'none', 'lzw': 'lzw', 'packbits': 'packbits'}


def _coverage_mask(images: List[ImageInfo], workwidth: int, workheight: int) -> np.ndarray:
    covered = np.zeros((workheight, workwidth), dtype=bool)
    for img in images:
        covered[img.ypos:img.ypos + img.height, img.xpos:img.xpos + img.width] |= img.mask
    return covered


def write_tiff(filename: str, out_channels: List[np.ndarray],
               images: List[ImageInfo], workwidth: int, workheight: int, workbpp: int,
               min_left: int, min_top: int, xres: float, yres: float,
               geotiff: Optional[GeoTIFFInfo], compression: str = 'none',
               bigtiff: bool = False, nomask: bool = False, verbosity: int = 1):
    covered  = _coverage_mask(images, workwidth, workheight)
    max_val  = 255 if workbpp == 8 else 65535
    dtype    = np.uint8 if workbpp == 8 else np.uint16

    rgba = np.zeros((workheight, workwidth, 4), dtype=dtype)
    rgba[..., 0] = out_channels[0]
    rgba[..., 1] = out_channels[1]
    rgba[..., 2] = out_channels[2]
    rgba[..., 3] = max_val if nomask else np.where(covered, max_val, 0).astype(dtype)

    comp_name = _COMPRESSION_MAP.get(compression.lower(), 'none')

    extra = []
    if xres > 0:
        extra.append((296, 3, 1, 2, True))  # ResolutionUnit = inch
    if geotiff is not None and geotiff.set:
        scale     = np.array([geotiff.x_cell_res, geotiff.y_cell_res, 0.0], dtype=np.float64)
        tiepoints = np.array([0.0, 0.0, 0.0,
                               min_left * geotiff.x_cell_res,
                               -min_top * geotiff.y_cell_res,
                               0.0], dtype=np.float64)
        extra += [
            (33550, 12, len(scale),     scale.tobytes(),     False),
            (33922, 12, len(tiepoints), tiepoints.tobytes(), False),
        ]

    opts = dict(
        photometric='rgb',
        planarconfig='contig',
        rowsperstrip=64,
        compression=comp_name,
        extrasamples=() if nomask else (2,),
        extratags=extra,
        metadata={},
    )
    if xres > 0:
        opts['resolution'] = (xres, yres)

    with tifffile.TiffWriter(filename, bigtiff=bigtiff) as tw:
        tw.write(rgba, **opts)


def write_jpeg(filename: str, out_channels: List[np.ndarray],
               images: List[ImageInfo], workwidth: int, workheight: int,
               quality: int = 75, nomask: bool = False, verbosity: int = 1):
    covered = _coverage_mask(images, workwidth, workheight)
    bgr = np.stack([out_channels[2], out_channels[1], out_channels[0]], axis=2)
    if not nomask:
        np.multiply(bgr, covered[:, :, np.newaxis], out=bgr)
    try:
        import cv2 as _cv2
        _cv2.imwrite(filename, bgr, [_cv2.IMWRITE_JPEG_QUALITY, quality])
    except ImportError:
        from PIL import Image
        Image.fromarray(bgr[:, :, ::-1], mode='RGB').save(filename, format='JPEG', quality=quality)


# ============================================================
#  Pseudowrap helpers (_core)
# ============================================================

def _pseudowrap_split(images: List[ImageInfo]) -> List[ImageInfo]:
    img   = images[0]
    split = img.width >> 1
    right_w = img.width - split

    right_img = ImageInfo(
        filename=img.filename, bpp=img.bpp,
        width=right_w, height=img.height, xpos=0, ypos=img.ypos,
        channels=[ch[:, split:].copy() for ch in img.channels],
        mask=img.mask[:, split:].copy(),
        xres=img.xres, yres=img.yres, geotiff=img.geotiff,
    )
    left_img = ImageInfo(
        filename=img.filename, bpp=img.bpp,
        width=split, height=img.height, xpos=img.width - split, ypos=img.ypos,
        channels=[ch[:, :split].copy() for ch in img.channels],
        mask=img.mask[:, :split].copy(),
        xres=img.xres, yres=img.yres, geotiff=img.geotiff,
    )
    return [left_img, right_img]


def _pseudowrap_seam(workwidth: int, workheight: int) -> np.ndarray:
    split      = workwidth >> 1
    assignment = np.zeros((workheight, workwidth), dtype=np.uint8)
    assignment[:, :workwidth - split] = 1
    assignment[:, workwidth - split:] = 0
    return assignment


# ============================================================
#  Main pipeline (go)
# ============================================================

def go(input_files: List[str], output_file: Optional[str] = None,
       max_levels: int = 1_000_000, sub_levels: int = 0, wideblend: bool = False,
       workbpp_cmd: int = 0, crop: bool = True, bgr: bool = False,
       reverse: bool = False, simple_seam: bool = False, content_seam: bool = False,
       nomask: bool = False,
       compression: str = 'none', jpeg_quality: int = -1, bigtiff: bool = False,
       seam_save_filename: Optional[str] = None, seam_load_filename: Optional[str] = None,
       xor_filename: Optional[str] = None, timing: bool = False,
       no_output: bool = False, exposure_correct: bool = False,
       saturation_correct: bool = False,
       verbosity: int = 1,
       print_func: Callable = print) -> dict:

    t0 = time.perf_counter()
    n  = len(input_files)
    if n == 0:   raise ValueError("No input files specified")
    if n > 255:  raise ValueError("Too many (>255) input images")

    if verbosity >= 1:
        print_func("  opening images...")

    t1 = time.perf_counter()
    images, workbpp, xres, yres, geotiff, workwidth, workheight = load_images(
        input_files, verbosity=verbosity, print_func=print_func)
    if timing:
        print_func(f"  load: {time.perf_counter() - t1:.3f}s")

    if not images:
        raise ValueError("No valid input files")

    if workbpp_cmd in (8, 16):
        workbpp = 8 if (workbpp_cmd == 16 and jpeg_quality >= 0) else workbpp_cmd

    min_left = min_top = 0
    if crop:
        min_left, min_top, workwidth, workheight = tighten(images)

    pseudowrap = (n == 1)
    if pseudowrap:
        if verbosity >= 1:
            print_func("  Only one image; pseudo-wrapping mode")
        images = _pseudowrap_split(images)
        levels = pseudowrap_levels(workwidth)
    else:
        levels = compute_levels(images, workwidth, workheight, wideblend, max_levels, sub_levels)

    if verbosity >= 1:
        print_func(f"  {workwidth}x{workheight}, {workbpp}bpp, {levels} levels")

    t1 = time.perf_counter()
    if pseudowrap:
        assignment = _pseudowrap_seam(workwidth, workheight)
    else:
        assignment, _ = compute_seams(
            images, workwidth, workheight,
            reverse=reverse, simple_seam=simple_seam, content_seam=content_seam,
            seam_load_filename=seam_load_filename,
            seam_save_filename=seam_save_filename,
            xor_filename=xor_filename, verbosity=verbosity, print_func=print_func)
    if timing:
        print_func(f"  seaming: {time.perf_counter() - t1:.3f}s")

    out_channels = None
    corrections: dict = {}

    if not no_output:
        t1 = time.perf_counter()
        out_channels = blend(
            images=images, assignment=assignment,
            workwidth=workwidth, workheight=workheight,
            levels=levels, workbpp=workbpp, bgr=bgr, verbosity=verbosity,
            exposure_correct=exposure_correct,
            saturation_correct=saturation_correct,
            out_info=corrections, print_func=print_func)
        if timing:
            print_func(f"  blend: {time.perf_counter() - t1:.3f}s")

        if pseudowrap:
            split = workwidth >> 1
            out_channels = [np.roll(ch, split, axis=1) for ch in out_channels]
            images[0].xpos  = 0
            images[0].width = workwidth
            images[0].mask  = np.ones((images[0].height, workwidth), dtype=bool)
            images = images[:1]

        if output_file:
            if verbosity >= 1:
                print_func(f"  writing {output_file}...")
            t1  = time.perf_counter()
            ext = output_file.rsplit('.', 1)[-1].lower()
            if ext in ('jpg', 'jpeg'):
                write_jpeg(output_file, out_channels, images, workwidth, workheight,
                           quality=jpeg_quality if jpeg_quality >= 0 else 75,
                           nomask=nomask, verbosity=verbosity)
            elif ext in ('tif', 'tiff'):
                write_tiff(output_file, out_channels, images, workwidth, workheight,
                           workbpp, min_left, min_top, xres, yres, geotiff,
                           compression=compression or 'none', bigtiff=bigtiff,
                           nomask=nomask, verbosity=verbosity)
            else:
                raise ValueError(f"Unknown output extension: .{ext}")
            if timing:
                print_func(f"  write: {time.perf_counter() - t1:.3f}s")

    if timing:
        print_func(f"  total: {time.perf_counter() - t0:.3f}s")

    return dict(images=images, assignment=assignment, out_channels=out_channels,
                workwidth=workwidth, workheight=workheight, workbpp=workbpp,
                min_left=min_left, min_top=min_top, xres=xres, yres=yres, geotiff=geotiff,
                corrections=corrections if (exposure_correct or saturation_correct) else {})


# ============================================================
#  CLI
# ============================================================

def _help():
    print("Usage: multiblend.py [options] -o OUTPUT INPUT [INPUT ...]")
    print()
    print("Options:")
    print("   -l X           X > 0: cap pyramid levels at X")
    print("                  X < 0: reduce auto level count by -X")
    print("   -d DEPTH       override output bit depth (8 or 16)")
    print("   -o FILE        output file (.tif/.tiff or .jpg/.jpeg)")
    print("  --nocrop        do not crop to content bounding box")
    print("  --bgr           swap R/B channel order in output")
    print("  --wideblend     base level count on output image dimensions")
    print("  --compression=X TIFF compression: none (default), lzw, packbits")
    print("  --nomask        omit alpha channel from output")
    print("  --bigtiff       write BigTIFF format")
    print("  --reverse       reverse image priority (last = highest)")
    print("  --simple-seam   use nearest-centre seaming instead of Voronoi EDT")
    print("  --save-seams F  save seam map PNG to F")
    print("  --load-seams F  load seam map PNG from F")
    print("  --save-xor F    save coverage XOR map PNG to F")
    print("  --no-output     skip blend/write (use with --save-seams)")
    print("  --timing        print per-stage timings")
    print("  -q / --quiet    suppress progress output")
    print("  -v / --verbose  increase verbosity")
    print("  -h / --help     show this help")
    print()
    print("Pass a single image to blend across the left/right boundary (pseudo-wrap).")
    sys.exit(0)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    _hdr_verbosity = 1 + argv.count('-v') + argv.count('--verbose') \
                       - argv.count('-q') - argv.count('--quiet')
    if _hdr_verbosity >= 1:
        print()
        print(f"multiblend.py v{__version__} (based on David Horman's multiblend)")
        print("-" * 60)

    if not argv or argv[0] in ('-h', '--help', '/?'):
        _help()

    if len(argv) < 2:
        sys.exit("not enough arguments (try --help)")

    if _hdr_verbosity >= 1:
        print()

    max_levels       = 1_000_000
    sub_levels       = 0
    depth            = 0
    crop             = True
    bgr              = False
    wideblend        = False
    reverse          = False
    simple_seam      = False
    nomask           = False
    bigtiff          = False
    timing           = False
    verbosity        = 1
    compression      = 'none'
    jpeg_quality     = -1
    output_file      = None
    seam_save        = None
    seam_load        = None
    xor_save         = None
    no_output        = False
    exposure_correct    = False
    saturation_correct  = False
    content_seam        = False

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in ('-h', '--help', '/?'):
            _help()
        elif arg == '-d':
            i += 1; depth = int(argv[i])
            if depth not in (8, 16):
                sys.exit("invalid output depth (must be 8 or 16)")
        elif arg == '-l':
            i += 1; val = int(argv[i])
            if val >= 0: max_levels = max(1, val)
            else:        sub_levels = -val
        elif arg == '--nocrop':      crop        = False
        elif arg == '--bgr':         bgr         = True
        elif arg == '--wideblend':   wideblend   = True
        elif arg == '--nomask':      nomask      = True
        elif arg == '--bigtiff':     bigtiff     = True
        elif arg == '--reverse':     reverse     = True
        elif arg in ('--simple-seam', '--simpleseam'): simple_seam = True
        elif arg == '--timing':      timing           = True
        elif arg == '--exposure':      exposure_correct    = True
        elif arg == '--saturation':    saturation_correct  = True
        elif arg == '--content-seam':  content_seam     = True
        elif arg == '--no-output':   no_output   = True
        elif arg in ('-v', '--verbose'): verbosity += 1
        elif arg in ('-q', '--quiet'):   verbosity -= 1
        elif arg in ('-w', '-a', '--no-ciecam'):
            print(f"  ignoring enblend option {arg}")
        elif arg.startswith('-f') or arg.startswith('--primary-seam-generator'):
            print(f"  ignoring enblend option {arg}")
        elif arg in ('--save-seams', '--saveseams'):
            i += 1; seam_save = argv[i]
        elif arg in ('--load-seams', '--loadseams'):
            i += 1; seam_load = argv[i]
        elif arg in ('--save-xor', '--savexor'):
            i += 1; xor_save = argv[i]
        elif arg.startswith('--compression'):
            comp_str = arg.split('=', 1)[1] if '=' in arg else (i := i + 1) or argv[i]
            if comp_str.lstrip('-').isdigit():
                jpeg_quality = int(comp_str)
            elif comp_str.lower() in ('none', 'lzw', 'packbits'):
                compression = comp_str.lower()
            else:
                sys.exit(f"unknown compression: {comp_str}")
        elif arg in ('-o', '--output'):
            i += 1; output_file = argv[i]
            ext = output_file.rsplit('.', 1)[-1].lower()
            if ext in ('jpg', 'jpeg'):
                if jpeg_quality < 0: jpeg_quality = 75
            elif ext in ('tif', 'tiff'):
                pass
            else:
                sys.exit(f"unknown file extension: .{ext}")
            i += 1; break
        elif arg == '--':
            i += 1; break
        else:
            sys.exit(f"unknown argument \"{arg}\"")

        i += 1

    if output_file is None and seam_save is None:
        sys.exit("no output file specified (-o)")

    input_files = argv[i:]
    if not input_files:
        sys.exit("no input files specified")
    if len(input_files) > 255:
        sys.exit("too many (>255) input images")

    try:
        go(
            input_files=input_files, output_file=output_file,
            max_levels=max_levels, sub_levels=sub_levels, wideblend=wideblend,
            workbpp_cmd=depth, crop=crop, bgr=bgr, reverse=reverse,
            simple_seam=simple_seam, content_seam=content_seam, nomask=nomask,
            compression=compression, jpeg_quality=jpeg_quality, bigtiff=bigtiff,
            seam_save_filename=seam_save, seam_load_filename=seam_load,
            xor_filename=xor_save, timing=timing, no_output=no_output,
            exposure_correct=exposure_correct,
            saturation_correct=saturation_correct, verbosity=verbosity,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.exit(str(exc))


if __name__ == '__main__':
    main()

