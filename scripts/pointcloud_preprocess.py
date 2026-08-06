#!/usr/bin/env python3
"""RealSense point cloud preprocessing for AnyDexGrasp eye-in-hand pipeline.

Applies workspace crop, RANSAC table-plane removal, and statistical outlier
removal to a camera-frame point cloud before it enters the CGR grasp network.

All functions operate on (N,3) float32 numpy arrays in camera coordinates.
Each step returns (points_clean, keep_mask) where keep_mask is a bool array
of the same length as the input, with True for kept points.

Frame cache
-----------
The real-robot augmentation loop (get_ggarray_features) calls get_grasp() 11x
per frame with different augment_mat. Preprocessing is augment-independent, so
we cache the result keyed by a hash of the depth frame content.  This is
transparent to callers — call preprocess_point_cloud_cached() inside get_grasp.
"""

import hashlib
import time
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def crop_workspace_xy(points, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)):
    # type: (np.ndarray, Tuple[float,float], Tuple[float,float]) -> Tuple[np.ndarray, np.ndarray]
    """Remove points outside the camera-frame X/Y workspace.

    Returns (points_cropped, keep_mask).
    """
    mask_x = (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1])
    mask_y = (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
    keep = mask_x & mask_y
    return points[keep], keep


def remove_table_plane(
    points,
    plane_dist_thresh=0.008,
    ransac_max_iter=100,
    min_inliers_ratio=0.15,
    table_clearance=0.01,
    sample_size=50000,
):
    # type: (np.ndarray, float, int, float, float, int) -> Tuple[np.ndarray, np.ndarray]
    """RANSAC dominant-plane fitting, remove the plane and everything below it.

    The plane normal is oriented toward the camera origin (0,0,0 in camera
    frame) so that points *above* the table have positive signed distance.

    When no dominant plane is found (inlier ratio < *min_inliers_ratio*) the
    function returns all points unchanged and prints a warning.

    Parameters
    ----------
    points : (N,3) float32
    plane_dist_thresh : float
        RANSAC inlier distance threshold in metres.
    table_clearance : float
        Minimum signed distance above the plane to keep.  Points exactly on
        the table surface and below are discarded.
    min_inliers_ratio : float
        Fraction of subsampled points that must be inliers for the plane to
        be accepted as the table.

    Returns (points_above_table, keep_mask).
    """
    import open3d as o3d

    n = points.shape[0]
    if n < 3:
        return points, np.ones(n, dtype=bool)

    # Subsample for speed — a dominant plane is still dominant in a 50k subset.
    if n > sample_size:
        rng = np.random.default_rng()
        sample_idx = rng.choice(n, sample_size, replace=False)
    else:
        sample_idx = np.arange(n)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[sample_idx])

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=plane_dist_thresh,
        ransac_n=3,
        num_iterations=ransac_max_iter,
    )
    a, b, c, d = plane_model  # a*x + b*y + c*z + d = 0

    # Normalize (Open3D may not guarantee unit normal for all versions)
    norm = np.sqrt(a * a + b * b + c * c)
    if norm < 1e-12:
        print("[preprocess] ⚠ RANSAC returned degenerate plane, skipping table removal")
        return points, np.ones(n, dtype=bool)
    a, b, c, d = a / norm, b / norm, c / norm, d / norm

    inlier_ratio = len(inliers) / len(sample_idx)
    if inlier_ratio < min_inliers_ratio:
        print(
            f"[preprocess] ⚠ No dominant plane found "
            f"(inlier_ratio={inlier_ratio:.3f} < {min_inliers_ratio}), "
            f"keeping all points"
        )
        return points, np.ones(n, dtype=bool)

    # Orient normal toward the camera (origin in camera frame).
    # signed_distance(origin) = d  (since a*0 + b*0 + c*0 + d = d).
    if d < 0:
        a, b, c, d = -a, -b, -c, -d

    # Signed distances for ALL points (not just the subsample).
    sd = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    keep = sd > table_clearance

    n_removed = n - keep.sum()
    print(
        f"[preprocess] Table removed: {n_removed}/{n} points "
        f"(inlier_ratio={inlier_ratio:.3f}, clearance={table_clearance:.3f}m)"
    )
    return points[keep], keep


def remove_statistical_outliers(
    points,
    nb_neighbors=20,
    std_ratio=2.0,
    max_points=200000,
):
    # type: (np.ndarray, int, float, int) -> Tuple[np.ndarray, np.ndarray]
    """Open3D statistical outlier removal.

    When *points* exceeds *max_points* it is randomly subsampled first —
    acceptable because the subsequent 5 mm voxelisation already merges
    nearby points.

    Returns (points_clean, keep_mask_local) where *keep_mask_local* is
    relative to the (possibly subsampled) input.
    """
    import open3d as o3d

    n = points.shape[0]
    if n < nb_neighbors:
        return points, np.ones(n, dtype=bool)

    if n > max_points:
        rng = np.random.default_rng()
        idx = rng.choice(n, max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    clean_cloud, inlier_ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    keep_local = np.zeros(len(pts), dtype=bool)
    keep_local[np.array(inlier_ind)] = True

    n_removed = len(pts) - len(inlier_ind)
    if n_removed > 0:
        print(
            f"[preprocess] SOR removed {n_removed}/{len(pts)} outliers "
            f"(nb={nb_neighbors}, std={std_ratio})"
        )

    if n > max_points:
        # Map local keep back to original indexing.
        keep = np.zeros(n, dtype=bool)
        keep[idx[keep_local]] = True
        return points[keep], keep
    else:
        return points[keep_local], keep_local


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preprocess_point_cloud(
    points,
    cfg,  # type: Dict
):
    # type: (np.ndarray, Dict) -> Tuple[np.ndarray, np.ndarray]
    """Run the full preprocessing pipeline on camera-frame points.

    Parameters
    ----------
    points : (N,3) float32
        Already Z-depth-masked points in camera frame.
    cfg : dict
        Keys:
        - crop_xy : bool (default False)
        - cloud_ws_x_range, cloud_ws_y_range : tuple
        - remove_table : bool (default False)
        - plane_dist_thresh, table_clearance, plane_min_inliers_ratio,
          ransac_max_iter, plane_sample
        - denoise : bool (default False)
        - sor_nb_neighbors, sor_std_ratio, sor_max_points

    Returns
    -------
    (points_clean, keep_mask)
        keep_mask is bool (N,) — used to filter DEBUG colors identically.
    """
    N = points.shape[0]
    keep = np.ones(N, dtype=bool)

    # 1. Workspace X/Y crop
    if cfg.get("crop_xy", False):
        t0 = time.time()
        points, crop_mask = crop_workspace_xy(
            points,
            x_range=cfg.get("cloud_ws_x_range", (-0.5, 0.5)),
            y_range=cfg.get("cloud_ws_y_range", (-0.5, 0.5)),
        )
        # Map back to original indexing
        orig_idx = np.where(keep)[0]
        keep[orig_idx[~crop_mask]] = False
        print(f"[preprocess] crop_xy: {points.shape[0]} points ({time.time()-t0:.3f}s)")

    # 2. RANSAC table-plane removal
    if cfg.get("remove_table", False):
        t0 = time.time()
        points, table_mask = remove_table_plane(
            points,
            plane_dist_thresh=cfg.get("plane_dist_thresh", 0.008),
            ransac_max_iter=cfg.get("ransac_max_iter", 100),
            min_inliers_ratio=cfg.get("plane_min_inliers_ratio", 0.15),
            table_clearance=cfg.get("table_clearance", 0.01),
            sample_size=cfg.get("plane_sample", 50000),
        )
        orig_idx = np.where(keep)[0]
        keep[orig_idx[~table_mask]] = False
        print(f"[preprocess] remove_table: {points.shape[0]} points ({time.time()-t0:.3f}s)")

    # 3. Statistical outlier removal
    if cfg.get("denoise", False):
        t0 = time.time()
        points, sor_mask = remove_statistical_outliers(
            points,
            nb_neighbors=cfg.get("sor_nb_neighbors", 20),
            std_ratio=cfg.get("sor_std_ratio", 2.0),
            max_points=cfg.get("sor_max_points", 200000),
        )
        orig_idx = np.where(keep)[0]
        keep[orig_idx[~sor_mask]] = False
        print(f"[preprocess] denoise: {points.shape[0]} points ({time.time()-t0:.3f}s)")

    return points, keep


# ---------------------------------------------------------------------------
# Frame cache for the augmentation loop
# ---------------------------------------------------------------------------

_frame_cache = {"key": None, "points": None, "keep_mask": None}


def preprocess_point_cloud_cached(points, cfg, depth_frame):
    # type: (np.ndarray, Dict, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    """Cached wrapper around preprocess_point_cloud.

    The augmentation loop (get_ggarray_features) passes the *same* depth
    array to 11 successive get_grasp calls.  Caching on depth content
    avoids re-running expensive preprocessing 10 extra times per frame.

    *depth_frame* must be the raw uint16 depth image (720,1280) used to
    derive *points* — its byte content forms the cache key.
    """
    # Build a cache key from depth content + preprocessing parameters.
    param_key = (
        cfg.get("crop_xy", False),
        cfg.get("cloud_ws_x_range", (-0.5, 0.5)),
        cfg.get("cloud_ws_y_range", (-0.5, 0.5)),
        cfg.get("remove_table", False),
        cfg.get("plane_dist_thresh", 0.008),
        cfg.get("table_clearance", 0.01),
        cfg.get("plane_min_inliers_ratio", 0.15),
        cfg.get("ransac_max_iter", 100),
        cfg.get("plane_sample", 50000),
        cfg.get("denoise", False),
        cfg.get("sor_nb_neighbors", 20),
        cfg.get("sor_std_ratio", 2.0),
        cfg.get("sor_max_points", 200000),
    )
    depth_hash = hashlib.md5(depth_frame.tobytes()).hexdigest()
    key = (depth_hash, param_key)

    if _frame_cache["key"] == key:
        print("[preprocess] cache hit — reusing preprocessed cloud")
        return _frame_cache["points"], _frame_cache["keep_mask"]

    print("[preprocess] cache miss — running preprocessing")
    pts_clean, keep = preprocess_point_cloud(points, cfg)
    _frame_cache["key"] = key
    _frame_cache["points"] = pts_clean
    _frame_cache["keep_mask"] = keep
    return pts_clean, keep
