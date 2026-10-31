"""Shared, hardware-free grasp plans and versioned observation artifacts.

Transforms T_A_B map coordinates from B to A. Lengths are metres, RPY radians.
The demo executes the saved plan; it never silently changes its orientation.
"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import grasp_common as gc

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_PATH = ROOT / "logs/preview/observation.json"
APPROACH_Z = 0.12
LIFT_Z = 0.08
PATH_STEP = 0.003
MAX_RESULT_AGE = 120.0


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=str(path.parent), delete=False) as f:
            tmp = f.name
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, str(path))
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(calibration, intrinsics):
    paths = [Path(gc.default_config_path()), Path(calibration), Path(intrinsics),
             ROOT / "scripts/hand_bridge.py", ROOT / "hand_control/controller_ub.py",
             ROOT / "scripts/tools/compute_hand_geometry.py",
             ROOT / "scripts/grasp_common.py", Path(__file__),
             ROOT / "scripts/own_preview.py", ROOT / "scripts/grasp_decoding.py",
             ROOT / "scripts/robot_inspire.py", ROOT / "scripts/camera_snapshot.py",
             ROOT / "scripts/realsense.py", ROOT / "scripts/jog_manual.py",
             ROOT / "scripts/anydex_eih_grasp.py", ROOT / "scripts/MoveZ1.py",
             ROOT / "own_hand/own_hand_grasp.py", ROOT / "own_hand/collision.py"]
    paths += sorted((ROOT / "assets/own_urdf/dexterous_hand").rglob("*.urdf"))
    paths += sorted((ROOT / "assets/own_urdf/dexterous_hand/meshes").glob("*.STL"))
    return {str(p.resolve()): sha256(p) for p in paths}


def validate_rotation(value):
    R = np.asarray(value, dtype=float).reshape(3, 3)
    if (not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        raise ValueError("rotation_matrix must be a finite proper rotation")
    return R


def transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = validate_rotation(rotation)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    if not np.isfinite(T).all():
        raise ValueError("non-finite translation")
    return T


def pose_transform(pose):
    pose = np.asarray(pose, dtype=float).reshape(6)
    return transform(Rotation.from_euler("xyz", pose[:3]).as_matrix(), pose[3:])


def same_pose(a, b, position_tolerance=0.005, angle_tolerance=0.035):
    A, B = pose_transform(a), pose_transform(b)
    return bool(np.linalg.norm(A[:3, 3] - B[:3, 3]) <= position_tolerance and
                Rotation.from_matrix(A[:3, :3] @ B[:3, :3].T).magnitude() <= angle_tolerance)


def load_observation(path=OBSERVATION_PATH):
    if not Path(path).exists():
        return None
    with open(str(path)) as f:
        observation = json.load(f)
    if observation.get("state") != "ready":
        return None
    pose_transform(observation["flange_pose"])
    return observation


def depth_cloud(depth, intrinsics, max_depth=1.5):
    """Collision context: retain table, do not use network's table-removed cloud.

    Uses the same aligned pinhole projection as robot_inspire.get_grasp.
    Near-depth invalid pixels remain unknown, not certified empty space.
    """
    y, x = np.indices(depth.shape)
    z = np.asarray(depth, dtype=float) / 1000.0
    points = np.stack(((x - intrinsics["ppx"]) * z / intrinsics["fx"],
                       (y - intrinsics["ppy"]) * z / intrinsics["fy"], z), axis=-1)
    return points[np.isfinite(points).all(axis=-1) & (z > 0) & (z < max_depth)]


def linear_poses(start, end, step=PATH_STEP):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if step <= 0 or not np.allclose(start[:3], end[:3], atol=1e-8):
        raise ValueError("linear path requires positive step and constant orientation")
    n = max(1, int(np.ceil(np.linalg.norm(end[3:] - start[3:]) / step)))
    return np.linspace(start, end, n + 1)


def build_plan(candidate, calibration, flange_pose, approach_z=APPROACH_Z):
    """Resolve offsets once, and recover actual hand pose from commanded flange.

    Own-hand mount is physical flange<-base. Legacy Inspire compensation keeps
    its historical sign. Explicit own 4x4 mounts are supported by grasp_common.
    """
    cfg = gc.load_hand_config(model=candidate["hand_model"])
    target = gc.compute_grasp_target(
        candidate, np.asarray(calibration["R_cam2flange"]),
        np.asarray(calibration["t_cam2flange"]), flange_pose,
        cfg["hand_mount_z"], cfg["mount_yaw_deg"], cfg["orient_deviation_thresh_deg"],
        hand_mount_transform=gc.resolve_hand_mount(cfg))
    final = np.r_[target["rpy"], target["p_flange"]]
    pre = final + np.array([0, 0, 0, 0, 0, approach_z])
    lift = final + np.array([0, 0, 0, 0, 0, LIFT_Z])
    T_base_camera = pose_transform(flange_pose) @ transform(
        calibration["R_cam2flange"], calibration["t_cam2flange"])
    mount = gc.resolve_hand_mount(cfg)
    T_camera_hand = np.linalg.inv(T_base_camera) @ pose_transform(final) @ mount
    # Preserve every offset/fallback in the collision-checked geometry.
    direction_camera = T_base_camera[:3, :3].T @ np.array([0., 0., 1.])
    return {
        "final_pose": final.tolist(), "pregrasp_pose": pre.tolist(),
        "lift_pose": lift.tolist(),
        "descent_poses": linear_poses(pre, final).tolist(),
        "lift_poses": linear_poses(final, lift).tolist(),
        "T_camera_hand": T_camera_hand.tolist(),
        "pregrasp_direction_camera": direction_camera.tolist(),
        "approach_distance": float(approach_z),
        "used_network_orientation": bool(target["used_network"]),
        "orientation_deviation_deg": target["deviation_deg"],
        "collision_scope": "preshape_descent_only; arm/transit/closure not checked",
    }


def validate_result(result, observation, current_pose, now=None):
    now = time.time() if now is None else now
    if result.get("schema_version") != 2 or result.get("hand_model") != "own":
        raise ValueError("请重新运行 own 模式预览，生成新版抓取计划")
    if not observation or result.get("observation_id") != observation.get("id"):
        raise ValueError("抓取结果不属于当前观察轮次，请重新推理")
    captured = float(result["captured_at"])
    if (not np.isfinite(captured) or captured < observation["ready_at"] or
            not 0 <= now - captured <= MAX_RESULT_AGE):
        raise ValueError("抓取图像已过期或早于当前观察位，请重新推理")
    if not same_pose(result["capture_flange_pose"], observation["flange_pose"]):
        raise ValueError("采集位姿与当前观察记录不一致")
    if not same_pose(current_pose, result["capture_flange_pose"]):
        raise ValueError("机械臂已离开采集位姿，请返回观察位并重新推理")
    if not result.get("source_hashes"):
        raise ValueError("缺少模型/配置版本记录")
    for path, expected in result["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("配置或手模型已改变，请重新推理: " + path)
    if sha256(result["capture_bundle"]) != result["capture_bundle_sha256"]:
        raise ValueError("采集文件与抓取计划不一致，请重新推理")
    if not result.get("candidates"):
        raise ValueError("没有通过自研手碰撞检查的候选")


def check_plan_ik(plan, check_ik, bounds, max_joint_step=0.35):
    """Read-only IK; local seed continuity and limits are checked by the SDK.

    MoveL may still fail at runtime; sampled feasibility isn't path execution.
    The first solve is global, subsequent solves use the preceding solution.
    """
    poses = plan["descent_poses"] + plan["lift_poses"][1:]
    seed = None
    for index, pose in enumerate(poses):
        pose = np.asarray(pose, dtype=float).reshape(6)
        if not np.isfinite(pose).all() or any(
                not low <= v <= high for v, (low, high) in zip(pose[3:], bounds)):
            return False, "workspace:%d" % index
        ok, q = check_ik(pose, seed)
        if not ok:
            return False, "ik:%d" % index
        q = np.asarray(q, dtype=float).reshape(6)
        if not np.isfinite(q).all():
            return False, "invalid_ik:%d" % index
        if seed is not None and np.max(np.abs(q - seed)) > max_joint_step:
            return False, "joint_jump:%d" % index
        seed = q
    return True, "sampled_ik_passed"
