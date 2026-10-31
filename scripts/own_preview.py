"""Own-hand branch: map candidates before collision, save resolved plans/replay."""
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup

import grasp_pipeline as gp

sys.path.insert(0, str(gp.ROOT / "own_hand"))
import own_hand_grasp as ohg
from collision import HandCollisionDetector

# Preserve ws's existing hand_bridge interpretation explicitly. These are
# transfer proposals, not learned scores for our hand; Type 2 means Ring here.
OWN_TYPE_MAP = {i: i for i in range(1, 9)}


def run(args, ggarray, scores, types, predicted_depths, features,
        depths, colors, frame_info, observation, started_at):
    with open(args.calibration) as f:
        calibration = json.load(f)
    with open(args.intrinsics) as f:
        intrinsics = json.load(f)
    flange = (observation["flange_pose"] if observation else gp.gc.OBSERVE_RPYXYZ)
    run_dir = Path(args.output).resolve().parent / "runs" / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    scene = gp.depth_cloud(depths, intrinsics)
    bundle = run_dir / "capture.npz"
    np.savez_compressed(str(bundle), depth=depths, color=colors, scene_points=scene,
                        ggarray=ggarray, scores=scores, network_types=types,
                        predicted_depths=predicted_depths, grasp_features=features)
    detector = HandCollisionDetector(scene, margin=args.collision_margin,
                                     step=gp.PATH_STEP) if args.collision else None
    accepted, decisions, tested = [], [], []
    two_fingers = GraspGroup(ggarray)
    order = np.argsort(-np.asarray(scores), kind="stable")[:args.candidate_limit]
    for index in order:
        g = two_fingers[int(index)]
        network_type = int(types[index])
        own_type = OWN_TYPE_MAP[network_type]
        translation, rotation = ohg.base_pose_from_grasp(g, own_type)
        candidate = {
            "candidate_id": int(index), "score": float(scores[index]),
            "score_source": "inspire_transfer", "score_calibrated_for_own_hand": False,
            "network_grasp_type": network_type, "grasp_type": own_type,
            "hand_model": "own", "pose_frame": "camera", "pose_link": "base_link",
            "translation": translation.tolist(), "rotation_matrix": rotation.tolist(),
            "depth": float(g.depth), "depth_consumed_in_pose": True,
            "predicted_inspire_depth_unused": float(predicted_depths[index]),
            "width": float(g.width), "two_finger_score": float(g.score),
            "two_finger_translation": np.asarray(g.translation).tolist(),
            "two_finger_rotation_matrix": np.asarray(g.rotation_matrix).tolist(),
            "two_finger_width": float(g.width), "two_finger_depth": float(g.depth),
            "preshape_motors": list(ohg.motor_positions(own_type).values()),
            "closed_motors": list(ohg.motor_positions(own_type, "closed").values()),
        }
        plan = gp.build_plan(candidate, calibration, flange)
        collision = (detector.check(own_type, plan) if detector else
                     {"passed": False, "model": "unchecked", "scope": "none"})
        candidate["plan"], candidate["collision"] = plan, collision
        candidate["collision_checked"] = detector is not None
        tested.append(candidate)
        decisions.append({"candidate_id": int(index), "grasp_type": own_type,
                          "reason": ("collision" if not collision["passed"] else "accepted"),
                          "collision": collision})
        if detector is None or collision["passed"]:
            accepted.append(candidate)
        if len(accepted) >= args.topk:
            break
    result = dict(accepted[0]) if accepted else {}
    result.update({
        "schema_version": 2, "hand_model": "own", "run_id": run_dir.name,
        "status": "candidates_ready" if accepted else "no_collision_free_candidate",
        "created_at": time.time(), **frame_info,
        "observation_id": observation["id"] if observation else None,
        "capture_flange_pose": list(flange),
        "capture_pose_source": "settled_observation" if observation else "offline_assumed",
        "source_hashes": gp.source_hashes(args.calibration, args.intrinsics),
        "calibration_path": str(Path(args.calibration).resolve()),
        "intrinsics_path": str(Path(args.intrinsics).resolve()),
        "calibration": calibration,
        "capture_bundle": str(bundle), "capture_bundle_sha256": gp.sha256(bundle),
        "num_output_grasps": len(accepted), "candidates": accepted,
        "decisions": decisions, "tested_candidates": tested,
        "elapsed_sec": time.time() - started_at,
        "collision_checked": detector is not None,
    })
    # A preview started in one round cannot publish into a newer round.
    if observation:
        current = gp.load_observation(args.observation)
        if current is None or current["id"] != observation["id"]:
            raise RuntimeError("推理期间观察轮次已改变，请重新推理")
    gp.write_json(run_dir / "result.json", result)
    gp.write_json(args.output, result)
    print("[own] 检查 %d 个候选，保留 %d 个；回放：%s" %
          (len(decisions), len(accepted), run_dir / "result.json"))
    if not accepted:
        return 6
    chosen = accepted[0]
    print("[own] 预览 candidate_id=%d，type=%d，score=%.3f（Inspire迁移评分）" %
          (chosen["candidate_id"], chosen["grasp_type"], chosen["score"]))
    if args.visualize or args.save_debug_scene:
        T = np.asarray(chosen["plan"]["T_camera_hand"])
        mesh = ohg.hand_mesh_at_pose(T[:3, 3], T[:3, :3], chosen["grasp_type"],
                                     color=[1., .1, .1])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.08).transform(T)
        T_base_camera = gp.pose_transform(flange) @ gp.transform(
            calibration["R_cam2flange"], calibration["t_cam2flange"])
        T_camera_flange = np.linalg.inv(T_base_camera) @ gp.pose_transform(chosen["plan"]["final_pose"])
        flange_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1).transform(T_camera_flange)
        if args.save_debug_scene:
            o3d.io.write_point_cloud(str(run_dir / "scene.ply"), cloud)
            o3d.io.write_triangle_mesh(str(run_dir / "preshape.ply"), mesh)
        if args.visualize:
            o3d.visualization.draw_geometries([cloud.voxel_down_sample(.003), mesh, frame, flange_frame])
    return 0
