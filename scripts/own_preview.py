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
    print("[own] 保存 RGB-D 快照与候选，供离线排查...", flush=True)
    np.savez_compressed(str(bundle), depth=depths, color=colors, scene_points=scene,
                        ggarray=ggarray, scores=scores, network_types=types,
                        predicted_depths=predicted_depths, grasp_features=features)
    detector = HandCollisionDetector(scene, margin=args.collision_margin,
                                     step=gp.PATH_STEP) if args.collision else None
    print("[own] 碰撞场景 %d 点（保留桌面）；正在初始化自研手几何..." % len(scene),
          flush=True)
    accepted, decisions, tested = [], [], []
    two_fingers = GraspGroup(ggarray)
    order = np.argsort(-np.asarray(scores), kind="stable")[:args.candidate_limit]
    for rank, index in enumerate(order, 1):
        candidate_started = time.monotonic()
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
        print("[own] 候选 %d/%d，id=%d，type=%d：检查中..." %
              (rank, len(order), index, own_type), flush=True)
        collision = (detector.check(own_type, plan,
                                   full_count=getattr(args, "collision_full_count", False)) if detector else
                     {"passed": False, "model": "unchecked", "scope": "none"})
        if collision.get("first_collision"):
            contact = collision["first_collision"]
            contact["link_name"] = ohg.ALL_LINKS[contact["link_index"]]
        elapsed = time.monotonic() - candidate_started
        candidate["check_elapsed_sec"] = elapsed
        count_text = (str(collision.get("collision_points", 0)) if
                      collision.get("collision_points_exact", True) else
                      ">=%d" % collision["collision_points"])
        verdict = ("未检查（仅预览）" if detector is None else
                   "通过" if collision["passed"] else "拒绝，碰撞点 " + count_text)
        print("[own]   %s；%.2fs%s" %
              (verdict, elapsed, "；部件=" + collision["first_collision"]["link_name"]
               if collision.get("first_collision") else ""), flush=True)
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
    diagnostic_only = not accepted
    if diagnostic_only:
        print("[own] 没有通过碰撞检查的候选；排名第一的拒绝候选仅供诊断，禁止执行。", flush=True)
    chosen = accepted[0] if accepted else tested[0]
    print("[own] 预览 candidate_id=%d，type=%d，score=%.3f（Inspire迁移评分）" %
          (chosen["candidate_id"], chosen["grasp_type"], chosen["score"]))
    print("[own] 执行朝向=%s，偏差=%.1f°，最终法兰位置=%s" %
          ("网络朝向" if chosen["plan"]["used_network_orientation"] else "固定朝向回退",
           chosen["plan"]["orientation_deviation_deg"],
           np.round(chosen["plan"]["final_pose"][3:], 4).tolist()), flush=True)
    if args.visualize or args.save_debug_scene:
        T = np.asarray(chosen["plan"]["T_camera_hand"]).copy()
        contact = chosen["collision"].get("first_collision") if diagnostic_only else None
        if contact:
            T[:3, 3] += (np.asarray(chosen["plan"]["pregrasp_direction_camera"]) *
                         contact["sweep_offset_m"])
            print("[own] 红色手模型=检测到碰撞时的位姿；黄色点=碰撞点采样；"
                  "部件=%s，距最终位置 %.3fm" %
                  (contact["link_name"], contact["sweep_offset_m"]), flush=True)
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
            geometries = [cloud.voxel_down_sample(.003), mesh, frame, flange_frame]
            if contact:
                for point in contact["points_camera"][:30]:
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=.002)
                    marker.translate(point).paint_uniform_color([1., 1., 0.])
                    geometries.append(marker)
            title = ("Own hand: COLLISION - DIAGNOSTIC ONLY" if diagnostic_only else
                     "Own hand: candidate preview")
            print("[own] 打开 Open3D 窗口：" + title, flush=True)
            o3d.visualization.draw_geometries(geometries, window_name=title)
    return 6 if diagnostic_only else 0
