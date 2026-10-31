#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from multiprocessing import resource_tracker, shared_memory

import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation

import grasp_common as gc
import grasp_pipeline as gp
from camera_snapshot import read_snapshot, META_NAME


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one AnyDexGrasp Inspire inference pass (camera only, no robot control)."
    )
    parser.add_argument("--checkpoint_path", default="../logs/model/checkpoint.tar.18")
    parser.add_argument("--inspire_model_path", default="../logs/model/inspire_model/obj140")
    parser.add_argument("--inspire_mesh_json_path", default="../assets/inspire_urdf")
    parser.add_argument("--use_graspnet_v2", action="store_true", default=True)
    parser.add_argument("--half_views", action="store_true")
    parser.add_argument("--score_thresh", type=float, default=0.85)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50,
                        help="最多检查的自研手候选数")
    parser.add_argument("--collision-margin", type=float, default=0.003)
    parser.add_argument("--observation", default=str(gp.OBSERVATION_PATH))
    parser.add_argument("--voxel_size", type=float, default=0.005)

    parser.add_argument("--min_depth", type=float, default=0.2)
    parser.add_argument("--max_depth", type=float, default=0.6)
    
    parser.add_argument("--collision", action="store_true", default=True)
    parser.add_argument("--no-collision", dest="collision", action="store_false",
                        help="仅供预览；未检查的 own 结果不能自动执行")
    parser.add_argument("--random_grasp", action="store_true")
    parser.add_argument("--hand-model", dest="hand_model", choices=["inspire", "own"],
                        default=None,
                        help="抓取锚定几何: own=自研手URDF重锚定(推荐), inspire=查表旧行为; "
                             "缺省读 config/grasp_config.json 的 hand_model")
    parser.add_argument("--fixed-type", dest="fixed_type", type=int, default=None,
                        choices=range(1, 9),
                        help="只保留该类型对应的模型预测（不改写其他类型的评分）")
    parser.add_argument("--output", default="../logs/preview/latest_inspire_grasp.json")
    parser.add_argument("--intrinsics", default="../config/realsense_intrinsics.json")
    parser.add_argument("--calibration", default="../calibration/cam2flange_current.json",
                        help="眼在手上标定 JSON (cam2flange)，用于绘制法兰目标")
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.add_argument("--save_debug_scene", action="store_true")
    # ── Preprocessing ────────────────────────────────────────
    parser.add_argument("--preprocess", action="store_true",
                        help="Enable cloud preprocessing: workspace crop + RANSAC table removal + SOR denoise")
    parser.add_argument("--no_table_removal", action="store_true",
                        help="Skip RANSAC table removal (when --preprocess is on, table removal is on by default)")
    parser.add_argument("--no_denoise", action="store_true",
                        help="Skip SOR denoising (when --preprocess is on, denoising is on by default)")
    parser.add_argument("--cloud_ws_x_range", nargs=2, type=float, default=[-0.5, 0.5],
                        help="Camera-frame X bounds for input cloud crop (m)")
    parser.add_argument("--cloud_ws_y_range", nargs=2, type=float, default=[-0.5, 0.5],
                        help="Camera-frame Y bounds for input cloud crop (m)")
    # ── Grasp candidate workspace ────────────────────────────
    parser.add_argument("--grasp_ws_x_range", nargs=2, type=float, default=[-0.40, 0.40],
                        help="Camera-frame X bounds for grasp candidate filtering (m)")
    parser.add_argument("--grasp_ws_y_range", nargs=2, type=float, default=[-0.40, 0.40],
                        help="Camera-frame Y bounds for grasp candidate filtering (m)")
    # ── RANSAC ───────────────────────────────────────────────
    parser.add_argument("--plane_dist_thresh", type=float, default=0.008,
                        help="RANSAC plane inlier distance threshold (m)")
    parser.add_argument("--table_clearance", type=float, default=0.01,
                        help="Minimum distance above table to keep points (m)")
    parser.add_argument("--plane_min_inliers_ratio", type=float, default=0.15,
                        help="Minimum inlier ratio to accept dominant plane")
    # ── SOR ──────────────────────────────────────────────────
    parser.add_argument("--sor_nb_neighbors", type=int, default=20)
    parser.add_argument("--sor_std_ratio", type=float, default=2.0)
    return parser.parse_args()


def import_robot_inspire(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ dir for robot_inspire
    sys.argv = [
        "robot_inspire.py",
        "--checkpoint_path",
        args.checkpoint_path,
        "--inspire_model_path",
        args.inspire_model_path,
        "--inspire_mesh_json_path",
        args.inspire_mesh_json_path,
        "--robot_ip",
        "0.0.0.0",
    ]
    if args.use_graspnet_v2:
        sys.argv.append("--use_graspnet_v2")
    if args.half_views:
        sys.argv.append("--half_views")

    import robot_inspire as ri

    ri.RANDOM_GRASP = args.random_grasp
    ri.DEBUG = args.visualize or args.save_debug_scene
    return ri


def attach_shared_memory(name):
    shm = shared_memory.SharedMemory(name=name)
    # We only attach to memory owned by realsense.py. Avoid unlinking it here.
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def load_intrinsics(path):
    if not path or not os.path.exists(path):
        print("Using built-in AnyDexGrasp camera intrinsics.")
        return None
    with open(path, "r") as f:
        intr = json.load(f)
    print(
        "Using RealSense intrinsics:",
        f"fx={intr['fx']:.3f}",
        f"fy={intr['fy']:.3f}",
        f"cx={intr['ppx']:.3f}",
        f"cy={intr['ppy']:.3f}",
    )
    return intr


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def ensure_cloud(cloud, points_down):
    if cloud is not None:
        return cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_down.detach().cpu().numpy())
    return cloud


def make_grasp_frame(grasp, size=0.08):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(grasp.rotation_matrix)
    transform[:3, 3] = np.asarray(grasp.translation)
    frame.transform(transform)
    return frame


class _PoseShim:
    """让 make_grasp_frame 接受裸 translation/rotation（own 重锚定后的位姿）。"""

    def __init__(self, translation, rotation_matrix):
        self.translation = np.asarray(translation, dtype=float)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=float)


def save_debug_scene(output_json, cloud, hand_mesh, two_finger_geom, frame):
    base_dir = os.path.dirname(output_json)
    os.makedirs(base_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(base_dir, "latest_scene_cloud.ply"), cloud)
    o3d.io.write_triangle_mesh(os.path.join(base_dir, "latest_inspire_mesh.ply"), hand_mesh)
    o3d.io.write_triangle_mesh(os.path.join(base_dir, "latest_grasp_frame.ply"), frame)
    if two_finger_geom is not None:
        if isinstance(two_finger_geom, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(os.path.join(base_dir, "latest_two_finger_grasp.ply"), two_finger_geom)
        elif isinstance(two_finger_geom, o3d.geometry.LineSet):
            o3d.io.write_line_set(os.path.join(base_dir, "latest_two_finger_grasp.ply"), two_finger_geom)


def main():
    args = parse_args()
    if args.topk < 1 or args.candidate_limit < args.topk or args.collision_margin < 0:
        raise ValueError("需满足 candidate-limit >= topk >= 1 且碰撞裕量非负")
    ri = import_robot_inspire(args)
    hand_model = args.hand_model or gc.load_hand_config().get("hand_model", "inspire")
    print(f"[手模型] {hand_model}"
          + ("  (自研手URDF锚定: JSON顶层位姿=自研手base)" if hand_model == "own"
             else "  (Inspire查表锚定: 旧行为)"))

    print("[1/5] loading networks...")
    intrinsics = load_intrinsics(args.intrinsics)
    net = ri.get_net(args.checkpoint_path, use_v2=args.use_graspnet_v2)
    inspire_models = {} if args.random_grasp else ri.get_inspire_model(args.inspire_model_path)
    meshes_pcls = ri.load_meshes_pointcloud(args.inspire_mesh_json_path) \
        if args.collision and hand_model == "inspire" else None

    print("[2/5] opening RealSense shared memory...")
    shm_depth = attach_shared_memory("realsense_depth")
    shm_color = attach_shared_memory("realsense_color")
    shm_meta = None

    try:
        print("[3/5] running base grasp network...")
        t0 = time.time()
        try:
            shm_meta = attach_shared_memory(META_NAME)
        except FileNotFoundError:
            raise RuntimeError("请重启更新后的 realsense.py，以提供完整 RGB-D 快照")
        observation = gp.load_observation(args.observation)
        depths, colors, frame_info = read_snapshot(
            np.ndarray((720, 1280), dtype=np.uint16, buffer=shm_depth.buf),
            np.ndarray((720, 1280, 3), dtype=np.float32, buffer=shm_color.buf),
            np.ndarray((2,), dtype=np.int64, buffer=shm_meta.buf))
        if observation and frame_info["captured_at"] < observation["ready_at"]:
            raise RuntimeError("相机图像早于观察位就绪时间，请稍后重新推理")
        valid = ((depths / 1000.0 > args.min_depth) & (depths / 1000.0 < args.max_depth)).sum()
        print(
            "depth stats:",
            "min", int(depths.min()),
            "max", int(depths.max()),
            f"valid_{args.min_depth}_{args.max_depth}",
            int(valid),
            flush=True,
        )
        ggarray, cloud, points_down, grasp_features, sinput = ri.get_grasp(
            net,
            depths,
            SimpleNamespace(buf=colors),
            voxel_size=args.voxel_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            intrinsics=intrinsics,
            ws_x_range=tuple(args.grasp_ws_x_range),
            ws_y_range=tuple(args.grasp_ws_y_range),
            preprocess=args.preprocess,
            cloud_ws_x_range=tuple(args.cloud_ws_x_range),
            cloud_ws_y_range=tuple(args.cloud_ws_y_range),
            remove_table=args.preprocess and not args.no_table_removal,
            denoise=args.preprocess and not args.no_denoise,
            plane_dist_thresh=args.plane_dist_thresh,
            table_clearance=args.table_clearance,
            plane_min_inliers_ratio=args.plane_min_inliers_ratio,
            sor_nb_neighbors=args.sor_nb_neighbors,
            sor_std_ratio=args.sor_std_ratio,
        )
        if ggarray is None:
            print("No grasp detected before multifinger selection.")
            return 2

        ggarray = ggarray.cpu().numpy()
        grasp_features = grasp_features.cpu().numpy()
        print(f"base candidates: {len(ggarray)}")

        print("[4/5] selecting Inspire grasp type/depth...")
        ggarray, if_flip = ri.flip_ggarray(ggarray)
        grasp_features = np.c_[grasp_features, if_flip]
        keep = ~np.array(if_flip)
        ggarray = ggarray[keep]
        grasp_features = grasp_features[keep]
        if len(ggarray) == 0:
            print("No grasp left after flip filtering.")
            return 3

        source_index = ggarray[:, 0].argsort()[::-1][:1000]
        ggarray = ggarray[source_index]
        grasp_features = grasp_features[source_index]
        grasp_features_dic = ri.get_graspgroup_features(grasp_features, sinput)
        inspire_depth, inspire_type, scores, ggarray, grasp_features = ri.get_inspire_depth_type(
            inspire_models,
            grasp_features_dic,
            ggarray,
            grasp_features=grasp_features,
            fixed_type=args.fixed_type,
        )

        if not args.random_grasp:
            mask = scores > args.score_thresh
            ggarray = ggarray[mask]
            grasp_features = grasp_features[mask]
            inspire_depth = inspire_depth[mask]
            inspire_type = inspire_type[mask]
            scores = scores[mask]
        if len(ggarray) == 0:
            print(f"No Inspire grasp above score_thresh={args.score_thresh}.")
            return 4

        if hand_model == "own":
            from own_preview import run
            return run(args, ggarray, scores, inspire_type, inspire_depth, grasp_features,
                       depths, colors, frame_info, observation, t0)

        two_fingers_ggarray = GraspGroup(ggarray)
        two_fingers_object_ids = ggarray[:, 16]
        inspire_ggarray = ri.InspireHandRGraspGroup()
        inspire_ggarray.set_grasp_min_width(ri.MIN_GRASP_WIDTH)
        inspire_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, args.inspire_mesh_json_path)
        inspire_ggarray.object_ids = two_fingers_object_ids
        inspire_ggarray.scores = scores
        inspire_ggarray.depths = inspire_ggarray.depths + inspire_depth + ri.INSPIREHANDR_DEFAULT_DEPTH

        z_axis_index = inspire_ggarray.filter_grasp_group_by_z_axis(
            0.4, ws_x_range=tuple(args.grasp_ws_x_range), ws_y_range=tuple(args.grasp_ws_y_range))
        two_fingers_ggarray = two_fingers_ggarray[z_axis_index]
        grasp_features = grasp_features[z_axis_index]
        if len(inspire_ggarray) == 0:
            print("No Inspire grasp left after z-axis filtering.")
            return 5

        if args.collision:
            print("[5/5] running collision filtering...")
            detector = ri.ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
            inspire_ggarray, two_fingers_ggarray, empty_mask, min_width_index = detector.detect(
                inspire_ggarray,
                two_fingers_ggarray,
                args.inspire_mesh_json_path,
                meshes_pcls,
                min_grasp_width=ri.MIN_GRASP_WIDTH,
                VoxelGrid=ri.INSPIREHANDR_VOXElGRID,
                DEBUG=False,
                approach_dist=0.06,
                collision_thresh=0,
                adjust_gripper_centers=True,
            )
            inspire_ggarray = inspire_ggarray[empty_mask]
            two_fingers_ggarray = two_fingers_ggarray[empty_mask]
            grasp_features = grasp_features[min_width_index][empty_mask]
            if len(inspire_ggarray) == 0:
                print("No Inspire grasp left after collision filtering.")
                return 6
        else:
            print("[5/5] skipping collision filtering. Use --collision for the fuller pipeline.")

        order = np.argsort(inspire_ggarray.scores)[::-1][: args.topk]
        inspire_ggarray = inspire_ggarray[order]
        two_fingers_ggarray = two_fingers_ggarray[order]

        chosen = inspire_ggarray[0]
        chosen_two_finger = two_fingers_ggarray[0]

        result = {
            "score": float(chosen.score),
            "translation": to_jsonable(chosen.translation),
            "rotation_matrix": to_jsonable(chosen.rotation_matrix),
            "depth": float(chosen.depth),
            "width": float(chosen.width),
            "angle": to_jsonable(chosen.angle),
            "grasp_type": int(chosen.grasp_type),
            "two_finger_score": float(chosen_two_finger.score),
            "two_finger_translation": to_jsonable(chosen_two_finger.translation),
            "two_finger_rotation_matrix": to_jsonable(chosen_two_finger.rotation_matrix),
            "two_finger_width": float(chosen_two_finger.width),
            "two_finger_depth": float(chosen_two_finger.depth),
            "hand_model": hand_model,
            "num_output_grasps": int(len(inspire_ggarray)),
            "elapsed_sec": time.time() - t0,
            "collision_checked": bool(args.collision),
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        print("AnyDexGrasp preview OK")
        print(json.dumps(result, indent=2))
        print(f"saved: {args.output}")

        if args.visualize or args.save_debug_scene:
            cloud = ensure_cloud(cloud, points_down).voxel_down_sample(max(args.voxel_size, 0.005))
            hand_mesh = chosen.load_mesh(args.inspire_mesh_json_path, chosen_two_finger)
            hand_mesh.paint_uniform_color([1.0, 0.1, 0.1])
            frame = make_grasp_frame(chosen)
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
            try:
                two_finger_geom = chosen_two_finger.to_open3d_geometry()
                two_finger_geom.paint_uniform_color([0.1, 0.4, 1.0])
            except Exception:
                two_finger_geom = None

            # ── 法兰目标可视化：预览里确认的姿态 = 机器人会执行的姿态 ──
            # 绿色框=执行器将采用网络朝向；橙色框=将回退固定朝向；蓝点=指尖位置
            flange_frame = None
            tip_marker = None
            try:
                with open(args.calibration) as f:
                    calib = json.load(f)
                hand_cfg = gc.load_hand_config(model=hand_model)
                R_c2f = np.array(calib["R_cam2flange"], dtype=float)
                t_c2f = np.array(calib["t_cam2flange"], dtype=float)
                R_palm = np.asarray(result["rotation_matrix"], dtype=float).reshape(3, 3)
                p_palm_cam = np.asarray(result["translation"], dtype=float)
                # 基座系手动偏移（config offset_xyz，纯水平 +X 微调）转到相机系，与执行器
                # compute_grasp_target 同效果：基座偏移 Δb 在相机系表现为 R_c2f.T @ R_f2b.T @ Δb。
                R_f2b = Rotation.from_euler("xyz", gc.OBSERVE_RPYXYZ[:3]).as_matrix()
                offset_cam = R_c2f.T @ R_f2b.T @ np.asarray(gc.load_offset_xyz(), dtype=float)
                p_palm_cam = p_palm_cam + offset_cam
                # 基座系旋转调试偏移（config offset_rpy_deg，度，绕掌心中心）共轭进相机系：
                # 执行器命令的法兰旋转 = R_off @ R_f2b @ R_c2f @ R_palm @ R_mount，相机系表达 =
                # (R_f2b@R_c2f).T @ R_off @ (R_f2b@R_c2f) @ R_palm @ R_mount。零偏移时取 I，
                # 保证旧渲染逐字节不变（避免 R_f2b.T@R_f2b 的浮点非正交噪声）。
                offset_rpy_deg = np.asarray(gc.load_offset_rpy_deg(), dtype=float)
                R_off = Rotation.from_euler("xyz", np.deg2rad(offset_rpy_deg)).as_matrix()
                R_off_cam = (np.eye(3) if np.allclose(offset_rpy_deg, 0)
                             else R_c2f.T @ R_f2b.T @ R_off @ R_f2b @ R_c2f)
                R_mount = gc.compute_R_mount(hand_cfg["mount_yaw_deg"])
                # 相机系里法兰帧 = palm@R_mount：X 沿指尖，Y/Z 差 90°（安装滚转）。
                # 不能再乘 R_c2f（相机↔法兰标定）：那会多转约 90°，橙帧方向与红 mesh 对不上。
                R_flange_cam = R_off_cam @ R_palm @ R_mount
                approach_cam = R_flange_cam[:, 0]      # 指尖方向 = 法兰 +X（相机系）
                p_flange_cam = p_palm_cam - hand_cfg["hand_mount_z"] * approach_cam
                p_tip_cam = p_palm_cam + (hand_cfg["hand_fingertip_z"] - hand_cfg["hand_mount_z"]) * approach_cam

                # 基座系 z 高度补偿（config z_comp_knots，与执行器同：键=补偿前命令法兰 z）
                # 共轭进相机系：基座平移 Δb 在相机系表现为 R_c2f.T @ R_f2b.T @ Δb（同 offset_cam）。
                # 先以补偿前 z_base 取键（p_flange_cam 经完整相机→法兰→基座链换算，与执行器
                # compute_grasp_target 的 p_flange[2] 数值一致），再整手（palm/flange/tip）刚体平移 d_cam。
                z_base = (R_f2b @ (R_c2f @ p_flange_cam + t_c2f)
                          + np.asarray(gc.OBSERVE_RPYXYZ[3:6], dtype=float))[2]
                dz = gc.compute_z_comp(z_base, gc.load_z_comp_knots())
                if dz != 0.0:
                    d_cam = R_c2f.T @ R_f2b.T @ np.array([0.0, 0.0, dz])
                    p_palm_cam = p_palm_cam + d_cam
                    p_flange_cam = p_flange_cam + d_cam
                    p_tip_cam = p_tip_cam + d_cam

                # 偏差与执行器一致：基座系计算（与 compute_grasp_target 同式）。
                # 旧式 angle(R_c2f R_palm R_mount @ R_fixed^T R_f2b) 与执行器数值相同（共轭相似），
                # 但统一写法可保证预览的"是否回退"判断与执行器走同一条代码路径。
                R_flange_base = R_f2b @ R_c2f @ R_palm @ R_mount
                R_fixed = Rotation.from_euler("xyz", gc.FIXED_RPY).as_matrix()
                dev = float(np.abs(
                    Rotation.from_matrix(R_flange_base @ R_fixed.T).magnitude()) * 180.0 / np.pi)
                will_fallback = dev > hand_cfg["orient_deviation_thresh_deg"]

                flange_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
                T = np.eye(4)
                T[:3, :3] = R_flange_cam
                T[:3, 3] = p_flange_cam
                flange_frame.transform(T)
                flange_frame.paint_uniform_color([0.9, 0.6, 0.1] if will_fallback else [0.1, 0.8, 0.3])

                tip_marker = o3d.geometry.TriangleMesh.create_sphere(0.004, 12).translate(p_tip_cam)
                tip_marker.paint_uniform_color([0.1, 0.4, 1.0])

                mode = "将回退固定朝向" if will_fallback else "采用网络朝向"
                off_rpy_txt = ""
                if not np.allclose(offset_rpy_deg, 0):
                    off_rpy_txt = (f"  +offset_rpy=["
                                   f"{', '.join(f'{v:+.1f}°' for v in offset_rpy_deg)}]")
                print(f"[法兰目标] p_cam={p_flange_cam.round(3)} 与固定朝向夹角 {dev:.1f}° "
                      f"(阈值 {hand_cfg['orient_deviation_thresh_deg']}°) → {mode}"
                      f"{off_rpy_txt}",
                      flush=True)
            except Exception as e:
                print(f"[法兰目标] 可视化计算失败: {e}", flush=True)
                flange_frame = None
                tip_marker = None

            if args.save_debug_scene:
                save_debug_scene(args.output, cloud, hand_mesh, two_finger_geom, frame)
                print(f"debug scene saved under: {os.path.dirname(args.output)}")
            if args.visualize:
                geometries = [cloud, hand_mesh, frame, world_frame]
                if two_finger_geom is not None:
                    geometries.append(two_finger_geom)
                if flange_frame is not None:
                    geometries.append(flange_frame)
                if tip_marker is not None:
                    geometries.append(tip_marker)
                o3d.visualization.draw_geometries(geometries)
        return 0
    finally:
        shm_depth.close()
        shm_color.close()
        if shm_meta is not None:
            shm_meta.close()


if __name__ == "__main__":
    raise SystemExit(main())
