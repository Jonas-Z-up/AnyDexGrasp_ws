#!/usr/bin/env python3
"""Z1 + 自研手 装配渲染 — 按 config 当前定义的安装参数可视化装配，供真机照片比对。

内容:
  - Z1 机械臂（z1.urdf, pybullet IK 到指定法兰位姿, 默认观测位姿）
  - 自研手（base_link 放在"掌心点": 法兰 + hand_mount_z×法兰X轴,
    姿态 = 法兰 ∘ Rx(-mount_yaw)，与 grasp_common.compute_grasp_target 的定义互逆）
  - D435i 相机（青色长方体示意, 按 calibration/cam2flange_current.json 外参安装）
  - 法兰(白框下)/掌心/相机 三处坐标轴标记（RGB=XYZ）

输出 docs/arm_renders/assembly_{yaw0,yaw90,yaw180,yaw270,top,closeup}.png

用法:
  python scripts/tools/render_arm_hand_assembly.py                      # 观测位姿
  python scripts/tools/render_arm_hand_assembly.py --rpyxyz 0,0.6,0,0.35,0,0.25

注意: "掌心点"是管线里的虚拟点（由 Inspire 查表定义），手 base_link 的真实安装
位置在 config 中并无独立参数（误差由 jog 闭环吸收）。本图假设 base_link 原点
重合于掌心点——若真机照片中手明显更靠前/靠后或拧了一个角度，该差异量就是
hand_mount_z / mount_yaw_deg 与实物的偏差，正是要看的结论。
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS_ROOT, "scripts"))
sys.path.insert(0, os.path.join(WS_ROOT, "scripts", "tools"))

import grasp_common  # noqa: E402
from compute_hand_geometry import URDF_PATH as HAND_URDF, MESH_DIR as HAND_MESH_DIR  # noqa: E402

Z1_URDF = ("/home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_moveit_ws/"
           "src/z1_ros/z1_controller/config/z1.urdf")
Z1_MESH_ROOT = ("/home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_ws/"
                "src/unitree_ros/robots/z1_description")
CAM_JSON = os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")
OUT_DIR = os.path.join(WS_ROOT, "own_hand", "renders", "arm")
DEFAULT_RPYXYZ = "0,1.057,0,0.3,0,0.35"   # 演示循环的观测位姿


def patched_urdf_path(src, package_prefix, real_dir):
    with open(src, encoding="utf-8-sig") as f:
        text = f.read()
    patched = text.replace(package_prefix, real_dir.rstrip("/") + "/")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(patched)
    tmp.close()
    return tmp.name


def quat_from_R(R):
    return tuple(Rotation.from_matrix(R).as_quat())


def axis_frame_marker(p, pos, R, length=0.05, thick=0.004):
    """在 pos/R 处放 XYZ 三根短棒（RGB=XYZ）。"""
    colors = [(0.9, 0.15, 0.15, 1), (0.15, 0.8, 0.15, 1), (0.15, 0.35, 0.95, 1)]
    for k, color in enumerate(colors):
        d = R[:, k]
        center = np.asarray(pos) + d * length / 2
        # 单位 X 方向旋转到 d
        ax = np.array([1.0, 0, 0])
        v = np.cross(ax, d)
        c = float(np.dot(ax, d))
        if c > 0.999:
            Rm = np.eye(3)
        elif c < -0.999:
            Rm = Rotation.from_euler("z", np.pi).as_matrix()
        else:
            v = v / np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rm = np.eye(3) + K + K @ K * (1 / (1 + c))
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, thick, thick])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, thick, thick],
                                  rgbaColor=color)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                          basePosition=center.tolist(), baseOrientation=quat_from_R(Rm))


def frame_of_link(p, body, link):
    st = p.getLinkState(body, link, computeForwardKinematics=1)
    pos = np.array(st[4])
    R = np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
    return pos, R


def main():
    import pybullet as p

    ap = argparse.ArgumentParser(description="Z1+自研手装配渲染（按 config 安装参数）")
    ap.add_argument("--rpyxyz", default=DEFAULT_RPYXYZ,
                    help="法兰目标位姿 roll,pitch,yaw,x,y,z (m/rad)，逗号分隔")
    ap.add_argument("--hand-roll-deg", type=float, default=90.0,
                    help="手 base 相对掌心帧绕接近轴的附加滚转角(°)。实机比对定案: +90"
                         "（2026-09-07, 见 docs/自研手几何数据.md）。管线掌心帧是 Inspire "
                         "查表约定，与自研手 base 帧存在此固定差；将来绕过查表规划时需"
                         "折算进规划链")
    args = ap.parse_args()
    rpyxyz = np.array([float(v) for v in args.rpyxyz.split(",")])

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(os.path.dirname(HAND_MESH_DIR))

    # ── 场景与机械臂 ──
    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plane_shape,
                      baseVisualShapeIndex=p.createVisualShape(
                          p.GEOM_PLANE, rgbaColor=(0.75, 0.75, 0.75, 1)))
    z1 = p.loadURDF(patched_urdf_path(Z1_URDF, "package://z1_description/", Z1_MESH_ROOT),
                    useFixedBase=True)
    joint_idxs = [i for i in range(p.getNumJoints(z1))
                  if p.getJointInfo(z1, i)[2] == p.JOINT_REVOLUTE]
    ee = p.getNumJoints(z1) - 1

    orn_q = tuple(Rotation.from_euler("xyz", rpyxyz[:3]).as_quat())
    sol = p.calculateInverseKinematics(z1, ee, rpyxyz[3:6].tolist(), orn_q,
                                       maxNumIterations=800, residualThreshold=1e-6)
    for k, ji in enumerate(joint_idxs):
        p.resetJointState(z1, ji, sol[k])
    flange_pos, R_flange = frame_of_link(p, z1, ee)
    rpy_reached = Rotation.from_matrix(R_flange).as_euler("xyz")
    print(f"[IK] 目标 rpyxyz={np.round(rpyxyz,4).tolist()}")
    print(f"[IK] 到达 pos={np.round(flange_pos,4).tolist()} "
          f"rpy={np.round(rpy_reached,4).tolist()}")
    print(f"[IK] 关节角={np.round(sol[:len(joint_idxs)],3).tolist()}")

    # ── 掌心点与自研手（grasp_common 同源参数）──
    cfg = grasp_common.load_hand_config()
    mz, alpha = cfg["hand_mount_z"], np.deg2rad(cfg["mount_yaw_deg"])
    approach = R_flange[:, 0]
    palm_pos = flange_pos + mz * approach
    R_palm = R_flange @ Rotation.from_euler("x", -alpha).as_matrix()
    print(f"[安装] hand_mount_z={mz}  mount_yaw_deg={cfg['mount_yaw_deg']}  "
          f"(config/grasp_config.json)")

    hand = p.loadURDF(patched_urdf_path(HAND_URDF, "package://dexterous_hand/meshes/",
                                        HAND_MESH_DIR), useFixedBase=True)
    R_hand = R_palm @ Rotation.from_euler("x", np.deg2rad(args.hand_roll_deg)).as_matrix()
    p.resetBasePositionAndOrientation(hand, palm_pos.tolist(), quat_from_R(R_hand))
    print(f"[手滚转] hand-roll = {args.hand_roll_deg}° (自研手 base 帧相对掌心帧)")

    # ── D435i 相机（外参: cam → flange）──
    with open(CAM_JSON) as f:
        cam = json.load(f)
    R_c2f = np.array(cam["R_cam2flange"])
    t_c2f = np.array(cam["t_cam2flange"])
    cam_pos = flange_pos + R_flange @ t_c2f
    R_cam = R_flange @ R_c2f
    cam_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.045, 0.0125, 0.021])
    cam_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.045, 0.0125, 0.021],
                                  rgbaColor=(0.1, 0.85, 0.9, 0.9))
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cam_shape, baseVisualShapeIndex=cam_vis,
                      basePosition=cam_pos.tolist(), baseOrientation=quat_from_R(R_cam))
    print(f"[相机] cam2flange t={np.round(t_c2f*1000,1).tolist()}mm ({cam['date']})")

    # ── 坐标轴标记 ──
    axis_frame_marker(p, flange_pos, R_flange, length=0.06, thick=0.005)
    axis_frame_marker(p, palm_pos, R_hand, length=0.05, thick=0.004)
    axis_frame_marker(p, cam_pos, R_cam, length=0.035, thick=0.003)

    # ── 渲染 ──
    os.makedirs(OUT_DIR, exist_ok=True)
    target = ((flange_pos + palm_pos) / 2 + np.array([0.05, 0, 0])).tolist()
    views = {"yaw0": (0, -15, 1.1), "yaw90": (90, -15, 1.1), "yaw180": (180, -15, 1.1),
             "top": (0, -89, 1.1), "closeup": (90, -20, 0.5)}
    for name, (yaw, pitch, dist) in views.items():
        view = p.computeViewMatrixFromYawPitchRoll(target, distance=dist, yaw=yaw,
                                                   pitch=pitch, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=4 / 3, nearVal=0.05, farVal=5)
        img = p.getCameraImage(960, 720, viewMatrix=view, projectionMatrix=proj,
                               renderer=p.ER_TINY_RENDERER)[2]
        arr = np.reshape(img, (720, 960, 4))[:, :, :3].astype(np.uint8)
        from PIL import Image
        fp = os.path.join(OUT_DIR, f"assembly_{name}.png")
        Image.fromarray(arr).save(fp)
        print(f"[渲染] {fp}")
    p.disconnect()
    print(f"\n[假设] 手 base_link 原点重合于掌心点；若真机照片中手前后/拧角不符，"
          f"差异即 hand_mount_z / mount_yaw_deg 与实物的偏差。")


if __name__ == "__main__":
    main()
