"""Offline regressions. No camera, SDK connection, serial writes or motion.

Run: python -m unittest discover -s scripts/tests -v (conda anydex).
"""
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "own_hand")]
import grasp_common as gc
import grasp_pipeline as gp
from camera_snapshot import read_snapshot
from collision import HandCollisionDetector
from grasp_decoding import parse_preds, batch_viewpoint_params_to_matrix


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.calib = {"R_cam2flange": np.eye(3).tolist(), "t_cam2flange": [0, 0, 0]}
        self.candidate = dict(hand_model="own", grasp_type=2, score=.95,
                              translation=[.3, 0, .2], rotation_matrix=np.eye(3).tolist())

    def test_plan_closes_full_transform_with_offsets(self):
        mount = gp.transform(Rotation.from_euler("xyz", [.4, -.2, .1]).as_matrix(), [.04, .02, -.01])
        cfg = gc.load_hand_config(model="own")
        cfg["hand_mount_transform"] = mount.tolist()
        flange = [.2, -.3, .1, .12, -.05, .3]
        calib = dict(R_cam2flange=Rotation.from_euler("y", .5).as_matrix().tolist(),
                     t_cam2flange=[-.01, .02, .07])
        with patch.object(gc, "load_hand_config", return_value=cfg), \
                patch.object(gc, "load_offset_xyz", return_value=[.01, -.02, .03]), \
                patch.object(gc, "load_offset_rpy_deg", return_value=[5, 7, 0]), \
                patch.object(gc, "load_z_comp_knots", return_value=[[0, .002], [1, .002]]):
            plan = gp.build_plan(self.candidate, calib, flange)
        B_C = gp.pose_transform(flange) @ gp.transform(calib["R_cam2flange"], calib["t_cam2flange"])
        B_H = gp.pose_transform(plan["final_pose"]) @ mount
        np.testing.assert_allclose(B_C @ np.array(plan["T_camera_hand"]), B_H, atol=1e-10)
        expected_position = (B_C @ np.r_[self.candidate["translation"], 1])[:3] + [.01, -.02, .032]
        np.testing.assert_allclose(B_H[:3, 3], expected_position, atol=1e-10)
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(plan["descent_poses"], axis=0)[:, 3:], axis=1)), gp.PATH_STEP + 1e-10)

    def test_depth_is_not_applied_twice(self):
        a = gp.build_plan(self.candidate, self.calib, np.zeros(6))
        self.candidate["depth"] = .08
        b = gp.build_plan(self.candidate, self.calib, np.zeros(6))
        np.testing.assert_allclose(a["T_camera_hand"], b["T_camera_hand"])

    def test_raw_cloud_keeps_table_outside_network_depth(self):
        cloud = gp.depth_cloud(np.array([[400, 900, 0]], dtype=np.uint16),
                               dict(fx=100, fy=100, ppx=0, ppy=0))
        np.testing.assert_allclose(cloud[:, 2], [.4, .9])

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            gp.transform(np.diag([1, 1, -1]), [0, 0, 0])

    def test_sampled_ik_seeded_and_jump_rejected(self):
        plan = gp.build_plan(self.candidate, self.calib, np.zeros(6))
        calls = []
        def check(pose, seed):
            calls.append(seed)
            return True, np.zeros(6) if seed is None else seed + .01
        self.assertTrue(gp.check_plan_ik(plan, check, [(-1, 1)] * 3)[0])
        self.assertIsNone(calls[0])
        self.assertIsNotNone(calls[1])
        def jump(pose, seed):
            return True, np.zeros(6) if seed is None else np.ones(6)
        self.assertIn("joint_jump", gp.check_plan_ik(plan, jump, [(-1, 1)] * 3)[1])

    def test_workspace_failure_does_not_call_ik(self):
        plan = gp.build_plan(self.candidate, self.calib, np.zeros(6))
        check = Mock()
        self.assertFalse(gp.check_plan_ik(plan, check, [(0, .1)] * 3)[0])
        check.assert_not_called()


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.box = o3d.geometry.TriangleMesh.create_box(.02, .02, .02)
        self.plan = dict(T_camera_hand=np.eye(4).tolist(), approach_distance=.12,
                         pregrasp_direction_camera=[0., 0., 1.])

    def check(self, points, plan=None):
        detector = HandCollisionDetector(points, mesh_provider=lambda _: [self.box])
        return detector.check(2, plan or self.plan)

    def test_swept_obstacle_and_interior(self):
        result = self.check([[.01, .01, .07]])
        self.assertFalse(result["passed"])
        self.assertEqual(result["collision_points"], 1)
        self.assertEqual(result["surface_only_links"], 0)

    def test_clear_path(self):
        self.assertTrue(self.check([[.3, .3, .3]])["passed"])

    def test_early_rejection_matches_full_sweep_and_marks_lower_bound(self):
        points = [[.01, .01, z] for z in [.01, .04, .07, .10, .13]]
        points.append([.3, .3, .3])
        detector = HandCollisionDetector(points, mesh_provider=lambda _: [self.box])
        for threshold in [0, 2, 5]:
            fast = detector.check(2, self.plan, threshold)
            full = detector.check(2, self.plan, threshold, full_count=True)
            self.assertEqual(fast["passed"], full["passed"])
            self.assertLessEqual(fast["collision_points"], full["collision_points"])
            self.assertTrue(full["collision_points_exact"])
            if not fast["passed"]:
                self.assertTrue(fast["early_exit"])
                self.assertFalse(fast["collision_points_exact"])
            else:
                self.assertTrue(fast["collision_points_exact"])

    def test_contact_points_are_measured_camera_points(self):
        R = Rotation.from_euler("y", .5).as_matrix()
        T = gp.transform(R, [.3, .1, .2])
        point = R @ [.01, .01, .01] + T[:3, 3] + [.06, 0, 0]
        plan = dict(T_camera_hand=T.tolist(), approach_distance=.12,
                    pregrasp_direction_camera=[1., 0., 0.])
        result = self.check([point], plan)
        contact = result["first_collision"]
        self.assertIsNotNone(contact)
        np.testing.assert_allclose(contact["points_camera"], [point])
        self.assertGreater(contact["sweep_offset_m"], 0)

    def test_non_axis_direction_and_rotation(self):
        plan = dict(T_camera_hand=gp.transform(Rotation.from_euler("y", .5).as_matrix(), [.3, .1, .2]).tolist(),
                    approach_distance=.12, pregrasp_direction_camera=[1., 0., 0.])
        T = np.array(plan["T_camera_hand"])
        point = T[:3, :3] @ [.01, .01, .01] + T[:3, 3] + [.06, 0, 0]
        self.assertFalse(self.check([point], plan)["passed"])

    def test_real_hand_preshape_matches_commands_and_has_all_links(self):
        import own_hand_grasp as ohg
        from hand_bridge import PRESHAPE_LIBRARY
        motors = ohg.motor_positions(2)
        self.assertEqual(list(motors.values()), PRESHAPE_LIBRARY[2]["broadcast"])
        meshes = ohg.link_meshes_for_motors(motors)
        self.assertEqual(len(meshes), len(ohg.ALL_LINKS))
        self.assertTrue(all(not m.is_empty() for m in meshes))
        # A scene point on the real palm must be detected at the supplied pose.
        point = np.asarray(meshes[0].vertices)[0]
        detector = HandCollisionDetector([point])
        self.assertFalse(detector.check(2, self.plan)["passed"])


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source"
        self.source.write_text("config")
        self.now = time.time()
        self.pose = [0., 0., 0., .3, 0., .35]
        self.observation = dict(id="round1", state="ready", ready_at=self.now - 1, flange_pose=self.pose)
        self.result = dict(schema_version=2, hand_model="own", observation_id="round1",
                           captured_at=self.now, capture_flange_pose=self.pose,
                           source_hashes={str(self.source): gp.sha256(self.source)}, candidates=[{}],
                           capture_bundle=str(self.source), capture_bundle_sha256=gp.sha256(self.source))

    def validate(self):
        gp.validate_result(self.result, self.observation, self.pose, now=self.now)

    def test_valid_observation(self):
        self.validate()

    def test_previous_round_stale_future_and_moved_rejected(self):
        for field, value in [("observation_id", "round0"), ("captured_at", self.now - 121),
                             ("captured_at", self.now + 2)]:
            original = self.result[field]
            self.result[field] = value
            with self.assertRaises(ValueError):
                self.validate()
            self.result[field] = original
        moved = np.array(self.pose) + [0, 0, 0, .02, 0, 0]
        with self.assertRaises(ValueError):
            gp.validate_result(self.result, self.observation, moved, now=self.now)

    def test_changed_config_rejected(self):
        self.source.write_text("changed")
        with self.assertRaises(ValueError):
            self.validate()

    def test_atomic_json_preserves_previous_on_nan(self):
        path = Path(self.tmp.name) / "result.json"
        gp.write_json(path, {"valid": 1})
        with self.assertRaises(ValueError):
            gp.write_json(path, {"invalid": float("nan")})
        self.assertEqual(json.loads(path.read_text()), {"valid": 1})

    def test_snapshot_returns_copies_and_rejects_partial_write(self):
        depth, color = np.ones((2, 2), dtype=np.uint16), np.ones((2, 2, 3), dtype=np.float32)
        metadata = np.array([2, int(self.now * 1e9)], dtype=np.int64)
        d, c, info = read_snapshot(depth, color, metadata)
        depth[:] = 0
        self.assertTrue(d.all())
        self.assertEqual(info["frame_sequence"], 2)
        metadata[0] = 3
        with self.assertRaises(RuntimeError):
            read_snapshot(depth, color, metadata, timeout=.01)


class DecoderTests(unittest.TestCase):
    def test_background_filter_aligns_every_feature_column(self):
        torch.manual_seed(7)
        end = dict(before_generator=torch.randn(1, 3, 256), point_features=torch.randn(1, 3, 512),
                   sinput=SimpleNamespace(C=torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])),
                   stage1_objectness_pred=torch.tensor([[0., 1.], [1., 0.], [0., 1.]]),
                   stage2_seed_xyz=torch.rand(1, 3, 3), stage2_seed_inds=torch.tensor([[2, 0, 1]]),
                   stage2_view_xyz=torch.tensor([[[0., 0., -1.]] * 3]),
                   stage2_view_inds=torch.tensor([[12, 34, 56]]), stage2_view_scores=torch.rand(1, 3),
                   stage3_grasp_scores=torch.rand(1, 3, 48, 5), stage3_grasp_features=torch.randn(1, 3, 480),
                   stage3_normalized_grasp_widths=torch.rand(1, 3, 48, 5))
        old, old_features = parse_preds(end, filter_objectness=False)
        new, features = parse_preds(end)
        self.assertEqual(tuple(new[0].shape), (2, 15))
        self.assertTrue(torch.allclose(new[0], old[0][[0, 1]]))
        self.assertTrue(torch.allclose(features[0], old_features[0][[0, 1]]))
        rotations = old[0][:, 3:12].reshape(-1, 3, 3)
        self.assertTrue(torch.allclose(torch.linalg.det(rotations), torch.ones(3), atol=1e-6))
        end["stage1_objectness_pred"][:] = torch.tensor([1., 0.])
        self.assertEqual(parse_preds(end), ([], []))

    def test_fixed_type_uses_its_own_predictions(self):
        # Exercise production selector without importing legacy robot initializers.
        module = ast.parse((ROOT / "scripts/robot_inspire.py").read_text())
        node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "get_inspire_depth_type")
        env = dict(np=np, torch=torch, copy=copy, OrderedDict=OrderedDict,
                   RANDOM_GRASP=False, NUM_OF_INSPIRE_TYPE=8, NUM_OF_INSPIRE_DEPTH=4)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "selector", "exec"), env)
        class Model:
            def __init__(self, i):
                self.i = i
            def __call__(self, x):
                scores = torch.arange(20, device=x.device).repeat(len(x), 1).float() + self.i * 100
                return scores, None
        models = {"480": {str(i): Model(i) for i in range(1, 9)}}
        inputs = dict(grasp_preds_features=np.zeros((2, 480)), grasp_depths=np.array([0, 1], dtype=int))
        depth, types, scores, poses, features = env["get_inspire_depth_type"](
            models, inputs, np.arange(34).reshape(2, 17), np.array([[10], [20]]), fixed_type=2)
        self.assertTrue(np.all(types == 2))
        self.assertTrue(np.all((scores >= 200) & (scores < 220)))
        np.testing.assert_array_equal(poses[:, 0] // 17, features[:, 0] // 10 - 1)


class SDKTests(unittest.TestCase):
    def test_ik_is_read_only_and_strict_movel_never_falls_back(self):
        # Compile only class methods: no SDK/ROS import or constructor side effects.
        tree = ast.parse((ROOT / "scripts/MoveZ1.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        sdk = SimpleNamespace(postureToHomo=lambda p: p)
        env = dict(np=np, unitree_arm_interface=sdk)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "sdk_controller", "exec"), env)
        controller = object.__new__(env[cls.name])
        controller.safe_check, controller.move_speed = False, .15
        controller.arm = Mock()
        controller.arm_model = Mock()
        controller.arm_model.getJointQMin.return_value = -np.ones(6)
        controller.arm_model.getJointQMax.return_value = np.ones(6)
        controller.arm_model.inverseKinematics.return_value = (True, np.zeros(6))
        self.assertTrue(controller.check_ik([0] * 6)[0])
        controller.arm.MoveJ.assert_not_called()
        controller.arm.MoveL.assert_not_called()
        controller.arm.MoveL.return_value = False
        with self.assertRaises(RuntimeError):
            controller.move_l_strict([0] * 6)
        controller.arm.MoveJ.assert_not_called()
        self.assertEqual(controller.arm.MoveL.call_args[0], ([0] * 6, .15))


if __name__ == "__main__":
    unittest.main()


class ExecutionTests(unittest.TestCase):
    def test_rejected_preview_opens_diagnostic_without_executable_candidate(self):
        import own_preview
        contact = dict(link_index=0, sweep_offset_m=.06,
                       points_camera=[[0., 0., .4]])
        collision = dict(passed=False, collision_points=1,
                         collision_points_exact=False, early_exit=True,
                         first_collision=contact, model="own_urdf_preshape")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calib, intrinsics, output = tmp/'calib.json', tmp/'intrinsics.json', tmp/'latest.json'
            gp.write_json(calib, dict(R_cam2flange=np.eye(3).tolist(), t_cam2flange=[0, 0, 0]))
            gp.write_json(intrinsics, dict(fx=100, fy=100, ppx=0, ppy=0))
            args = SimpleNamespace(calibration=str(calib), intrinsics=str(intrinsics),
                                   output=str(output), collision=True, collision_margin=.003,
                                   candidate_limit=1, topk=1, visualize=True,
                                   save_debug_scene=False, collision_full_count=False)
            ggarray = np.array([[.95, .06, .02, .01, *np.eye(3).ravel(), .3, 0, .4, -1]])
            with patch.object(own_preview, 'HandCollisionDetector') as detector, \
                    patch.object(own_preview.ohg, 'hand_mesh_at_pose',
                                 return_value=o3d.geometry.TriangleMesh()) as mesh, \
                    patch.object(o3d.visualization, 'draw_geometries') as draw:
                detector.return_value.check.return_value = collision
                status = own_preview.run(args, ggarray, np.array([.9]), np.array([2]),
                                         np.array([.02]), np.zeros((1, 480)),
                                         np.array([[400]], dtype=np.uint16), np.zeros((1, 1, 3)),
                                         dict(captured_at=time.time(), frame_sequence=2), None, time.time())
            self.assertEqual(status, 6)
            result = json.loads(output.read_text())
            self.assertEqual(result['candidates'], [])
            self.assertEqual(result['num_output_grasps'], 0)
            self.assertEqual(result['status'], 'no_collision_free_candidate')
            draw.assert_called_once()
            self.assertIn('DIAGNOSTIC ONLY', draw.call_args.kwargs['window_name'])
            self.assertGreater(len(draw.call_args.args[0]), 4)  # Collision markers.
            plan = result['tested_candidates'][0]['plan']
            T = np.array(plan['T_camera_hand'])
            expected_position = T[:3, 3] + np.array(plan['pregrasp_direction_camera']) * .06
            np.testing.assert_allclose(mesh.call_args.args[0], expected_position)
            self.assertFalse(detector.return_value.check.call_args.kwargs['full_count'])

    def test_own_preview_archive_matches_offline_replay(self):
        import own_preview
        spec = importlib.util.spec_from_file_location('preview_replay', ROOT/'scripts/tools/replay_grasp.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calib, intrinsics, output = tmp/'calib.json', tmp/'intrinsics.json', tmp/'latest.json'
            gp.write_json(calib, dict(R_cam2flange=np.eye(3).tolist(), t_cam2flange=[0, 0, 0]))
            gp.write_json(intrinsics, dict(fx=100, fy=100, ppx=0, ppy=0))
            args = SimpleNamespace(calibration=str(calib), intrinsics=str(intrinsics),
                                   output=str(output), collision=True, collision_margin=.003,
                                   candidate_limit=1, topk=1, visualize=False, save_debug_scene=False)
            # GraspNet layout: score,width,height,depth,R(9),translation(3),object_id.
            ggarray = np.array([[.95, .06, .02, .01, *np.eye(3).ravel(), .3, 0, .4, -1]])
            status = own_preview.run(args, ggarray, np.array([.9]), np.array([2]),
                                     np.array([.02]), np.zeros((1, 480)),
                                     np.array([[1400]], dtype=np.uint16), np.zeros((1, 1, 3)),
                                     dict(captured_at=time.time(), frame_sequence=2), None, time.time())
            self.assertEqual(status, 0)
            result = json.loads(output.read_text())
            candidate = result['candidates'][0]
            self.assertEqual(candidate['grasp_type'], 2)
            self.assertTrue(candidate['depth_consumed_in_pose'])
            self.assertIsNone(result['observation_id'])
            self.assertEqual(gp.sha256(result['capture_bundle']), result['capture_bundle_sha256'])
            archive = Path(result['capture_bundle']).with_name('result.json')
            report = module.replay(str(archive))
            self.assertEqual(report['source_changes'], [])
            self.assertEqual(report['checks'][0]['previous'], report['checks'][0]['replayed'])

    def load_demo_functions(self):
        tree = ast.parse((ROOT / 'scripts/anydex_eih_grasp.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('grasp_once', 'execute_selected')]
        ns = dict(np=np, os=__import__('os'), json=json, time=SimpleNamespace(time=time.time, sleep=Mock()),
                  gp=gp, jog_manual=SimpleNamespace(descend_interruptible=Mock()),
                  X_RANGE=(-1, 1), Y_RANGE=(-1, 1), Z_RANGE=(-1, 1),
                  TRANSIT_RPYXYZ=np.zeros(6), DROP_RPYXYZ=np.zeros(6),
                  DESCEND_STEP_MM=10, GRASP_WAIT_S=0, HAND_WAIT_S=0, DROP_SETTLE_S=0, LIFT_Z=.08)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<demo-functions>', 'exec'), ns)
        return ns

    def test_all_ik_fail_sends_no_motion_and_records_reason(self):
        ns = self.load_demo_functions()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            candidate = dict(candidate_id=1, hand_model='own', grasp_type=2,
                             collision=dict(model='own_urdf_preshape', passed=True))
            plan = gp.build_plan(dict(hand_model='own', translation=[.3, 0, .2],
                                      rotation_matrix=np.eye(3).tolist()),
                                 dict(R_cam2flange=np.eye(3).tolist(), t_cam2flange=[0, 0, 0]), np.zeros(6))
            candidate['plan'] = plan
            gp.write_json(path, dict(run_id='test', candidates=[candidate], capture_bundle=str(Path(tmp)/'capture.npz'),
                                     calibration={}, capture_flange_pose=[0]*6))
            ns['DEFAULT_GRASP_JSON'] = str(path)
            arm, hand = Mock(), Mock()
            arm.getpose.return_value = np.zeros(6)
            arm.check_ik.return_value = (False, np.zeros(6))
            with patch.object(gp, 'load_observation', return_value={}), \
                    patch.object(gp, 'validate_result'), patch.object(gp, 'build_plan', return_value=plan):
                self.assertEqual(ns['grasp_once'](arm, hand, 1), 1)
            arm.movej.assert_not_called()
            arm.movel.assert_not_called()
            hand.close.assert_not_called()
            report = json.loads((Path(tmp)/'execution.json').read_text())
            self.assertEqual(report['status'], 'no_reachable_candidate')
            self.assertEqual(report['decisions'][0]['reason'], 'ik:0')

    def test_descent_failure_or_interrupt_never_closes_hand(self):
        ns = self.load_demo_functions()
        plan = dict(pregrasp_pose=[0, 0, 0, .3, 0, .4], final_pose=[0, 0, 0, .3, 0, .28],
                    lift_pose=[0, 0, 0, .3, 0, .36])
        with tempfile.TemporaryDirectory() as tmp:
            for effect, exc in [(RuntimeError('MoveL failed'), RuntimeError),
                                ((True, np.zeros(6)), KeyboardInterrupt)]:
                arm, hand = Mock(), Mock()
                descend = ns['jog_manual'].descend_interruptible
                descend.side_effect = effect if isinstance(effect, Exception) else None
                descend.return_value = effect
                with self.assertRaises(exc):
                    ns['execute_selected'](arm, hand, 1, 2, plan, {}, Path(tmp)/'execution.json')
                self.assertTrue(descend.call_args.kwargs['strict_linear'])
                hand.close.assert_not_called()
                arm.movel.assert_not_called()
                self.assertEqual(arm.movej.call_count, 1)

    def test_archived_capture_replays_without_hardware(self):
        spec = importlib.util.spec_from_file_location('offline_replay', ROOT/'scripts/tools/replay_grasp.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            bundle, result_path = Path(tmp)/'capture.npz', Path(tmp)/'result.json'
            np.savez_compressed(str(bundle), scene_points=np.array([[5., 5., 5.]]))
            candidate = dict(candidate_id=4, grasp_type=2, collision=dict(passed=True),
                             plan=dict(T_camera_hand=np.eye(4).tolist(), approach_distance=.12,
                                       pregrasp_direction_camera=[0, 0, 1]))
            gp.write_json(result_path, dict(schema_version=2, capture_bundle=str(bundle),
                                           capture_bundle_sha256=gp.sha256(bundle), calibration={},
                                           source_hashes={}, candidates=[candidate], tested_candidates=[candidate]))
            report = module.replay(str(result_path))
            self.assertTrue(report['diagnostic_only'])
            self.assertTrue(report['checks'][0]['replayed']['passed'])
            self.assertTrue((Path(tmp)/'replay.json').exists())
            self.assertFalse((Path(tmp)/'latest_inspire_grasp.json').exists())
