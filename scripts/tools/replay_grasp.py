#!/usr/bin/env python3
"""Replay saved own-hand collision plans without camera, network or robot.

python scripts/tools/replay_grasp.py logs/preview/runs/<id>/result.json
--replan recomputes transforms from current grasp_config and calibration.
Output is diagnostic only; never replaces the executable latest preview.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "own_hand"))
import grasp_pipeline as gp
from collision import HandCollisionDetector


def replay(path, replan=False, output=None):
    with open(path) as f:
        result = json.load(f)
    if result.get("schema_version") != 2:
        raise ValueError("仅支持新版 own 抓取记录")
    if gp.sha256(result["capture_bundle"]) != result["capture_bundle_sha256"]:
        raise ValueError("采集文件校验失败")
    with np.load(result["capture_bundle"], allow_pickle=False) as bundle:
        scene = bundle["scene_points"]
    detector = HandCollisionDetector(scene)
    calibration = result["calibration"]
    if replan:
        with open(result["calibration_path"]) as f:
            calibration = json.load(f)
    changed = [p for p, h in result["source_hashes"].items()
               if not Path(p).exists() or gp.sha256(p) != h]
    checks = []
    for candidate in result.get("tested_candidates", result["candidates"]):
        plan = (gp.build_plan(candidate, calibration, result["capture_flange_pose"])
                if replan else candidate["plan"])
        detector.margin = candidate["collision"].get("margin_m", .003)
        detector.step = candidate["collision"].get("sweep_step_m", gp.PATH_STEP)
        collision = detector.check(candidate["grasp_type"], plan,
                                   candidate["collision"].get("threshold", 0))
        checks.append({"candidate_id": candidate["candidate_id"],
                       "previous": candidate["collision"], "replayed": collision})
    report = {"diagnostic_only": True, "replanned": replan, "source_changes": changed,
              "checks": checks, "input": str(Path(path).resolve())}
    output = output or str(Path(path).with_name("replay.json"))
    gp.write_json(output, report)
    print("回放 %d 个候选，%d 个通过碰撞检查；变更源文件 %d 个。" %
          (len(checks), sum(c["replayed"]["passed"] for c in checks), len(changed)))
    print("报告: " + output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--replan", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    replay(args.result, args.replan, args.output)
