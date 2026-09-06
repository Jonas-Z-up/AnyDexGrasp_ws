# 结果归档说明

- **现行手眼外参以 `../cam2flange_current.json` 为准**（抓取管线消费点），本目录只放历史归档，勿再放入散文件。
- `calibration_charuco_current.json` — 2026-09-06 采用解的报告（ChArUco 31 组 + 自标定内参，DANIILIDIS，2.46mm/0.175°）。
- `cam2flange_horaud_archive.json` — 旧棋盘格方案（HORAUD）外参存档，曾被 `grasp_config.json` 的 offset_xyz=[0.072,0,-0.085] 总账补偿；新外参启用后该 offset 必须归零重测。
- 旧棋盘格标定的 txt 零件（内参/外参/位姿矩阵等）在 `../archive/legacy_results_txt/`。
- 各次标定/复测的原始数据和完整解算输出在 `../sessions/<会话>/`。
