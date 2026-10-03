# 本机 1000 step 评测

仓库来源：https://github.com/kuieless/AerialMetric.git
克隆提交：2ec0e6c6d252796ab3b454ebecf60ab53df87fd3

评测权重：`/data1/szq/moge2/权重/workspace/retrain-lora96-192-UElr2-no-ground-1800-20261002_182751/checkpoint/00001000.pt`
结果目录：`/data1/szq/moge2/benchmark/retrain-lora96-192-UElr2-no-ground-1800-20261002_182751-step1000`
完整评测日志：`/data1/szq/moge2/benchmark/retrain-lora96-192-UElr2-no-ground-1800-20261002_182751-step1000/benchmark.log`
GPU：0、1；训练目前已退出，是否从 1150 step 恢复等待用户选择。
Python：`/home/szq/miniconda3/envs/moge310/bin/python`。

## 当前任务

后台 `evaluate_step1000.py` 等待目标 checkpoint 大小连续三次稳定（每次间隔 10 秒），再验证 checkpoint 可读取且含 rank 96 LoRA，然后运行完整 `benchmark.sh`。状态写入 `step1000_status.json`，失败会保存非零退出状态。

```bash
cat /data1/szq/moge310/AerialMetric/local_runs/step1000_status.json
tail -f /data1/szq/moge310/AerialMetric/local_runs/step1000_watcher.log
```

正式评测覆盖原脚本全部 10 项任务：航拍无 GT 内参、航拍有 GT 内参、地面无 GT 内参、地面有 GT 内参；包含基座和新 LoRA 权重。保持原脚本的分辨率、batch、mask、全量采样和地面评测精度设置。

## 配置变更

- 项目目录取当前仓库位置，使用 moge310 的 Python，模型路径指向本次训练 00001000.pt。
- 地面和航拍测试数据沿用 `/data1/szq/Val`；7 个地面集共 4406 条索引的 image.jpg、depth.png、meta.json 存在性检查通过。
- 结果目录按本次实验独立保存，不删除已有目录；摘要按实际 checkpoint 文件名定位。
- 队列等待并检查每个任务退出码；航拍评测缺报告时失败，不再只打印完成。
- 两条 LoRA 加载路径使用 strict=True；当前训练模型的 model_config 与两条评测路径的 MODEL_CONFIG 完全一致。

## 已执行的启动检查

现有 50 step checkpoint 的单张 Decoupled（带 GT 内参和 mask）完整推理→提取→评测成功；单张 NYUv2（无 GT 内参）评测成功，LoRA 参数严格加载成功。
基座单张 Decoupled（无 GT 内参）与单张 NYUv2（带 GT 内参）也执行成功。
这些样本只用于确认流程能运行，不能用于判断 1000 step 的质量或与论文比较。
检查输出：`/data1/szq/moge2/benchmark/retrain-no-ground-step1000-smoke/`。

启动任务 PID：3587961。

## 2026-10-02 迁移

项目工作目录迁移至 /data1/szq/moge310；旧 /home 路径为兼容软链接。临时文件和模型缓存写入同一盘 tmp/cache。基座副本为 weights/vitl-normal.pt，已与 1000 step 权重中未训练的基座张量逐个对比。
