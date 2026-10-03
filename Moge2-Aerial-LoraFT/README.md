# MoGe2 Aerial LoRA Fine-tuning

[English](README_en.md) | [中文](README_ch.md)

Training code and reproduction record for the **no-ground run**, trained for **1000 iterations with seed 333** and evaluated **without camera intrinsics** (2026-10-02).

## Contents

`moge/` contains the training entry point and required model, loader, loss and utility modules. `configs/train.json` is the actual run configuration; `data/indices/` preserves its seven dataset indices. `provenance/` records run parameters, dependency versions, source hashes and report sources. Chinese comments and commented-out code were removed; executable ASTs remain identical to the originals. Evaluation code and weights remain outside this directory.

## Data

Training root: `/data1/szq/moge2/训练集/`. Ground datasets (Hypersim, MVS-Synth and TartanAir) were unavailable on this server and excluded.

| Dataset | Directory | Label | Weight | Index entries | Missing samples |
|---|---|---|---:|---:|---:|
| Syn-UE | Syn-train-UE | ground-b | 0.5 | 11899 | 0 |
| Syn-UE2 | Syn-train-UE2 | ground-b | 0.5 | 5084 | 0 |
| Syn-GES | Syn-train | SFM | 1 | 1998 | 0 |
| GAU | TrainingData_Final_MoGe-All | SFM | 3 | 17039 | 674 |
| FOV | FOV | SFM | 4 | 4398 | 0 |
| Lidar | B | B | 0.5 | 15195 | 0 |
| Lidar2 | B-2 | B | 4 | 3881 | 0 |

Each dataset uses this existing layout:

```text
<dataset>/train_index.txt              # One relative sample directory per line
<dataset>/<scene>/<sample>/image.jpg
<dataset>/<scene>/<sample>/depth.npy   # Depth in meters; H×W or H×W×1
<dataset>/<scene>/<sample>/meta.json   # {"intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]]}
```

Intrinsics are normalized: divide the first row of pixel intrinsics by W and the second by H. Images, depth and intrinsics must correspond after resizing/cropping. This run used preprocessed data; the complete raw-data conversion history was not verified. Dataset selection follows configuration weights, then samples uniformly within the chosen index. GAU's 674 missing entries were retained; failed reads become invalid samples with no configured loss.

Update dataset paths in `configs/train.json` for another machine. Check the existing layout with:

```bash
python tools/check_training_data.py --config configs/train.json
```

The checker is read-only and exits with code 1 for missing entries. Online augmentation, the 500 m depth cutoff and sampling settings are defined in the configuration and loader.

## Training

| Setting | Value |
|---|---|
| Model | MoGe v2, DINOv2 ViT-L/14 |
| LoRA | rank 96, alpha 192, dropout 0.1; qkv/proj/fc1/fc2 |
| Additional trainable module | scale_head |
| Seed | **333** (`set_seed(333, device_specific=True)`) |
| GPUs / precision | 2,3; DDP, bf16 |
| Effective batch | 2 GPUs × 4 images × 4 accumulation steps = 32 |
| Completed iterations | 1000 |
| Optimizer / learning rates | AdamW; backbone 1e-6, other group 1e-5 |
| Schedule | 200-step linear warmup, then 1600-step cosine; minimum LR 1e-7 |
| Save / visualization interval | 50 iteration indices |

Use the existing Python 3.10 `moge310` environment, or install `requirements.txt` in a compatible CUDA environment. Start a new run:

```bash
cd /home/szq/moge310/AerialMetric/Moge2-Aerial-LoraFT
MAIN_PROCESS_PORT=29633 bash train.sh --seed 333 --num_iterations 1000
```

The launcher creates a new workspace. Override `PYTHON_BIN`, `CONFIG_PATH`, `BASE_CHECKPOINT`, `TRAIN_WORKSPACE`, `CUDA_VISIBLE_DEVICES` or `MAIN_PROCESS_PORT` as needed. The base checkpoint defaults to `/data1/szq/moge310/weights/vitl-normal.pt`.

Training completed in approximately 1 hour 40 minutes; aerial evaluation completed on 2026-10-02 at 23:33 (Asia/Shanghai). Results use `00000999.pt`: iteration indices start at zero, so this checkpoint contains the completed 1000-iteration run. Saved checkpoints contain full model weights including LoRA, not EMA exports or complete optimizer/scheduler resume states.

## Evaluation

Use the parent AerialMetric repository: LoRA inference → depth extraction → dataset metrics. Inputs and GT match by scene and sample stem:

| Dataset | RGB input under `/data1/szq/Val/` | GT layout | Metadata / masks |
|---|---|---|---|
| Decoupled | `decoupled/<scene>/image/<id>.jpg` | `decoupled-norm/<scene>/<id>/depth.npy` | CSVs in decoupled; decoupled-masks |
| Oblique | `Oblique/<scene>/rgbs/<id>.jpg` | `Oblique/<scene>/depth/<id>.npy` | Oblique-masks |
| Wild | `Wild/<scene>/rgbs/<id>.jpg` | `Wild/<scene>/depth_1k/<id>.npy` (preferred) | Pseudo-GT; depth/ fallback |

All samples; batch 8; resize 0; intrinsics `none`; masks `load`. Decoupled/Oblique evaluate valid depths in (0.001,400) m; Wild uses 400 m and 800 m caps. GT scale alignment is disabled. Scores average per-image metrics; N counts evaluated images.

Re-run only the LoRA evaluation:

```bash
cd /home/szq/moge310/AerialMetric
export OPENCV_IO_ENABLE_OPENEXR=1
export TMPDIR=/data1/szq/moge310/tmp
export MPLCONFIGDIR=/data1/szq/moge310/cache/matplotlib
/home/szq/miniconda3/envs/moge310/bin/python \
  MoGe/moge/scripts/code-final/aerial_eval_cli.py \
  --model_type lora96 \
  --checkpoint "/data1/szq/moge2/权重/workspace/retrain-lora96-192-no-ground-seed333-1000iters-20261002_213320/checkpoint/00000999.pt" \
  --output_dir /data1/szq/moge2/benchmark/lora96-seed333-1000iters-no-intrinsics-recheck \
  --gpu 2 --resize 0 --batch_size 8 \
  --intrinsics_mode none --mask_mode load --cleanup_intermediate \
  --decoupled_input /data1/szq/Val/decoupled \
  --decoupled_gt /data1/szq/Val/decoupled-norm \
  --decoupled_csv_dir /data1/szq/Val/decoupled \
  --decoupled_mask_dir /data1/szq/Val/decoupled-masks \
  --oblique_input /data1/szq/Val/Oblique \
  --oblique_gt /data1/szq/Val/Oblique \
  --oblique_mask_dir /data1/szq/Val/Oblique-masks \
  --wild_input /data1/szq/Val/Wild \
  --wild_gt /data1/szq/Val/Wild
```

## Results

LoRA seed 333 after 1000 training iterations (checkpoint index 999). ↓ lower is better; ↑ higher is better. Wild a1/a2/a3 use thresholds 1.25, 1.25² and 1.25³. Original report precision is retained. All 266 result rows are included; full scene tables expand below.

### Overall: Decoupled and Oblique

| Dataset | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Decoupled | 4625 | 0.1262 | 14.992 | 0.479 | 0.852 | 0.034 | 0.900 | 0.094 |
| Oblique | 1417 | 0.1261 | 17.813 | 0.510 | 0.835 | 0.049 | 0.881 | 0.101 |

### Overall: Wild (pseudo-GT)

| Depth range | N | AbsRel ↓ | RMSE ↓ | a1 ↑ | a2 ↑ | a3 ↑ | N-RMSE ↓ | SI-Log ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 - 400m | 1170 | 0.2125 | 45.335 | 0.535 | 0.853 | 0.960 | 0.1950 | 0.072 |
| 0 - 800m | 1184 | 0.2213 | 73.365 | 0.525 | 0.842 | 0.949 | 0.1617 | 0.101 |

### Decoupled: scene groups

| Subset / scene | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cleaned_Dataset_Campus | 880 | 0.1186 | 12.723 | 0.452 | 0.869 | 0.059 | 0.946 | 0.076 |
| Cleaned_Dataset_Factory | 948 | 0.1036 | 11.866 | 0.574 | 0.917 | 0.049 | 0.928 | 0.111 |
| Cleaned_Dataset_Farm | 946 | 0.2138 | 27.212 | 0.246 | 0.578 | 0.019 | 0.796 | 0.107 |
| Cleaned_Dataset_Gress | 1851 | 0.0967 | 11.427 | 0.563 | 0.950 | 0.024 | 0.917 | 0.086 |

### Decoupled: pitch

| Subset / scene | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| High Oblique (-45 to -60) | 1169 | 0.1208 | 16.667 | 0.506 | 0.868 | 0.038 | 0.985 | 0.079 |
| Horizontal (>-45) | 561 | 0.1135 | 17.702 | 0.534 | 0.870 | 0.039 | 0.989 | 0.084 |
| Nadir (-90) | 1130 | 0.1367 | 13.624 | 0.416 | 0.831 | 0.030 | 0.666 | 0.131 |
| Oblique (-60 to -85) | 1765 | 0.1272 | 13.897 | 0.485 | 0.849 | 0.033 | 0.964 | 0.082 |

### Decoupled: altitude

| Subset / scene | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| High (>120m) | 1972 | 0.1222 | 17.669 | 0.455 | 0.854 | 0.029 | 0.894 | 0.098 |
| Mid (60-120m) | 2653 | 0.1292 | 13.002 | 0.498 | 0.850 | 0.038 | 0.904 | 0.090 |

### Oblique: categories

| Subset / scene | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [CAT] City | 842 | 0.1065 | 20.286 | 0.552 | 0.883 | 0.048 | 0.956 | 0.075 |
| [CAT] Natural | 297 | 0.1555 | 12.477 | 0.452 | 0.773 | 0.059 | 0.723 | 0.149 |
| [CAT] Rural | 278 | 0.1541 | 16.022 | 0.443 | 0.757 | 0.040 | 0.822 | 0.131 |

<details>
<summary>Oblique: all scenes</summary>

| Subset / scene | N | AbsRel ↓ | RMSE ↓ | a1.10 ↑ | a1.25 ↑ | SI-Log ↓ | Spear ↑ | N-RMSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interval5_HKairport01_cropped_downsampled | 6 | 0.0441 | 4.318 | 0.926 | 0.998 | 0.036 | 0.852 | 0.138 |
| park0 | 2 | 0.0491 | 2.451 | 0.993 | 1.000 | 0.024 | 0.317 | 0.193 |
| yingrenshi | 49 | 0.0504 | 6.780 | 0.859 | 0.995 | 0.049 | 0.993 | 0.039 |
| upper | 50 | 0.0538 | 9.480 | 0.806 | 0.998 | 0.038 | 0.915 | 0.072 |
| interval5_HKairport_GNSS02_cropped_downsampled | 5 | 0.0584 | 5.863 | 0.808 | 0.991 | 0.046 | 0.794 | 0.192 |
| hav | 25 | 0.0605 | 11.843 | 0.791 | 1.000 | 0.026 | 0.995 | 0.039 |
| park14 | 10 | 0.0617 | 4.021 | 0.886 | 1.000 | 0.019 | 0.212 | 0.091 |
| interval5_HKairport02_cropped_downsampled | 10 | 0.0626 | 6.080 | 0.785 | 0.994 | 0.047 | 0.825 | 0.130 |
| lfls2 | 69 | 0.0727 | 11.578 | 0.676 | 0.983 | 0.029 | 0.956 | 0.064 |
| ODM2-output | 30 | 0.0780 | 9.202 | 0.649 | 0.982 | 0.040 | 0.964 | 0.064 |
| interval5_HKairport03_cropped_downsampled | 6 | 0.0785 | 7.724 | 0.623 | 0.986 | 0.052 | 0.759 | 0.146 |
| interval5_HKairport_GNSS03_cropped_downsampled | 5 | 0.0786 | 7.203 | 0.749 | 0.992 | 0.042 | 0.855 | 0.166 |
| sziit | 88 | 0.0792 | 15.068 | 0.658 | 0.975 | 0.035 | 0.974 | 0.049 |
| SMBU | 44 | 0.0811 | 17.368 | 0.600 | 0.994 | 0.036 | 0.928 | 0.071 |
| bellus-output | 20 | 0.0824 | 11.550 | 0.669 | 0.971 | 0.067 | 0.757 | 0.148 |
| ODM1-output | 30 | 0.0825 | 9.948 | 0.606 | 0.961 | 0.046 | 0.949 | 0.068 |
| interval5_AMvalley03_cropped_downsampled | 7 | 0.0851 | 11.426 | 0.611 | 0.929 | 0.045 | 0.836 | 0.148 |
| R-PHD-output | 30 | 0.0876 | 9.242 | 0.606 | 0.977 | 0.025 | 0.790 | 0.129 |
| interval5_HKairport_GNSS_Evening_cropped_downsampled | 9 | 0.0892 | 7.957 | 0.532 | 0.982 | 0.040 | 0.736 | 0.194 |
| interval5_HKairport_GNSS01_cropped_downsampled | 8 | 0.0898 | 7.931 | 0.532 | 0.992 | 0.034 | 0.746 | 0.199 |
| park8 | 38 | 0.0946 | 3.880 | 0.614 | 0.940 | 0.076 | 0.666 | 0.171 |
| lfls | 51 | 0.1011 | 18.005 | 0.433 | 0.980 | 0.033 | 0.951 | 0.069 |
| interval5_AMtown03_cropped_downsampled | 5 | 0.1054 | 8.588 | 0.308 | 0.996 | 0.048 | 0.648 | 0.120 |
| longhua | 236 | 0.1060 | 25.809 | 0.557 | 0.896 | 0.051 | 0.943 | 0.082 |
| ODM6-output | 30 | 0.1078 | 14.472 | 0.531 | 0.896 | 0.061 | 0.899 | 0.109 |
| interval5_HKisland03_cropped_downsampled | 2 | 0.1091 | 8.410 | 0.523 | 0.964 | 0.055 | 0.781 | 0.142 |
| interval5_HKisland_GNSS03_cropped_downsampled | 2 | 0.1129 | 8.432 | 0.560 | 0.951 | 0.050 | 0.700 | 0.187 |
| ainterval5_AMtown02_cropped_downsampled | 10 | 0.1136 | 9.329 | 0.455 | 0.937 | 0.046 | 0.512 | 0.173 |
| interval5_AMvalley02_cropped_downsampled | 8 | 0.1168 | 14.490 | 0.433 | 0.840 | 0.047 | 0.887 | 0.159 |
| interval5_HKisland01_cropped_downsampled | 10 | 0.1253 | 8.805 | 0.457 | 0.922 | 0.058 | 0.802 | 0.127 |
| Artsci | 60 | 0.1254 | 18.976 | 0.464 | 0.801 | 0.050 | 0.969 | 0.087 |
| ainterval5_HKisland02_cropped_downsampled | 4 | 0.1268 | 9.246 | 0.545 | 0.836 | 0.060 | 0.586 | 0.173 |
| interval5_HKisland_GNSS02_cropped_downsampled | 5 | 0.1314 | 9.218 | 0.490 | 0.828 | 0.054 | 0.830 | 0.101 |
| BC2 | 24 | 0.1316 | 10.356 | 0.399 | 0.685 | 0.020 | 0.931 | 0.110 |
| sztu | 37 | 0.1366 | 29.862 | 0.337 | 0.796 | 0.032 | 0.971 | 0.097 |
| interval5_AMvalley01_cropped_downsampled | 24 | 0.1376 | 15.893 | 0.385 | 0.771 | 0.046 | 0.784 | 0.171 |
| ainterval5_AMtown01_cropped_downsampled | 24 | 0.1380 | 11.215 | 0.334 | 0.724 | 0.043 | 0.445 | 0.192 |
| lewis-output | 20 | 0.1539 | 31.376 | 0.364 | 0.693 | 0.079 | 0.995 | 0.061 |
| park5 | 1 | 0.1598 | 6.047 | 0.032 | 0.933 | 0.073 | 0.547 | 0.173 |
| interval5_HKisland_GNSS01_cropped_downsampled | 10 | 0.1627 | 11.316 | 0.375 | 0.740 | 0.056 | 0.891 | 0.098 |
| BC1 | 26 | 0.1645 | 13.726 | 0.220 | 0.637 | 0.021 | 0.905 | 0.115 |
| park10 | 11 | 0.1788 | 7.386 | 0.112 | 0.825 | 0.046 | 0.853 | 0.125 |
| polytech | 133 | 0.1855 | 28.851 | 0.283 | 0.625 | 0.083 | 0.963 | 0.099 |
| interval5_HKisland_GNSS_Evening_cropped_downsampled | 15 | 0.1915 | 12.852 | 0.317 | 0.689 | 0.051 | 0.929 | 0.125 |
| park13 | 28 | 0.1972 | 14.197 | 0.355 | 0.607 | 0.101 | 0.775 | 0.252 |
| caliterra-output | 20 | 0.2073 | 14.922 | 0.206 | 0.474 | 0.052 | 0.826 | 0.170 |
| sceneca-output | 20 | 0.2275 | 14.859 | 0.255 | 0.568 | 0.040 | 0.565 | 0.225 |
| park9 | 30 | 0.3171 | 11.682 | 0.346 | 0.443 | 0.037 | 0.318 | 0.112 |
| L1 | 30 | 0.5092 | 65.324 | 0.000 | 0.000 | 0.059 | 0.913 | 0.145 |

</details>

<details>
<summary>Wild: all scenes, 0 - 400m</summary>

| Scene | N | AbsRel ↓ | RMSE ↓ | a1 ↑ | a2 ↑ | a3 ↑ | N-RMSE ↓ | SI-Log ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| videoplayback8_scene_035 | 15 | 0.0212 | 9.140 | 0.996 | 1.000 | 1.000 | 0.0305 | 0.036 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_538 | 7 | 0.0310 | 4.758 | 1.000 | 1.000 | 1.000 | 0.0752 | 0.018 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_297 | 7 | 0.0324 | 7.995 | 0.999 | 1.000 | 1.000 | 0.0327 | 0.029 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_303 | 7 | 0.0332 | 12.552 | 0.993 | 0.999 | 1.000 | 0.0400 | 0.046 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_869 | 7 | 0.0424 | 9.579 | 1.000 | 1.000 | 1.000 | 0.0402 | 0.044 |
| videoplayback8_scene_022 | 13 | 0.0455 | 10.854 | 0.988 | 0.998 | 1.000 | 0.0319 | 0.056 |
| CHICHESTER_STATION___Clip_Pack_292___Chichester_City_Centre_Drone_Stock_Footage_29_mB9fTvxqYxg_scene_000 | 40 | 0.0462 | 15.620 | 0.978 | 0.993 | 0.996 | 0.0804 | 0.086 |
| scene_068 | 6 | 0.0479 | 21.196 | 1.000 | 1.000 | 1.000 | 0.3334 | 0.031 |
| videoplayback8_scene_021 | 12 | 0.0486 | 15.817 | 0.986 | 0.999 | 1.000 | 0.0510 | 0.049 |
| videoplayback8_scene_023 | 6 | 0.0529 | 12.463 | 0.992 | 0.999 | 1.000 | 0.0368 | 0.052 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_004 | 21 | 0.0539 | 7.212 | 1.000 | 1.000 | 1.000 | 0.0417 | 0.021 |
| scene_041 | 10 | 0.0539 | 16.517 | 0.994 | 0.998 | 1.000 | 0.0505 | 0.048 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_243 | 7 | 0.0624 | 19.447 | 0.914 | 0.999 | 1.000 | 0.0552 | 0.093 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_280 | 7 | 0.0686 | 19.336 | 0.980 | 0.998 | 1.000 | 0.0571 | 0.057 |
| videoplayback5_scene_013 | 7 | 0.0701 | 16.288 | 0.966 | 1.000 | 1.000 | 0.0588 | 0.077 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_932 | 8 | 0.0780 | 22.944 | 0.958 | 0.998 | 1.000 | 0.0666 | 0.088 |
| videoplayback8_scene_036 | 16 | 0.0836 | 15.998 | 0.974 | 0.997 | 0.999 | 0.0435 | 0.085 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_445 | 7 | 0.0897 | 14.784 | 1.000 | 1.000 | 1.000 | 0.1026 | 0.030 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_367 | 8 | 0.0917 | 5.735 | 0.999 | 1.000 | 1.000 | 0.0665 | 0.026 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_368 | 5 | 0.0977 | 8.264 | 0.995 | 0.999 | 1.000 | 0.0954 | 0.037 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_302 | 7 | 0.0999 | 22.319 | 0.980 | 0.999 | 1.000 | 0.0723 | 0.045 |
| videoplayback5_scene_014 | 7 | 0.1106 | 22.075 | 0.993 | 1.000 | 1.000 | 0.0651 | 0.037 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_401 | 7 | 0.1140 | 34.331 | 0.996 | 1.000 | 1.000 | 0.1362 | 0.039 |
| Anfield_Stadium_Development_Liverpool_FC_Phase_2_Anfield_Road_March_2022_Drone_Footage_D0H2_Z6uDB4_scene_002 | 11 | 0.1198 | 21.566 | 0.983 | 1.000 | 1.000 | 0.0685 | 0.031 |
| videoplayback8_scene_045 | 25 | 0.1221 | 36.506 | 0.824 | 0.978 | 0.996 | 0.1052 | 0.140 |
| scene_039 | 13 | 0.1231 | 38.853 | 0.957 | 0.996 | 0.999 | 0.2533 | 0.026 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_158 | 7 | 0.1238 | 43.620 | 0.741 | 0.846 | 0.991 | 0.1502 | 0.204 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_035 | 32 | 0.1239 | 29.450 | 0.874 | 0.998 | 1.000 | 0.0958 | 0.066 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_627 | 7 | 0.1240 | 26.968 | 0.960 | 0.999 | 1.000 | 0.0978 | 0.051 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_284 | 7 | 0.1254 | 16.924 | 0.900 | 0.999 | 1.000 | 0.0486 | 0.075 |
| videoplayback8_scene_016 | 13 | 0.1273 | 28.929 | 0.993 | 1.000 | 1.000 | 0.1053 | 0.024 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_036 | 34 | 0.1289 | 41.772 | 0.884 | 1.000 | 1.000 | 0.1531 | 0.076 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_025 | 7 | 0.1325 | 19.515 | 0.957 | 0.999 | 1.000 | 0.0562 | 0.054 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_010 | 7 | 0.1368 | 25.186 | 0.949 | 1.000 | 1.000 | 0.0760 | 0.041 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_313 | 7 | 0.1426 | 8.692 | 0.984 | 1.000 | 1.000 | 0.1302 | 0.038 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_103 | 7 | 0.1443 | 35.081 | 0.853 | 0.992 | 0.999 | 0.1113 | 0.080 |
| Orbit_of_Manchester_United_Old_Trafford_Football_Stadium_Stock_Video_Footage_Clip_pack_24_D9_BYX6Rw1_gLo_scene_000 | 40 | 0.1499 | 30.469 | 0.871 | 0.997 | 1.000 | 0.1022 | 0.044 |
| videoplayback8_scene_025 | 12 | 0.1505 | 35.849 | 0.798 | 1.000 | 1.000 | 0.1389 | 0.019 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_275 | 7 | 0.1519 | 19.410 | 0.976 | 1.000 | 1.000 | 0.0662 | 0.041 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_089 | 7 | 0.1533 | 12.413 | 0.887 | 0.981 | 0.995 | 0.0736 | 0.105 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_400 | 7 | 0.1642 | 56.515 | 0.541 | 1.000 | 1.000 | 0.2266 | 0.108 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_235 | 7 | 0.1646 | 19.811 | 0.924 | 0.997 | 0.999 | 0.1190 | 0.042 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_017 | 24 | 0.1691 | 42.175 | 0.676 | 0.998 | 1.000 | 0.1397 | 0.064 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_128 | 7 | 0.1733 | 27.021 | 0.610 | 0.997 | 1.000 | 0.1393 | 0.080 |
| Norwich_Cathedral_Drone_Stock_Footage_f9BLyX_st6c_scene_000 | 36 | 0.1786 | 35.941 | 0.613 | 0.959 | 0.995 | 0.1048 | 0.119 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_030 | 26 | 0.1835 | 33.459 | 0.585 | 0.977 | 0.991 | 0.1015 | 0.132 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_319 | 7 | 0.1844 | 41.602 | 0.689 | 0.999 | 1.000 | 0.1401 | 0.045 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_249 | 7 | 0.1909 | 31.080 | 0.667 | 1.000 | 1.000 | 0.1959 | 0.024 |
| Nottingham_Council_House_Drone_Stock_Footage_OAm0PPiLZwc_scene_000 | 30 | 0.1976 | 42.528 | 0.489 | 1.000 | 1.000 | 0.1348 | 0.051 |
| scene_025 | 22 | 0.1987 | 55.759 | 0.653 | 0.688 | 0.956 | 0.2624 | 0.062 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_184 | 7 | 0.2122 | 51.232 | 0.466 | 0.998 | 1.000 | 0.1693 | 0.071 |
| videoplayback8_scene_017 | 13 | 0.2138 | 50.406 | 0.310 | 0.999 | 1.000 | 0.1881 | 0.040 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_306 | 7 | 0.2148 | 46.406 | 0.423 | 0.976 | 0.998 | 0.1613 | 0.081 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_001 | 4 | 0.2165 | 36.671 | 0.506 | 0.977 | 0.988 | 0.1140 | 0.131 |
| EPIC_REVEAL___Clip_Pack_292___Chichester_City_Centre_Drone_Stock_Footage_9_BUxiJCgyZIk_scene_000 | 17 | 0.2196 | 23.937 | 0.538 | 0.901 | 0.985 | 0.0686 | 0.167 |
| Nottingham_Trent_Bridge_Cricket_Ground_Drone_Stock_Footage_nSMBAHFn9lc_scene_000 | 25 | 0.2257 | 51.181 | 0.225 | 0.997 | 1.000 | 0.1853 | 0.031 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_731 | 5 | 0.2340 | 65.696 | 0.323 | 1.000 | 1.000 | 0.3674 | 0.093 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_432 | 7 | 0.2380 | 50.683 | 0.181 | 0.992 | 0.998 | 0.1659 | 0.064 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_005 | 7 | 0.2420 | 79.016 | 0.029 | 1.000 | 1.000 | 0.5064 | 0.011 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1039 | 6 | 0.2551 | 45.633 | 0.115 | 0.919 | 0.999 | 0.1723 | 0.083 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_044 | 7 | 0.2559 | 63.237 | 0.004 | 0.999 | 1.000 | 0.3958 | 0.028 |
| videoplayback5_scene_010 | 7 | 0.2609 | 38.160 | 0.297 | 1.000 | 1.000 | 0.1429 | 0.046 |
| videoplayback8_scene_038 | 14 | 0.2682 | 71.854 | 0.021 | 0.991 | 0.999 | 0.2694 | 0.045 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_891 | 8 | 0.2737 | 73.563 | 0.004 | 0.999 | 1.000 | 0.2847 | 0.030 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_262 | 6 | 0.2741 | 58.298 | 0.022 | 0.985 | 0.996 | 0.1602 | 0.086 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_330 | 7 | 0.2757 | 56.730 | 0.045 | 0.902 | 0.994 | 0.1895 | 0.085 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_701 | 5 | 0.2874 | 73.301 | 0.006 | 0.994 | 1.000 | 0.2950 | 0.026 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_370 | 7 | 0.2893 | 40.068 | 0.111 | 0.744 | 0.991 | 0.1143 | 0.108 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_009 | 7 | 0.2947 | 59.545 | 0.054 | 0.880 | 0.996 | 0.1808 | 0.153 |
| Warwick_Castle_Drone_Stock_Footage_VfU63IOpmVQ_scene_000 | 27 | 0.2987 | 65.649 | 0.054 | 0.801 | 0.997 | 0.1991 | 0.096 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_320 | 7 | 0.2994 | 63.596 | 0.038 | 0.842 | 0.999 | 0.1993 | 0.084 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_355 | 8 | 0.2997 | 60.820 | 0.010 | 0.891 | 1.000 | 0.1943 | 0.040 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_040 | 7 | 0.2999 | 56.299 | 0.002 | 0.994 | 1.000 | 0.2151 | 0.035 |
| Carrow_Road_Stock_Footage___Norwich_City_Football_Club_zFeFd19qHdg_scene_000 | 48 | 0.3022 | 64.640 | 0.003 | 0.980 | 0.999 | 0.2197 | 0.032 |
| scene_097 | 9 | 0.3026 | 48.416 | 0.027 | 0.925 | 0.997 | 0.1727 | 0.071 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1014 | 8 | 0.3110 | 61.554 | 0.001 | 0.967 | 0.999 | 0.2120 | 0.034 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_686 | 5 | 0.3132 | 64.437 | 0.322 | 0.440 | 0.931 | 0.1727 | 0.265 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_839 | 7 | 0.3264 | 41.845 | 0.000 | 0.905 | 0.999 | 0.2845 | 0.043 |
| scene_098 | 9 | 0.3316 | 58.969 | 0.005 | 0.796 | 0.992 | 0.1806 | 0.095 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_036 | 7 | 0.3330 | 68.110 | 0.001 | 0.855 | 1.000 | 0.2189 | 0.040 |
| scene_106 | 8 | 0.3361 | 105.840 | 0.005 | 0.692 | 0.999 | 0.7398 | 0.064 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_868 | 7 | 0.3421 | 42.606 | 0.045 | 0.646 | 0.996 | 0.1309 | 0.088 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_088 | 7 | 0.3447 | 65.654 | 0.002 | 0.766 | 0.995 | 0.2187 | 0.072 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_890 | 8 | 0.3452 | 44.055 | 0.019 | 0.609 | 0.963 | 0.1246 | 0.104 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_207 | 7 | 0.3520 | 48.278 | 0.001 | 0.589 | 0.999 | 0.1339 | 0.060 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1003 | 7 | 0.3539 | 80.756 | 0.258 | 0.491 | 0.721 | 0.3516 | 0.320 |
| Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_004 | 6 | 0.3603 | 72.525 | 0.241 | 0.319 | 0.816 | 0.2532 | 0.268 |
| videoplayback2_scene_002 | 5 | 0.3708 | 52.991 | 0.028 | 0.292 | 0.994 | 0.1699 | 0.095 |
| videoplayback8_scene_018 | 12 | 0.3760 | 100.368 | 0.002 | 0.219 | 0.997 | 0.4135 | 0.053 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_020 | 29 | 0.3862 | 3.259 | 0.886 | 0.972 | 0.981 | 0.0465 | 0.180 |
| scene_034 | 12 | 0.3984 | 139.199 | 0.002 | 0.208 | 0.955 | 1.2334 | 0.062 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_043 | 7 | 0.4033 | 92.573 | 0.001 | 0.048 | 0.995 | 0.3465 | 0.046 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_031 | 20 | 0.4126 | 102.197 | 0.000 | 0.028 | 0.996 | 0.4066 | 0.037 |
| scene_104 | 7 | 0.4197 | 145.648 | 0.055 | 0.241 | 0.701 | 1.0883 | 0.125 |
| scene_019 | 11 | 0.4386 | 157.301 | 0.002 | 0.262 | 0.577 | 1.5447 | 0.035 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_016 | 5 | 0.4764 | 98.437 | 0.002 | 0.023 | 0.553 | 0.4109 | 0.079 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_441 | 7 | 0.4789 | 60.241 | 0.001 | 0.001 | 0.683 | 0.1816 | 0.057 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_041 | 21 | 0.4876 | 88.727 | 0.002 | 0.007 | 0.532 | 0.2660 | 0.095 |
| Wembley_Stadium_Drone_Stock_Footage_bQ6i1otGOfE_scene_000 | 16 | 0.4996 | 144.486 | 0.002 | 0.002 | 0.290 | 0.8345 | 0.051 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_022 | 7 | 0.5226 | 85.983 | 0.001 | 0.001 | 0.084 | 0.2704 | 0.056 |

</details>

<details>
<summary>Wild: all scenes, 0 - 800m</summary>

| Scene | N | AbsRel ↓ | RMSE ↓ | a1 ↑ | a2 ↑ | a3 ↑ | N-RMSE ↓ | SI-Log ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| videoplayback8_scene_035 | 15 | 0.0214 | 9.399 | 0.996 | 1.000 | 1.000 | 0.0281 | 0.036 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_538 | 7 | 0.0310 | 4.758 | 1.000 | 1.000 | 1.000 | 0.0752 | 0.018 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_297 | 7 | 0.0324 | 7.995 | 0.999 | 1.000 | 1.000 | 0.0327 | 0.029 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_303 | 7 | 0.0348 | 16.361 | 0.991 | 0.999 | 1.000 | 0.0312 | 0.051 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_869 | 7 | 0.0424 | 9.579 | 1.000 | 1.000 | 1.000 | 0.0402 | 0.044 |
| CHICHESTER_STATION___Clip_Pack_292___Chichester_City_Centre_Drone_Stock_Footage_29_mB9fTvxqYxg_scene_000 | 40 | 0.0472 | 18.643 | 0.978 | 0.993 | 0.996 | 0.0958 | 0.090 |
| videoplayback8_scene_021 | 12 | 0.0503 | 19.561 | 0.986 | 0.999 | 1.000 | 0.0349 | 0.053 |
| videoplayback8_scene_022 | 13 | 0.0510 | 28.270 | 0.980 | 0.997 | 0.999 | 0.0382 | 0.071 |
| scene_068 | 6 | 0.0520 | 30.761 | 1.000 | 1.000 | 1.000 | 0.0664 | 0.037 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_004 | 21 | 0.0539 | 7.212 | 1.000 | 1.000 | 1.000 | 0.0417 | 0.021 |
| scene_041 | 10 | 0.0560 | 19.782 | 0.992 | 0.998 | 1.000 | 0.0418 | 0.051 |
| videoplayback8_scene_023 | 6 | 0.0564 | 19.876 | 0.990 | 0.999 | 1.000 | 0.0282 | 0.057 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_243 | 7 | 0.0669 | 33.539 | 0.911 | 0.996 | 0.998 | 0.0446 | 0.103 |
| videoplayback5_scene_013 | 7 | 0.0701 | 16.288 | 0.966 | 1.000 | 1.000 | 0.0588 | 0.077 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_280 | 7 | 0.0815 | 42.982 | 0.946 | 0.997 | 0.999 | 0.0674 | 0.091 |
| videoplayback8_scene_036 | 16 | 0.0866 | 21.885 | 0.962 | 0.997 | 0.999 | 0.0549 | 0.089 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_445 | 7 | 0.0897 | 14.784 | 1.000 | 1.000 | 1.000 | 0.1026 | 0.030 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_367 | 8 | 0.0917 | 5.735 | 0.999 | 1.000 | 1.000 | 0.0665 | 0.026 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_368 | 5 | 0.0977 | 8.264 | 0.995 | 0.999 | 1.000 | 0.0954 | 0.037 |
| videoplayback5_scene_014 | 7 | 0.1106 | 22.383 | 0.994 | 1.000 | 1.000 | 0.0609 | 0.037 |
| scene_039 | 13 | 0.1176 | 38.739 | 0.973 | 0.998 | 1.000 | 0.1367 | 0.029 |
| Anfield_Stadium_Development_Liverpool_FC_Phase_2_Anfield_Road_March_2022_Drone_Footage_D0H2_Z6uDB4_scene_002 | 11 | 0.1184 | 25.714 | 0.984 | 1.000 | 1.000 | 0.0474 | 0.034 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_401 | 7 | 0.1215 | 55.982 | 0.993 | 1.000 | 1.000 | 0.0858 | 0.039 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_302 | 7 | 0.1232 | 65.546 | 0.898 | 0.995 | 0.999 | 0.0925 | 0.117 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_284 | 7 | 0.1237 | 25.824 | 0.899 | 0.999 | 1.000 | 0.0395 | 0.096 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_627 | 7 | 0.1252 | 41.278 | 0.950 | 0.999 | 1.000 | 0.0614 | 0.054 |
| videoplayback8_scene_016 | 13 | 0.1273 | 29.162 | 0.993 | 1.000 | 1.000 | 0.0954 | 0.024 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_158 | 7 | 0.1288 | 52.237 | 0.741 | 0.842 | 0.987 | 0.1798 | 0.214 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_035 | 32 | 0.1337 | 55.865 | 0.837 | 0.982 | 0.995 | 0.0789 | 0.107 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_025 | 7 | 0.1372 | 41.548 | 0.935 | 0.995 | 0.997 | 0.0651 | 0.110 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_010 | 7 | 0.1397 | 44.264 | 0.913 | 0.999 | 1.000 | 0.0605 | 0.057 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_313 | 7 | 0.1426 | 8.692 | 0.984 | 1.000 | 1.000 | 0.1302 | 0.038 |
| videoplayback8_scene_045 | 25 | 0.1477 | 75.007 | 0.747 | 0.927 | 0.977 | 0.1183 | 0.183 |
| videoplayback8_scene_025 | 12 | 0.1510 | 51.588 | 0.801 | 1.000 | 1.000 | 0.0784 | 0.025 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_275 | 7 | 0.1519 | 19.410 | 0.976 | 1.000 | 1.000 | 0.0662 | 0.041 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_089 | 7 | 0.1537 | 13.798 | 0.887 | 0.981 | 0.995 | 0.0814 | 0.107 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_036 | 34 | 0.1549 | 77.145 | 0.728 | 0.999 | 1.000 | 0.1146 | 0.090 |
| Orbit_of_Manchester_United_Old_Trafford_Football_Stadium_Stock_Video_Footage_Clip_pack_24_D9_BYX6Rw1_gLo_scene_000 | 40 | 0.1554 | 55.621 | 0.835 | 0.995 | 0.999 | 0.0797 | 0.071 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_932 | 8 | 0.1566 | 116.137 | 0.877 | 0.935 | 0.956 | 0.2311 | 0.215 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_103 | 7 | 0.1567 | 59.057 | 0.802 | 0.974 | 0.998 | 0.0826 | 0.110 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_235 | 7 | 0.1646 | 19.811 | 0.924 | 0.997 | 0.999 | 0.1190 | 0.042 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_030 | 26 | 0.1709 | 50.984 | 0.641 | 0.977 | 0.991 | 0.0700 | 0.162 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_128 | 7 | 0.1733 | 27.021 | 0.610 | 0.997 | 1.000 | 0.1393 | 0.080 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_017 | 24 | 0.1779 | 58.039 | 0.637 | 0.988 | 0.999 | 0.0868 | 0.080 |
| Norwich_Cathedral_Drone_Stock_Footage_f9BLyX_st6c_scene_000 | 36 | 0.1870 | 62.151 | 0.578 | 0.954 | 0.992 | 0.0870 | 0.153 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_249 | 7 | 0.1909 | 31.080 | 0.667 | 1.000 | 1.000 | 0.1959 | 0.024 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_731 | 5 | 0.1924 | 95.065 | 0.540 | 0.975 | 0.992 | 0.1643 | 0.194 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_400 | 7 | 0.1958 | 92.158 | 0.449 | 0.934 | 1.000 | 0.1802 | 0.138 |
| Nottingham_Council_House_Drone_Stock_Footage_OAm0PPiLZwc_scene_000 | 30 | 0.1970 | 67.665 | 0.508 | 0.998 | 1.000 | 0.0946 | 0.088 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_319 | 7 | 0.1976 | 77.433 | 0.574 | 0.998 | 1.000 | 0.1111 | 0.062 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_001 | 4 | 0.1979 | 54.193 | 0.579 | 0.980 | 0.989 | 0.0751 | 0.161 |
| videoplayback8_scene_017 | 13 | 0.2159 | 73.537 | 0.317 | 0.997 | 0.999 | 0.1101 | 0.057 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_306 | 7 | 0.2187 | 69.559 | 0.416 | 0.971 | 0.994 | 0.1012 | 0.091 |
| EPIC_REVEAL___Clip_Pack_292___Chichester_City_Centre_Drone_Stock_Footage_9_BUxiJCgyZIk_scene_000 | 17 | 0.2258 | 32.863 | 0.535 | 0.895 | 0.981 | 0.0885 | 0.183 |
| Nottingham_Trent_Bridge_Cricket_Ground_Drone_Stock_Footage_nSMBAHFn9lc_scene_000 | 25 | 0.2267 | 78.386 | 0.231 | 0.996 | 0.999 | 0.1159 | 0.050 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_184 | 7 | 0.2289 | 79.087 | 0.386 | 0.994 | 1.000 | 0.1470 | 0.083 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_005 | 7 | 0.2388 | 93.575 | 0.052 | 1.000 | 1.000 | 0.2644 | 0.013 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_432 | 7 | 0.2504 | 93.357 | 0.174 | 0.947 | 0.997 | 0.1323 | 0.104 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1039 | 6 | 0.2551 | 45.633 | 0.115 | 0.919 | 0.999 | 0.1723 | 0.083 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_044 | 7 | 0.2559 | 63.237 | 0.004 | 0.999 | 1.000 | 0.3958 | 0.028 |
| videoplayback5_scene_010 | 7 | 0.2614 | 38.533 | 0.294 | 1.000 | 1.000 | 0.1443 | 0.046 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_262 | 6 | 0.2661 | 82.273 | 0.062 | 0.986 | 0.996 | 0.1077 | 0.099 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_330 | 7 | 0.2807 | 84.920 | 0.053 | 0.873 | 0.993 | 0.1214 | 0.105 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_891 | 8 | 0.2835 | 112.332 | 0.011 | 0.973 | 0.999 | 0.1706 | 0.064 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_701 | 5 | 0.2838 | 104.115 | 0.007 | 0.993 | 1.000 | 0.1606 | 0.029 |
| videoplayback8_scene_038 | 14 | 0.2875 | 124.175 | 0.017 | 0.870 | 0.999 | 0.1862 | 0.075 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_320 | 7 | 0.2922 | 91.929 | 0.102 | 0.832 | 0.995 | 0.1280 | 0.156 |
| Carrow_Road_Stock_Footage___Norwich_City_Football_Club_zFeFd19qHdg_scene_000 | 48 | 0.2959 | 97.611 | 0.042 | 0.967 | 0.999 | 0.1406 | 0.097 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_355 | 8 | 0.2976 | 83.008 | 0.019 | 0.893 | 0.999 | 0.1164 | 0.051 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_040 | 7 | 0.2999 | 56.299 | 0.002 | 0.994 | 1.000 | 0.2147 | 0.035 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1014 | 8 | 0.3016 | 96.289 | 0.051 | 0.954 | 0.999 | 0.1394 | 0.120 |
| scene_097 | 9 | 0.3026 | 48.416 | 0.027 | 0.925 | 0.997 | 0.1727 | 0.071 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_370 | 7 | 0.3040 | 77.722 | 0.122 | 0.731 | 0.976 | 0.1707 | 0.234 |
| Warwick_Castle_Drone_Stock_Footage_VfU63IOpmVQ_scene_000 | 27 | 0.3192 | 113.233 | 0.046 | 0.677 | 0.991 | 0.1552 | 0.116 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_868 | 7 | 0.3260 | 116.900 | 0.193 | 0.714 | 0.993 | 0.1645 | 0.306 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_839 | 7 | 0.3264 | 41.845 | 0.000 | 0.905 | 0.999 | 0.2845 | 0.043 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_009 | 7 | 0.3266 | 116.542 | 0.114 | 0.827 | 0.965 | 0.2023 | 0.276 |
| scene_098 | 9 | 0.3285 | 63.264 | 0.015 | 0.801 | 0.991 | 0.0883 | 0.111 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_686 | 5 | 0.3310 | 109.579 | 0.291 | 0.437 | 0.916 | 0.1726 | 0.325 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_036 | 7 | 0.3330 | 68.123 | 0.001 | 0.855 | 1.000 | 0.2171 | 0.040 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_088 | 7 | 0.3339 | 103.458 | 0.013 | 0.777 | 0.992 | 0.1478 | 0.126 |
| scene_104 | 7 | 0.3348 | 149.292 | 0.239 | 0.511 | 0.839 | 0.2830 | 0.263 |
| videoplayback2_scene_002 | 5 | 0.3378 | 70.552 | 0.122 | 0.423 | 0.995 | 0.1300 | 0.168 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_890 | 8 | 0.3395 | 73.890 | 0.068 | 0.637 | 0.958 | 0.1227 | 0.203 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_207 | 7 | 0.3460 | 73.306 | 0.028 | 0.613 | 0.998 | 0.1111 | 0.158 |
| Wembley_Stadium__Drone_Stock_Footage_4K_iNW_9vUCFRg_scene_004 | 6 | 0.3501 | 88.733 | 0.249 | 0.385 | 0.835 | 0.1566 | 0.289 |
| scene_025 | 29 | 0.3528 | 215.449 | 0.494 | 0.603 | 0.740 | 1.3244 | 0.106 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_1003 | 7 | 0.3576 | 82.786 | 0.253 | 0.488 | 0.722 | 0.3436 | 0.327 |
| videoplayback8_scene_018 | 12 | 0.3817 | 158.893 | 0.001 | 0.188 | 0.998 | 0.2472 | 0.052 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_020 | 29 | 0.3862 | 3.259 | 0.886 | 0.972 | 0.981 | 0.0465 | 0.180 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_043 | 7 | 0.4010 | 129.984 | 0.001 | 0.082 | 0.991 | 0.1948 | 0.061 |
| scene_034 | 12 | 0.4026 | 190.637 | 0.002 | 0.210 | 0.940 | 0.3707 | 0.079 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_016 | 5 | 0.4116 | 112.833 | 0.113 | 0.235 | 0.683 | 0.2325 | 0.245 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_031 | 20 | 0.4152 | 149.157 | 0.000 | 0.023 | 0.993 | 0.2290 | 0.042 |
| scene_019 | 11 | 0.4309 | 228.018 | 0.019 | 0.270 | 0.624 | 0.4451 | 0.042 |
| scene_106 | 15 | 0.4628 | 251.097 | 0.008 | 0.410 | 0.531 | 0.6151 | 0.055 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_441 | 7 | 0.4789 | 61.173 | 0.001 | 0.002 | 0.682 | 0.1472 | 0.059 |
| Wembley_Stadium_Drone_Stock_Footage_bQ6i1otGOfE_scene_000 | 16 | 0.4885 | 198.544 | 0.034 | 0.063 | 0.339 | 0.3464 | 0.255 |
| Best_of_Italy_8K_Ultra_HD_Drone_Video_kCmF1DzyZTI_scene_041 | 21 | 0.4922 | 178.783 | 0.024 | 0.049 | 0.462 | 0.2437 | 0.219 |
| TOP_40___Most_Beautiful_Countries_in_EUROPE_8K_ULTRA_HD_X69yHbtXncQ_scene_022 | 7 | 0.5030 | 105.585 | 0.029 | 0.052 | 0.167 | 0.1516 | 0.181 |

</details>

Report paths and SHA-256 hashes: `provenance/result_sources.json`. Run parameters: `provenance/training_run.json`. Validation passed for training CLI imports, Python/shell syntax, unchanged executable ASTs and all report rows.
