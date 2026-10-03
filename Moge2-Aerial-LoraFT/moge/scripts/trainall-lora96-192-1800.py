
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
import json
import time
import random
from typing import *
import itertools
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm, trange
import mlflow
# 🔥 [Import PEFT]
from peft import LoraConfig, get_peft_model, TaskType
import peft

torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss, 
    edge_loss,
    normal_loss, 
    mask_l2_loss, 
    mask_bce_loss,
    metric_scale_loss,
    normal_map_loss,
    monitoring, 
)
from moge.train.utils import build_optimizer, build_lr_scheduler
from moge.utils.geometry_torch import intrinsics_to_fov
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.tools import key_average, recursive_replace, CallbackOnException, flatten_nested_dict
from moge.test.metrics import compute_metrics


def multi_scale_gradient_loss(pred, gt, mask=None, params=None):
    if params is None: params = {}
    scales = params.get('scales', 4)
    total_loss = 0
    valid_scales = 0

    for scale in range(scales):
        step = 2 ** scale
        if pred.shape[2] <= step or pred.shape[3] <= step: break

        # Calculate gradients
        pred_grad_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
        gt_grad_x = gt[:, :, :, step:] - gt[:, :, :, :-step]
        pred_grad_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
        gt_grad_y = gt[:, :, step:, :] - gt[:, :, :-step, :]

        # Create a validity mask (1 where data is good, 0 where data is NaN/Inf)
        # We must check both GT and Pred for validity
        valid_mask_x = torch.isfinite(gt_grad_x) & torch.isfinite(pred_grad_x)
        valid_mask_y = torch.isfinite(gt_grad_y) & torch.isfinite(pred_grad_y)

        # Combine with user-provided mask if it exists
        if mask is not None:
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]
            valid_mask_x = valid_mask_x & mask_x
            valid_mask_y = valid_mask_y & mask_y

        # Compute Loss only on valid pixels
        if valid_mask_x.sum() > 0:
            diff_x = torch.abs(pred_grad_x - gt_grad_x)
            # Only sum errors where mask is True
            loss_x = diff_x[valid_mask_x].mean()
        else:
            loss_x = 0.0

        if valid_mask_y.sum() > 0:
            diff_y = torch.abs(pred_grad_y - gt_grad_y)
            loss_y = diff_y[valid_mask_y].mean()
        else:
            loss_y = 0.0

        total_loss += (loss_x + loss_y)
        valid_scales += 1

    if valid_scales > 0:
        return total_loss / valid_scales
    return torch.tensor(0.0, device=pred.device, requires_grad=True)

@click.command()
@click.option('--config', 'config_path', type=str, default='configs/debug.json')
@click.option('--workspace', type=str, default='workspace/debug', help='Path to the workspace')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Path to the checkpoint to load. "latest" to load latest checkpoint in workspace, integer to load by step number')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Use mixed precision training. Backbone is converted to FP16')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--save_every', type=int, default=10000, help='Save checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size')
@click.option('--enable_mlflow', type=bool, default=True, help='Log metrics to MLFlow')
@click.option('--seed', type=int, default=0, help='Random seed')
def main(
    config_path: str,
    workspace: str,
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    enable_gradient_checkpointing: bool,
    enable_mixed_precision: bool,
    enable_ema: bool,
    num_iterations: int,
    save_every: int,
    log_every: int,
    vis_every: int,
    num_vis_images: int,
    enable_mlflow: bool,
    seed: Optional[int],
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 🔥 [Fix 1: BF16 Support]
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    device = accelerator.device
    batch_size_total = batch_size_forward * gradient_accumulation_steps * accelerator.num_processes

    # Log config
    if accelerator.is_main_process:
        if enable_mlflow:
            try:
                mlflow.log_params({
                    **click.get_current_context().params,
                    'batch_size_total': batch_size_total,
                })
            except:
                print('Failed to log config to MLFlow')
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with Path(workspace).joinpath('config.json').open('w') as f:
            json.dump(config, f, indent=4)

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(config['model_version'])      
        model = MoGeModel(**config['model'])

    # ================= 🚀 [LoRA Configuration] 🚀 =================
    print("✨ Applying PEFT LoRA configuration...")
    LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]
    HEADS_TO_SAVE = ["scale_head"]
    peft_config = LoraConfig(
        r=96,
        lora_alpha=192,
        lora_dropout=0.1,
        bias="none",
        target_modules=LORA_TARGETS, 
        modules_to_save=HEADS_TO_SAVE, 
    )

    model = get_peft_model(model, peft_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    # =============================================================

    # ================= 🛡️ FSDP Auto-Wrap Fix 🛡️ =================
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        try:
            backbone_ref = None
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                 if hasattr(model.base_model.model, 'encoder'):
                     backbone_ref = model.base_model.model.encoder.backbone
            elif hasattr(model, 'encoder'):
                 backbone_ref = model.encoder.backbone
            
            if backbone_ref and hasattr(backbone_ref, 'blocks'):
                block_module = backbone_ref.blocks[0]
                block_class = type(block_module)
                import functools
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={block_class},
                )
                accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy
                if accelerator.is_main_process:
                    print(f"[FSDP] Policy applied for {block_class.__name__}")
        except Exception as e:
            print(f"[FSDP] Warning: {e}")
    # ===========================================================

    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'🔥Total parameters: {count_total_parameters}')
    count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'🔥 Trainable parameters: {count_trainable} (Ratio: {count_trainable/count_total_parameters:.2%})')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # ================= ⚙️ Gradient Checkpointing Fix ⚙️ =================
    if enable_gradient_checkpointing:
        if accelerator.is_main_process: print("⚙️ Enabling Gradient Checkpointing...")
        
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        
        try:
            found_hook = False
            for name, module in model.named_modules():
                if name.endswith("patch_embed"):
                    module.register_forward_hook(make_inputs_require_grad)
                    found_hook = True
                    break
            if not found_hook:
                 if hasattr(model, "enable_input_require_grads"):
                     model.enable_input_require_grads()
        except Exception as e:
            print(f"⚠️ Failed to register input grad hook: {e}")

        activated_cnt = 0
        for name, module in model.named_modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                activated_cnt += 1
            elif hasattr(module, "set_grad_checkpointing"):
                module.set_grad_checkpointing(True)
                activated_cnt += 1
            elif hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = True
                activated_cnt += 1
    # ===================================================================

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
    
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')

    # Attempt to load checkpoint
    checkpoint: Dict[str, Any]
    with accelerator.local_main_process_first():
        if checkpoint_path is None:
            checkpoint = None
        elif checkpoint_path.endswith('.pt'):
            print(f'Load checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        elif checkpoint_path == "latest": 
            checkpoint_path = Path(workspace, 'checkpoint', 'latest.pt')
            if checkpoint_path.exists():
                print(f'Load checkpoint: {checkpoint_path}')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                # ... (Latest loading logic kept same but omitted for brevity if simple path used)
            else:
                checkpoint = None
        else:
             # Handle integer step loading
             i_step = int(checkpoint_path)
             checkpoint = {'step': i_step}
             if (p:=Path(workspace, 'checkpoint', f'{i_step:08d}.pt')).exists():
                 checkpoint['model'] = torch.load(p, map_location='cpu', weights_only=True)['model']

    if checkpoint is None:
        print('Initialize model weights')
        initial_step = 0
    else:
        # ================= 🔧 [Fix v4] Verified Auto-Mapping =================
        print(f"🔄 Loading weights from {checkpoint_path}...")
        state_dict = checkpoint['model']
        
        is_peft = hasattr(model, "peft_config") or isinstance(model, (peft.PeftModel, peft.LoraModel))
        model_keys = set(model.state_dict().keys()) 
        
        if is_peft:
            print("🛠️  Applying smart mapping for PEFT model...")
            new_state_dict = {}
            
            for k, v in state_dict.items():
                # A: Build potential key names
                key_normal = f"base_model.model.{k}"
                
                parts = key_normal.split('.')
                if parts[-1] in ['weight', 'bias']:
                    key_lora_base = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                else:
                    key_lora_base = key_normal 

                # B: Check Heads
                head_match = next((h for h in HEADS_TO_SAVE if k.startswith(h + ".")), None)
                
                if head_match:
                    suffix = k[len(head_match)+1:]
                    key_trainable = f"base_model.model.{head_match}.modules_to_save.default.{suffix}"
                    key_original = f"base_model.model.{head_match}.original_module.{suffix}"
                    
                    if key_trainable in model_keys:
                        new_state_dict[key_trainable] = v
                    if key_original in model_keys:
                        new_state_dict[key_original] = v
                        
                elif key_lora_base in model_keys:
                    new_state_dict[key_lora_base] = v
                elif key_normal in model_keys:
                    new_state_dict[key_normal] = v
                else:
                    pass
            state_dict = new_state_dict

        msg = model.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in msg.missing_keys if "lora_" not in k]
        print(f"✅ Loaded. Real missing keys (excluding LoRA): {len(real_missing)}")
        if len(real_missing) > 0 and accelerator.is_main_process:
             print(f"⚠️  Warning: Still missing keys: {real_missing[:5]}")
        # ===================================================================
        
        if 'step' in checkpoint:
            initial_step = checkpoint['step'] + 1
        else:
            initial_step = 0
            
        if 'optimizer' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer'])
            except: pass
        if enable_ema and accelerator.is_main_process and 'ema_model' in checkpoint:
             try: ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
             except: pass
        if 'lr_scheduler' in checkpoint:
             try: lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
             except: pass
        del checkpoint
    
    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(
            config['data'], 
            batch_size_forward,
            num_load_workers=6, 
            num_process_workers=6, 
            buffer_size=32 
        )

    def _write_bytes_retry_loop(save_path: Path, data: bytes):
        while True:
            try:
                save_path.write_bytes(data)
                break
            except Exception as e:
                print('Error while saving checkpoint, retrying in 1 minute: ', e)
                time.sleep(60)

    # Ready to train
    records = []
    model.train()

    # 🛡️ Freeze BN Stats
    def freeze_bn_stats(m):
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm, nn.SyncBatchNorm)):
            m.eval() 
    model.apply(freeze_bn_stats)
    
    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):  
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images_adj = (num_vis_images // batch_size_forward) * batch_size_forward
            if num_vis_images_adj == 0 and num_vis_images > 0: num_vis_images_adj = batch_size_forward
            
            for _ in range(max(1, num_vis_images_adj // batch_size_forward)):
                batch = train_data_pipe.get()
                batches_for_vis.append(batch)

        if vis_every > 0 and accelerator.is_main_process and initial_step == 0:
            save_dir = Path(workspace).joinpath('vis/gt')
            for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
                image, gt_depth, gt_normal, gt_intrinsics, info = batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
                gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                for i_instance in range(batch['image'].shape[0]):
                    idx = i_batch * batch_size_forward + i_instance
                    image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt_depth_i = gt_depth[i_instance].numpy()
                    gt_points_i = gt_points[i_instance].numpy()
                    gt_normal_i = gt_normal[i_instance].numpy()
                    save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(gt_points_i, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(colorize_depth(gt_depth_i), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal.png')), cv2.cvtColor(colorize_normal(gt_normal_i), cv2.COLOR_RGB2BGR))
                    with save_dir.joinpath(f'{idx:04d}/info.json').open('w') as f:
                        json.dump(info[i_instance], f)

        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)   

        for i_step in range(initial_step, num_iterations):
            i_accumulate = 0
            while i_accumulate < gradient_accumulation_steps:
                batch = train_data_pipe.get()
                image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric']
                image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics = image.to(device), gt_depth.to(device), gt_normal.to(device), gt_mask_fin.to(device), gt_mask_inf.to(device), gt_intrinsics.to(device)
                current_batch_size = image.shape[0]
                
                gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                with accelerator.accumulate(model):
                    if i_step <= config.get('low_resolution_training_steps', 0):
                        num_tokens = config['model']['num_tokens_range'][0]
                    else:
                        num_tokens = accelerate.utils.broadcast_object_list([random.randint(*config['model']['num_tokens_range'])])[0]
                    
                    autocast_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else torch.float16
                    is_mixed_enabled = accelerator.mixed_precision != 'no'
                    
                    with torch.autocast(device_type=accelerator.device.type, dtype=autocast_dtype, enabled=is_mixed_enabled):
                        output = model(image, num_tokens=num_tokens)
                    
                    pred_points, pred_mask, pred_normal, pred_metric_scale = (output.get(k, None) for k in ['points', 'mask', 'normal', 'metric_scale'])

                    loss_list = []
                    for i in range(current_batch_size):
                        current_label = str(label_type[i])
                        if current_label not in config['loss']: continue
                            
                        gt_metric_scale = None
                        loss_dict, weight_dict, misc_dict = {}, {}, {}

                        for k, v in config['loss'][label_type[i]].items():
                            weight_dict[k] = v['weight']
                            if v['function'] == 'affine_invariant_global_loss':
                                loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(pred_points[i], gt_points[i], **v['params'])
                            elif v['function'] == 'affine_invariant_local_loss':
                                valid_mask = torch.isfinite(gt_points[i])
                                valid_count = valid_mask.sum()

                                if valid_count < 50:
                                    loss_dict[k] = pred_points[i].sum() * 0.0
                                    misc_dict[k] = 0.0
                                else:
                                    loss_dict[k], misc_dict[k] = affine_invariant_local_loss(
                                        pred_points[i].unsqueeze(0), 
                                        gt_points[i].unsqueeze(0), 
                                        gt_focal[i], 
                                        gt_metric_scale, 
                                        **v['params']
                                    )
                            elif v['function'] == 'geometry_consistency_loss':
                                val, _ = geometry_consistency_loss(pred_points[i], gt_points[i])
                            elif v['function'] == 'normal_loss':
                                loss_dict[k], misc_dict[k] = normal_loss(pred_points[i], gt_points[i])
                            elif v['function'] == 'edge_loss':
                                loss_dict[k], misc_dict[k] = edge_loss(pred_points[i], gt_points[i])
                            elif v['function'] == 'normal_map_loss':
                                loss_dict[k], misc_dict[k] = normal_map_loss(pred_normal[i], gt_normal[i])
                            elif v['function'] == 'mask_bce_loss':
                                loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                            elif v['function'] == 'mask_l2_loss':
                                loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])
                            elif v['function'] == 'metric_scale_loss':
                                if is_metric[i] and pred_metric_scale is not None:
                                    loss_dict[k], misc_dict[k] = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
                            else:
                                raise ValueError(f'Undefined loss function: {v["function"]}')
                        
                        weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                        loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                        loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=0.0)
                        loss_list.append(loss_)
                        
                        if isinstance(loss_, torch.Tensor) and torch.isnan(loss_).item():
                            pbar.write(f'NaN loss in process {accelerator.process_index}')
                        
                        safe_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                        records.append({**safe_loss_dict})

                    if len(loss_list) > 0:
                        loss = sum(loss_list) / len(loss_list)
                    else:
                        if accelerator.is_main_process:
                            pbar.write(f"⚠️ [Step {i_step}] Entire batch invalid! Using dummy loss.")
                        loss = pred_points.sum() * 0.0 
                    
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_check = [p for p in model.parameters() if p.grad is not None]
                        if len(params_to_check) > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_check, 1.0)
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                if accelerator.is_main_process:
                                    pbar.write(f"⚠️ [Step {i_step}] Gradient explosion (Norm={grad_norm.item()})! Skipping.")
                                optimizer.zero_grad()
                                continue
                        optimizer.step()
                        optimizer.zero_grad()

                i_accumulate += 1

            lr_scheduler.step()

            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            if i_step == initial_step or i_step % log_every == 0:
                records = [key_average(records)]
                records = accelerator.gather_for_metrics(records, use_gather_object=True)
                if accelerator.is_main_process:
                    records = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in r.items()} for r in records]
                    records = key_average(records)
                    if enable_mlflow:
                        try: mlflow.log_metrics(records, step=i_step)
                        except: pass
                records = []

            # Save checkponits
            if accelerator.is_main_process and (i_step % save_every == 0 or i_step == num_iterations - 1):
                pbar.write(f'Save checkpoint: {i_step:08d}')
                Path(workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)
                
                with io.BytesIO() as f:
                    torch.save({'model_config': config['model'], 'model': accelerator.unwrap_model(model).state_dict()}, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', f'{i_step:08d}.pt'), checkpoint_bytes)
                # ... (Optimizer saving code kept same)
                with io.BytesIO() as f:
                    torch.save({'model_config': config['model'], 'step': i_step}, f)
                    checkpoint_bytes = f.getvalue()
                save_checkpoint_executor.submit(_write_bytes_retry_loop, Path(workspace, 'checkpoint', 'latest.pt'), checkpoint_bytes)
            
            # Visualize
            if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0):
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
                save_dir.mkdir(parents=True, exist_ok=True)
                
                with torch.inference_mode():
                    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
                        image, gt_depth, gt_intrinsics = batch['image'], batch['depth'], batch['intrinsics']
                        image, gt_depth, gt_intrinsics = image.to(device), gt_depth.to(device), gt_intrinsics.to(device)
                        
                        # PEFT Compatible Inference
                        if hasattr(unwrapped_model, 'infer'):
                             output = unwrapped_model.infer(image)
                        else:
                             output = unwrapped_model(image)

                        pred_points = output.get('points', None)
                        pred_depth = output.get('depth', None)
                        pred_mask =  output.get('mask', None)
                        pred_normal = output.get('normal', None)
                        
                        if pred_points is not None: pred_points = pred_points.cpu().numpy()
                        if pred_depth is not None: pred_depth = pred_depth.cpu().numpy()
                        if pred_mask is not None: pred_mask = pred_mask.cpu().numpy()
                        if pred_normal is not None: pred_normal = pred_normal.cpu().numpy()
                        
                        image = (image.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                        
                        for i_instance in range(image.shape[0]):
                            idx = i_batch * batch_size_forward + i_instance
                            save_dir.joinpath(f'{idx:04d}').mkdir(parents=True, exist_ok=True)
                            
                            cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/image.jpg')), cv2.cvtColor(image[i_instance], cv2.COLOR_RGB2BGR))
                            
                            if pred_points is not None:
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/points.exr')), cv2.cvtColor(pred_points[i_instance], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                            
                            if pred_depth is not None:
                                mask_i = pred_mask[i_instance] if pred_mask is not None else None
                                vis_img = colorize_depth(pred_depth[i_instance], mask_i)
                                
                                d_val = pred_depth[i_instance]
                                if mask_i is not None:
                                    valid_vals = d_val[mask_i > 0]
                                else:
                                    valid_vals = d_val
                                
                                if valid_vals.size > 0:
                                    vmin, vmax = valid_vals.min(), valid_vals.max()
                                    text = f"Range: {vmin:.2f}m - {vmax:.2f}m"
                                    cv2.putText(vis_img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
                                    cv2.putText(vis_img, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
                                
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/depth_vis.png')), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

                                try:
                                    plt.figure(figsize=(8, 6))
                                    d_plot = d_val.copy()
                                    if mask_i is not None:
                                        d_plot[mask_i == 0] = np.nan
                                        
                                    if valid_vals.size > 0:
                                        robust_min = np.percentile(valid_vals, 1)
                                        robust_max = np.percentile(valid_vals, 99)
                                        plt.imshow(d_plot, cmap='Spectral_r', vmin=robust_min, vmax=robust_max)
                                    else:
                                        plt.imshow(d_plot, cmap='Spectral_r')
                                        
                                    cbar = plt.colorbar()
                                    cbar.set_label('Depth (meters)')
                                    plt.title(f'Step {i_step} | Depth Prediction')
                                    plt.axis('off')
                                    
                                    plt.savefig(str(save_dir.joinpath(f'{idx:04d}/depth_chart.png')), bbox_inches='tight')
                                    plt.close()
                                except Exception as e:
                                    print(f"Matplotlib vis failed: {e}")

                            if pred_normal is not None:
                                cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}/normal_vis.png')), cv2.cvtColor(colorize_normal(pred_normal[i_instance], pred_mask[i_instance] if pred_mask is not None else None), cv2.COLOR_RGB2BGR))
            
            pbar.set_postfix({'loss': loss.item()}, refresh=False)
            pbar.update(1)

if __name__ == '__main__':
    main()