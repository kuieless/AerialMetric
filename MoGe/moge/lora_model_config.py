"""Architecture constants for the released MoGe-2 Aerial LoRA checkpoint.

This is intentionally limited to inference-required model and adapter settings.
Training datasets, optimizer settings, schedulers, and losses are not needed at
benchmark time.
"""

MODEL_VERSION = "v2"

MODEL_CONFIG = {
    "encoder": {
        "backbone": "dinov2_vitl14",
        "intermediate_layers": [5, 11, 17, 23],
        "dim_out": 1024,
    },
    "neck": {
        "dim_in": [1026, 2, 2, 2, 2],
        "dim_out": None,
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 2, 2, 2, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "points_head": {
        "dim_in": [1024, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 3],
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "normal_head": {
        "dim_in": [1024, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 3],
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "mask_head": {
        "dim_in": [1024, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 1],
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    },
    "scale_head": {"dims": [1024, 1024, 1024, 1]},
    "remap_output": "exp",
    "num_tokens_range": [1200, 3600],
}

LORA_TARGET_MODULES = ["qkv", "proj", "fc1", "fc2"]
LORA_MODULES_TO_SAVE = ["scale_head"]
LORA_ALPHA_MULTIPLIER = 2
