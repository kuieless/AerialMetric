# Reference: https://github.com/microsoft/MoGe
from pathlib import Path
from typing import Dict, Optional

import click
import torch
import utils3d
from peft import LoraConfig, get_peft_model

from moge.model import import_model_class_by_version
from moge.lora_model_config import (
    MODEL_CONFIG, MODEL_VERSION, LORA_TARGET_MODULES,
    LORA_MODULES_TO_SAVE, LORA_ALPHA_MULTIPLIER,
)
from moge.test.baseline import MGEBaselineInterface


class Baseline(MGEBaselineInterface):
    """
    MoGe-2 LoRA baseline wrapper for the ground benchmark.

    This wrapper keeps the LoRA path independent from the plain metric baseline.
    """

    def __init__(
        self,
        lora_weight_path: str,
        lora_rank: int,
        resolution_level: int,
        use_fp16: bool,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = LORA_ALPHA_MULTIPLIER * self.lora_rank
        self.resolution_level = int(resolution_level)
        self.use_fp16 = bool(use_fp16)

        MoGeModel = import_model_class_by_version(MODEL_VERSION)
        self.model = MoGeModel(**MODEL_CONFIG)

        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias="none",
            target_modules=LORA_TARGET_MODULES,
            modules_to_save=LORA_MODULES_TO_SAVE,
        )
        self.model = get_peft_model(self.model, peft_config)

        checkpoint_path = Path(lora_weight_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot find LoRA checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        new_state_dict = {}
        model_keys = set(self.model.state_dict().keys())
        for k, v in state_dict.items():
            if k in model_keys:
                new_state_dict[k] = v
                continue

            prefixed_k = f"base_model.model.{k}"
            if prefixed_k in model_keys:
                new_state_dict[prefixed_k] = v
                continue

            parts = prefixed_k.split(".")
            if parts[-1] in {"weight", "bias"}:
                base_injected_k = ".".join(parts[:-1] + ["base_layer", parts[-1]])
                if base_injected_k in model_keys:
                    new_state_dict[base_injected_k] = v
                    continue

            for head in LORA_MODULES_TO_SAVE:
                if k.startswith(head):
                    suffix = k[len(head) + 1 :]
                    trainable_k = f"base_model.model.{head}.modules_to_save.default.{suffix}"
                    if trainable_k in model_keys:
                        new_state_dict[trainable_k] = v
                        break

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        if self.use_fp16:
            self.model.half()

    @click.command()
    @click.option("--lora_weight", "lora_weight_path", type=click.Path(), required=True, help="Path to the LoRA checkpoint (.pt).")
    @click.option("--lora_rank", type=int, default=96, show_default=True, help="LoRA rank used for the adapter.")
    @click.option("--resolution_level", type=int, default=9, show_default=True, help="MoGe2 resolution level [0..9].")
    @click.option("--fp16", "use_fp16", is_flag=True, help="Enable mixed precision inference.")
    @click.option("--device", type=str, default="cuda:0", show_default=True, help="Device to use.")
    @staticmethod
    def load(
        lora_weight_path: str,
        lora_rank: int = 96,
        resolution_level: int = 9,
        use_fp16: bool = False,
        device: str = "cuda:0",
    ):
        return Baseline(
            lora_weight_path=lora_weight_path,
            lora_rank=lora_rank,
            resolution_level=resolution_level,
            use_fp16=use_fp16,
            device=device,
        )

    def _infer_impl(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor],
    ) -> Dict[str, torch.Tensor]:
        if intrinsics is not None:
            fov_x, _ = utils3d.pt.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None

        output = self.model.infer(
            image,
            fov_x=fov_x,
            resolution_level=self.resolution_level,
            use_fp16=self.use_fp16,
        )

        result = {
            "points_metric": output["points"],
            "depth_metric": output["depth"],
            "intrinsics": output["intrinsics"],
        }
        if "normal" in output:
            result["normal"] = output["normal"]
        if "mask" in output:
            result["mask"] = output["mask"]
        return result

    @torch.inference_mode()
    def infer(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._infer_impl(image=image, intrinsics=intrinsics)

    @torch.inference_mode()
    def infer_for_evaluation(
        self,
        image: torch.FloatTensor,
        intrinsics: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._infer_impl(image=image, intrinsics=intrinsics)
