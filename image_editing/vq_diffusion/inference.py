"""
DICE Editing with VQ-Diffusion.

Applies the DICE inversion + editing algorithm using VQ-Diffusion (multinomial diffusion).
Requires the VQ-Diffusion codebase to be installed first (see README.md).

Usage:
    python inference.py --lamb 0.5 --gpu 0
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
import torchvision
import tqdm
from PIL import Image

# VQ-Diffusion imports (requires VQ-Diffusion codebase)
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import get_model_parameters_info
from image_synthesis.data.utils.image_preprocessor import DalleTransformerPreprocessor


class VQDiffusionDICE:
    """VQ-Diffusion model wrapper with DICE editing support."""
    
    def __init__(self, config, path):
        self.info = self._load_model(model_path=path, config_path=config)
        self.model = self.info["model"]
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_model(self, model_path, config_path):
        """Load model from config and checkpoint."""
        model_name = os.path.basename(config_path).replace(".yaml", "")
        config = load_yaml_config(config_path)
        model = build_model(config)
        
        print(get_model_parameters_info(model))
        
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
        else:
            print(f"Model path: {model_path} does not exist.")
            exit(1)
        
        epoch = ckpt.get("last_epoch", ckpt.get("epoch", 0))
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        
        if "ema" in ckpt:
            print("Loading EMA model")
            ema_model = model.get_ema_model()
            ema_model.load_state_dict(ckpt["ema"], strict=False)
        
        return {"model": model, "epoch": epoch, "model_name": model_name}

    def edit_with_dice(self, image, text, target_text, truncation_rate,
                       save_root, save_base_name, guidance_scale=5.0):
        """
        Run DICE editing on an image.
        
        Args:
            image: Input image tensor.
            text: Source text description.
            target_text: Target text for editing.
            truncation_rate: Sampling truncation rate.
            save_root: Output directory.
            save_base_name: Output filename.
            guidance_scale: CFG scale.
        """
        os.makedirs(save_root, exist_ok=True)
        
        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = True
        self.model.transformer.prior_rule = 0
        self.model.transformer.prior_weight = 0
        
        data_src = {"text": [text], "image": image}
        data_tgt = {"text": [target_text], "image": image}
        
        add_string = "r"
        res = self.model.edit_content_dps(
            batch=data_src,
            target_batch=data_tgt,
            filter_ratio=0,
            sample_type="top" + str(truncation_rate) + add_string,
        )
        
        content = res["content"].cpu()
        torchvision.utils.save_image(
            content, os.path.join(save_root, save_base_name + ".png"), normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICE Editing with VQ-Diffusion")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
    parser.add_argument("--end", type=int, default=100, help="End index in dataset")
    parser.add_argument("--lamb", type=float, default=0.5, help="Lambda (noise weight)")
    parser.add_argument("--if_lamb_schedule", action="store_true", default=False,
                        help="Use lambda schedule")
    parser.add_argument("--config", type=str, default="configs/ithq.yaml",
                        help="Model config path")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/vq_diffusion/ithq_learnable.pth",
                        help="Model checkpoint path")
    parser.add_argument("--dataset_file", type=str, default="dataset/mapping_file.json",
                        help="PIE-Bench mapping file")
    parser.add_argument("--image_dir", type=str, default="dataset/annotation_images",
                        help="Source image directory")
    parser.add_argument("--output_dir", type=str, default="outputs_vqdiff",
                        help="Output directory")
    args = parser.parse_args()
    
    device = "cuda"
    
    # Load model
    vqdiff = VQDiffusionDICE(config=args.config, path=args.checkpoint)
    vqdiff.model.transformer.lamb = args.lamb
    vqdiff.model.transformer.if_lamb_schedule = args.if_lamb_schedule
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    with open(args.dataset_file, "r") as f:
        editing_instruction = json.load(f)
    
    editing_list = list(editing_instruction.items())
    
    for key, item in tqdm.tqdm(editing_list[args.start:args.end]):
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(args.image_dir, item["image_path"])
        
        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.uint8)
        transform = DalleTransformerPreprocessor(size=256, phase="val")
        image = transform(image=image)["image"]
        image = np.transpose(image.astype(np.float32), (2, 0, 1))
        image = torch.tensor(image, device=device).unsqueeze(0)
        
        save_dir = os.path.dirname(image_path.replace(args.image_dir, args.output_dir))
        save_name = os.path.basename(image_path).split(".")[0]
        
        vqdiff.edit_with_dice(
            image, original_prompt, editing_prompt,
            truncation_rate=1.0,
            save_root=save_dir,
            save_base_name=save_name,
            guidance_scale=5.0)
        
        torch.cuda.empty_cache()
    
    print(f"Done! Results saved to {args.output_dir}")
