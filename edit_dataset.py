from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        
        # Convert path to absolute path and validate
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")
            
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        # Get all demo directories
        try:
            demo_dirs = [d for d in os.listdir(self.path) if d.startswith("demo_")]
            if not demo_dirs:
                raise ValueError(f"No demo directories found in {self.path}")
                
            demo_dirs.sort(key=lambda x: int(x.split("_")[1]))
            
            # Get all samples
            self.samples = []
            for demo_dir in demo_dirs:
                demo_path = os.path.join(self.path, demo_dir)
                sample_dirs = [d for d in os.listdir(demo_path) if d.startswith("sample_")]
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(demo_path, sample_dir)
                    self.samples.append(sample_path)
                    
            if not self.samples:
                raise ValueError(f"No samples found in any demo directory in {self.path}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {self.path}: {str(e)}")

        # Split the samples
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.samples))
        idx_1 = math.floor(split_1 * len(self.samples))
        self.samples = self.samples[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample_path = self.samples[i]
        
        # Load prompt
        with open(os.path.join(sample_path, "prompt.json")) as fp:
            prompt = json.load(fp)["edit"]

        # Load images
        image_0 = Image.open(os.path.join(sample_path, "1_0.jpg"))
        image_1 = Image.open(os.path.join(sample_path, "1_1.jpg"))

        # Resize images
        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        # Convert to tensor and normalize
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        # Apply random crop and flip
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        
        # Convert path to absolute path and validate
        self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset directory not found: {self.path}")
            
        self.res = res

        # Get all demo directories
        try:
            demo_dirs = [d for d in os.listdir(self.path) if d.startswith("demo_")]
            if not demo_dirs:
                raise ValueError(f"No demo directories found in {self.path}")
                
            demo_dirs.sort(key=lambda x: int(x.split("_")[1]))
            
            # Get all samples
            self.samples = []
            for demo_dir in demo_dirs:
                demo_path = os.path.join(self.path, demo_dir)
                sample_dirs = [d for d in os.listdir(demo_path) if d.startswith("sample_")]
                for sample_dir in sample_dirs:
                    sample_path = os.path.join(demo_path, sample_dir)
                    self.samples.append(sample_path)
                    
            if not self.samples:
                raise ValueError(f"No samples found in any demo directory in {self.path}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {self.path}: {str(e)}")

        # Split the samples
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.samples))
        idx_1 = math.floor(split_1 * len(self.samples))
        self.samples = self.samples[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample_path = self.samples[i]
        
        # Load prompt
        with open(os.path.join(sample_path, "prompt.json")) as fp:
            prompt = json.load(fp)["edit"]

        # Load input image
        image_0 = Image.open(os.path.join(sample_path, "1_0.jpg"))

        # Resize image
        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        # Convert to tensor and normalize
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, edit=prompt)
