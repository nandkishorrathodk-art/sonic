"""PyTorch Dataset & DataLoaders for AVCS Training."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from forge.vision.cortex.data.synthetic import ProceduralScreenGenerator, SyntheticScreenData


class ScreenDataset(Dataset):
    """Generates synthetic screen samples for supervised and self-supervised training."""

    def __init__(
        self,
        num_samples: int = 1000,
        width: int = 640,
        height: int = 360,
        latent_dim: int = 256,
        affordance_grid_size: tuple[int, int] = (32, 32),
    ) -> None:
        self.num_samples = num_samples
        self.width = width
        self.height = height
        self.latent_dim = latent_dim
        self.affordance_grid_size = affordance_grid_size
        self.generator = ProceduralScreenGenerator(width=width, height=height)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate current frame
        data = self.generator.generate(seed=idx)

        # Generate consecutive frame with temporal mutation (cursor movement or minor animation)
        data_next = self.generator.generate(category=data.category, seed=idx + 100000)

        # Convert frame to Float Tensor (3, H, W) normalized to [0, 1]
        frame_t = torch.from_numpy(data.frame).permute(2, 0, 1).float() / 255.0
        frame_next = torch.from_numpy(data_next.frame).permute(2, 0, 1).float() / 255.0

        # Downsample affordance mask to affordance_grid_size using nearest neighbor
        mask_t = torch.from_numpy(data.affordance_mask).unsqueeze(0).unsqueeze(0).float()
        mask_down = F.interpolate(mask_t, size=self.affordance_grid_size, mode="nearest").squeeze().long()

        # Target grounding coordinate from first interactive point (if available)
        if data.interactive_points:
            pt = data.interactive_points[0]
            target_coord = torch.tensor([pt[0], pt[1]], dtype=torch.float32)
        else:
            target_coord = torch.tensor([0.5, 0.5], dtype=torch.float32)

        # Real query embedding derived from the sample's target class name. The
        # grounding head must learn from language that actually names the target;
        # random noise here would train it on nothing.
        from forge.vision.cortex.core.modulation import encode_text_query

        has_target = bool(data.interactive_points)
        query_text = data.interactive_points[0][2] if has_target else data.category
        query_embed = encode_text_query(query_text, self.latent_dim)[0]

        return {
            "frame_t": frame_t,
            "frame_next": frame_next,
            "affordance_mask": mask_down,
            "category_idx": torch.tensor(data.category_idx, dtype=torch.long),
            "target_coord": target_coord,
            "query_embed": query_embed,
            "grounding_valid": torch.tensor(has_target, dtype=torch.bool),
        }


def create_dataloader(
    num_samples: int = 500,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ScreenDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

