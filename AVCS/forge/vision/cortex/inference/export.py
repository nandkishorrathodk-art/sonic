"""Model Export Utilities (TorchScript & ONNX) for TensorRT Cloud Deployment."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from forge.vision.cortex.model import ArtificialVisualCortex


class ExportableNeuralCortex(nn.Module):
    """Purely functional neural forward pass for ONNX, TorchScript, and TensorRT."""

    def __init__(self, cortex: ArtificialVisualCortex) -> None:
        super().__init__()
        self.backbone = cortex.backbone
        self.dorsal = cortex.dorsal
        self.ventral = cortex.ventral
        self.language = cortex.language

    def forward(
        self,
        peripheral: torch.Tensor,
        foveal: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fused_feat, _, _ = self.backbone(peripheral, foveal)
        affordance_logits = self.dorsal(fused_feat)
        ventral_out = self.ventral(fused_feat)
        _, query_coords = self.language(fused_feat, text_embedding)

        return (
            ventral_out["latent"],
            affordance_logits,
            ventral_out["category_logits"],
            ventral_out["text_density"],
            query_coords,
        )


def export_torchscript(
    model: ArtificialVisualCortex,
    save_path: str = "runs/avcs_cortex.pt",
) -> str:
    """Exports model to optimized TorchScript trace."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    wrapper = ExportableNeuralCortex(model).eval()

    p = torch.randn(1, 3, model.cfg.retina.peripheral_size[0], model.cfg.retina.peripheral_size[1])
    f = torch.randn(1, 3, model.cfg.retina.foveal_size[0], model.cfg.retina.foveal_size[1])
    t = torch.randn(1, model.cfg.latent_dim)

    traced = torch.jit.trace(wrapper, (p, f, t))
    traced.save(save_path)
    print(f"TorchScript model exported successfully: {save_path}")
    return save_path


def export_onnx(
    model: ArtificialVisualCortex,
    save_path: str = "runs/avcs_cortex.onnx",
    opset_version: int = 17,
) -> str:
    """Exports model to ONNX format for TensorRT compilation."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    wrapper = ExportableNeuralCortex(model).eval()

    p = torch.randn(1, 3, model.cfg.retina.peripheral_size[0], model.cfg.retina.peripheral_size[1])
    f = torch.randn(1, 3, model.cfg.retina.foveal_size[0], model.cfg.retina.foveal_size[1])
    t = torch.randn(1, model.cfg.latent_dim)

    torch.onnx.export(
        wrapper,
        (p, f, t),
        save_path,
        input_names=["peripheral", "foveal", "text_embedding"],
        output_names=["latent", "affordance_logits", "category_logits", "text_density", "query_coords"],
        dynamic_axes={
            "peripheral": {0: "batch_size"},
            "foveal": {0: "batch_size"},
            "text_embedding": {0: "batch_size"},
        },
        opset_version=opset_version,
    )
    print(f"ONNX model exported successfully: {save_path}")
    return save_path
