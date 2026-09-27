"""YOLO-style Loss Function for DTSeek:
Combines:
1. Classification Loss (L_cls): CrossEntropy for predicted category.
2. Localization Span Loss (L_box): L1 + GIoU loss on character span (center, width) for non-background tokens.
3. Objectness / Confidence Loss (L_conf): Binary Cross-Entropy matching detection presence.
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def generalized_iou_1d(pred_bounds: torch.Tensor, target_bounds: torch.Tensor) -> torch.Tensor:
    """1D Generalized IoU loss for spans [start, end] in [0, 1].
    
    Args:
        pred_bounds: [B, 2] (start, end)
        target_bounds: [B, 2] (start, end)
    """
    p_start, p_end = pred_bounds[:, 0], pred_bounds[:, 1]
    t_start, t_end = target_bounds[:, 0], target_bounds[:, 1]

    inter_start = torch.max(p_start, t_start)
    inter_end = torch.min(p_end, t_end)
    inter_len = (inter_end - inter_start).clamp(min=0.0)

    p_len = (p_end - p_start).clamp(min=1e-6)
    t_len = (t_end - t_start).clamp(min=1e-6)
    union_len = p_len + t_len - inter_len

    iou = inter_len / union_len.clamp(min=1e-6)

    # Convex hull (smallest enclosing segment)
    hull_start = torch.min(p_start, t_start)
    hull_end = torch.max(p_end, t_end)
    hull_len = (hull_end - hull_start).clamp(min=1e-6)

    giou = iou - (hull_len - union_len) / hull_len
    return 1.0 - giou  # Loss in [0, 2]


class YOLODetectionLoss(nn.Module):
    def __init__(self, lambda_cls: float = 1.0, lambda_box: float = 2.0, lambda_conf: float = 0.5):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf

    def forward(
        self,
        pred_logits: torch.Tensor,       # [B, TotalQueries]
        pred_spans: torch.Tensor,        # [B, TotalQueries, 2] (center, width)
        pred_bounds: torch.Tensor,       # [B, TotalQueries, 2] (start, end)
        pred_conf: torch.Tensor,         # [B, 1]
        target_labels: torch.Tensor,     # [B] (0: null, 1..3)
        target_spans: torch.Tensor,      # [B, 2] (center, width)
        target_bounds: torch.Tensor,     # [B, 2] (start, end)
    ) -> Dict[str, torch.Tensor]:
        # 1. Classification Loss
        l_cls = F.cross_entropy(pred_logits, target_labels)

        # 2. Box Regression Loss (Only for non-background classes: label > 0)
        pos_mask = (target_labels > 0)
        if pos_mask.sum() > 0:
            # Select the predicted span corresponding to the target label query
            idx = target_labels[pos_mask]
            b_idx = torch.arange(len(target_labels), device=pred_spans.device)[pos_mask]
            
            p_span_pos = pred_spans[b_idx, idx]
            t_span_pos = target_spans[pos_mask]

            p_bounds_pos = pred_bounds[b_idx, idx]
            t_bounds_pos = target_bounds[pos_mask]

            l1_loss = F.l1_loss(p_span_pos, t_span_pos)
            giou_loss = generalized_iou_1d(p_bounds_pos, t_bounds_pos).mean()
            l_box = l1_loss + giou_loss
        else:
            l_box = torch.tensor(0.0, device=pred_logits.device)

        # 3. Objectness / Confidence Loss
        target_conf = pos_mask.float().unsqueeze(-1)
        l_conf = F.binary_cross_entropy(pred_conf, target_conf)

        total_loss = self.lambda_cls * l_cls + self.lambda_box * l_box + self.lambda_conf * l_conf

        return {
            "loss": total_loss,
            "loss_cls": l_cls,
            "loss_box": l_box,
            "loss_conf": l_conf,
        }
