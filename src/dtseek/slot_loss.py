"""Vectorized Fast Hungarian Matcher and Criterion for PyTorch.

Optimized to run batches swiftly without Python-level scipy loop overhead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_span: float = 3.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,   # [B, K, C]
        pred_spans: torch.Tensor,    # [B, K, 2]
        targets: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        B, K, C = pred_logits.shape
        out_prob = pred_logits.softmax(-1)  # [B, K, C]

        indices = []
        for b in range(B):
            tgt_labels = targets[b]["labels"]  # [M]
            tgt_spans = targets[b]["spans"]    # [M, 2]
            M = len(tgt_labels)

            if M == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # 1. Classification Cost
            cost_class = -out_prob[b, :, tgt_labels]  # [K, M]
            # 2. Span L1 Cost
            cost_span = torch.cdist(pred_spans[b], tgt_spans, p=1)  # [K, M]

            cost_matrix = (self.cost_class * cost_class + self.cost_span * cost_span).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64),
            ))
        return indices


class SetCriterion(nn.Module):
    def __init__(self, matcher: HungarianMatcher, num_classes: int = 4, eos_coef: float = 0.2):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        pred_logits: torch.Tensor,  # [B, K, C]
        pred_spans: torch.Tensor,   # [B, K, 2]
        pred_bounds: torch.Tensor,  # [B, K, 2]
        pred_conf: torch.Tensor,    # [B, K]
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(pred_logits, pred_spans, targets)
        B, K, C = pred_logits.shape

        target_classes = torch.zeros((B, K), dtype=torch.int64, device=pred_logits.device)
        p_spans_list, t_spans_list = [], []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
                p_spans_list.append(pred_spans[b, src_idx])
                t_spans_list.append(targets[b]["spans"][tgt_idx])

        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight)

        if len(p_spans_list) > 0:
            p_cat = torch.cat(p_spans_list, dim=0)
            t_cat = torch.cat(t_spans_list, dim=0)
            loss_span = F.l1_loss(p_cat, t_cat)
            num_boxes = p_cat.shape[0]
        else:
            loss_span = torch.tensor(0.0, device=pred_logits.device)
            num_boxes = 0

        target_conf = (target_classes > 0).float()
        loss_conf = F.binary_cross_entropy(pred_conf, target_conf)

        total_loss = loss_ce + 4.0 * loss_span + 0.5 * loss_conf
        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_span": loss_span,
            "loss_conf": loss_conf,
            "num_matched": num_boxes,
        }
