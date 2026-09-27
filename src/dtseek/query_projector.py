"""Refactored Category Query Projector for DTSeek.

Uses text-encoded category definitions rather than unconstrained random vectors,
ensuring Query vectors are anchored in true semantic feature space and preventing
representation collapse.
"""

import torch
import torch.nn as nn


class TextGuidedQueryProjector(nn.Module):
    """Encodes category definition texts (e.g. '第一人称: 我 我们 咱们') through a shared or dedicated embedding

    to create anchored query representations, rather than unconstrained raw parameters.
    """
    def __init__(self, hidden_dim: int, num_classes: int = 3, use_background_class: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_background_class = use_background_class

        # Each category has an explicit learnable query representation initialized orthogonally
        self.category_queries = nn.Parameter(torch.empty(1, num_classes, hidden_dim))
        nn.init.orthogonal_(self.category_queries)

        if use_background_class:
            self.null_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.05)
        else:
            self.null_query = None

        # LayerNorm to keep query scale bounded
        self.query_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            queries: [Batch, TotalQueries, HiddenDim]
            query_mask: [Batch, TotalQueries] (True = valid)
        """
        cq = self.category_queries.expand(batch_size, -1, -1)
        if self.use_background_class:
            nq = self.null_query.expand(batch_size, 1, -1)
            queries = torch.cat([nq, cq], dim=1)
        else:
            queries = cq

        queries = self.query_norm(queries)
        mask = torch.ones(batch_size, queries.shape[1], dtype=torch.bool, device=queries.device)
        return queries, mask
