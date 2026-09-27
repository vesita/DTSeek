"""Pipeline for DTSeek: supporting dynamic task switching and multiple tokenizers."""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .model import DTSeekConfig, DTSeekModel


class TaskCartridge:
    """Represents a pluggable task head / vocabulary with its own category descriptions."""

    def __init__(self, task_name: str, categories: List[Dict[str, str]], tokenizer=None, encoder=None):
        """
        Args:
            task_name: e.g. "triage", "sentiment", "medical_diagnosis"
            categories: List of dicts, e.g. [{"name": "billing", "desc": "invoices and payments"}, ...]
        """
        self.task_name = task_name
        self.categories = categories
        self.category_names = [c["name"] for c in categories]
        self.tokenizer = tokenizer
        self.encoder = encoder
        self._cached_embeddings: Optional[torch.Tensor] = None

    def get_embeddings(self, device: torch.device, hidden_dim: int) -> torch.Tensor:
        """Returns [1, NumClasses, HiddenDim] class query representations."""
        if self._cached_embeddings is not None:
            return self._cached_embeddings.to(device)

        if self.encoder is not None and self.tokenizer is not None:
            texts = [f"{c['name']}: {c.get('desc', '')}" for c in self.categories]
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.encoder(**inputs)
                # Mean pool
                mask = inputs["attention_mask"].unsqueeze(-1)
                emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            self._cached_embeddings = emb.unsqueeze(0)  # [1, C, D]
            return self._cached_embeddings
        else:
            # Fallback to random projection / trainable codebook if no label encoder
            gen = torch.Generator().manual_seed(abs(hash(self.task_name)) % (2**31))
            emb = torch.randn(1, len(self.categories), hidden_dim, generator=gen, device=device) * 0.05
            self._cached_embeddings = emb
            return self._cached_embeddings


class DTSeekEngine:
    """High-level Engine coordinating Doc Encoder and pluggable Task Cartridges."""

    def __init__(
        self,
        doc_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 384,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hidden_dim = hidden_dim

        # 1. Doc Input Pipeline (Doc Tokenizer A + Doc Encoder)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_name)
        self.doc_encoder = AutoModel.from_pretrained(doc_model_name).to(self.device)
        self.doc_encoder.eval()

        # 2. DTSeek Cross-Decoder & Heads
        self.config = DTSeekConfig(hidden_dim=hidden_dim, num_heads=6, num_decoder_layers=2)
        self.model = DTSeekModel(self.config).to(self.device)
        self.model.eval()

        # 3. Dynamic Task Registry
        self.tasks: Dict[str, TaskCartridge] = {}

    def register_task(
        self,
        task_name: str,
        categories: List[Dict[str, str]],
        tokenizer=None,
        encoder=None,
    ) -> None:
        """Register or hot-swap a task cartridge with custom vocabulary/labels."""
        if tokenizer is None:
            tokenizer = self.doc_tokenizer
        if encoder is None:
            encoder = self.doc_encoder
        cartridge = TaskCartridge(task_name, categories, tokenizer=tokenizer, encoder=encoder)
        self.tasks[task_name] = cartridge

    def decide(
        self,
        doc_text: str,
        task: Union[str, List[Dict[str, str]]],
        max_doc_len: int = 1024,
    ) -> Dict[str, Any]:
        """Runs single-pass decision over document text against specified task categories."""
        # 1. Resolve Task Cartridge
        if isinstance(task, str):
            if task not in self.tasks:
                raise ValueError(f"Task '{task}' not registered.")
            cartridge = self.tasks[task]
        else:
            cartridge = TaskCartridge(task_name="ad_hoc", categories=task, tokenizer=self.doc_tokenizer, encoder=self.doc_encoder)

        # 2. Encode Document (Single forward pass to produce Doc Memory Cache)
        doc_inputs = self.doc_tokenizer(
            doc_text,
            max_length=max_doc_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            doc_memory = self.doc_encoder(**doc_inputs).last_hidden_state  # [1, L_doc, D]
            doc_mask = doc_inputs["attention_mask"].bool()

            # 3. Query Embeddings from Task Cartridge
            class_embeddings = cartridge.get_embeddings(self.device, self.hidden_dim)

            # 4. Cross-Decoder Forward
            outputs = self.model(
                class_embeddings=class_embeddings,
                doc_memory=doc_memory,
                doc_mask=doc_mask,
            )

        probs = outputs["probs"][0].cpu().tolist()
        best_idx = int(outputs["best_index"][0].item())
        confidence = float(outputs["confidence"][0].item())
        is_bg = bool(outputs["is_background"][0].item())

        # Map back to category name
        if self.config.use_background_class:
            cat_names = ["_null_background_"] + cartridge.category_names
        else:
            cat_names = cartridge.category_names

        category_probs = {cat_names[i]: probs[i] for i in range(len(cat_names))}
        chosen_cat = cat_names[best_idx]

        return {
            "task": cartridge.task_name,
            "prediction": chosen_cat,
            "is_background": is_bg,
            "confidence": confidence,
            "probabilities": category_probs,
        }
