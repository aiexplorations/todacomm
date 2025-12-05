"""
Transformer model implementations with layer-wise feature extraction.

Supports GPT-2, BERT, and custom transformer architectures with
HuggingFace integration and efficient extraction on Apple Silicon (MLX).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

from .base import BaseModel, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for transformer models."""
    
    model_type: Literal["gpt2", "bert", "distilbert", "custom"] = "gpt2"
    model_name_or_path: str = "gpt2"  # HuggingFace model name or local path
    num_labels: int = 2  # For classification tasks
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    vocab_size: int = 50257
    
    # Task-specific
    task: Literal["lm", "classification", "qa"] = "lm"
    
    # Extraction config
    extract_attention_weights: bool = True
    extract_ffn_outputs: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TransformerConfig":
        """Create from dictionary."""
        return cls(**config_dict)


class TransformerModel(BaseModel):
    """
    Transformer model wrapper with layer-wise feature extraction.
    
    Supports:
    - GPT-2 variants (small, medium)
    - BERT variants (base, distilbert)
    - Custom transformers
    
    Optimized for Apple Silicon (M4 Pro) with efficient batching.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained model from HuggingFace
        if config.task == "lm":
            if "gpt2" in config.model_name_or_path.lower():
                self.model = GPT2LMHeadModel.from_pretrained(config.model_name_or_path)
            else:
                self.model = AutoModel.from_pretrained(config.model_name_or_path)
        elif config.task == "classification":
            self.model = BertForSequenceClassification.from_pretrained(
                config.model_name_or_path,
                num_labels=config.num_labels
            )
        elif config.task == "qa":
            self.model = BertForQuestionAnswering.from_pretrained(config.model_name_or_path)
        else:
            raise ValueError(f"Unknown task: {config.task}")
        
        # Get base transformer (handles different model structures)
        if hasattr(self.model, "transformer"):
            self.transformer = self.model.transformer  # GPT-2
        elif hasattr(self.model, "bert"):
            self.transformer = self.model.bert  # BERT
        elif hasattr(self.model, "distilbert"):
            self.transformer = self.model.distilbert  # DistilBERT
        else:
            self.transformer = self.model
        
        # Cache for layer names
        self._layer_names = self._build_layer_names()
        
        # Hook storage for intermediate activations
        self._hooks = []
        self._cached_features = {}
    
    def _build_layer_names(self) -> List[str]:
        """Build list of extractable layer names."""
        layer_names = ["embedding"]
        
        # Add transformer layers
        if hasattr(self.transformer, "h"):  # GPT-2
            num_layers = len(self.transformer.h)
            for i in range(num_layers):
                layer_names.append(f"layer_{i}")
                if self.config.extract_attention_weights:
                    layer_names.append(f"layer_{i}_attention")
                if self.config.extract_ffn_outputs:
                    layer_names.append(f"layer_{i}_ffn")
        elif hasattr(self.transformer, "layer"):  # BERT/DistilBERT
            num_layers = len(self.transformer.layer)
            for i in range(num_layers):
                layer_names.append(f"layer_{i}")
                if self.config.extract_attention_weights:
                    layer_names.append(f"layer_{i}_attention")
                if self.config.extract_ffn_outputs:
                    layer_names.append(f"layer_{i}_ffn")
        elif hasattr(self.transformer, "encoder"):  # Some BERT variants
            if hasattr(self.transformer.encoder, "layer"):
                num_layers = len(self.transformer.encoder.layer)
                for i in range(num_layers):
                    layer_names.append(f"layer_{i}")
        
        layer_names.append("final")
        return layer_names
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Model outputs (logits)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state
    
    def extract_layer_features(
        self,
        input_ids: torch.Tensor,
        layer_name: str,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract features from a specific layer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            layer_name: Name of layer to extract
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Features from specified layer [batch_size, seq_len, hidden_size]
            or [batch_size, hidden_size] if pooled
        """
        self.validate_layer_name(layer_name)
        
        # Use output_hidden_states for efficient extraction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=self.config.extract_attention_weights,
                **kwargs
            )
        
        # Extract based on layer name
        if layer_name == "embedding":
            # Get embedding layer output
            if hasattr(self.transformer, "wte"):  # GPT-2
                return self.transformer.wte(input_ids)
            elif hasattr(self.transformer, "embeddings"):  # BERT
                return self.transformer.embeddings(input_ids)
        
        elif layer_name == "final":
            # Final layer output
            return outputs.hidden_states[-1]
        
        elif layer_name.startswith("layer_") and "_attention" not in layer_name and "_ffn" not in layer_name:
            # Intermediate layer output
            layer_idx = int(layer_name.split("_")[1])
            return outputs.hidden_states[layer_idx + 1]  # +1 because first is embedding
        
        elif "_attention" in layer_name:
            # Attention weights
            layer_idx = int(layer_name.split("_")[1])
            if outputs.attentions is not None:
                # Average over attention heads: [batch, heads, seq, seq] -> [batch, seq, seq]
                return outputs.attentions[layer_idx].mean(dim=1)
            else:
                raise ValueError("Attention weights not available. Set extract_attention_weights=True")
        
        elif "_ffn" in layer_name:
            # FFN output (requires custom hook - simplified for now)
            layer_idx = int(layer_name.split("_")[1])
            # Return layer output as proxy (can be enhanced with hooks)
            return outputs.hidden_states[layer_idx + 1]
        
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
    
    def get_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return self._layer_names
    
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pool_strategy: Literal["mean", "cls", "last", "none"] = "mean",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient multi-layer extraction in single forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            layer_names: Layers to extract (None = all)
            attention_mask: Attention mask
            pool_strategy: How to pool sequence dimension
                - "mean": Average over sequence
                - "cls": Use [CLS] token (BERT)
                - "last": Use last token (GPT-2)
                - "none": Keep full sequence
            
        Returns:
            Dictionary of layer features
        """
        if layer_names is None:
            layer_names = self.get_layer_names()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=self.config.extract_attention_weights,
                **kwargs
            )
        
        features = {}
        
        for layer_name in layer_names:
            # Extract features
            if layer_name == "embedding":
                if hasattr(self.transformer, "wte"):
                    feats = self.transformer.wte(input_ids)
                elif hasattr(self.transformer, "embeddings"):
                    feats = self.transformer.embeddings(input_ids)
            elif layer_name == "final":
                feats = outputs.hidden_states[-1]
            elif layer_name.startswith("layer_") and "_" not in layer_name[6:]:
                layer_idx = int(layer_name.split("_")[1])
                feats = outputs.hidden_states[layer_idx + 1]
            else:
                # Use extract_layer_features for complex cases
                feats = self.extract_layer_features(input_ids, layer_name, attention_mask, **kwargs)
            
            # Apply pooling
            if pool_strategy == "mean" and feats.dim() == 3:
                # Average over sequence dimension
                if attention_mask is not None:
                    # Masked average
                    mask_expanded = attention_mask.unsqueeze(-1).expand(feats.size())
                    sum_feats = (feats * mask_expanded).sum(dim=1)
                    sum_mask = mask_expanded.sum(dim=1)
                    feats = sum_feats / sum_mask.clamp(min=1)
                else:
                    feats = feats.mean(dim=1)
            elif pool_strategy == "cls" and feats.dim() == 3:
                # Use first token ([CLS])
                feats = feats[:, 0, :]
            elif pool_strategy == "last" and feats.dim() == 3:
                # Use last token
                if attention_mask is not None:
                    # Get last non-padding token
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    feats = feats[torch.arange(feats.size(0)), seq_lengths]
                else:
                    feats = feats[:, -1, :]
            # else: pool_strategy == "none", keep as is
            
            features[layer_name] = feats
        
        return features


def load_pretrained_transformer(
    model_name: str,
    task: Literal["lm", "classification", "qa"] = "lm",
    num_labels: int = 2,
    device: str = "cpu"
) -> TransformerModel:
    """
    Convenience function to load a pre-trained transformer.
    
    Args:
        model_name: HuggingFace model name (e.g., "gpt2", "bert-base-uncased")
        task: Task type
        num_labels: Number of labels for classification
        device: Device to load model on
        
    Returns:
        TransformerModel instance
    """
    config = TransformerConfig(
        model_name_or_path=model_name,
        task=task,
        num_labels=num_labels
    )
    model = TransformerModel(config)
    model.to(device)
    model.eval()
    return model
