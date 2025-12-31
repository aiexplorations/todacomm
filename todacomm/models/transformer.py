"""
Transformer model implementations with layer-wise feature extraction.

Supports GPT-2, BERT, OPT, Pythia, SmolLM, Qwen2, GPT-Neo and other
HuggingFace transformer architectures with efficient extraction.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Literal, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

from .base import BaseModel, ModelConfig


# Model types that use AutoModelForCausalLM
CAUSAL_LM_TYPES = {"gpt2", "opt", "pythia", "smollm", "qwen2", "gpt-neo", "llama", "mistral", "gemma"}
# Model types that use BERT-style loading
BERT_TYPES = {"bert", "distilbert", "roberta"}


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for transformer models."""

    model_type: str = "gpt2"  # More flexible - any HuggingFace model type
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
    - GPT-2, DistilGPT-2
    - BERT, DistilBERT, RoBERTa
    - OPT (Meta)
    - Pythia (EleutherAI)
    - SmolLM (HuggingFace)
    - Qwen2 (Alibaba)
    - GPT-Neo (EleutherAI)
    - Any HuggingFace causal LM

    Optimized for Apple Silicon (MPS) with efficient batching.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Load pre-trained model from HuggingFace
        if config.task == "lm":
            # Use AutoModelForCausalLM for most LM models
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name_or_path,
                    trust_remote_code=True,  # Needed for some models like Qwen2
                    torch_dtype=torch.float32,  # Convert BFloat16 to Float32 for MPS compatibility
                )
            except Exception:
                # Fallback to AutoModel for models without LM head
                self.model = AutoModel.from_pretrained(
                    config.model_name_or_path,
                    trust_remote_code=True
                )
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
        self.transformer = self._get_base_transformer()

        # Cache for layer names
        self._layer_names = self._build_layer_names()

        # Hook storage for intermediate activations
        self._hooks = []
        self._cached_features = {}

    def _get_base_transformer(self):
        """Extract the base transformer from different model architectures."""
        model = self.model

        # Try common attribute names for the base transformer
        for attr in ["transformer", "model", "bert", "distilbert", "roberta",
                     "gpt_neox", "decoder", "encoder"]:
            if hasattr(model, attr):
                base = getattr(model, attr)
                # Some models have nested structure (e.g., model.model)
                if hasattr(base, "layers") or hasattr(base, "h") or hasattr(base, "layer"):
                    return base
                # Check one level deeper
                for inner_attr in ["layers", "h", "layer", "decoder"]:
                    if hasattr(base, inner_attr):
                        return base

        # Fallback: return the model itself
        return model
    
    def _build_layer_names(self) -> List[str]:
        """Build list of extractable layer names."""
        layer_names = ["embedding"]

        # Find the layers attribute - different models use different names
        num_layers = self._get_num_layers()

        for i in range(num_layers):
            layer_names.append(f"layer_{i}")
            if self.config.extract_attention_weights:
                layer_names.append(f"layer_{i}_attention")
            if self.config.extract_ffn_outputs:
                layer_names.append(f"layer_{i}_ffn")

        layer_names.append("final")
        return layer_names

    def _get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        # Try different attribute names used by various architectures
        for attr in ["h", "layers", "layer"]:
            if hasattr(self.transformer, attr):
                return len(getattr(self.transformer, attr))

        # Check for encoder attribute (some BERT variants)
        if hasattr(self.transformer, "encoder"):
            encoder = self.transformer.encoder
            for attr in ["layer", "layers"]:
                if hasattr(encoder, attr):
                    return len(getattr(encoder, attr))

        # Check model config as fallback
        if hasattr(self.model, "config"):
            config = self.model.config
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(config, attr):
                    return getattr(config, attr)

        # Default fallback
        return 12

    def _get_layers_module(self):
        """Get the module containing transformer layers."""
        for attr in ["h", "layers", "layer"]:
            if hasattr(self.transformer, attr):
                return getattr(self.transformer, attr)

        if hasattr(self.transformer, "encoder"):
            encoder = self.transformer.encoder
            for attr in ["layer", "layers"]:
                if hasattr(encoder, attr):
                    return getattr(encoder, attr)

        return None

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for input_ids across different architectures."""
        # Try various embedding attribute names
        # GPT-2 style
        if hasattr(self.transformer, "wte"):
            return self.transformer.wte(input_ids)
        # BERT style
        if hasattr(self.transformer, "embeddings"):
            return self.transformer.embeddings(input_ids)
        # OPT/Pythia/GPT-Neo style (embed_tokens)
        if hasattr(self.transformer, "embed_tokens"):
            return self.transformer.embed_tokens(input_ids)
        # Some models have it nested
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens(input_ids)
        # Fallback: use first hidden state
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[0]

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
        # Handle different output types
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif hasattr(outputs, "start_logits"):
            # QA model returns start_logits and end_logits
            return outputs.start_logits
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        else:
            # Return the raw outputs if no standard attribute found
            return outputs[0]
    
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
            # Get embedding layer output - try various architectures
            return self._get_embeddings(input_ids)
        
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
                feats = self._get_embeddings(input_ids)
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
