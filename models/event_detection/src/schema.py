import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from torch import tensor


@dataclass
class SingleLabelClassificationForwardOutput:
    loss: Optional[tensor] = None
    prediction_logits: tensor = None
    encoded_features: Optional[tensor] = None
    attentions: Optional[Tuple[tensor]] = None


@dataclass
class TSNEFeature:
    final_hidden_states: List
    encoded_features: List
    labels: List


@dataclass
class AgentPolicyOutput:
    y_predict: List
    y_true: List
    loss: float
    tsne_feature: Optional[TSNEFeature] = None
    test_input_text: Optional[List] = None


@dataclass
class PrototypicalNetworksForwardOutput:
    loss: Optional[tensor] = None
    distance: Optional[tensor] = None


@dataclass
class InputFeature:
    input_ids: tensor
    attention_mask: tensor
    labels: Optional[tensor] = None


@dataclass
class EncodedFeature:
    encoded_feature: tensor
    labels: Optional[tensor] = None


@dataclass
class TransformedFeature:
    transformed_feature: tensor
    labels: Optional[tensor] = None


@dataclass
class HeadOutput:
    output: tensor
    labels: Optional[tensor] = None


@dataclass
class FeatureToVisualize:
    feature: np.array
    labels: Optional[List[str]] = None


@dataclass
class ClassificationResult:
    acc: float
    f1_macro: float
    f1_micro: float
    f1_per_class: dict
    precision_macro: float
    recall_macro: float
    path_to_plot: str
    loss: Optional[float]
    other: Optional[float]


@dataclass
class LayerConfig:
    n_in: int = 768
    n_out: int = 20


@dataclass
class LayersConfig:
    from_pretrained: str = "bert-base-uncased"
    layers: Dict = LayerConfig #FIXME


@dataclass
class ContrastiveConfig:
    contrastive_loss_ratio: float = 0
    temperature: float = 0.07
    base_temperature: float = 0.07
    contrast_mode: str = "all"
    L2_normalize_encoded_feature: bool = True


@dataclass
class ModelConfig:
    layers: LayersConfig
    contrastive: ContrastiveConfig
    num_transformer_layers: int = 2
    freeze_transformer_layers: Any = None
    learning_rate: float = 0.0001
    dropout_rate: float = 0.5
    epochs: int = 5
    output_path: str = "./outputs/"


@dataclass
class DataConfig:
    name: str = "tweet_eval"
    config: str = "emotion"
    batch_size: int = 32
    label_column: str = "label"
    subset: str = 1


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    name: str = "emotion"
    seed: int = 42
