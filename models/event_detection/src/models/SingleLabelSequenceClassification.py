from itertools import chain
from omegaconf import DictConfig
from torch.nn import Module, CrossEntropyLoss, Identity
from transformers import AutoModel, AdamW, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from models.event_detection.src.models import SequenceClassification
from models.event_detection.src.models.heads import DenseLayerHead, DenseLayerContrastiveHead
from models.event_detection.src.schema import SingleLabelClassificationForwardOutput, InputFeature, EncodedFeature
from models.event_detection.src.losses.supervised_contrastive_loss import SupervisedContrastiveLoss


class SingleLabelSequenceClassification(SequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelSequenceClassification, self).__init__(cfg)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=True)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()

    def forward(self,
                input_feature: InputFeature,
                mode: str) -> SingleLabelClassificationForwardOutput:
        output = self.encoder(input_ids=input_feature.input_ids,
                              attention_mask=input_feature.attention_mask).pooler_output
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature, mode=mode)

        if mode == "train":
            loss = self.loss(head_output.output, input_feature.labels)
            loss.backward()
            self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "validation":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "test":
            return SingleLabelClassificationForwardOutput(prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        else:
            raise ValueError(f"mode {mode} is not one of train, validation or test.")

    def instantiate_encoder(self) -> PreTrainedModel:
        encoder: PreTrainedModel = AutoModel.from_pretrained(self.cfg.model.from_pretrained)
        encoder = self.trim_encoder_layers(encoder, self.cfg.model.num_transformer_layers)
        encoder = self.freeze_encoder(encoder, self.cfg.model.freeze_transformer_layers)
        return encoder

    def instantiate_feature_transformer(self) -> Module:
        # this returns an empty Module
        return Identity()

    def instantiate_classification_head(self) -> DenseLayerHead:
        return DenseLayerHead(self.cfg)


class SingleLabelContrastiveSequenceClassification(SingleLabelSequenceClassification):
    def __init__(self, cfg: DictConfig):
        super(SingleLabelContrastiveSequenceClassification, self).__init__(cfg)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(cfg.model.from_pretrained, normalization=True)
        params = chain(self.encoder.parameters(), self.classification_head.parameters())
        self.optimizer = AdamW(params, lr=cfg.model.learning_rate)
        self.loss = CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=cfg.model.contrastive.temperature,
                                                          base_temperature=cfg.model.contrastive.base_temperature,
                                                          contrast_mode=cfg.model.contrastive.contrast_mode)
        self.contrastive_loss_ratio = cfg.model.contrastive.contrastive_loss_ratio

    def forward(self,
                input_feature: InputFeature,
                mode: str) -> SingleLabelClassificationForwardOutput:
        output = self.encoder(input_ids=input_feature.input_ids,
                              attention_mask=input_feature.attention_mask).pooler_output
        encoded_feature: EncodedFeature = EncodedFeature(encoded_feature=output, labels=input_feature.labels)
        head_output = self.classification_head(encoded_feature, mode=mode)

        if mode == "train":
            loss = self.loss(head_output.output, input_feature.labels)
            new_shape = (
                int(head_output.output.shape[0]/(self.cfg.augmenter.num_samples+1)),
                self.cfg.augmenter.num_samples+1,
                head_output.output.shape[-1]
            )
            contrastive_features = head_output.output.reshape(new_shape)
            contrastive_loss = self.contrastive_loss(contrastive_features, input_feature.labels[:int(head_output.output.shape[0]/(self.cfg.augmenter.num_samples+1))])
            total_loss = (1 - self.contrastive_loss_ratio) * loss + self.contrastive_loss_ratio * contrastive_loss
            total_loss.backward()
            self.optimizer.step()
            return SingleLabelClassificationForwardOutput(loss=total_loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "validation":
            loss = self.loss(head_output.output, input_feature.labels)
            return SingleLabelClassificationForwardOutput(loss=loss.item(), prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        elif mode == "test":
            return SingleLabelClassificationForwardOutput(prediction_logits=head_output.output,
                                                          encoded_features=encoded_feature.encoded_feature)
        else:
            raise ValueError(f"mode {mode} is not one of train, validation or test.")

    def instantiate_classification_head(self) -> DenseLayerContrastiveHead:
        return DenseLayerContrastiveHead(self.cfg)