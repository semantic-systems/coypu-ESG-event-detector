import torch
from omegaconf import DictConfig
from torch import tensor
from models.event_detection.src.models.heads import Head
from models.event_detection.src.schema import EncodedFeature, HeadOutput


class PrototypicalHead(Head):
    def __init__(self, cfg: DictConfig):
        super(PrototypicalHead, self).__init__()
        self.cfg = cfg

    def forward(
        self,
        support_features: EncodedFeature,
        query_features: EncodedFeature,
    ) -> HeadOutput:

        support_labels: tensor = support_features.labels
        support_feature: tensor = support_features.encoded_feature
        query_feature: tensor = query_features.encoded_feature

        labels_within_episode = torch.unique(support_labels).tolist()
        # Prototype i is the mean of all support features vector with label i

        z_proto = torch.cat(
            [
                support_feature[torch.nonzero(support_labels == label)].mean(0)
                for label in labels_within_episode
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(query_feature, z_proto)
        # prediction =
        logits = -dists
        return HeadOutput(output=logits)
