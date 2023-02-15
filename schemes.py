from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from stores.download import LocalMetaConfig


@dataclass
class PublicMetaConfig:
    package: str


@dataclass
class ModelConfig:
    type: str
    meta: Union[LocalMetaConfig, PublicMetaConfig]


@dataclass
class Config:
    event_type_detector: Optional[ModelConfig] = None
    event_argument_extractor: Optional[ModelConfig] = None
    event_extractor: Optional[ModelConfig] = None


@dataclass
class EventDetectorOutput:
    tweet: str
    event_type: Optional[str]
    wikidata_link: Optional[str]


@dataclass
class LinkedEntity:
    entity: Optional[str]
    id: Optional[str]
    label: Optional[str]
    description: Optional[str]


@dataclass
class EventExtractorOutput:
    tweet: str
    event_type: Optional[str]
    wikidata_link: Optional[str]


@dataclass
class EventArgumentExtractorOutput:
    tweet: str
    event_arguments: Optional[List[LinkedEntity]]


@dataclass
class Time:
    value: str


@dataclass
class Location:
    value: str

