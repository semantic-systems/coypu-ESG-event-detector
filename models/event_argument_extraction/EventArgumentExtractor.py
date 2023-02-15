from schemes import EventArgumentExtractorOutput, LinkedEntity
from models import BaseComponent


class BaseEventArgumentExtractor(BaseComponent):
    def __init__(self):
        super(BaseEventArgumentExtractor).__init__()

    def forward(self, tweet:str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=[LinkedEntity(entity=None,
                                                                          id=None,
                                                                          label=None,
                                                                          description=None)])


class BaseEventTemporalInformationExtractor(BaseComponent):
    def __init__(self):
        super(BaseEventTemporalInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=[LinkedEntity(entity=None,
                                                                          id=None,
                                                                          label=None,
                                                                          description=None)])


class BaseEventGeoSpatialInformationExtractor(BaseComponent):
    def __init__(self):
        super(BaseEventGeoSpatialInformationExtractor).__init__()

    def forward(self, tweet: str) -> EventArgumentExtractorOutput:
        return EventArgumentExtractorOutput(tweet=tweet,
                                            event_arguments=[LinkedEntity(entity=None,
                                                                          id=None,
                                                                          label=None,
                                                                          description=None)])
