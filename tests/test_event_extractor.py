from unittest.mock import patch

import pytest

from event_extractor import EventExtractor
from schemes import EventExtractorOutput, EventArgumentExtractorOutput


def test_instantiate_event_extractor(event_type_detector_instance, event_argument_extractor_openie_instance):
    extractor = EventExtractor(event_detector=event_type_detector_instance,
                               event_argument_extractor=event_argument_extractor_openie_instance)
    assert extractor


def test_event_extractor_infer(event_extractor_instance):
    with patch.object(EventExtractor, 'write_json') as mock:
        mock.return_value = True
        output = event_extractor_instance.infer("fake tweet")
        assert isinstance(output, EventExtractorOutput)


@pytest.mark.skip(reason="wip")
def test_event_extractor_infer2(event_extractor_instance):
    with patch.object(EventExtractor, 'write_json') as mock:
        mock.return_value = True
        with patch(EventExtractor, 'event_argument_extractor', 'forward') as mock_event_argument_extractor:
            mock_event_argument_extractor.return_value = EventArgumentExtractorOutput(tweet="bla",
                                                                                      event_graph=[["bla"]],
                                                                                      event_arguments=["bla"],
                                                                                      wikidata_links={"bla": "bla"})
            output = event_extractor_instance.infer("fake tweet")
            assert isinstance(output, EventExtractorOutput)
