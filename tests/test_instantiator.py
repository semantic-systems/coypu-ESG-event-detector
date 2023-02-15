from unittest.mock import patch

import pytest

from event_extractor import Instantiator


def test_instantiator_valid_event_detector_path_and_openie(valid_pretrained_event_detector_path):
    with patch.object(Instantiator, 'set_up_directory') as mock:
        mock.return_value = True
        instantiator = Instantiator(event_detector_model_path=valid_pretrained_event_detector_path,
                                    event_argument_extractor_model_path="openie")
    assert instantiator


def test_instantiator_invalid_event_detector_path_and_openie(invalid_pretrained_event_detector_path):
    with patch.object(Instantiator, 'set_up_directory') as mock:
        mock.return_value = True
        with pytest.raises(AssertionError):
            Instantiator(event_detector_model_path=invalid_pretrained_event_detector_path,
                         event_argument_extractor_model_path="openie")


@pytest.mark.skip(reason="pretrained event argument extractor not implemented")
def test_instantiator_valid_event_detector_path_and_event_argument_extractor_path(valid_pretrained_event_detector_path,
                                                                                  valid_pretrained_event_argument_extractor_path):
    with patch.object(Instantiator, 'set_up_directory') as mock:
        mock.return_value = True
        instantiator = Instantiator(event_detector_model_path=valid_pretrained_event_detector_path,
                                    event_argument_extractor_model_path=valid_pretrained_event_argument_extractor_path)
    assert instantiator


@pytest.mark.skip(reason="pretrained event argument extractor not implemented")
def test_instantiator_valid_event_detector_path_and_invalid_event_argument_extractor_path(
        valid_pretrained_event_detector_path,
        invalid_pretrained_event_argument_extractor_path):
    with patch.object(Instantiator, 'set_up_directory') as mock:
        mock.return_value = True
        with pytest.raises(AssertionError):
            Instantiator(event_detector_model_path=valid_pretrained_event_detector_path,
                         event_argument_extractor_model_path=invalid_pretrained_event_argument_extractor_path)
