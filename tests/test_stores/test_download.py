from stores.download import download_from_google_drive, LocalMetaConfig


def test_download_pretrained_model_from_google_drive(pretrained_model_google_drive_url,
                                                     pretrained_model_output_path):
    assert len(pretrained_model_output_path.listdir()) == 0
    config = LocalMetaConfig(google_drive_link=pretrained_model_google_drive_url,
                             directory_to_store=pretrained_model_output_path)
    download_from_google_drive(config)
    assert len(pretrained_model_output_path.listdir()) == 1
