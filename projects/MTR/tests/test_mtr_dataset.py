from awml_pred.datasets import MTRDataset


def test_mtr_dataset(dummy_dataset_cfg: dict) -> None:
    """
    Test MTRDataset.

    Args:
    ----
        dummy_dataset_cfg (dict): Fixture of dataset config.
    """
    _ = MTRDataset(**dummy_dataset_cfg)
