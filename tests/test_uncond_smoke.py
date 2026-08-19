import torch


def test_import():
    from wavtts.model import CFM, DiT  # noqa: F401

    assert torch.tensor(1.0).item() == 1.0
