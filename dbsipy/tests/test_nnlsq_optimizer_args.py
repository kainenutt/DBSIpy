from __future__ import annotations

import torch

from dbsipy.nn.leastsquares import nnlsq


class _DummyCfg:
    pass


def test_nnlsq_accepts_dbsi_config_metadata_key() -> None:
    # Tiny problem: 1 voxel, 2 volumes, 2 basis functions.
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([[0.25, 0.75]], dtype=torch.float32)

    # Minimal optimizer args; include DBSI_CONFIG which should be ignored by the validator.
    x = nnlsq(
        A=A,
        b=b,
        optimizer_args={
            "lr": 1e-3,
            "epochs": 1,
            "loss": "mse",
            "DBSI_CONFIG": _DummyCfg(),
        },
        device="cpu",
    )

    assert x.shape[0] == 1
    assert x.ndim == 2
