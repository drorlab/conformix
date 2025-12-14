# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pytest
import torch
from torch_geometric.data import Batch, Data

from bioemu.sde_lib import SDE, CosineVPSDE

EPS = 1e-5


@pytest.fixture(scope="function")
def tiny_state_batch() -> Batch:
    return Batch.from_data_list([_make_sample(i) for i in range(0, 10)])


def _make_sample(bigness) -> Data:
    foo_per_sample = 3 * (bigness + 1)
    return Data(foo=torch.randn(foo_per_sample, 3))


def test_sde(tiny_state_batch):
    """Tests correct shapes for all methods of the SDE class"""
    x: torch.Tensor = tiny_state_batch.foo
    sde: SDE = CosineVPSDE()

    batch_size = x.shape[0]
    batch_idx = None

    t = torch.rand(batch_size) * (sde.T - EPS) + EPS

    def _check_shapes(drift, diffusion):
        assert drift.shape == x.shape
        assert diffusion.shape[0] == x.shape[0]

    # Forward SDE methods
    drift, diffusion = sde.sde(x, t, batch_idx)
    _check_shapes(drift, diffusion)

    mean, std = sde.marginal_prob(x, t, batch_idx)
    _check_shapes(mean, std)

    z = sde.prior_sampling(x.shape)
    assert z.shape == x.shape
