# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path

import hydra
import numpy as np
import torch
import yaml

# This script is for checking if the DiGConditionalScoreModel in this directory matches the implementation in
# feynman/projects/sampling. It needs the old `sampling` environment to run, because the old `sampling` code has extra dependences.


def test_digconditional_score_model(default_batch):
    config_path = Path(__file__).parent / "tiny_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model: torch.nn.Module = hydra.utils.instantiate(config["score_model"])
    state_dict_path = Path(__file__).parent / "state_dict.ptkeep"
    # Uncomment below to update saved state dict.
    # with open(state_dict_path, "wb") as f:
    #     torch.save(model.state_dict(), f)
    with open(state_dict_path, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    model.eval()
    model_output = model(default_batch, t=torch.tensor([0.0] * 10))
    model_output = {
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in model_output.items()
    }
    expected_path = Path(__file__).parent / "expected.npz"
    # Uncomment below to update expected model output.
    # with open(expected_path, "wb") as f:
    #     np.savez(f, **model_output)
    expected_output = np.load(expected_path)
    assert set(model_output.keys()) == set(expected_output.keys())
    for k in model_output.keys():
        if k == "system_id":
            assert model_output[k] == expected_output[k].tolist()
        else:
            assert np.allclose(model_output[k], expected_output[k], atol=1e-5)
