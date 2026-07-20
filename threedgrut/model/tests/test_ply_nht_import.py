# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

from threedgrut.model.features import Features
from threedgrut.model.model import MixtureOfGaussians


def _write_geometry_ply(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    data = np.array(
        [
            (1.0, 2.0, 3.0, -1.0, 0.1, 0.2, 0.3, -2.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0),
            (4.0, 5.0, 6.0, 2.0, 0.4, 0.5, 0.6, -5.0, -6.0, -7.0, 0.0, 1.0, 0.0, 0.0),
        ],
        dtype=dtype,
    )
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)
    return data


def _nht_model_stub(seed=42):
    return SimpleNamespace(
        feature_type=Features.Type.NHT,
        conf=OmegaConf.create(
            {
                "seed_initialization": seed,
                "model": {"nht_features": {"init_min": -1.0, "init_max": 1.0}},
            }
        ),
        device="cpu",
        positions=torch.nn.Parameter(torch.empty((0, 3))),
        rotation=torch.nn.Parameter(torch.empty((0, 4))),
        scale=torch.nn.Parameter(torch.empty((0, 3))),
        density=torch.nn.Parameter(torch.empty((0, 1))),
        particle_feature_dim=48,
        ray_feature_dim=24,
    )


def _sh_model_stub():
    return SimpleNamespace(
        feature_type=Features.Type.SH,
        device="cpu",
        positions=torch.nn.Parameter(torch.empty((0, 3))),
        rotation=torch.nn.Parameter(torch.empty((0, 4))),
        scale=torch.nn.Parameter(torch.empty((0, 3))),
        density=torch.nn.Parameter(torch.empty((0, 1))),
        features_albedo=torch.nn.Parameter(torch.empty((0, 3))),
        features_specular=torch.nn.Parameter(torch.empty((0, 0))),
        max_n_features=0,
    )


def test_nht_import_transfers_geometry_and_initializes_features_deterministically(tmp_path):
    ply_path = tmp_path / "geometry.ply"
    source = _write_geometry_ply(ply_path)

    model_a = _nht_model_stub()
    model_b = _nht_model_stub()
    MixtureOfGaussians.init_from_ply(model_a, str(ply_path), init_model=False)
    MixtureOfGaussians.init_from_ply(model_b, str(ply_path), init_model=False)

    np.testing.assert_allclose(model_a.positions.detach().numpy(), np.stack((source["x"], source["y"], source["z"]), axis=1))
    np.testing.assert_allclose(model_a.density.detach().numpy().squeeze(1), source["opacity"])
    np.testing.assert_allclose(
        model_a.scale.detach().numpy(), np.stack((source["scale_0"], source["scale_1"], source["scale_2"]), axis=1)
    )
    np.testing.assert_allclose(
        model_a.rotation.detach().numpy(), np.stack((source["rot_0"], source["rot_1"], source["rot_2"], source["rot_3"]), axis=1)
    )
    assert model_a.features.shape == (2, 48)
    assert model_a.n_active_features == 24
    torch.testing.assert_close(model_a.features, model_b.features)


def test_sh_import_retains_dc_features(tmp_path):
    ply_path = tmp_path / "geometry.ply"
    source = _write_geometry_ply(ply_path)
    model = _sh_model_stub()

    MixtureOfGaussians.init_from_ply(model, str(ply_path), init_model=False)

    np.testing.assert_allclose(
        model.features_albedo.detach().numpy(),
        np.stack((source["f_dc_0"], source["f_dc_1"], source["f_dc_2"]), axis=1),
    )
    assert model.features_specular.shape == (2, 0)
