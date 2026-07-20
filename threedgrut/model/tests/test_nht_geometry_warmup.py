# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from threedgrut.trainer import Trainer3DGRUT


class _Strategy:
    def __init__(self):
        self.suspended = False

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.suspended = False


def _trainer_stub():
    trainer = object.__new__(Trainer3DGRUT)
    position = torch.nn.Parameter(torch.ones(1))
    feature = torch.nn.Parameter(torch.ones(1))
    trainer.model = SimpleNamespace(
        optimizer=SimpleNamespace(
            param_groups=[
                {"name": "positions", "lr": 0.4, "params": [position]},
                {"name": "features", "lr": 0.2, "params": [feature]},
            ]
        )
    )
    trainer.strategy = _Strategy()
    trainer._geometry_warmup_steps = 1
    trainer._geometry_lr_scale = 0.1
    trainer._in_geometry_warmup = False
    trainer._in_color_refine = False
    return trainer, position


def test_nht_geometry_warmup_freezes_then_scales_geometry_learning_rate_once():
    trainer, position = _trainer_stub()

    trainer._apply_geometry_warmup(0)
    assert trainer.strategy.suspended
    assert trainer.model.optimizer.param_groups[0]["lr"] == 0.0
    assert trainer.model.optimizer.param_groups[1]["lr"] == 0.2

    position.grad = torch.ones_like(position)
    trainer._zero_geometry_warmup_grads()
    assert position.grad is None

    trainer._apply_geometry_warmup(1)
    assert not trainer.strategy.suspended

    # Mimic the scheduler restoring the base learning rate, then apply the
    # post-warmup multiplier once at the end of the iteration.
    trainer.model.optimizer.param_groups[0]["lr"] = 0.4
    trainer._apply_geometry_warmup(1, apply_lr_scale=True)
    assert trainer.model.optimizer.param_groups[0]["lr"] == pytest.approx(0.04)

    trainer._apply_geometry_warmup(2)
    assert trainer.model.optimizer.param_groups[0]["lr"] == pytest.approx(0.04)
