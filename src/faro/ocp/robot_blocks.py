"""Independent robot runtime blocks and deterministic flat-vector layouts."""
from dataclasses import dataclass
from collections.abc import Mapping

import casadi as cs
import numpy as np
import pinocchio.casadi as cpin

from faro.ocp.kinematics import RobotKinematics, qrob_integrate


@dataclass
class RobotBlock:
    rid: str
    cfg: object
    model: object
    data: object
    cmodel: object
    cdata: object
    nq: int
    nv: int
    kinematics: RobotKinematics
    integrate: cs.Function
    collision_model: object


def setup_robot_blocks(scene, collision_models=None):
    """Keep frame names local and mutable Pinocchio data private to each instance."""
    blocks = {}
    for rid, cfg in scene.robots.items():
        model = cfg.pin_model
        cmodel = cpin.Model(model)
        collision_model = (collision_models or {}).get(rid, cfg.collision_model)
        frames = {("ee", patch): patch for patch in cfg.patch_ees}
        frames.update({("collision", name): capsule.frame
                       for name, capsule in collision_model.capsules.items()})
        velocity_frames = {patch: patch for patch in cfg.patch_ees}
        blocks[rid] = RobotBlock(
            rid, cfg, model, model.createData(), cmodel, cmodel.createData(),
            model.nq, model.nv, RobotKinematics(cmodel, frames, velocity_frames),
            qrob_integrate(cmodel), collision_model,
        )
    return blocks


class BlockLayout:
    """Ordered named slices, shared by symbolic assembly and numeric packing."""

    def __init__(self, sizes: Mapping):
        self.sizes = dict(sizes)
        self.slices = {}
        offset = 0
        for name, size in self.sizes.items():
            if int(size) < 0:
                raise ValueError(f"Negative block size for {name!r}")
            self.slices[name] = slice(offset, offset + int(size))
            offset += int(size)
        self.size = offset

    @classmethod
    def from_symbols(cls, symbols):
        return cls({name: int(value.numel()) for name, value in symbols.items()})

    def pack(self, values):
        if set(values) != set(self.sizes):
            raise ValueError("Block keys do not match the layout")
        arrays = []
        for name, size in self.sizes.items():
            value = np.asarray(values[name], dtype=float)
            if value.size != size or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid block {name!r}: expected {size} finite values")
            arrays.append(value.reshape(-1))
        return np.concatenate(arrays) if arrays else np.zeros(0)

    def unpack(self, vector):
        value = np.asarray(vector)
        if value.shape[-1] != self.size:
            raise ValueError(f"Expected final dimension {self.size}, got {value.shape}")
        return {name: value[..., block] for name, block in self.slices.items()}
