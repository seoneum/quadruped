
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp

try:
    import mujoco
    from mujoco import mjx
except Exception as e:
    raise RuntimeError('mujoco/mujoco-mjx required. Install with `uv add mujoco mujoco-mjx jax jaxlib`.')


@dataclass
class MJXEnvCfg:
    mjcf_path: str
    actuator_names: Sequence[str]
    dt: float = 0.002
    horizon: int = 1000
    obs_joint_qpos: Sequence[str] | None = None
    obs_joint_qvel: Sequence[str] | None = None


def _build_model(mjcf_path: str):
    mjc_m = mujoco.MjModel.from_xml_path(mjcf_path)
    mjc_d = mujoco.MjData(mjc_m)
    # to MJX
    mx_m = mjx.put_model(mjc_m)
    return mjc_m, mjc_d, mx_m


def _name_to_id(model: mujoco.MjModel, obj: int, names: Sequence[str]) -> np.ndarray:
    ids = []
    for n in names:
        i = mujoco.mj_name2id(model, obj, n)
        if i < 0:
            raise ValueError(f'Name {n} not found')
        ids.append(i)
    return np.array(ids, dtype=np.int32)


class MJXParallelEnv:
    def __init__(self, cfg: MJXEnvCfg, num_envs: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self.mjc_m, self.mjc_d, self.mx_m = _build_model(cfg.mjcf_path)
        self.act_ids = _name_to_id(self.mjc_m, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.actuator_names)
        # observation joints (default: all dof)
        if cfg.obs_joint_qpos is None:
            self.qpos_ids = np.arange(self.mjc_m.nq, dtype=np.int32)
        else:
            self.qpos_ids = _name_to_id(self.mjc_m, mujoco.mjtObj.mjOBJ_JOINT, cfg.obs_joint_qpos)
        if cfg.obs_joint_qvel is None:
            self.qvel_ids = np.arange(self.mjc_m.nv, dtype=np.int32)
        else:
            self.qvel_ids = _name_to_id(self.mjc_m, mujoco.mjtObj.mjOBJ_JOINT, cfg.obs_joint_qvel)
        # initial state
        self.mx_d0 = mjx.make_data(self.mx_m)

        def reset_key(key):
            # small noise in qpos/qvel
            d = self.mx_d0
            return d

        self.reset_key = jax.jit(jax.vmap(reset_key))

        def step_fn(d, ctrl):
            d = d.replace(ctrl=ctrl)
            d = mjx.step(self.mx_m, d)
            return d

        self.step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: jax.Array):
        keys = jax.random.split(key, self.num_envs)
        d = self.reset_key(keys)
        return d

    def obs(self, d) -> jax.Array:
        # concatenate qpos and qvel slices
        qpos = d.qpos[:, :]
        qvel = d.qvel[:, :]
        return jnp.concatenate([qpos, qvel], axis=-1)

    def step(self, d, action: jax.Array, torque_limit: float = 2.3):
        # action expected shape: (num_envs, n_act)
        # scale to actuator torque limits (N·m)
        ctrl = jnp.clip(action, -1.0, 1.0) * torque_limit
        d2 = self.step_fn(d, ctrl)
        # simple reward: forward x-velocity of base, minus torque^2 cost
        velx = d2.qvel[:, 0]
        reward = velx - 0.001 * jnp.sum((ctrl/torque_limit)**2, axis=-1)
        done = jnp.zeros((ctrl.shape[0],), dtype=bool)
        return d2, reward, done
