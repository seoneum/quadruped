
from __future__ import annotations
import os
import time
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from mjx_env import MJXEnvCfg, MJXParallelEnv
from utils import Normalizer, Saver

@dataclass
class ARSCfg:
    n_envs: int = 64
    horizon: int = 500
    n_directions: int = 32
    step_size: float = 0.02
    noise_std: float = 0.03
    top_b: int = 16


def rollout(env: MJXParallelEnv, key, theta, horizon, normalizer: Normalizer | None = None):
    key, k = jax.random.split(key)
    d = env.reset(k)
    def body(carry, t):
        d, rew, key = carry
        obs = env.obs(d); obs_np = np.asarray(obs);
        if normalizer is not None:
            normalizer.update_batch(obs_np); obs_np = normalizer.normalize(obs_np)
        obs = jnp.asarray(obs_np)
        # normalize observations (running mean/std can be added)
        act = obs @ theta
        d2, r, done = env.step(d, act, torque_limit=2.3)
        return (d2, rew + r, key), None
    (d, ret, _), _ = jax.lax.scan(body, (d, jnp.zeros((env.num_envs,)), key), None, length=horizon)
    return ret.mean()


def train(mjcf_path: str, actuator_names: list[str], cfg: ARSCfg):
    env = MJXParallelEnv(MJXEnvCfg(mjcf_path=mjcf_path, actuator_names=actuator_names, horizon=cfg.horizon), num_envs=cfg.n_envs)
    normalizer = Normalizer(env.mx_m.nq + env.mx_m.nv)
    saver = Saver()
    key = jax.random.PRNGKey(0)
    obs_dim = env.mx_m.nq + env.mx_m.nv
    act_dim = len(actuator_names)
    theta = jnp.zeros((obs_dim, act_dim))

    best = -1e9
    import csv, os
    log_path = os.path.join('rl','logs','training.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # header
    open(log_path, 'a').close()
    for it in range(1000):
        key, k_dirs, k_roll = jax.random.split(key, 3)
        deltas = jax.random.normal(k_dirs, (cfg.n_directions, obs_dim, act_dim))
        def eval_dir(delta, plus):
            th = theta + (cfg.noise_std * delta if plus else -cfg.noise_std * delta)
            return rollout(env, k_roll, th, cfg.horizon, normalizer)
        rets_plus = jax.vmap(lambda d: eval_dir(d, True))(deltas)
        rets_minus = jax.vmap(lambda d: eval_dir(d, False))(deltas)
        # select top directions
        std = jnp.std(jnp.concatenate([rets_plus, rets_minus])) + 1e-8
        scores = jnp.maximum(rets_plus, rets_minus)
        idx = jnp.argsort(-scores)[:cfg.top_b]
        step = jnp.zeros_like(theta)
        for i in idx:
            step = step + (rets_plus[i] - rets_minus[i]) * deltas[i]
        theta = theta + cfg.step_size / (cfg.top_b * std) * step
        ret = rollout(env, k_roll, theta, cfg.horizon, normalizer)
        best = max(best, float(ret))
        print(f"Iter {it}: ret={float(ret):.3f} best={best:.3f}")
        with open(log_path, 'a', newline='') as f:
            import csv
            csv.writer(f).writerow([it, float(ret), float(best)])
        if it % 10 == 0:
            path = saver.save(it, np.asarray(theta), {'cfg': cfg.__dict__})
            print('checkpoint', path, 'theta_norm', float(jnp.linalg.norm(theta)))

if __name__ == '__main__':
    # Example using the simple quadruped.xml with 8 actuators
    here = os.path.dirname(__file__)
    mjcf = os.path.join(here, '..', 'ros2_ws', 'src', 'open_quadruped_sim_mjx', 'open_quadruped_sim_mjx', 'assets', 'quadruped.xml')
    # Order should match MJCF actuator names, action in [-1,1]
    actuators = [
        'hip_front_left','knee_front_left','hip_front_right','knee_front_right',
        'hip_back_left','knee_back_left','hip_back_right','knee_back_right'
    ]
    train(mjcf, actuators, ARSCfg())
