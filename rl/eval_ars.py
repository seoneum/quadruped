
from __future__ import annotations
import os, sys
import numpy as np
import jax
import jax.numpy as jnp
from mjx_env import MJXEnvCfg, MJXParallelEnv

# Evaluate a saved linear policy theta on N environments for given horizon

def load_theta(path: str) -> np.ndarray:
    if path.endswith('.npz'):
        d = np.load(path)
        return d['theta']
    raise ValueError('Unsupported checkpoint format: ' + path)


def eval_return(env: MJXParallelEnv, theta: jnp.ndarray, horizon: int) -> float:
    key = jax.random.PRNGKey(42)
    d = env.reset(key)
    ret = jnp.zeros((env.num_envs,))
    def body(carry, t):
        d, ret = carry
        obs = env.obs(d)
        act = obs @ theta
        d2, r, _ = env.step(d, act, torque_limit=2.3)
        return (d2, ret + r), None
    (d, ret), _ = jax.lax.scan(body, (d, ret), None, length=horizon)
    return float(ret.mean())

if __name__ == '__main__':
    here = os.path.dirname(__file__)
    # Evaluate against the large Quardred MJCF by default
    mjcf = os.path.join(here, '..', 'ros2_ws', 'src', 'open_quadruped_sim_mjx', 'open_quadruped_sim_mjx', 'assets', 'Quardred_08272115_minimum.xml')
    # Order per leg: [hip, knee] x [fl, fr, bl, br]
    actuators: list[str] = [
        'act_Left_Hip_Joint','act_Lower_Leg_33',
        'act_Right_Hip_Joint','act_Lower_Leg_134',
        'act_Left_Hip_Joint_1','act_Lower_Leg_1',
        'act_Right_Hip_Joint_1','act_Lower_Leg_4_1',
    ]
    theta_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'checkpoints', 'latest.npz')
    theta = jnp.asarray(load_theta(theta_path))
    env = MJXParallelEnv(MJXEnvCfg(mjcf_path=mjcf, actuator_names=actuators, horizon=500), num_envs=64)
    avg_ret = eval_return(env, theta, horizon=500)
    print('Average return:', avg_ret)
