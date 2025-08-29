#!/usr/bin/env python3
# ==============================================================================
# 0. JAX 환경 설정 (가장 먼저 실행)
# ==============================================================================
# float64(배정밀도)를 비활성화하여 GPU에서 float32로 연산 속도 향상
from jax import config
config.update("jax_enable_x64", False)

import os
import argparse
import time
from dataclasses import replace
from typing import Tuple, Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from jax.flatten_util import ravel_pytree
from tqdm import trange
import mujoco
from mujoco import mjx

# ==============================================================================
# 체크포인트 유틸리티 함수
# - 학습 진행 상황을 .npz 파일로 안전하게 저장하고 불러옵니다.
# ==============================================================================
def save_checkpoint(path, theta, key, it, obs_dim, act_dim, meta=None, compressed=False):
    """학습 상태를 안전하게 저장합니다 (임시 파일 생성 후 원자적 교체)."""
    dirpath = os.path.dirname(os.path.abspath(path))
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    tmp_path = path + ".tmp"
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp_path, "wb") as f:
        saver(
            f,
            theta=np.asarray(theta, dtype=np.float32),
            key=np.asarray(key, dtype=np.uint32),
            iter=np.int64(it),
            obs_dim=np.int32(obs_dim),
            act_dim=np.int32(act_dim),
            meta=np.asarray(meta or {}, dtype=object),
            version=np.int32(1),
        )
    os.replace(tmp_path, path)

def load_checkpoint(path):
    """저장된 체크포인트를 불러옵니다."""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        ckpt = {
            "theta": jnp.array(d["theta"]),
            "obs_dim": int(d["obs_dim"]),
            "act_dim": int(d["act_dim"]),
            "iter": int(d["iter"]) if "iter" in d.files else 0,
            "key": jnp.array(d["key"]) if "key" in d.files else None,
            "meta": d["meta"].item() if "meta" in d.files else {},
        }
        return ckpt
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {path}. Error: {e}")
        return None

# ==============================================================================
# 1. MJX 기반 병렬 시뮬레이션 환경 생성
# - JAX의 vmap을 활용하여 수백 개의 환경을 GPU에서 병렬로 실행합니다.
# ==============================================================================
def make_mjx_env(xml_path: str,
                 action_repeat: int = 4,
                 z_threshold: float = 0.25,
                 ctrl_cost_weight: float = 1e-3,
                 alive_bonus: float = 0.05,
                 contact_penalty_weight: float = 2e-3,
                 tilt_penalty_weight: float = 1e-3,
                 forward_axis: str = "x",
                 forward_sign: int = 1,
                 target_speed: float = 0.8,
                 overspeed_weight: float = 0.5
                 ):
    m = mujoco.MjModel.from_xml_path(xml_path)
    mm = mjx.put_model(m)

    # 충돌 페널티 계산을 위한 지오메트리 ID 설정
    ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if ground_id == -1: ground_id = 0
    NON_FOOT_GEOM_NAMES = [
        'torso', 'front_left_leg', 'front_right_leg', 'back_left_leg', 'back_right_leg',
        'front_left_shin', 'front_right_shin', 'back_left_shin', 'back_right_shin'
    ]
    non_foot_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) for name in NON_FOOT_GEOM_NAMES]
    non_foot_geom_ids_jnp = jnp.array([i for i in non_foot_geom_ids if i != -1])

    # 액션 스케일링을 위한 제어 범위 설정
    ctrl_low = jnp.array(m.actuator_ctrlrange[:, 0], dtype=jnp.float32)
    ctrl_high = jnp.array(m.actuator_ctrlrange[:, 1], dtype=jnp.float32)
    ctrl_center = (ctrl_low + ctrl_high) / 2.0
    ctrl_half = (ctrl_high - ctrl_low) / 2.0

    def scale_action(a_unit: jnp.ndarray) -> jnp.ndarray:
        """정책의 출력(-1~1)을 실제 모터 제어 범위로 변환합니다."""
        return ctrl_center + ctrl_half * jnp.tanh(a_unit)

    def obs_from_dd(dd: Any) -> jnp.ndarray:
        """데이터에서 관측(observation) 벡터를 추출합니다."""
        # 학습/뷰어 일관성: root free joint(7) 제외 + 모든 관절 속도
        return jnp.concatenate([dd.qpos[7:], dd.qvel], axis=0).astype(jnp.float32)

    # 초기 상태 값들
    qpos0 = jnp.array(m.qpos0, dtype=jnp.float32)
    zero_qvel = jnp.zeros((m.nv,), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((m.nu,), dtype=jnp.float32)

    # 보상 함수 관련 설정
    axis2idx = {"x": 0, "y": 1, "z": 2}
    fwd_idx = axis2idx[forward_axis]
    fwd_sign = jnp.float32(forward_sign)
    tgt_speed = jnp.float32(target_speed)
    over_w = jnp.float32(overspeed_weight)

    def reset_single(key: jax.Array) -> Tuple[Tuple[Any, ...], jnp.ndarray]:
        """단일 환경을 리셋합니다."""
        dd = mjx.make_data(mm)
        noise = 0.01 * random.normal(key, (qpos0.shape[0],), dtype=jnp.float32)
        qpos = qpos0 + noise.at[:7].set(0.0) # 루트에는 노이즈 미적용
        dd = replace(dd, qpos=qpos, qvel=zero_qvel, ctrl=zero_ctrl)
        obs = obs_from_dd(dd)
        return (dd, obs, jnp.array(False), zero_ctrl), obs

    def step_single(state, a_unit: jnp.ndarray):
        """단일 환경의 한 스텝을 진행합니다."""
        dd, _, done_prev, _ = state
        ctrl = scale_action(a_unit)

        def integrate_if_alive(dd_in):
            """에피소드가 끝나지 않았을 때만 물리 시뮬레이션을 진행합니다."""
            def body_fn(carry, _):
                return mjx.step(mm, replace(carry, ctrl=ctrl)), None
            dd_out, _ = lax.scan(body_fn, dd_in, xs=None, length=action_repeat)
            return dd_out

        dd = lax.cond(done_prev, lambda d: d, integrate_if_alive, dd)
        obs = obs_from_dd(dd)

        # --- 보상(Reward) 함수 ---
        forward_speed = (dd.qvel[fwd_idx] * fwd_sign).astype(jnp.float32)
        reward_forward = jnp.minimum(forward_speed, tgt_speed)
        overspeed = jnp.maximum(forward_speed - tgt_speed, 0.0)
        overspeed_pen = over_w * (overspeed ** 2)
        ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(ctrl))

        # 발 이외의 부위가 바닥에 닿으면 페널티
        contact = dd._impl.contact
        g1, g2 = contact.geom1, contact.geom2
        ids = non_foot_geom_ids_jnp
        is_nonfoot = ((g1[:, None] == ids[None, :]).any(axis=1) | (g2[:, None] == ids[None, :]).any(axis=1))
        is_ground = (g1 == ground_id) | (g2 == ground_id)
        valid = jnp.arange(g1.shape[0], dtype=jnp.int32) < dd._impl.ncon
        num_bad_contacts = jnp.sum(is_nonfoot & is_ground & valid)
        contact_penalty = contact_penalty_weight * num_bad_contacts.astype(jnp.float32)

        # 몸통이 기울어지면 페널티
        torso_angular_velocity = dd.qvel[3:5]
        tilt_penalty = tilt_penalty_weight * jnp.sum(jnp.square(torso_angular_velocity))

        reward = alive_bonus + reward_forward - overspeed_pen - ctrl_cost - contact_penalty - tilt_penalty

        # --- 종료(Done) 조건 ---
        z = dd.qpos[2]
        isnan = jnp.any(jnp.isnan(dd.qpos)) | jnp.any(jnp.isnan(dd.qvel))
        done_now = jnp.logical_or(isnan, z < z_threshold)
        done = jnp.logical_or(done_prev, done_now)

        new_state = (dd, obs, done, ctrl)
        return new_state, (obs, reward, done)

    # vmap을 사용하여 단일 환경 함수들을 병렬 배치 버전으로 변환
    reset_batch = jax.jit(jax.vmap(reset_single))
    step_batch = jax.jit(jax.vmap(step_single))

    env_info = {"obs_dim": int(m.nq - 7 + m.nv), "act_dim": int(m.nu)}
    return reset_batch, step_batch, env_info

# ==============================================================================
# 2. 정책 정의 및 ARS(Augmented Random Search) 학습 루프
# ==============================================================================
def make_policy_fns(obs_dim: int, act_dim: int) -> Tuple[Callable, Callable]:
    """선형 정책(Linear Policy) 함수를 생성합니다."""
    params_pytree_shape = (jnp.zeros((obs_dim, act_dim)), jnp.zeros((act_dim,)))
    def ravel_pytree_partial(pytree): return ravel_pytree(pytree)[0]
    _, unravel_fn = ravel_pytree(params_pytree_shape)

    @jax.jit
    def policy_apply_fn(theta: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        """주어진 정책 파라미터(theta)와 관측(obs)으로 액션을 계산합니다."""
        W, b = unravel_fn(theta)
        return jnp.tanh(obs @ W + b)
    return ravel_pytree_partial, policy_apply_fn

def ars_train(xml_path: str, **kwargs):
    """ARS 알고리즘으로 정책을 학습합니다."""
    seed, num_envs, episode_length = kwargs['seed'], kwargs['num_envs'], kwargs['episode_length']
    steps_per_iter = (2 * kwargs["num_dirs"] * num_envs * episode_length * kwargs["action_repeat"])
    print(f"[Config] steps/iter ≈ {steps_per_iter:,}  | dir_chunk={kwargs.get('dir_chunk')}")

    # 환경 및 정책 초기화
    reset_batch, step_batch, info = make_mjx_env(
        xml_path=xml_path,
        **{k: kwargs[k] for k in ['action_repeat', 'z_threshold', 'ctrl_cost_weight', 'alive_bonus',
                                  'contact_penalty_weight', 'tilt_penalty_weight', 'forward_axis',
                                  'forward_sign', 'target_speed', 'overspeed_weight']}
    )
    obs_dim, act_dim = info["obs_dim"], info["act_dim"]
    key = random.PRNGKey(seed)
    ravel, policy_apply = make_policy_fns(obs_dim, act_dim)
    key, kW = random.split(key)
    W0 = 0.01 * random.normal(kW, (obs_dim, act_dim))
    b0 = jnp.zeros((act_dim,))
    theta = ravel((W0, b0))
    start_it = 0

    # 체크포인트에서 이어하기
    if kwargs.get("resume", False):
        ckpt = load_checkpoint(kwargs["save_path"])
        if ckpt is not None:
            if ckpt["obs_dim"] != obs_dim or ckpt["act_dim"] != act_dim:
                raise ValueError(f"Checkpoint dim mismatch: ckpt({ckpt['obs_dim']},{ckpt['act_dim']}) vs model({obs_dim},{act_dim})")
            theta, start_it = ckpt["theta"], ckpt["iter"]
            if ckpt["key"] is not None: key = ckpt["key"]
            print(f"[Resume] Loaded checkpoint from {kwargs['save_path']} at iter {start_it}")

    @jax.jit
    def rollout_return(theta_: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
        """주어진 정책으로 롤아웃을 실행하고 누적 보상을 반환합니다."""
        state, obs = reset_batch(keys)
        def body(carry, _):
            st, ob, ret = carry
            act = policy_apply(theta_, ob)
            st, (ob_next, r, done) = step_batch(st, act)
            return (st, ob_next, ret + r * (1.0 - done)), None
        (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(num_envs)), None, length=episode_length)
        return returns

    def make_eval_chunk_fn(rollout_return_fn, noise_std):
        """메모리 관리를 위해 탐색 방향(delta)을 청크 단위로 평가하는 함수를 생성합니다."""
        @jax.jit
        def eval_chunk(theta_, deltas_chunk, keys_chunk):
            def eval_one(delta, keys):
                r_plus = rollout_return_fn(theta_ + noise_std * delta, keys).mean()
                r_minus = rollout_return_fn(theta_ - noise_std * delta, keys).mean()
                return r_plus, r_minus
            return jax.vmap(eval_one)(deltas_chunk, keys_chunk)
        return eval_chunk

    eval_chunk = make_eval_chunk_fn(rollout_return, kwargs["noise_std"])

    # --- 메인 학습 루프 ---
    total_new_iters = kwargs["iterations"]
    ckpt_every = kwargs["ckpt_every"] or kwargs["eval_every"]
    pbar = trange(total_new_iters, desc=f"ARS training (start iter: {start_it})")
    for it_local in pbar:
        global_it = start_it + it_local
        key, k_delta, k_dirs = random.split(key, 3)
        
        # 1. 탐색 방향(델타) 샘플링
        deltas = random.normal(k_delta, (kwargs["num_dirs"], theta.shape[0]))
        base_keys = random.split(k_dirs, kwargs["num_dirs"] * num_envs).reshape(kwargs["num_dirs"], num_envs, 2)

        # 2. 델타 방향으로 정책 평가
        dir_chunk = kwargs.get("dir_chunk")
        if dir_chunk:
            R_plus_list, R_minus_list = [], []
            for s in range(0, kwargs["num_dirs"], dir_chunk):
                e = min(s + dir_chunk, kwargs["num_dirs"])
                r_p, r_m = eval_chunk(theta, deltas[s:e], base_keys[s:e])
                R_plus_list.append(r_p); R_minus_list.append(r_m)
            R_plus = jnp.concatenate(R_plus_list, axis=0)
            R_minus = jnp.concatenate(R_minus_list, axis=0)
        else:
            R_plus, R_minus = eval_chunk(theta, deltas, base_keys)

        # 3. 정책 업데이트
        scores = jnp.maximum(R_plus, R_minus)
        topk_idx = jnp.argsort(scores)[-kwargs["top_dirs"]:]
        R_plus_top, R_minus_top, deltas_top = R_plus[topk_idx], R_minus[topk_idx], deltas[topk_idx]
        reward_std = jnp.std(jnp.concatenate([R_plus_top, R_minus_top])) + 1e-8
        weighted = (R_plus_top - R_minus_top)[:, None] * deltas_top
        grad_est = jnp.mean(weighted, axis=0) / reward_std
        theta += kwargs["step_size"] * grad_est
        pbar.set_postfix({"mean+": f"{R_plus.mean():.2f}", "mean-": f"{R_minus.mean():.2f}", "best": f"{scores.max():.2f}"})

        # 4. 평가 및 저장
        if (global_it + 1) % kwargs["eval_every"] == 0 or it_local == total_new_iters - 1:
            key, k_eval = random.split(key)
            eval_keys = random.split(k_eval, num_envs)
            ret = rollout_return(theta, eval_keys).mean()
            print(f"\n[Iter {global_it+1}] Eval Return: {float(ret):.2f}")

        if (global_it + 1) % ckpt_every == 0 or it_local == total_new_iters - 1:
            save_checkpoint(
                kwargs["save_path"], theta=theta, key=key, it=global_it + 1,
                obs_dim=obs_dim, act_dim=act_dim,
                meta={k: kwargs.get(k) for k in ["xml_path", "forward_axis", "forward_sign", "target_speed", "overspeed_weight"]},
            )

# ==============================================================================
# 3. 추론(Inference) 및 시각화(Viewer)
# ==============================================================================
def run_inference(xml_path: str, ckpt_path: str, **kwargs):
    """학습된 정책의 평균 성능을 렌더링 없이 계산합니다."""
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None: raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    theta, obs_dim, act_dim = ckpt["theta"], ckpt["obs_dim"], ckpt["act_dim"]
    _, policy_apply = make_policy_fns(obs_dim, act_dim)

    env_params = {k: kwargs[k] for k in ['action_repeat', 'z_threshold', 'ctrl_cost_weight', 'alive_bonus',
                                         'contact_penalty_weight', 'tilt_penalty_weight', 'forward_axis',
                                         'forward_sign', 'target_speed', 'overspeed_weight']}
    reset_batch, step_batch, _ = make_mjx_env(xml_path=xml_path, **env_params)

    keys = random.split(random.PRNGKey(kwargs['seed']), kwargs['num_envs'])
    state, obs = reset_batch(keys)
    def body(carry, _):
        st, ob, ret = carry
        act = policy_apply(theta, ob)
        st, (ob_next, r, done) = step_batch(st, act)
        return (st, ob_next, ret + r * (1.0 - done)), None
    (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(kwargs['num_envs'], dtype=jnp.float32)), None, length=kwargs['episode_length'])
    print(f"Inference mean return: {float(jnp.mean(returns)):.2f}")

def run_viewer(xml_path: str, ckpt_path: str, action_repeat: int):
    """MuJoCo Viewer로 학습된 정책의 움직임을 시각화합니다."""
    try:
        from mujoco import viewer as mj_viewer
    except Exception as e:
        raise RuntimeError("MuJoCo viewer 모듈을 찾을 수 없습니다. 'pip install -U mujoco glfw' 후 재시도하세요.") from e

    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None: raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    theta, obs_dim, act_dim = np.array(ckpt["theta"]), ckpt["obs_dim"], ckpt["act_dim"]
    
    # 학습과 동일한 방식으로 theta에서 W, B 복원
    w_size = obs_dim * act_dim
    W, B = theta[:w_size].reshape(obs_dim, act_dim), theta[w_size:]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # 학습과 동일한 액션 스케일링
    low, high = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    ctrl_center, ctrl_half = 0.5 * (low + high), 0.5 * (high - low)

    def obs_from_md(data): return np.concatenate([data.qpos[7:], data.qvel]).astype(np.float32)
    def scale_action_np(a_unit): return ctrl_center + ctrl_half * np.tanh(a_unit)

    def reset_env():
        mujoco.mj_resetData(m, d)
        noise = 0.01 * np.random.randn(d.qpos.shape[0])
        noise[:7] = 0.0
        d.qpos[:] += noise
        mujoco.mj_forward(m, d)

    reset_env()
    with mj_viewer.launch_passive(m, d) as v:
        def key_callback(keycode):
            if keycode in (ord('R'), ord('r')): reset_env()
        v.user_key_callback = key_callback
        while v.is_running():
            step_start = time.time()
            obs = obs_from_md(d)
            # 학습과 동일한 정책 적용 (obs @ W + b)
            a_unit = np.tanh(obs @ W + B)
            d.ctrl[:] = scale_action_np(a_unit)
            for _ in range(action_repeat):
                mujoco.mj_step(m, d)
            v.sync()
            time_until_next_step = m.opt.timestep * action_repeat - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# ==============================================================================
# 4. 스크립트 실행 메인 블록
# - 커맨드 라인 인자를 파싱하여 학습, 추론, 시각화 중 하나를 실행합니다.
# ==============================================================================
def parse_args():
    """스크립트 실행을 위한 커맨드 라인 인자를 정의하고 파싱합니다."""
    p = argparse.ArgumentParser(description="MJX + ARS trainer/viewer for quadruped")
    p.add_argument("--xml", type=str, required=True, dest="xml_path", help="로봇 모델 XML 파일 경로")
    p.add_argument("--infer", action="store_true", help="추론 모드 실행")
    p.add_argument("--view", action="store_true", help="시각화(뷰어) 모드 실행")
    p.add_argument("--resume", action="store_true", help="저장된 체크포인트에서 학습 이어하기")
    p.add_argument("--seed", type=int, default=0, help="랜덤 시드")
    
    # --- ARS 하이퍼파라미터 ---
    p.add_argument("--num-envs", type=int, default=64, help="병렬 환경 개수")
    p.add_argument("--num-dirs", type=int, default=16, help="탐색 방향(델타) 개수")
    p.add_argument("--top-dirs", type=int, default=4, help="업데이트에 사용할 상위 델타 개수")
    p.add_argument("--episode-length", type=int, default=150, help="에피소드 최대 길이")
    p.add_argument("--iterations", type=int, default=100, help="총 학습 반복 횟수")
    p.add_argument("--dir-chunk", type=int, default=16, help="메모리 관리를 위한 델타 평가 청크 크기")
    p.add_argument("--step-size", type=float, default=0.02, help="학습률 (learning rate)")
    p.add_argument("--noise-std", type=float, default=0.03, help="탐색 노이즈 표준편차")

    # --- 환경 파라미터 ---
    p.add_argument("--action-repeat", type=int, default=2, help="액션 반복 횟수")
    p.add_argument("--z-threshold", type=float, default=0.25, help="종료 조건이 되는 최소 높이")
    p.add_argument("--ctrl-cost-weight", type=float, default=1e-4, help="제어 비용 가중치")
    p.add_argument("--alive-bonus", type=float, default=0.05, help="생존 보너스")
    p.add_argument("--contact-penalty-weight", type=float, default=2e-3, help="잘못된 충돌 페널티 가중치")
    p.add_argument("--tilt-penalty-weight", type=float, default=1e-3, help="몸통 기울어짐 페널티 가중치")
    p.add_argument("--forward-axis", choices=["x", "y", "z"], default="x", help="전진 방향 축")
    p.add_argument("--forward-sign", type=int, choices=[-1, 1], default=1, help="전진 방향 부호")
    p.add_argument("--target-speed", type=float, default=0.5, help="목표 전진 속도")
    p.add_argument("--overspeed-weight", type=float, default=1.0, help="과속 페널티 가중치")

    # --- 저장 및 평가 ---
    p.add_argument("--eval-every", type=int, default=10, help="평가 주기 (iteration)")
    p.add_argument("--ckpt-every", type=int, default=10, help="체크포인트 저장 주기 (iteration)")
    p.add_argument("--save-path", type=str, default="ars_policy.npz", help="체크포인트 저장 경로")
    p.add_argument("--gl-backend", type=str, choices=["egl", "glfw", "osmesa"], default=None, help="MuJoCo 렌더링 백엔드")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # GL 백엔드 설정: viewer면 glfw, 아니면 명시 없을 때 osmesa 기본(서버 환경에서 안전)
    if args.view:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = args.gl_backend or os.environ.get("MUJOCO_GL", "osmesa")

    if args.view:
        run_viewer(xml_path=args.xml_path, ckpt_path=args.save_path, action_repeat=args.action_repeat)
    elif args.infer:
        run_inference(xml_path=args.xml_path, ckpt_path=args.save_path, **vars(args))
    else:
        ars_train(**vars(args))

