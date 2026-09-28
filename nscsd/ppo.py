"""Single-agent PPO control (actor-critic + GAE) with the same shield, reward,
scenario prior, validation battery, champion selection and env-step budget as
NS-CSD.  It isolates the contribution of GRPO + continuous solution discovery
from that of simply removing Sentinel's co-evolving attacker.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import env as E
from . import scenarios as S
from .agents import NetDefender
from .policy import DefenderNet
from .rollout import validation_score
from .shield import repair


@dataclass
class PPOConfig:
    seed: int = 42
    updates: int = 420
    n_envs: int = 32
    T: int = 100
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    shield_penalty: float = 0.0
    reward: str = "score"
    eval_every: int = 17
    physics: str = "inline"


def train_ppo(cfg: PPOConfig, log=print):
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    net = DefenderNet()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    champion, champion_score, champion_iter = copy.deepcopy(net), -np.inf, 0
    history, env_steps = [], 0
    n, T = cfg.n_envs, cfg.T
    for u in range(1, cfg.updates + 1):
        scen = S.random_scenarios(rng, n)
        (typ, inten, mut), tape = S.contexts(rng, scen, T)
        obs = E.initial_obs(tape.legit0)
        O, TH, FO, DR, RW = [], [], [], [], []
        for t in range(T):
            thr, focus, drop = net.act(obs)
            O.append(obs); TH.append(thr); FO.append(focus); DR.append(drop)
            s_thr, s_focus, s_drop, rep, _ = repair(obs, thr, focus, drop)
            obs, info = E.step(obs, t, tape, (typ[:, t], inten[:, t], mut[:, t]), (s_thr, s_focus, s_drop),
                               np.zeros(n, bool), cfg.physics)
            RW.append(info[cfg.reward] - cfg.shield_penalty * rep)
        env_steps += n * T
        obs_t = torch.as_tensor(np.stack(O), dtype=torch.float32)          # (T, n, 12)
        with torch.no_grad():
            v = net.critic(obs_t).squeeze(-1).numpy()
            v_last = net.critic(torch.as_tensor(obs, dtype=torch.float32)).squeeze(-1).numpy()
        rew = np.stack(RW)
        adv = np.zeros_like(rew)
        last = np.zeros(n)
        for t in reversed(range(T)):
            nv = v_last * 0.0 if t == T - 1 else v[t + 1]  # episode truncates: bootstrap 0 like Sentinel's done flag
            delta = rew[t] + cfg.gamma * nv - v[t]
            last = delta + cfg.gamma * cfg.lam * last
            adv[t] = last
        ret = adv + v
        f = lambda x: torch.as_tensor(np.asarray(x).reshape(T * n, *np.asarray(x).shape[2:]))
        o, th, fo, dr = obs_t.reshape(T * n, -1), f(np.stack(TH)).float(), f(np.stack(FO)).long(), f(np.stack(DR)).float()
        a, r = f(adv).float(), f(ret).float()
        a = (a - a.mean()) / (a.std() + 1e-8)
        th, dr = th.clamp(1e-4, 1 - 1e-4), dr.clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            old = net.log_prob(o, th, fo, dr)
        N = o.shape[0]
        for _ in range(cfg.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                lp = net.log_prob(o[mb], th[mb], fo[mb], dr[mb])
                ratio = torch.exp(lp - old[mb])
                pg = -torch.min(ratio * a[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a[mb]).mean()
                vl = F.mse_loss(net.critic(o[mb]).squeeze(-1), r[mb])
                ent = net.entropy(o[mb]).mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if u % cfg.eval_every == 0 or u == cfg.updates:
            val, per = validation_score(NetDefender(copy.deepcopy(net), shielded=True), cfg.seed, physics=cfg.physics)
            if val > champion_score:
                champion, champion_score, champion_iter = copy.deepcopy(net), val, u
            history.append(dict(iter=u, env_steps=env_steps, val=val, champion=champion_score,
                                champion_iter=champion_iter, **{f"val_{k}": x for k, x in per.items()}))
            log(f"[ppo seed {cfg.seed}] upd {u:4d} steps {env_steps/1e6:.2f}M val {val:+.3f} "
                f"champ {champion_score:+.3f}@{champion_iter}")
    return dict(policy=net, champion=champion, champion_score=champion_score, history=history,
                config=asdict(cfg), best_tree=None)
