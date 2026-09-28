"""Neural policies.

``DefenderNet`` keeps the exact parameter layout of Sentinel's ``ActorCritic``
(two 64-unit ReLU layers, Beta heads for threshold and drop rate, a categorical
head for the filtering focus) so Sentinel's released checkpoints load with
``load_state_dict``.  GRPO never uses the critic; it is kept only so the same
class can host both kinds of checkpoint.  ``AttackerNet`` mirrors Sentinel's
PPO attacker and is used only to replay Sentinel's own evaluation protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

EPS = 1e-4


class DefenderNet(nn.Module):
    def __init__(self, obs_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.actor_shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden), nn.ReLU())
        self.threshold_alpha = nn.Linear(hidden, 1)
        self.threshold_beta = nn.Linear(hidden, 1)
        self.focus_logits = nn.Linear(hidden, 3)
        self.drop_prob_alpha = nn.Linear(hidden, 1)
        self.drop_prob_beta = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def heads(self, obs):
        h = self.actor_shared(obs)
        ta = F.softplus(self.threshold_alpha(h)).squeeze(-1) + 1.0
        tb = F.softplus(self.threshold_beta(h)).squeeze(-1) + 1.0
        fl = self.focus_logits(h)
        da = F.softplus(self.drop_prob_alpha(h)).squeeze(-1) + 1.0
        db = F.softplus(self.drop_prob_beta(h)).squeeze(-1) + 1.0
        return ta, tb, fl, da, db

    def dists(self, obs):
        ta, tb, fl, da, db = self.heads(obs)
        return Beta(ta, tb), Categorical(logits=fl), Beta(da, db)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False, generator=None):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        ta, tb, fl, da, db = self.heads(obs_t)
        if deterministic:
            thr = ta / (ta + tb)
            focus = fl.argmax(-1)
            drop = da / (da + db)
        else:
            thr = Beta(ta, tb).sample()
            focus = Categorical(logits=fl).sample()
            drop = Beta(da, db).sample()
        thr = thr.clamp(EPS, 1 - EPS)
        drop = drop.clamp(EPS, 1 - EPS)
        return thr.double().numpy(), focus.numpy().astype(int), drop.double().numpy()

    def log_prob(self, obs, thr, focus, drop):
        t, f, d = self.dists(obs)
        return t.log_prob(thr) + f.log_prob(focus) + d.log_prob(drop)

    def entropy(self, obs):
        t, f, d = self.dists(obs)
        return t.entropy() + f.entropy() + d.entropy()


def kl_to(ref: DefenderNet, pol: DefenderNet, obs):
    """Analytic KL(pi || ref) summed over the three independent action heads."""
    tp, fp, dp = pol.dists(obs)
    with torch.no_grad():
        tr, fr, dr = ref.dists(obs)
    kl = torch.distributions.kl_divergence
    return kl(tp, tr) + kl(fp, fr) + kl(dp, dr)


class AttackerNet(nn.Module):
    """Sentinel's PPO attacker architecture (evaluation replay only)."""

    def __init__(self, obs_dim: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.type_logits = nn.Linear(hidden, 6)
        self.intensity_alpha = nn.Linear(hidden, 1)
        self.intensity_beta = nn.Linear(hidden, 1)
        self.mutation_alpha = nn.Linear(hidden, 1)
        self.mutation_beta = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)

    @torch.no_grad()
    def act(self, obs):
        h = self.net(torch.as_tensor(obs, dtype=torch.float32))
        typ = self.type_logits(h).argmax(-1)
        ia = F.softplus(self.intensity_alpha(h)).squeeze(-1) + 1.0
        ib = F.softplus(self.intensity_beta(h)).squeeze(-1) + 1.0
        ma = F.softplus(self.mutation_alpha(h)).squeeze(-1) + 1.0
        mb = F.softplus(self.mutation_beta(h)).squeeze(-1) + 1.0
        return typ.numpy().astype(int), (ia / (ia + ib)).double().numpy(), (ma / (ma + mb)).double().numpy()


def load_sentinel_defender(path) -> DefenderNet:
    net = DefenderNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def load_sentinel_attacker(path) -> AttackerNet:
    net = AttackerNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return net
