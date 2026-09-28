"""Train one NS-CSD variant (or the PPO control) for one seed and save artefacts.

    python scripts/train.py --method nscsd --seed 42
    python scripts/train.py --method no_discovery --seed 42 --iters 50   # quick run
"""
import argparse
import json
import os
import pickle
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nscsd.crystallize import describe  # noqa: E402
from nscsd.grpo import Config, train  # noqa: E402

torch.set_num_threads(1)

VARIANTS = {
    "nscsd": {},
    "no_discovery": dict(discovery=False),
    "no_elite": dict(elite=False),
    "no_anchor": dict(anchor=False),
    "no_crn": dict(crn=False),
    "no_shield": dict(shield=False),
    "grpo_vanilla": dict(discovery=False, elite=False, anchor=False, shield_discovery=False),
    "nscsd_safe": dict(sev_w=2.0),
    "nscsd_sentinel_reward": dict(reward="reward"),
    "no_shield_discovery": dict(shield_discovery=False),
    # Delta variants targeting the severe-outage failure under overload
    "nscsd_budget": dict(sev_budget=0.05),
    "nscsd_safemember": dict(safe_member=True),
    "nscsd_budget_safemember": dict(sev_budget=0.05, safe_member=True),
    "nscsd_cgrpo": dict(sev_budget=0.05, cost_norm="channel"),
    "nscsd_cgrpo_safemember": dict(sev_budget=0.05, cost_norm="channel", safe_member=True),
    # STRADDLE: safety-straddling group-relative advantages with discrete minimax-regret (MaxMC) levels
    "straddle": dict(curriculum="maxmc_discrete", sev_budget=0.05, safe_member=True),
    "straddle_nosafe": dict(curriculum="maxmc_discrete", sev_budget=0.05),
    "straddle_cgrpo": dict(curriculum="maxmc_discrete", sev_budget=0.05, cost_norm="channel"),
    "maxmc_only": dict(curriculum="maxmc_discrete"),
    "straddle_archive": dict(sev_budget=0.05, safe_member=True),
    # simpler competitor to the safe-side member: keep entropy up so groups straddle by chance
    "straddle_entropy": dict(curriculum="maxmc_discrete", sev_budget=0.05, ent_coef=0.03),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="nscsd", choices=list(VARIANTS) + ["ppo", "ppo_sentinel_reward"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--physics", default="inline", choices=["inline", "upstream"])
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    out = os.path.join(args.out, args.physics, args.method, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)

    if args.method.startswith("ppo"):
        from nscsd.ppo import PPOConfig, train_ppo
        cfg = PPOConfig(seed=args.seed, physics=args.physics)
        if args.method == "ppo_sentinel_reward":
            cfg.reward = "reward"
        if args.iters:
            cfg.updates = args.iters
        res = train_ppo(cfg)
    else:
        cfg = Config(seed=args.seed, physics=args.physics, **VARIANTS[args.method])
        if args.iters:
            cfg.iters = args.iters
        res = train(cfg)

    torch.save(res["policy"].state_dict(), os.path.join(out, "final.pt"))
    torch.save(res["champion"].state_dict(), os.path.join(out, "champion.pt"))
    pd.DataFrame(res["history"]).to_csv(os.path.join(out, "history.csv"), index=False)
    meta = {"config": res["config"], "champion_score": float(res["champion_score"]),
            "champion_shield": res.get("champion_shield"), "final_shield": res.get("final_shield"),
            "shield_log": res.get("shield_log")}
    if res.get("best_tree") is not None:
        with open(os.path.join(out, "tree.pkl"), "wb") as f:
            pickle.dump(res["best_tree"], f)
        with open(os.path.join(out, "tree_rules.txt"), "w") as f:
            f.write(describe(res["best_tree"]))
        meta["tree_score"] = float(res["best_tree_score"])
        meta["tree_fidelity"] = {k: float(v) for k, v in res["best_tree_fid"].items()}
    if res.get("archive") is not None:
        meta["archive"] = res["archive"].composition()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
