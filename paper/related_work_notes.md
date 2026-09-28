# Related-work notes: Neuro-Symbolic Continuous Solution Discovery with GRPO

56 verified references (see `references.bib`). How I checked them: web search on 2026-09-28. For each one I matched title, authors, venue/year and arXiv ID/DOI against the arXiv, publisher, proceedings or repository page named in the entry's `note`. Direct fetches of arxiv.org and api.crossref.org were blocked by the session egress proxy, so I relied on the indexed pages that search returned. Where I could not confirm full first names, the bib entry uses initials or "and others".

Format: **key**: what it contributes. *Relation to our framework.*

## 1. GRPO origin, variants, off-policy guidance, GRPO outside LLMs

- **shao2024deepseekmath**: Introduces GRPO. It is a critic-free PPO variant whose advantages are rewards normalized within a group of samples for the same prompt; the paper also presents DeepSeekMath 7B. *Support: this is the primary citation for our GRPO defender. Our groups are forked simulator rollouts from one state instead of LLM completions of one prompt.*
- **guo2025deepseekr1**: DeepSeek-R1 (Nature 2025) shows that large-scale GRPO-based RL with verifiable rewards produces reasoning behavior without human-labelled reasoning traces. *Support: evidence that critic-free group-relative RL is stable at scale, which motivates dropping PPO's value network.*
- **yu2025dapo**: DAPO adds clip-higher, dynamic sampling (it drops groups with zero advantage variance), token-level loss and overlong-reward shaping to GRPO. *Support/design: dynamic sampling corresponds to our regret curriculum removing scenarios the defender always wins or always loses, since those groups give zero advantage.*
- **liu2025understanding**: Identifies the length and std-normalization biases in GRPO and proposes Dr. GRPO, which removes them. *Design: justifies our choice of advantage normalization (mean-only or std-normalized) and needs an ablation.*
- **zheng2025gspo**: GSPO uses sequence-level importance ratios and clipping in place of token-level ones, which stabilizes large and MoE RL training. *Contrast/design: our episodes are multi-step control trajectories, so the trajectory-level versus step-level ratio question GSPO raises applies directly.*
- **yan2025luffy**: LUFFY mixes off-policy demonstrations into on-policy GRPO groups and uses regularized importance sampling (policy shaping) so the policy does not imitate them superficially. *Direct basis for archive-elite injection: our "demonstrations" are rollouts of archive elites, not traces from a stronger model.*
- **khanda2025grpocontinuous**: A theoretical framework for extending GRPO to continuous-action robotic control (trajectory clustering, state-aware advantages). *Support: GRPO outside LLMs. It is theory-heavy with limited experiments, and our network-control setting is a distinct application.*
- **oliveira2025critics**: The first systematic study of GRPO in classical single-task RL benchmarks (discrete and continuous), compared with PPO. *Support/caution: gives evidence on where critic-free advantages work or fail in classic MDPs. Cite it when arguing GRPO is viable for a non-LLM defender.*
- **wang2026gr2po**: GR2PO is critic-free GRPO for continuous robot control. It uses discounted returns from parallel rollouts, normalizes them per time index within the group, and applies clipped updates. *Closest non-LLM algorithmic neighbour (group-normalized returns from parallel rollouts). It has no common random numbers, archive or curriculum.*
- **girgis2026constrainedgrpo**: Extends GRPO to constrained MDPs with Lagrangian-style cost handling. *Alternative to ours: constraint satisfaction inside the objective, whereas we use a runtime symbolic shield. Candidate baseline or related-work contrast.*
- **cao2025secloop**: SecLoop is an LLM-driven security-automation loop for zero-touch networks. Its SA-GRPO variant (no std division, no KL term) refines security strategies from group feedback over parallel executions. *Novelty threat: GRPO already applied to network security in ZTN/6G. It differs because it fine-tunes an LLM strategy generator, not a numeric DDoS-mitigation policy, and has no QD archive, curriculum, shield or crystallized rules.*
- **chen2025sec**: Self-Evolving Curriculum trains a non-stationary bandit over problem categories, using absolute advantage as a learning-progress reward to select GRPO training data. *Partial overlap: GRPO paired with an adaptive curriculum, but over static LLM prompts and without a QD archive or UED-style scenario generation.*
- **li2025darling**: DARLING puts a learned semantic-diversity signal into the GRPO reward to optimize quality and diversity jointly. *Contrast: this is diversity inside the reward. We keep diversity in an external MAP-Elites archive and keep the GRPO reward purely task-based.*

## 2. PPO, GAE, leave-one-out baselines, common random numbers

- **schulman2017ppo**: PPO, with a clipped surrogate objective and a learned critic. *Baseline: the Sentinel defender is PPO. GRPO reuses PPO's clipped surrogate.*
- **schulman2016gae**: GAE, an exponentially weighted advantage estimator that trades bias against variance using a value function. *Contrast: this is the critic-based advantage we remove. Ablation baseline (PPO+GAE).*
- **ahmadian2024back**: Shows that REINFORCE/RLOO-style estimators match or beat PPO for RLHF, and that many PPO components are unnecessary there. *Support: justifies critic-free group baselines. RLOO is the natural ablation of GRPO's group mean.*
- **kool2019buy4**: Original leave-one-out REINFORCE baseline: sample k times and use the mean of the other samples as each sample's baseline. *Support: theoretical ancestor of group-relative advantages.*
- **ng2000pegasus**: PEGASUS fixes the random seeds of the simulator ("scenarios"), turning a stochastic (PO)MDP into a deterministic one for policy evaluation and search. *Direct justification for common random numbers in our forked rollouts: group members share the exogenous randomness (traffic and attack seed), so advantage differences reflect the actions taken.*

## 3. Unsupervised environment design / curricula

- **dennis2020paired**: Formalizes UED. PAIRED trains an adversary to generate levels that maximize regret, using an antagonist agent. *Contrast: PAIRED is still a learned adversary, which is what we avoid (Sentinel's RL attacker). We use its minimax-regret objective without training a generator.*
- **jiang2021plr**: Prioritized Level Replay replays levels with high estimated learning potential (for example positive value loss or GAE magnitude) from a rolling buffer. *Direct basis for our scenario buffer. We replace the value-loss score with a critic-free regret proxy (archive-champion return minus defender return).*
- **jiang2021robustplr**: Replay-guided adversarial environment design. It casts PLR as UED (Dual Curriculum Design), shows that training only on curated levels ("robust PLR") gives minimax-regret guarantees, and proposes REPAIRED. *Support: theoretical justification for our regret-driven replay curriculum.*
- **parkerholder2022accel**: ACCEL makes small edits to high-regret levels so that complexity compounds over training. *Direct basis for our scenario mutation operator (mutating attack parameters such as rate, vector mix and pulse pattern).*
- **rutherford2024noregrets**: Shows that common regret approximations track success rate more than true regret, and proposes learnability-based sampling (p(1-p)). *Design caution: our regret proxy should be validated. Learnability sampling is a cheap alternative and should be in the ablations.*
- **wang2019poet**: POET co-evolves environments and agent solutions, with transfer between pairs, producing an open-ended curriculum. *Related: environment-solution pairing and stepping-stone transfer are the ancestors of our archive plus scenario discovery. POET keeps an environment population in place of an RL adversary.*

## 4. Quality-diversity and continuous solution discovery

- **mouret2015mapelites**: MAP-Elites keeps the best solution in each cell of a user-chosen behaviour-descriptor grid. *Direct basis for the Solution Archive (cells defined by, for example, collateral-drop rate by rule complexity or attack family).*
- **pugh2016qd**: Frames quality diversity as a paradigm (novelty search with local competition, MAP-Elites). *Background: defines QD terminology and justifies the archive's monotone elitism.*
- **nilsson2021pgame**: PGA-MAP-Elites uses TD3 policy-gradient steps as a variation operator inside MAP-Elites. *Closest QD-with-gradients prior work. Ours differs: GRPO (critic-free) is the improvement engine, and the archive stores symbolic trees as well as networks.*
- **romeraparedes2024funsearch**: FunSearch runs evolutionary program search with an LLM mutator and an automated evaluator over an island-based program database, and reached new cap-set results. *Analogy: a "continuous solution discovery" loop with a monotone program database. Our crystallized decision-tree programs play the role of FunSearch programs.*
- **novikov2025alphaevolve**: AlphaEvolve is an evolutionary coding agent in which LLMs edit code under evaluator feedback, discovering algorithms and infrastructure optimizations. *Analogy: evaluator-gated, archive-based continual improvement of executable artefacts. It supports framing our method as discovery rather than equilibrium-seeking.*
- **hughes2024openendedness**: Position paper that defines open-endedness as novelty plus learnability for an observer and argues it is essential for ASI. *Framing: our archive and curriculum are designed to be open-ended in this sense, with learnability ensured by regret filtering.*

## 5. Co-evolution pathologies and population-based self-play

- **rosin1997newmethods**: Introduces competitive fitness sharing, shared sampling and the hall of fame for competitive coevolution. *Contrast: Sentinel's Hall-of-Fame attacker comes from this line. We explain why a hall of fame alone does not prevent late-stage collapse.*
- **ficici1998challenges**: Identifies arms-race failure modes including mediocre stable states and the conditions for sustained progress. *Support for motivation: a theoretical account of the co-evolutionary collapse Sentinel reports.*
- **watson2001minimal**: Minimal-substrate models show disengagement, loss of gradient and intransitive cycling in coevolution. *Support: names the pathologies (disengagement, cycling, focusing) that our non-adversarial curriculum avoids.*
- **balduzzi2019openended**: A geometric view of functional-form games. Nontransitive games cause strategic cycles, and PSRO_rN uses niching to build diverse effective populations. *Support: cycling in nontransitive attacker-defender games motivates replacing naive co-evolution with a population or archive plus explicit diversity.*
- **lanctot2017psro**: PSRO computes best responses to meta-strategy mixtures over policy populations, and joint-policy correlation measures overfitting to opponents. *Baseline/contrast: the principled game-theoretic alternative to co-evolution. It still needs an RL attacker and best-response oracles, which we drop.*
- **heinrich2015fsp**: Fictitious Self-Play, a sample-based fictitious play in extensive-form games, best-responding to average opponent strategies. *Contrast: an averaging-opponent method that could fix Sentinel's instability but keeps the adversarial RL loop.*
- **pinto2017rarl**: RARL trains a protagonist against a learned destabilizing adversary for robustness. *Contrast: adversarial RL robustness, the paradigm Sentinel follows and we replace with regret-based scenario discovery.*

## 6. Safe RL, shielding, constrained RL, neuro-symbolic / programmatic RL

- **alshiekh2018shielding**: Safety shields synthesized from temporal-logic specifications override unsafe actions during learning and execution. *Support: the formal basis for the symbolic safety shield we inherit from Sentinel.*
- **konighofer2021online**: Online shielding for stochastic systems computes the safety of actions at runtime rather than precomputing the shield. *Support: relevant if our shield is evaluated online in the forked simulator.*
- **achiam2017cpo**: CPO is a trust-region policy search with near-constraint-satisfaction guarantees for CMDPs. *Alternative/baseline: constraint-based safety compared with shield-based safety.*
- **garcia2015saferl**: A taxonomy of safe RL: modified optimality criterion versus modified exploration process. *Background: positions the shield (exploration and execution modification) plus the KL anchor.*
- **bastani2018viper**: VIPER extracts decision-tree policies from a DNN oracle with Q-weighted DAgger, making them verifiable. *Direct basis for rule crystallization (Sentinel's depth-4 trees). Our archive stores such trees as first-class elites.*
- **verma2018pirl**: PIRL represents policies as programs in a DSL, found by neurally directed program search. *Support: programmatic policies as interpretable, verifiable artefacts, which is the "program" half of our archive.*
- **acharya2023neurosymbolic**: Survey of neurosymbolic RL and planning. *Background: situates our neural-snapshot plus symbolic-tree archive within the NeSy-RL taxonomy.*

## 7. DRL for DDoS / SDN security, ZSM, O-RAN, XAI for 6G

- **doriguzzi2020lucid**: LUCID is a lightweight CNN DDoS detector (IEEE TNSM). *Context/baseline: a detection-side contrast. Our work covers mitigation control, and detection outputs can be state features.*
- **liu2018drlddos**: DDPG-based DDoS flooding mitigation in SDN via OpenFlow. *Baseline lineage: early DRL mitigation using an actor-critic, the kind of critic we remove.*
- **malialis2015distributed**: Multi-agent RL router throttling for DDoS response, using difference rewards and task decomposition. *Baseline lineage: the classical RL DDoS throttling formulation.*
- **simpson2020perhost**: Per-host DDoS mitigation by direct-control RL (IEEE TNSM). The agent chooses per-flow drop probabilities. *Closest prior TNSM DRL-mitigation work and a candidate baseline or environment design reference.*
- **akbari2020atmos**: ATMoS is an RL-based autonomous threat-mitigation framework in SDN (NOMS). *Related: autonomous mitigation in NSM venues.*
- **shao2025adados**: AdaDoS trains an adversarial RL DoS attacker against detectors in SDN, formulated as a competitive game. *Contrast: shows the adversarial-RL attacker paradigm is still active. Our curriculum replaces a learned attacker with regret-guided scenario generation.*
- **nguyen2023drlcyber**: Survey of DRL for cyber security (CPS, intrusion detection, multi-agent game-theoretic defense). *Background survey.*
- **benzaid2020zsm**: AI-driven ETSI ZSM, covering closed-loop automation and the limitations and risks of AI in ZSM. *Motivation: trustworthiness requirements for autonomous NSM, which our shield, crystallized rules and monotone archive address.*
- **liyanage2023oran**: Survey of Open RAN security challenges and opportunities. *Context: 6G/O-RAN threat surface. Crystallized rules could be deployed as xApp/rApp policies.*
- **guo2020xai6g**: Argues XAI is necessary for trust in 6G autonomous networks (IEEE Commun. Mag.). *Motivation: XAI requirement for 6G.*
- **wang2021xai6g**: Survey of XAI for 6G technologies and use cases. *Background: XAI-for-6G taxonomy.*
- **senevirathna2025xaib5gsec**: XAI for B5G security survey (IEEE COMST 2025). *Background: positions interpretable (crystallized-rule) security controllers within XAI-for-security.*

## Positioning

- **Fixing co-evolutionary collapse without an adversary.** Co-evolution (Rosin & Belew; Ficici & Pollack; Watson & Pollack), self-play (FSP), PSRO and adversarial RL (RARL, AdaDoS, Sentinel) all keep a learned attacker and inherit cycling, disengagement or forgetting. We replace the attacker with a regret-driven scenario curriculum (PLR/ACCEL-style), which carries minimax-regret guarantees (Jiang 2021) and does not require equilibrium dynamics to stay stable.
- **Critic-free RL for network control.** GRPO has been validated mainly for LLM post-training. Non-LLM uses (Khanda 2025, de Oliveira 2025, GR2PO 2026) are recent and target robotics or classic control. We are, as far as this search found, the first to use GRPO for a numeric network-security or DDoS-mitigation control policy. Common random numbers (PEGASUS-style shared seeds) across forked simulator rollouts are a variance-reduction step specific to simulator-based groups that the LLM-GRPO literature does not need.
- **A monotone, mixed neural/symbolic QD archive.** MAP-Elites and PGA-ME archive neural policies, and FunSearch/AlphaEvolve archive programs. We could not find a work whose archive holds both neural snapshots and crystallized decision-tree programs (VIPER-style) as competing elites. This gives an interpretable champion at every point in training and removes the need for Sentinel's post-hoc benchmark checkpoint selection.
- **Archive-guided GRPO.** LUFFY injects off-policy expert traces into GRPO groups. We inject rollouts of our own archive elites, a self-generated teacher that improves monotonically, and add a KL anchor to the current archive champion. This combines mixed-policy GRPO with QD elitism, which we did not find combined anywhere.
- **Safety as a shield, not a constraint.** Constrained GRPO (Girgis 2026) and CPO build safety into the objective. We keep a runtime symbolic shield (Alshiekh 2018; Könighofer 2021), so safety holds during both exploration and deployment, and crystallized rules can be checked against it.
- **XAI for 6G/ZSM.** XAI-for-6G and ZSM surveys (Guo 2020; Wang 2021; Senevirathna 2025; Benzaid & Taleb 2020) call for transparent, trustworthy closed-loop automation. Our crystallized-rule elites are directly auditable controllers, not post-hoc explanations.
- **Novelty claims to hedge:** (i) "first GRPO for network security" is false in general, because SecLoop/SA-GRPO (Cao et al. 2025) applies GRPO to LLM-based security automation in zero-touch networks. Claim instead "first GRPO-trained numeric mitigation policy / first critic-free group-relative defender for DDoS." (ii) "first GRPO outside LLMs" is false (Khanda 2025; de Oliveira 2025; GR2PO 2026). (iii) "first GRPO with a curriculum" is false for LLMs (Self-Evolving Curriculum, Chen 2025, plus prompt-replay and learnability-filtering works; DAPO's dynamic sampling is a simple form). Claim instead "first regret-driven UED-style scenario generation (PLR/ACCEL mutation) for GRPO in a simulator." (iv) Diversity-aware GRPO exists (DARLING), but it puts diversity in the reward, not in an external MAP-Elites archive.

## Novelty-threat summary

1. **cao2025secloop** (arXiv 2512.09485, also on IEEE Xplore doc 11299517): GRPO (SA-GRPO) for security automation in zero-touch 6G networks. **Most direct threat** to any "GRPO for network security" claim. It differs from our work in using an LLM policy and having no QD archive, curriculum, shield or decision trees.
2. **wang2026gr2po**, **khanda2025grpocontinuous** and **oliveira2025critics**: GRPO in non-LLM control using group-normalized returns from parallel rollouts. These pre-empt any "GRPO outside LLMs" claim. None of them uses common random numbers, archives or curricula, as far as the abstracts show.
3. **chen2025sec** (and related GRPO prompt-curriculum works found in search but not added to the bib: Prompt Replay arXiv 2603.21177, AdaCuRL arXiv 2511.09478): GRPO combined with adaptive curricula, but only for LLM prompts.
4. **nilsson2021pgame**: gradient-based policy improvement inside MAP-Elites. This is prior art for "RL plus QD archive", but it is actor-critic (TD3), not GRPO.
5. We found **no** work combining GRPO with a MAP-Elites/QD archive of mixed neural and symbolic elites, or GRPO with PLR/ACCEL-style UED in a simulator. Searches run: "GRPO quality-diversity MAP-Elites archive", "GRPO prioritized level replay unsupervised environment design", "GRPO elite archive injection hall of fame", "GRPO interpretable decision tree distillation". Re-run these searches before submission.

## Unverifiable / not included

- **Alfatemi et al., "Sentinel: A Neuro-Symbolic Co-Evolutionary Framework for Trustworthy Network and Service Management Against DDoS Attacks", IEEE TNSM vol. 23, 2026, DOI 10.1109/TNSM.2026.3729146.** Web searches by title keywords, author name and exact DOI returned no matching page, so I could not verify it and did not add it to `references.bib`. It is presumably very recent or early-access and not yet indexed. The authors should add it by hand from IEEE Xplore or their own copy.
- **Wang et al. 2021 (arXiv 2112.04698)**: included as an arXiv preprint only. Its journal version, if one exists, was not confirmed.
- **Guo 2020 (IEEE Commun. Mag.)**: volume, issue and pages were verified, but the DOI was not shown on any page checked, so it is omitted (IEEE Xplore document 9141213 and arXiv ID are given instead).
- **Ahmadian et al. 2024**: the ACL Anthology DOI was not displayed and is omitted (URL and arXiv ID are given).
- **Acharya et al. (IEEE TAI)**: the DOI is verified, but the volume and issue were not confirmed and are omitted.
- **Conference versions of DAPO, Dr. GRPO, LUFFY and GSPO**: possible NeurIPS/COLM 2025 publications were not confirmed, so these are cited as arXiv preprints.
- **Könighofer et al. 2021**: an arXiv ID (2012.09539) was not confirmed, so only the Springer DOI is given.
