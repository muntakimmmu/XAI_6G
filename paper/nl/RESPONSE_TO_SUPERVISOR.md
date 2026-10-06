# IMBANG letter: response to supervisor comments

Reviewed copy: the earlier MIZAN draft annotated by A. Mahmud (four pages).
Revised version: `paper/nl/imbang_letter.pdf` (four pages, abstract 96 words).

| # | Location in reviewed copy | Comment | Change in the IMBANG version |
|---|---|---|---|
| 1 | Title, "MIZAN" underlined | — | Method renamed to IMBANG (the earlier name was already in use). |
| 2 | Sec. I, "most complete recent design" | "is the advanced recent design" | Now "Sentinel [7] is the most advanced recent design." |
| 3 | Table I, "MIZAN (ours)" struck through | — | "(ours)" removed; the row reads **IMBANG**. |
| 4 | Fig. 1 caption | "Explain this in journal body paragraphs" | Caption shortened to one line. Sec. II now opens with a paragraph that explains the figure: UEs and botnet, gNB/UPF link, edge scrubber in front of the MEC service, policy → shield → scrubber loop, and digital-twin training. |
| 5 | Sec. I, contributions | "three key contributions" | Now "This letter makes three key contributions." |
| 6 | "formalises ... confirms" | "maybe change to derive or validate" | Now "it **derives** this constraint-cancellation effect and **validates** it on 80 trained checkpoints". |
| 7 | "(Arabic for balance)" struck through | — | Language gloss removed. The name is introduced only through its acronym: **I**nterpretable, **M**onotone-safe, **B**udgeted, **A**dversary-free, **N**euro-symbolic **G**RPO. |
| 8 | "compares" | "use validate, evaluate or benchmark" | Now "it **benchmarks** IMBANG against ...". |
| 9 | Contributions | "in II", "in III" | Each contribution now cites its section: cancellation (Sec. III), method (Sec. IV), evaluation (Sec. V). The numbers come from `\ref` and match the paper. The margin notes read "II"/"III", but the cancellation result is in Sec. III and the method in Sec. IV. |
| 10 | Sec. II, "We use the setting of [7], illustrated in Fig. 1" | "We use system model as in [7]" | Now "We use the system model of [7], shown in Fig. 1." |
| 11 | Sec. II (margin line) | "explain which ones is our new proposition or new derivation" | New closing sentences in Sec. II: the traffic, observation and service models, Eq. (1), the score b_t and the shield are taken unchanged from [7]. The new elements are the budgeted formulation (2), Proposition 1 (cancellation) and the safe-side construction with Proposition 2. |
| 12 | Sec. V heading "Evaluation" | "DISCUSSION" | Section renamed "Results and Discussion". |
| 13 | Sec. VI "Discussion" struck through | "EXPLAIN IN PART V" | Separate Discussion section removed. Its content now sits in Sec. V: "Why one member suffices" and "Interpretability" paragraphs, and the scope of Prop. 2 as a limitation item. |
| 14 | Near Fig. 3 | "IF YOU HAVE 1 MORE FIGURE IS GOOD" | New **Fig. 3 (operating curve)**: severe outages and leakage against attack intensity for Sentinel, Constrained GRPO and IMBANG (10 seeds, 20 episodes per point, identical traffic). At i = 1, outages are 143 (Sentinel), 84 (C-GRPO) and 40 (IMBANG) per 500 windows. IMBANG leaks less than Sentinel at every intensity. Up to i = 0.9, Sentinel is marginally lower on outages, and the text says so. Script: `scripts/intensity_sweep.py`; data: `results/intensity_sweep_by_seed.csv`. |
| 15 | Sec. VII Conclusion | "REDUCE TO HALF" | Conclusion cut from nine lines to four. |

Also carried over from the reviewed copy: the author list and affiliations
(M. Rahaman, A. Mahmud, A. Chehri).

Checks on the revised PDF:
- 4 pages; all columns end at the bottom margin.
- Abstract 96 words (limit 75–100).
- All fonts embedded; no LaTeX warnings.
- No five-word phrase in the body matches the Sentinel paper. The only matches are A. Chehri's affiliation line, since he is also a Sentinel author.
