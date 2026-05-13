# Presentation Assets — Upload Bundle for Claude Design

This folder contains the complete set of images for the deck. Drag the **whole folder** into your Claude Design session (`claude.ai/design`) before pasting `../PRESENTATION_BRIEF.md`.

## What you still need to add (6 files)

These are the TensorBoard / cumulative-reward screenshots from the planning chat. Save each one with the exact filename below:

| Filename | What it shows | Slide |
|---|---|---|
| `tb_loss_dqn1.png` | TB `Loss/TD_Error`, run `run_poker_dqn_1_20260416_155300` (orange, ~308M steps, smoothed ≈ 27.2) | 13 (top-left) |
| `tb_loss_dqn2.png` | TB `Loss/TD_Error`, run `run_poker_dqn_2_20260416_155300` (cyan, ~306M steps, smoothed ≈ 27.1) | 13 (top-right) |
| `tb_loss_dqn3.png` | TB `Loss/TD_Error`, run `run_poker_dqn_3_20260416_155300` (yellow, ~304M steps, smoothed ≈ 23.1) | 13 (bottom-left) |
| `tb_loss_dqn4.png` | TB `Loss/TD_Error`, run `run_poker_dqn_4_20260416_155300` (green, ~306M steps, smoothed ≈ 24.9) | 13 (bottom-right) |
| `cumulative_reward_run1.png` | Cumulative Performance over Training — agent 4 ends at −0.028 | 14 (left) |
| `cumulative_reward_run2.png` | Cumulative Performance over Training — all four agents converge near 0 | 14 (right) |

How to save: in the chat where you attached them, right-click → "Save Image As…" or drag from the chat into Finder. macOS Preview's "File → Export…" lets you rename in one step.

## What's already here (14 files, copied from `results/` and `eval_logs/`)

### Range progression (5 files) — Slide 15
| Filename | Source | What it shows |
|---|---|---|
| `range_progression_agent1.png` | `results/progression_agent1.png` | Push-frequency heatmap evolution for agent 1 across training milestones |
| `range_progression_agent2.png` | `results/progression_agent2.png` | Same, agent 2 |
| `range_progression_agent3.png` | `results/progression_agent3.png` | Same, agent 3 |
| `range_progression_agent4.png` | `results/progression_agent4.png` | Same, agent 4 |
| `range_progression_summary.png` | `results/progression_summary.png` | Aggregate push-frequency vs training step |

### Final range cards (4 files) — Slide 16
| Filename | Source | What it shows |
|---|---|---|
| `range_snapshot_agent1.png` | `results/new_run/snapshots/range_snapshot_agent1.png` | Final-checkpoint 13×13 push-frequency heatmap, agent 1 |
| `range_snapshot_agent2.png` | `results/new_run/snapshots/range_snapshot_agent2.png` | Same, agent 2 |
| `range_snapshot_agent3.png` | `results/new_run/snapshots/range_snapshot_agent3.png` | Same, agent 3 |
| `range_snapshot_agent4.png` | `results/new_run/snapshots/range_snapshot_agent4.png` | Same, agent 4 |

### ICM analysis (4 files) — Slides 17, 18, 19
| Filename | Source | What it shows |
|---|---|---|
| `icm_stack_sweep.png` | `results/new_run/icm/icm_stack_sweep.png` | Exact (dashed) vs learned (solid) ICM equity as one stack varies 1–95 BB |
| `icm_scatter.png` | `results/new_run/icm/icm_scatter.png` | Learned vs exact equity across random stack distributions |
| `icm_progression.png` | `results/new_run/icm/icm_progression.png` | ICMNet output vs training step (drift / collapse) |
| `icm_decision_diff.png` | `results/new_run/icm/icm_decision_diff.png` | Push/fold decision divergence when DQN is fed learned vs exact ICM |

### Tournament evaluation (1 file) — Slide 20
| Filename | Source | What it shows |
|---|---|---|
| `eval_baselines_final.png` | `eval_logs/poker_dqn_1_20260416_153431_011792/final.png` | Final-checkpoint evaluation curves (DQN vs five rule-based baselines) |

## Once everything is in place

1. Open `claude.ai/design` and start a new session.
2. Drag this entire folder into the upload area — all 20 PNGs go in at once.
3. Paste the contents of `../PRESENTATION_BRIEF.md` as your first message.
4. Iterate as the brief suggests in §7. Export → PPTX when done.
