"""
Generate a standalone PDF report for the 9.3M training run analysis.
Covers: range card snapshots, push frequency progression, ICM comparison.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.realpath(os.path.join(HERE, '..'))  # results/new_run
OUTPUT = os.path.join(HERE, "9M_Analysis_Report.pdf")

# Image paths
IMG_SNAP = {i: os.path.join(RESULTS, f"9M_snapshots/range_snapshot_agent{i}.png") for i in range(1, 5)}
IMG_PROG = os.path.join(RESULTS, "9M_snapshots/progression_summary.png")
IMG_ICM_SWEEP = os.path.join(RESULTS, "9M_icm/icm_stack_sweep.png")
IMG_ICM_SCATTER = os.path.join(RESULTS, "9M_icm/icm_scatter.png")
IMG_ICM_DIFF = os.path.join(RESULTS, "9M_icm/icm_decision_diff.png")

# ── Styles ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
s_title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=24, spaceAfter=4*mm)
s_subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=13,
                             textColor=HexColor("#555555"), spaceAfter=10*mm,
                             alignment=TA_CENTER)
s_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17,
                       spaceBefore=8*mm, spaceAfter=4*mm)
s_h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                       spaceBefore=6*mm, spaceAfter=3*mm)
s_h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11,
                       spaceBefore=4*mm, spaceAfter=2*mm)
s_body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                         leading=14, spaceAfter=3*mm)
s_code = ParagraphStyle("Code", parent=styles["Code"], fontSize=8,
                         leading=11, fontName="Courier",
                         backColor=HexColor("#f4f4f4"),
                         borderPadding=4, spaceAfter=3*mm, leftIndent=6*mm)
s_caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                            textColor=HexColor("#555555"),
                            alignment=TA_CENTER, spaceAfter=5*mm)
s_tc = ParagraphStyle("TC", parent=styles["Normal"], fontSize=8, leading=10, fontName="Courier")
s_th = ParagraphStyle("TH", parent=styles["Normal"], fontSize=8, leading=10, fontName="Courier-Bold")

def P(text, style=s_body):
    return Paragraph(text, style)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc"),
                      spaceBefore=3*mm, spaceAfter=3*mm)

def img(path, width_cm=17):
    return Image(path, width=width_cm*cm, height=width_cm*cm*0.75, kind="proportional")

def make_table(headers, rows, col_widths=None):
    data = [[P(h, s_th) for h in headers]]
    for row in rows:
        data.append([P(str(c), s_tc) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t

def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    story = []

    # ═══════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 25*mm))
    story.append(P("Poker DQN — 9.3M Training Run Analysis", s_title))
    story.append(P("Push/Fold Range Evolution, ICM Module Evaluation,<br/>"
                    "and Strategic Behavior Assessment", s_subtitle))
    story.append(HR())
    meta = [
        ("Training run", "Post-bugfix run (20260416_155300_916856)"),
        ("Duration", "9,300,000 tournaments (checkpoints every 100k)"),
        ("Agents", "4 independent Poker_DQN agents (no weight sharing)"),
        ("Architecture", "ICMNet(8&rarr;128&rarr;4, softmax) + DuelingDQN(12&rarr;128&rarr;128&rarr;2)"),
        ("Action space", "Binary: all-in or fold"),
        ("Prize pool", "[+1.5, +0.5, -0.5, -1.5] (zero-sum)"),
        ("Date", "2026-04-20"),
    ]
    for label, val in meta:
        story.append(P(f"<b>{label}:</b> {val}"))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════
    story.append(P("Table of Contents", s_h1))
    toc = [
        "1. Executive Summary",
        "2. Push Frequency Progression (0 &rarr; 9.3M)",
        "3. Range Card Snapshots (2M / 4.5M / 7M / 9.3M)",
        "4. ICM Module Analysis",
        "   4.1 Stack Sweep",
        "   4.2 Scatter Plot",
        "   4.3 Decision Impact",
        "   4.4 Property Comparison",
        "5. Key Findings and Interpretation",
        "6. Recommendations",
    ]
    for line in toc:
        story.append(P(line))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 1. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    story.append(P("1. Executive Summary", s_h1))
    story.append(P(
        "This report analyzes the complete 9.3M tournament training run of "
        "four independent Poker_DQN agents playing push/fold poker. Key findings:"
    ))
    story.append(P(
        "<b>Range evolution:</b> All agents converge from wide, noisy ranges "
        "(40-50% push at 100k) to tight, hand-strength-correlated ranges "
        "by 9.3M. Agents 1-3 settle at 1-8% push frequency (very tight, "
        "premium-only), while Agent 4 remains the loosest at ~15%, exploiting "
        "the tight opponents."
    ))
    story.append(P(
        "<b>Continued learning 6M&rarr;9.3M:</b> Agents 2 and 3 show "
        "interesting late-training dynamics. Agent 2 loosens from 1% to 11% "
        "push between 7M and 9.3M, and Agent 3 rebounds from 2% to 8%. "
        "This suggests strategy adaptation is still ongoing and equilibrium "
        "has not been reached."
    ))
    story.append(P(
        "<b>ICM module:</b> The learned ICMNet has collapsed to constant "
        "one-hot outputs for all agents, providing no stack-dependent equity "
        "information. The DQN makes decisions almost entirely from the 8-dim "
        "base state. Agents 2 and 4 show 100% push agreement when swapping "
        "learned ICM for exact ICM. Agent 3 is a notable exception: only "
        "3.6% agreement, indicating its DQN has uniquely adapted to use the "
        "collapsed ICM value as a meaningful signal."
    ))

    # ═══════════════════════════════════════════════════════════════════
    # 2. PUSH FREQUENCY PROGRESSION
    # ═══════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(P("2. Push Frequency Progression", s_h1))
    story.append(P(
        "Push frequency is tracked at every 100k checkpoint, broken down by "
        "position (4 positions, position 0 = earliest to act, position 3 = "
        "latest). The plot covers the full 0&rarr;9.3M training trajectory."
    ))
    story.append(img(IMG_PROG, 17))
    story.append(P("Figure 1: Push frequency vs training tournaments, per position.", s_caption))

    story.append(P(
        "<b>Agent 1:</b> Rapidly tightens from ~40% to under 5% by 2M. "
        "Remains very tight (1-4%) through 9.3M. Positions are closely "
        "clustered, suggesting limited positional awareness."
    ))
    story.append(P(
        "<b>Agent 2:</b> Similar rapid tightening to 1-4% by 3M. A notable "
        "late divergence occurs after 7M where position 3 (latest position) "
        "rises sharply, indicating emerging positional adaptation &mdash; "
        "the agent learns to push wider from late position."
    ))
    story.append(P(
        "<b>Agent 3:</b> Tightens to 1-3% by 4M, then shows rebound "
        "activity after 5M. By 9.3M, push frequency is 5-10% with some "
        "position differentiation starting to emerge."
    ))
    story.append(P(
        "<b>Agent 4:</b> The loosest agent throughout. Starts at ~50%, "
        "drops to 10-15% by 5M, then stabilizes. Shows the most "
        "positional spread, pushing wider from later positions. This "
        "is strategically sound &mdash; exploiting the tight opponents."
    ))

    # ═══════════════════════════════════════════════════════════════════
    # 3. RANGE CARD SNAPSHOTS
    # ═══════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(P("3. Range Card Snapshots", s_h1))
    story.append(P(
        "13&times;13 heatmaps of P(all-in) for every starting hand, "
        "averaged across all 4 positions, at 4 milestones: 2M, 4.5M, 7M, "
        "9.3M. Green = high push probability, red = fold. Parameters: "
        "stack=25BB, 4 active players, shortest stack, softmax T=1.0."
    ))

    for agent_id in range(1, 5):
        story.append(P(f"3.{agent_id} Agent {agent_id}", s_h2))
        story.append(img(IMG_SNAP[agent_id], 17))
        story.append(P(f"Figure {agent_id+1}: Agent {agent_id} range snapshots at 2M / 4.5M / 7M / 9.3M.", s_caption))

    # Agent commentary
    story.append(P("Range Card Observations:", s_h2))
    story.append(make_table(
        ["Agent", "2M", "4.5M", "7M", "9.3M", "Trend"],
        [
            ["Agent 1", "9%", "1%", "2%", "4%",
             "Very tight; only premium pairs + AK/AQ"],
            ["Agent 2", "7%", "4%", "1%", "11%",
             "Tightens then loosens; late adaptation"],
            ["Agent 3", "11%", "2%", "6%", "8%",
             "V-shaped; rebounds with wider suited range"],
            ["Agent 4", "46%", "41%", "20%", "15%",
             "Consistently loosest; clear hand-strength gradient"],
        ],
        col_widths=[22*mm, 14*mm, 14*mm, 14*mm, 14*mm, 92*mm],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(P(
        "<b>Agent 4</b> shows the clearest hand-strength gradient throughout "
        "training: high pairs and broadway hands are pushed most, with "
        "probability decreasing smoothly toward low cards. This is the "
        "expected pattern for rational play."
    ))
    story.append(P(
        "<b>Agents 2 and 3</b> exhibit a \"V-shaped\" trajectory: they "
        "tighten aggressively in mid-training (4.5M-7M) then loosen again by "
        "9.3M. This suggests the agents are still adapting to each other's "
        "strategies and equilibrium has not been reached."
    ))

    # ═══════════════════════════════════════════════════════════════════
    # 4. ICM MODULE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(P("4. ICM Module Analysis", s_h1))
    story.append(P(
        "The ICMNet (Linear(8,128)&rarr;ReLU&rarr;Linear(128,4)&rarr;Softmax) "
        "is intended to learn tournament equity from chip stacks. It is trained "
        "<b>indirectly</b> via the DQN's TD-error loss &mdash; no supervised "
        "ICM targets. We compare its outputs against exact Malmuth-Harville ICM "
        "at the 9.3M checkpoint."
    ))

    # 4.1 Stack Sweep
    story.append(P("4.1 Stack Sweep", s_h2))
    story.append(P(
        "Player 0's stack is swept from 1BB to 95BB (others fixed at 25BB). "
        "Left axis: learned ICMNet output. Right axis: exact ICM equity."
    ))
    story.append(img(IMG_ICM_SWEEP, 17))
    story.append(P("Figure 6: Stack sweep &mdash; equity vs stack size at 9.3M.", s_caption))
    story.append(P(
        "Exact ICM (black) shows the expected concave monotonic curve. "
        "All 4 learned ICMNets output near-constant values regardless of "
        "stack size &mdash; the same collapse pattern observed at 6.3M "
        "persists through 9.3M. Additional training did not recover the "
        "ICM module."
    ))

    # 4.2 Scatter
    story.append(P("4.2 Scatter Plot", s_h2))
    story.append(img(IMG_ICM_SCATTER, 15))
    story.append(P("Figure 7: Learned vs exact ICM across 100 random stack distributions.", s_caption))
    story.append(P(
        "Points cluster at y&asymp;0 and y&asymp;1 (horizontal bands) rather "
        "than along the diagonal. No correlation between learned and exact ICM. "
        "The pattern is identical to the 6.3M analysis."
    ))

    # 4.3 Decision Impact
    story.append(PageBreak())
    story.append(P("4.3 Decision Impact", s_h2))
    story.append(P(
        "The DQN is fed learned ICM (left column) vs exact ICM (center) and "
        "the push/fold decisions are compared (right = difference). "
        "Greedy action, position-averaged, 25BB."
    ))
    story.append(img(IMG_ICM_DIFF, 17))
    story.append(P("Figure 8: Decision impact &mdash; learned ICM vs exact ICM at 9.3M.", s_caption))

    story.append(make_table(
        ["Agent", "Push agreement", "Mean |diff|", "Observation"],
        [
            ["Agent 1", "141/169 (83.4%)", "0.175",
             "Moderate divergence; exact ICM shifts some hands"],
            ["Agent 2", "169/169 (100.0%)", "0.077",
             "Identical decisions despite different ICM values"],
            ["Agent 3", "6/169 (3.6%)", "0.780",
             "MAJOR divergence; learned ICM pushes wide, exact folds"],
            ["Agent 4", "169/169 (100.0%)", "0.025",
             "Identical decisions; DQN ignores ICM entirely"],
        ],
        col_widths=[22*mm, 38*mm, 25*mm, 85*mm],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(P(
        "<b>Agent 3 is a critical finding.</b> At 9.3M (vs 98.8% agreement at "
        "6.3M), Agent 3 now shows only 3.6% agreement. This means its DQN has "
        "become <b>dependent</b> on the collapsed ICM output &mdash; it has "
        "learned to interpret the constant [0, 0, 1, 0] vector as a signal to "
        "push wider. When exact ICM is substituted (a different 4-dim vector), "
        "the DQN's decisions change dramatically. The Agent 3 range with "
        "exact ICM is extremely tight (center column), while with learned ICM "
        "it pushes at 8%. The collapsed ICM acts as an accidental \"push more\" "
        "bias that the DQN has adapted to."
    ))
    story.append(P(
        "<b>Agents 2 and 4</b> show the opposite: 100% agreement, meaning "
        "their DQNs have fully absorbed the constant ICM bias into their "
        "weights and are invariant to the ICM input."
    ))

    # 4.4 Property Comparison
    story.append(P("4.4 Property Comparison", s_h2))
    story.append(P("<b>Monotonicity</b> (bigger stack &rarr; higher equity):", s_h3))
    story.append(make_table(
        ["", "Exact ICM", "Agent 1", "Agent 2", "Agent 3", "Agent 4"],
        [["Monotonicity", "100.0%", "52.5%", "46.2%", "45.3%", "51.0%"]],
        col_widths=[30*mm, 30*mm, 25*mm, 25*mm, 25*mm, 25*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P("All agents ~50% &mdash; equivalent to random. No improvement from 6.3M."))

    story.append(P("<b>Rank ordering agreement and Spearman rho</b>:", s_h3))
    story.append(make_table(
        ["Metric", "Agent 1", "Agent 2", "Agent 3", "Agent 4"],
        [
            ["Rank match", "3.0%", "6.0%", "1.0%", "8.0%"],
            ["Spearman rho", "0.082", "-0.100", "-0.078", "0.004"],
        ],
        col_widths=[35*mm, 35*mm, 35*mm, 35*mm, 35*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P("Near chance level (4.2%) for rank match. No correlation."))

    story.append(P("<b>Key scenario outputs at 9.3M</b>:", s_h3))
    story.append(P(
        "Each agent's ICMNet outputs a fixed one-hot vector regardless of stacks:"
    ))
    story.append(make_table(
        ["Agent", "Collapsed to", "Equal stacks output", "One dominant output"],
        [
            ["Agent 1", "Player 3", "[0.000 0.022 0.000 0.978]", "[0.612 0.001 0.000 0.386]"],
            ["Agent 2", "Player 2", "[0.035 0.000 0.965 0.000]", "[0.107 0.000 0.893 0.000]"],
            ["Agent 3", "Player 2", "[0.000 0.000 0.998 0.001]", "[0.000 0.002 0.996 0.002]"],
            ["Agent 4", "Player 0", "[1.000 0.000 0.000 0.000]", "[1.000 0.000 0.000 0.000]"],
        ],
        col_widths=[22*mm, 28*mm, 55*mm, 55*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P(
        "Agent 1 shows slight sensitivity to extreme stack imbalances (the "
        "\"one dominant\" scenario shifts mass from player 3 to player 0), "
        "but this is insufficient for meaningful ICM estimation."
    ))

    # ═══════════════════════════════════════════════════════════════════
    # 5. KEY FINDINGS
    # ═══════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(P("5. Key Findings and Interpretation", s_h1))

    story.append(P("5.1 Strategy is still evolving at 9.3M", s_h2))
    story.append(P(
        "The V-shaped push frequency curves of Agents 2 and 3 (tighten then "
        "loosen) indicate the self-play dynamics have not reached a Nash "
        "equilibrium. The agents are still adapting to each other's strategies. "
        "Agent 4 acts as the exploiter, remaining loose against tight opponents. "
        "The tightening of Agent 4 from 46% to 15% suggests the other agents "
        "have partially adapted, making exploitation harder."
    ))

    story.append(P("5.2 Positional awareness is emerging", s_h2))
    story.append(P(
        "The progression plot shows position lines spreading apart after ~6M "
        "for Agents 2 and 4. Later position (acting last, more information) "
        "correlates with higher push frequency &mdash; this is correct poker "
        "strategy. Agents 1 and 3 show less positional differentiation."
    ))

    story.append(P("5.3 ICM collapse is permanent without intervention", s_h2))
    story.append(P(
        "The ICMNet remained collapsed from 6.3M through 9.3M for all agents. "
        "3 million additional training tournaments produced no recovery. This "
        "confirms the collapse is a stable equilibrium caused by softmax "
        "saturation and the indirect training signal. The collapse will not "
        "self-resolve with more training."
    ))

    story.append(P("5.4 Agent 3 developed ICM-dependent behavior", s_h2))
    story.append(P(
        "Between 6.3M (98.8% decision agreement) and 9.3M (3.6% agreement), "
        "Agent 3's DQN learned to use its collapsed ICM output [0,0,1,0] as "
        "a meaningful feature. This is not true ICM learning &mdash; the DQN "
        "has simply adapted to treat this constant vector as a \"push wider\" "
        "signal. When replaced with exact ICM values, the agent becomes "
        "extremely tight. This is an important cautionary finding: the DQN "
        "can become dependent on arbitrary constant features, creating fragile "
        "behavior."
    ))

    story.append(P("5.5 DQN base state carries the strategic intelligence", s_h2))
    story.append(P(
        "For Agents 1, 2, and 4, push/fold decisions are 83-100% identical "
        "regardless of ICM input. The 8-dim base state (hand ranks, suited "
        "flag, stack/BB, active players, shortest flag, position, call/pot) "
        "is sufficient for the agents' learned strategies. The DQN's own "
        "<font face='Courier'>stack_norm</font> feature implicitly captures "
        "stack-depth dependent play."
    ))

    # ═══════════════════════════════════════════════════════════════════
    # 6. RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    story.append(P("6. Recommendations", s_h1))

    story.append(P("6.1 Continue training", s_h2))
    story.append(P(
        "Strategy is still evolving. The V-shaped trajectories of Agents 2-3 "
        "suggest more training could yield further strategic refinement. "
        "Consider running to 15-20M to see if push frequencies stabilize."
    ))

    story.append(P("6.2 Fix the ICM module", s_h2))
    story.append(P(
        "The current indirect training approach has failed. Options, in order "
        "of implementation simplicity:"
    ))
    recs = [
        ("<b>Supervised pre-training:</b>", "Pre-train ICMNet against exact "
         "Malmuth-Harville targets, then freeze or fine-tune with small LR."),
        ("<b>Auxiliary loss:</b>", "Add L = L_TD + &lambda;&middot;L_ICM "
         "during RL training, where L_ICM is MSE against exact ICM."),
        ("<b>Remove softmax:</b>", "Replace with linear or ReLU output to "
         "prevent saturation collapse."),
        ("<b>Direct injection:</b>", "Feed exact ICM values directly as "
         "features, bypassing the network entirely."),
        ("<b>Remove ICMNet:</b>", "Use the simpler 8-dim DQNAgent if ICM "
         "awareness is not needed."),
    ]
    for i, (title, body) in enumerate(recs, 1):
        story.append(P(f"{i}. {title} {body}"))

    story.append(P("6.3 Monitor Agent 3's ICM dependency", s_h2))
    story.append(P(
        "Agent 3's newly developed dependency on the collapsed ICM vector "
        "is a fragility risk. If this agent were deployed or its weights "
        "transferred, the constant ICM bias would need to be preserved. "
        "This underscores the need to fix the ICM module before production use."
    ))

    # ── Reproducibility
    story.append(Spacer(1, 8*mm))
    story.append(HR())
    story.append(P("Reproducibility", s_h2))
    story.append(P(
        "# Range snapshots<br/>"
        "python analysis/range_snapshots.py --milestones 2000000 4500000 7000000 9300000<br/><br/>"
        "# Push frequency progression<br/>"
        "python analysis/range_progression.py --max-tournaments 9300000<br/><br/>"
        "# ICM comparison<br/>"
        "python analysis/icm_compare.py --max-tournaments 9300000",
        s_code
    ))
    story.append(P(
        "All evaluations use checkpoints from run 20260416_155300_916856. "
        "Random stack distributions use seed=42 for reproducibility."
    ))

    doc.build(story)
    print(f"PDF saved to {OUTPUT}")

if __name__ == "__main__":
    build()
