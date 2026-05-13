"""
Generate a standalone PDF of the ICM Analysis Report using reportlab.
Embeds all 3 figures inline so the PDF is fully self-contained.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "ICM_Analysis_Report.pdf")

IMG_SWEEP = os.path.join(HERE, "icm_stack_sweep.png")
IMG_SCATTER = os.path.join(HERE, "icm_scatter.png")
IMG_DIFF = os.path.join(HERE, "icm_decision_diff.png")

# ── Styles ─────────────────────────────────────────────────────────────────

styles = getSampleStyleSheet()

s_title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=22,
                          spaceAfter=6*mm)
s_subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"],
                             fontSize=12, textColor=HexColor("#555555"),
                             spaceAfter=8*mm, alignment=TA_CENTER)
s_h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
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
                         borderPadding=4, spaceAfter=3*mm,
                         leftIndent=6*mm)
s_caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9,
                            textColor=HexColor("#555555"),
                            alignment=TA_CENTER, spaceAfter=5*mm)
s_table_cell = ParagraphStyle("TC", parent=styles["Normal"], fontSize=8,
                               leading=10, fontName="Courier")
s_table_hdr = ParagraphStyle("TH", parent=styles["Normal"], fontSize=8,
                              leading=10, fontName="Courier-Bold")

# ── Helpers ────────────────────────────────────────────────────────────────

def P(text, style=s_body):
    return Paragraph(text, style)

def HR():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc"),
                      spaceBefore=3*mm, spaceAfter=3*mm)

def img(path, width_cm=17):
    return Image(path, width=width_cm*cm,
                 height=width_cm*cm * 0.75,  # approximate aspect
                 kind="proportional")

def make_table(headers, rows, col_widths=None):
    data = [[P(h, s_table_hdr) for h in headers]]
    for row in rows:
        data.append([P(str(c), s_table_cell) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# ── Document ───────────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    story = []

    # ── Title page ─────────────────────────────────────────────────────
    story.append(Spacer(1, 30*mm))
    story.append(P("ICM Module Analysis Report", s_title))
    story.append(P("Learned ICMNet vs Exact Malmuth-Harville ICM", s_subtitle))
    story.append(HR())

    meta = [
        ("Training run", "Post-bugfix 6.3M tournaments (20260416_155300_916856)"),
        ("Agents", "4 independent agents (no weight sharing)"),
        ("Checkpoint", "Final checkpoint at 6,300,000 tournaments"),
        ("Date", "2026-04-19"),
        ("Script", "analysis/icm_compare.py"),
    ]
    for label, val in meta:
        story.append(P(f"<b>{label}:</b> {val}", s_body))

    story.append(PageBreak())

    # ── 1. Background ──────────────────────────────────────────────────
    story.append(P("1. Background", s_h1))

    story.append(P("1.1 What is ICM?", s_h2))
    story.append(P(
        "The <b>Independent Chip Model (ICM)</b> is the standard method for "
        "converting tournament chip stacks into prize equity. The "
        "<b>Malmuth-Harville</b> variant recursively computes the probability "
        "that each player finishes in each position (proportional to chip "
        "share), then weights those probabilities by the prize pool."
    ))
    story.append(P(
        "For a 4-player sit-and-go with prize pool [+1.5, +0.5, -0.5, -1.5]:"
    ))
    story.append(P(
        "&bull; A player with more chips has higher equity (monotonic).<br/>"
        "&bull; Each additional chip is worth less than the previous (diminishing "
        "returns / concavity).<br/>"
        "&bull; Equal stacks produce equal equities."
    ))

    story.append(P("1.2 Architecture", s_h2))
    story.append(P(
        "The Poker_DQN model has two neural-network components:"
    ))
    story.append(P(
        "<b>ICMNet:</b> Linear(8, 128) &rarr; ReLU &rarr; Linear(128, 4) "
        "&rarr; Softmax<br/>"
        "&nbsp;&nbsp;Input: 4 normalized stacks + 4-dim prize pool = 8<br/>"
        "&nbsp;&nbsp;Output: 4-dim probability simplex", s_code
    ))
    story.append(P(
        "<b>DuelingDQN:</b> Linear(12, 128) &rarr; ReLU &rarr; "
        "Linear(128, 128) &rarr; ReLU<br/>"
        "&nbsp;&nbsp;&rarr; Value head(1) + Advantage head(2)<br/>"
        "&nbsp;&nbsp;Input: 8-dim base state + 4-dim ICM output = 12<br/>"
        "&nbsp;&nbsp;Output: Q(all-in), Q(fold)", s_code
    ))
    story.append(P(
        "The ICMNet is trained <b>indirectly</b>: there is no supervised ICM "
        "loss. Gradients flow from the DQN's TD-error loss backward through "
        "the concatenated state into the ICMNet weights. The question is: "
        "<b>what equity function did the ICMNet learn, and how does it compare "
        "to exact ICM?</b>"
    ))

    # ── 2. Methodology ─────────────────────────────────────────────────
    story.append(P("2. Methodology", s_h1))
    story.append(P(
        "Four analyses were performed on the final checkpoint (6.3M tournaments):"
    ))
    story.append(make_table(
        ["Analysis", "What it measures"],
        [
            ["Stack sweep",
             "How each function responds as one player's stack varies "
             "(1-95 BB), others fixed at 25BB"],
            ["Scatter plot",
             "Relationship between exact ICM and learned ICM across "
             "100 random stack distributions"],
            ["Decision impact",
             "Push/fold decisions when same DQN is fed learned ICM "
             "output vs exact ICM output"],
            ["Property comparison",
             "Monotonicity, rank ordering agreement, and key scenario "
             "outputs"],
        ],
        col_widths=[35*mm, 135*mm],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(P(
        "All evaluations use 100 random stack distributions generated via "
        "Dirichlet sampling (total 200 chips, minimum 2 chips per player, "
        "seed=42 for reproducibility)."
    ))

    # ── 3. Results ─────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("3. Results", s_h1))

    # 3.1 Stack Sweep
    story.append(P("3.1 Stack Sweep", s_h2))
    story.append(P(
        "Player 0's stack is swept from 1BB to 95BB while the other 3 "
        "players are fixed at 25BB. The plot shows the learned ICMNet "
        "softmax output (colored, left axis) against exact ICM raw equity "
        "and normalized equity (black, right axis)."
    ))
    story.append(img(IMG_SWEEP, 17))
    story.append(P("Figure 1: Stack sweep &mdash; equity vs stack size.", s_caption))
    story.append(P(
        "<b>Exact ICM</b> shows the expected concave monotonic curve: equity "
        "rises from -1.5 (last-place) to +1.5 (first-place) with diminishing "
        "returns per additional chip. This is the classic ICM pressure effect."
    ))
    story.append(P(
        "<b>Learned ICMNet</b> outputs near-constant values regardless of "
        "stack size for all 4 agents:"
    ))
    story.append(make_table(
        ["Agent", "Learned output for player 0", "Interpretation"],
        [
            ["Agent 1", "~0.00 (flat)", "All mass on player index 3"],
            ["Agent 2", "~0.03-0.07 (flat)", "All mass on player index 2"],
            ["Agent 3", "~0.00 (flat)", "All mass on player index 2"],
            ["Agent 4", "~1.00 (flat)", "All mass on player index 0"],
        ],
        col_widths=[25*mm, 55*mm, 90*mm],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(P(
        "The learned ICMNet shows <b>no sensitivity to stack sizes</b>. It "
        "has collapsed to a near-constant one-hot vector that does not vary "
        "as stacks change."
    ))

    # 3.2 Scatter
    story.append(PageBreak())
    story.append(P("3.2 Scatter Plot", s_h2))
    story.append(P(
        "For 100 random stack distributions, both exact ICM (normalized) and "
        "learned ICMNet output are computed for all 4 players. Each dot "
        "represents one player in one distribution (400 dots per agent). "
        "A perfect match would place all points on the diagonal."
    ))
    story.append(img(IMG_SCATTER, 16))
    story.append(P("Figure 2: Learned ICMNet vs exact ICM scatter.", s_caption))
    story.append(P(
        "Points cluster at y&asymp;0 and y&asymp;1 rather than along the "
        "diagonal, confirming the ICMNet outputs near-binary values with no "
        "correlation to exact ICM equity. Agent 1 shows the most spread but "
        "still no meaningful relationship."
    ))

    # 3.3 Decision Impact
    story.append(P("3.3 Decision Impact", s_h2))
    story.append(P(
        "For each agent, the 13&times;13 push/fold range card is computed "
        "twice: once using the learned ICMNet output, once replacing it with "
        "exact ICM (normalized to simplex). Both use greedy action selection, "
        "position-averaged across all 4 positions, at 25BB stack depth."
    ))
    story.append(img(IMG_DIFF, 17))
    story.append(P(
        "Figure 3: Decision impact &mdash; learned ICM (left), exact ICM "
        "(center), difference (right).", s_caption
    ))
    story.append(make_table(
        ["Agent", "Push agreement", "Mean |diff|", "Interpretation"],
        [
            ["Agent 1", "148/169 (87.6%)", "0.154",
             "Most divergence; exact ICM widens push range on high cards"],
            ["Agent 2", "168/169 (99.4%)", "0.007",
             "Near-identical decisions"],
            ["Agent 3", "167/169 (98.8%)", "0.059",
             "Near-identical decisions"],
            ["Agent 4", "169/169 (100.0%)", "0.015",
             "Perfectly identical decisions"],
        ],
        col_widths=[22*mm, 38*mm, 25*mm, 85*mm],
    ))
    story.append(Spacer(1, 3*mm))
    story.append(P(
        "<b>Key finding:</b> Despite the ICMNet learning nothing resembling "
        "actual ICM equity, <b>87.6% to 100% of push/fold decisions remain "
        "the same</b> when swapping in exact ICM. The DQN has learned to "
        "largely <b>ignore the ICM features</b> and make decisions based on "
        "the 8-dim base state (hand strength, stack, position, etc.)."
    ))

    # 3.4 Property Comparison
    story.append(PageBreak())
    story.append(P("3.4 Property Comparison", s_h2))

    story.append(P("<b>Monotonicity</b> (bigger stack &rarr; higher equity):", s_h3))
    story.append(make_table(
        ["", "Exact ICM", "Agent 1", "Agent 2", "Agent 3", "Agent 4"],
        [["Monotonicity", "100.0%", "53.3%", "49.7%", "48.2%", "49.2%"]],
        col_widths=[30*mm, 30*mm, 25*mm, 25*mm, 25*mm, 25*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P(
        "Exact ICM is perfectly monotonic. All learned ICMNets are ~50%, "
        "equivalent to random chance (a coin flip predicts &ldquo;bigger "
        "stack = higher equity&rdquo; 50% of the time)."
    ))

    story.append(P("<b>Rank ordering agreement</b>:", s_h3))
    story.append(make_table(
        ["Agent 1", "Agent 2", "Agent 3", "Agent 4"],
        [["5.0%", "6.0%", "1.0%", "2.0%"]],
        col_widths=[40*mm, 40*mm, 40*mm, 40*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P(
        "Random chance produces the correct ranking of 4 players 1/24 = "
        "4.2% of the time. All agents are near this baseline."
    ))

    story.append(P("<b>Spearman rank correlation</b>:", s_h3))
    story.append(make_table(
        ["Agent 1", "Agent 2", "Agent 3", "Agent 4"],
        [["0.098", "-0.022", "-0.026", "-0.026"]],
        col_widths=[40*mm, 40*mm, 40*mm, 40*mm],
    ))
    story.append(Spacer(1, 2*mm))
    story.append(P("All near zero &mdash; no rank correlation."))

    story.append(P("<b>Key scenario outputs</b> (final checkpoint):", s_h3))
    for agent_id, data in [
        (1, [
            ["Equal [50 50 50 50]",   "[0.250 0.250 0.250 0.250]", "[0.000 0.116 0.000 0.884]"],
            ["Desc. [80 60 40 20]",   "[0.438 0.347 0.215 0.000]", "[0.000 0.130 0.000 0.870]"],
            ["Dominant [140 20 20 20]","[1.000 0.000 0.000 0.000]", "[0.512 0.009 0.000 0.479]"],
            ["Short [66 66 66 2]",    "[0.333 0.333 0.333 0.000]", "[0.000 0.232 0.000 0.768]"],
        ]),
        (2, [
            ["Equal [50 50 50 50]",   "[0.250 0.250 0.250 0.250]", "[0.057 0.000 0.943 0.000]"],
            ["Desc. [80 60 40 20]",   "[0.438 0.347 0.215 0.000]", "[0.035 0.000 0.965 0.000]"],
            ["Dominant [140 20 20 20]","[1.000 0.000 0.000 0.000]", "[0.089 0.000 0.911 0.000]"],
            ["Short [66 66 66 2]",    "[0.333 0.333 0.333 0.000]", "[0.046 0.000 0.954 0.000]"],
        ]),
        (3, [
            ["Equal [50 50 50 50]",   "[0.250 0.250 0.250 0.250]", "[0.000 0.000 1.000 0.000]"],
            ["Desc. [80 60 40 20]",   "[0.438 0.347 0.215 0.000]", "[0.000 0.000 1.000 0.000]"],
            ["Dominant [140 20 20 20]","[1.000 0.000 0.000 0.000]", "[0.000 0.000 1.000 0.000]"],
            ["Short [66 66 66 2]",    "[0.333 0.333 0.333 0.000]", "[0.000 0.000 1.000 0.000]"],
        ]),
        (4, [
            ["Equal [50 50 50 50]",   "[0.250 0.250 0.250 0.250]", "[1.000 0.000 0.000 0.000]"],
            ["Desc. [80 60 40 20]",   "[0.438 0.347 0.215 0.000]", "[1.000 0.000 0.000 0.000]"],
            ["Dominant [140 20 20 20]","[1.000 0.000 0.000 0.000]", "[1.000 0.000 0.000 0.000]"],
            ["Short [66 66 66 2]",    "[0.333 0.333 0.333 0.000]", "[1.000 0.000 0.000 0.000]"],
        ]),
    ]:
        story.append(P(f"<b>Agent {agent_id}</b>:"))
        story.append(make_table(
            ["Scenario", "Exact (norm)", "Learned"],
            data,
            col_widths=[50*mm, 60*mm, 60*mm],
        ))
        story.append(Spacer(1, 2*mm))

    # ── 4. Analysis ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(P("4. Analysis", s_h1))

    story.append(P("4.1 Why did the ICMNet collapse?", s_h2))
    story.append(P(
        "The ICMNet is trained via <b>indirect gradient flow</b>: the only "
        "learning signal comes from the DQN's TD-error loss, which "
        "backpropagates through the concatenated 12-dim state into the ICMNet "
        "weights. There is no direct supervision telling the ICMNet what "
        "&ldquo;correct&rdquo; equities look like."
    ))
    story.append(P("Several factors contribute to the collapse:"))
    story.append(P(
        "<b>1. No supervised loss.</b> Without an explicit ICM target, the "
        "ICMNet only receives gradients that improve Q-value prediction. If "
        "the DQN can predict Q-values adequately from the 8-dim base state "
        "alone, the ICM gradients become uninformative noise."
    ))
    story.append(P(
        "<b>2. Softmax saturation.</b> The softmax output forces 4 outputs "
        "to sum to 1. Once one logit dominates, the softmax saturates and "
        "gradients for all outputs shrink exponentially (vanishing gradient "
        "through softmax). This creates a stable equilibrium where the "
        "collapsed one-hot output persists."
    ))
    story.append(P(
        "<b>3. DQN adaptation.</b> Even if the ICMNet produced useful "
        "information early in training, the DQN adapts to whatever the "
        "ICMNet currently outputs. Once collapse begins, the DQN compensates "
        "by relying more on the base state, further reducing the gradient "
        "signal to the ICMNet."
    ))
    story.append(P(
        "<b>4. Symmetry breaking.</b> Each agent's ICMNet independently "
        "collapses to a different one-hot vector (Agents 2-3 to index 2, "
        "Agent 4 to index 0, Agent 1 mixed). The initial random weights "
        "determine which index &ldquo;wins&rdquo; the softmax competition."
    ))

    story.append(P("4.2 What does the DQN actually use?", s_h2))
    story.append(P(
        "The decision impact analysis shows near-identical push/fold "
        "decisions whether the DQN receives learned or exact ICM values. "
        "This confirms:"
    ))
    story.append(P(
        "&bull; The <b>8-dim base state</b> (hand ranks, suited flag, stack "
        "in BB, active players, shortest stack flag, position, call/pot ratio) "
        "carries essentially all decision-relevant information.<br/>"
        "&bull; The <b>4-dim ICM output</b> has been absorbed into the DQN's "
        "bias terms as a constant offset.<br/>"
        "&bull; The DQN's own <font face='Courier'>stack_norm</font> feature "
        "already captures much of what ICM would provide for push/fold "
        "decisions at a given stack depth."
    ))

    story.append(P("4.3 Is the learned ICM better or worse than exact ICM?", s_h2))
    story.append(P(
        "Neither. The learned ICMNet has not learned any equity function at "
        "all. It outputs a constant vector that carries no information about "
        "relative chip positions. Exact ICM correctly captures monotonicity, "
        "diminishing returns, and stack ordering. The learned ICMNet captures "
        "none of these."
    ))
    story.append(P(
        "However, this does not mean the overall model performs poorly. The "
        "DQN has learned reasonable push/fold decisions using the base state "
        "features alone. The ICMNet simply does not contribute."
    ))

    # ── 5. Recommendations ─────────────────────────────────────────────
    story.append(P("5. Implications and Recommendations", s_h1))

    story.append(P(
        "The ICMNet module is <b>non-functional</b> in its current form. It "
        "acts as a constant bias vector that the DQN has learned to work "
        "around. This is an architectural issue, not a training duration "
        "issue: the collapse happens early and is self-reinforcing."
    ))

    story.append(P("Potential fixes:", s_h2))

    fixes = [
        ("<b>Supervised pre-training.</b>",
         "Pre-train the ICMNet against exact Malmuth-Harville targets before "
         "starting RL. Then either freeze it or fine-tune with a small "
         "learning rate alongside the DQN."),
        ("<b>Auxiliary ICM loss.</b>",
         "Add a supervised loss term alongside the TD loss: "
         "L = L_TD + &lambda; &middot; L_ICM, where L_ICM compares ICMNet "
         "output against exact ICM values computed from observed stacks."),
        ("<b>Remove the softmax.</b>",
         "Replace the softmax output with raw linear or ReLU outputs that "
         "don't have the saturation problem. The DQN can learn to interpret "
         "unnormalized values."),
        ("<b>Direct ICM injection.</b>",
         "Skip the learned network entirely and feed exact ICM values as "
         "features. This guarantees correct equity information."),
        ("<b>Remove ICMNet entirely.</b>",
         "If the DQN performs adequately with the 8-dim base state, the "
         "ICMNet adds complexity without benefit. The simpler DQNAgent class "
         "already exists as an alternative."),
    ]
    for i, (title, body) in enumerate(fixes, 1):
        story.append(P(f"{i}. {title} {body}"))

    # ── 6. Reproducibility ─────────────────────────────────────────────
    story.append(Spacer(1, 5*mm))
    story.append(HR())
    story.append(P("Reproducibility", s_h2))
    story.append(P("cd pokerDQN<br/>"
                    "python analysis/icm_compare.py --output-dir results/new_run/icm",
                    s_code))
    story.append(P(
        "Fixed random seed (42) for stack generation. All evaluations use "
        "the final checkpoint at 6,300,000 tournaments from run "
        "20260416_155300_916856."
    ))

    # ── Build ──────────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF saved to {OUTPUT}")


if __name__ == "__main__":
    build()
