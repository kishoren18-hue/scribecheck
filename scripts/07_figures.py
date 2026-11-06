"""Paper figures (PNG) from results/*.csv."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#d6d5cf",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.6})


def fig_report_card():
    c = pd.read_csv(R / "report_card.csv")
    c = c[~c.system.isin(["LED", "LED-pubmed"])]
    # verified counts where every flag was adjudicated; others scaled by sample precision
    L = pd.read_csv(R / "hardcheck_labels.csv")
    prec_other = (L[~L.system.isin(["GPT-4", "Human reference"])].label == "T").mean()
    for col in ["hc_any_per_note", "hc_any_lo", "hc_any_hi"]:
        c[col] = c[col] * prec_other
    for s_ in ["GPT-4", "Human reference"]:
        p = (L[L.system == s_].label == "T").mean()
        raw = c.loc[c.system == s_, ["hc_any_per_note", "hc_any_lo", "hc_any_hi"]] / prec_other
        c.loc[c.system == s_, ["hc_any_per_note", "hc_any_lo", "hc_any_hi"]] = raw * p
    c = c.sort_values("hc_any_per_note")
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(c))
    ax[0].barh(y, c.hc_any_per_note, color=BLUE, height=0.55)
    ax[0].errorbar(c.hc_any_per_note, y, xerr=[c.hc_any_per_note - c.hc_any_lo, c.hc_any_hi - c.hc_any_per_note],
                   fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax[0].set_yticks(y, c.system); ax[0].set_xlabel("Verified invented drug / dose / side claims per note")
    ax[0].set_title("Claims the visit never supported (95% CI)", loc="left", fontsize=10)
    ax[1].scatter(c.drug_coverage, c.hc_any_per_note, s=46, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
    for r in c.itertuples():
        ax[1].annotate(r.system, (r.drug_coverage, r.hc_any_per_note), xytext=(5, 3), textcoords="offset points",
                       fontsize=8, color=INK2)
    ax[1].set_xlabel("Drug coverage (share of visit's drugs in note)"); ax[1].set_ylabel("Verified claims per note")
    ax[1].set_title("Clean notes are often short notes", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_report_card.png", dpi=200); plt.close(fig)


def fig_detector():
    d = pd.read_csv(R / "detector_quality.csv").set_index("detector")
    types = ["U_FAB", "U_MED", "U_TIME", "U_LAT", "U_DOSE", "U_NUM", "G_MED", "G_NUM", "G_LAT", "NEG_FLIP", "RETRACT"]
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    x = np.arange(len(types)); w = 0.38
    ax.bar(x - w / 2 - 0.01, [d.loc["lexical", "mAUROC_" + t] for t in types], w, color=GRAY, label="Lexical grounding check")
    ax.bar(x + w / 2 + 0.01, [d.loc["SC+R", "mAUROC_" + t] for t in types], w, color=BLUE, label="ScribeCheck SC+R")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--"); ax.axvline(5.5, color="#d6d5cf", lw=1)
    ax.text(2.5, 1.02, "ungrounded errors", ha="center", color=INK2, fontsize=8.5)
    ax.text(8, 1.02, "grounded errors (every word was said)", ha="center", color=INK2, fontsize=8.5)
    ax.set_xticks(x, types, rotation=0, fontsize=8); ax.set_ylim(0.3, 1.08)
    ax.set_ylabel("Stratum-matched AUROC"); ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig2_detector_by_type.png", dpi=200); plt.close(fig)


def fig_shift():
    s = pd.read_csv(R / "shift_summary.csv")
    s = s[(s.score == "s_SC") & (s.alpha == 0.1)]
    order = ["in-dist", "asr:0.1", "asr:0.3", "paired:asr_real", "mts", "mix:crude", "mix:subtle", "asr:0.2+subtle"]
    lab = ["In-distribution", "Sim. ASR 10%", "Sim. ASR 30%", "Real ASR", "New corpus (MTS)", "Crude vendor",
           "Subtle vendor", "Subtle + ASR 20%"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    x = np.arange(len(order)); w = 0.38
    for k, (m, col, name) in enumerate([("CRC", BLUE, "CRC"), ("CRC-HP", ORANGE, "CRC-HP (90% conf.)")]):
        v = [s[(s.target == t) & (s.method == m)].risk.values[0] for t in order]
        wl = [s[(s.target == t) & (s.method == m)].workload.values[0] for t in order]
        ax[0].bar(x + (k - 0.5) * (w + 0.02), v, w, color=col, label=name)
        ax[1].bar(x + (k - 0.5) * (w + 0.02), wl, w, color=col, label=name)
    ax[0].axhline(0.10, color=INK, lw=1, ls="--"); ax[0].text(-0.4, 0.104, "target α = 0.10", fontsize=8, color=INK)
    for a, t in zip(ax, ["Realized miss rate on deployment notes", "Review workload (share of sentences flagged)"]):
        a.set_xticks(x, lab, rotation=35, ha="right", fontsize=8); a.set_title(t, loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_shift.png", dpi=200); plt.close(fig)


def fig_grounded():
    s = pd.read_csv(R / "shift_summary.csv")
    s = s[(s.score == "s_SC") & (s.alpha == 0.1) & (s.method == "CRC")]
    order = ["in-dist", "mix:subtle", "mix:crude", "asr:0.2+subtle", "mts"]
    lab = ["In-dist.", "Subtle vendor", "Crude vendor", "Subtle + ASR", "New corpus"]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2 - 0.01, [s[s.target == t].miss_ungrounded.values[0] for t in order], w, color=GRAY, label="Ungrounded errors")
    ax.bar(x + w / 2 + 0.01, [s[s.target == t].miss_grounded.values[0] for t in order], w, color=ORANGE, label="Grounded errors")
    ax.set_xticks(x, lab, fontsize=8.5); ax.set_ylabel("Share of errors never flagged")
    ax.set_title("A valid average guarantee hides who gets missed (CRC, α = 0.10)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "fig4_grounded_miss.png", dpi=200); plt.close(fig)


def fig_recal():
    r = pd.read_csv(R / "recalibration.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    names = {"mix:subtle": "Subtle vendor", "asr:0.2+subtle": "Subtle + ASR 20%", "mts": "New corpus (MTS)"}
    for (t, name), col in zip(names.items(), [BLUE, ORANGE, AQUA]):
        d = r[r.target == t]
        ax[0].plot(d.m, d.viol, marker="o", ms=5, lw=2, color=col, label=name)
        ax[1].plot(d.m, d.workload, marker="o", ms=5, lw=2, color=col, label=name)
    for a, t in zip(ax, ["P(miss rate > α) after local recalibration", "Review workload"]):
        a.set_xscale("log"); a.set_xlabel("Labeled notes from the deployment site (m)"); a.set_title(t, loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig5_recalibration.png", dpi=200); plt.close(fig)


def fig_hpr():
    c = pd.read_csv(R / "real_epistemic_claims.csv")
    c = c[~c.system.isin(["LED", "LED-pubmed"])]
    g = c.groupby(["system", "status"]).size().unstack(fill_value=0)
    for k in ["preserved", "inflated", "omitted"]:
        if k not in g:
            g[k] = 0
    g = g.loc[g.preserved.add(g.inflated).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(g)); left = np.zeros(len(g))
    for k, col, name in [("preserved", BLUE, "Uncertainty kept"), ("inflated", ORANGE, "Stated as certain"),
                         ("omitted", "#d6d5cf", "Diagnosis left out")]:
        ax.barh(y, g[k], left=left, color=col, height=0.6, label=name, edgecolor=SURF, linewidth=2)
        left += g[k].values
    ax.set_yticks(y, g.index); ax.set_xlabel("Hedged diagnoses from the visit (n = 23 across 22 encounters)")
    ax.set_title("What each system did with the clinician's uncertainty", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.45, -0.42))
    fig.tight_layout(); fig.savefig(FIG / "fig6_hedge_preservation.png", dpi=200); plt.close(fig)



def fig_mixrobust(det="SC"):
    s = pd.read_csv(R / f"mixrobust_summary_{det}.csv")
    al = pd.read_csv(R / f"shift_alarm_{det}.csv")
    meths = ["CRC", "Alarm-gated", "MR-CRC(h=0.4)", "MR-CRC(h=0.8)", "TC-CRC"]
    lab = ["CRC (marginal)", "Alarm-gated", "Trust dial h = 0.4", "Trust dial h = 0.8", "Type-conditional"]
    conds = ["in-dist", "mix:crude", "asr:0.2", "mix:subtle", "asr:0.2+subtle", "mts"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    cols = [GRAY, AQUA, "#86b6ef", BLUE, ORANGE]
    offs = [(6, 4), (6, -12), (6, 4), (-80, 10), (8, -4)]
    for m, l, c, o in zip(meths, lab, cols, offs):
        d = s[s.method == m].set_index("target").loc[conds]
        ax[0].scatter(d.workload.mean(), d.risk.max(), s=70, color=c, edgecolor=SURF, linewidth=1.5, zorder=3)
        ax[0].annotate(l, (d.workload.mean(), d.risk.max()), xytext=o, textcoords="offset points", fontsize=8, color=INK2)
    ax[0].axhline(0.10, color=INK, lw=1, ls="--"); ax[0].text(0.2, 0.104, "target α = 0.10", fontsize=8)
    ax[0].set_xlabel("Mean reviewer workload across 6 conditions"); ax[0].set_ylabel("Worst-condition miss rate")
    ax[0].set_title("The price of a guarantee that survives shift", loc="left", fontsize=10)
    ax[0].set_xlim(0.15, 1.0); ax[0].set_ylim(0, 0.17)
    names = {"in-dist": "No shift", "mix:crude": "Crude vendor", "asr:0.2": "ASR 20%", "mix:subtle": "Subtle vendor",
             "asr:0.2+subtle": "Subtle + ASR", "mts": "New corpus"}
    a = al.set_index("target").loc[conds]
    y = np.arange(len(conds))
    ax[1].barh(y - 0.2, a.alarm_mix_only, 0.38, color="#86b6ef", label="Mix signal only")
    ax[1].barh(y + 0.2, a.alarm_rate, 0.38, color=BLUE, label="Mix or error-rate signal")
    ax[1].set_yticks(y, [names[c] for c in conds]); ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.05)
    ax[1].set_xlabel("Share of deployments where the alarm fires")
    ax[1].set_title("Label-free shift alarm", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig7_mix_robust.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_report_card, fig_detector, fig_shift, fig_grounded, fig_recal, fig_hpr, fig_mixrobust]:
        f(); print("ok", f.__name__)
