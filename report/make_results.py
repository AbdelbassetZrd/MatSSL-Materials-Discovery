"""Generate LaTeX macros and tables from the experiment outputs, so that every number in
the report comes directly from data/predictions/*. Run: uv run python report/make_results.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "predictions"
PROC = ROOT / "data" / "processed"
GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
TASKS = {"metallicity": "Metal", "magnetism": "Mag", "stability": "Stab"}
POOLS = {"indist": "same database (labels hidden)", "gnome": "GNoME"}
METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Composite_Score"]


def esc(s: str) -> str:
    return (str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("*", r"$^*$").replace("#", r"\#"))


def pm(m, s, d=3):
    if pd.isna(s):
        return f"{m:.{d}f}"
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    macros = []
    values = {}

    def mac(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        values[name] = value

    # dataset sizes
    for task, key in [*TASKS.items(), ("band_gap", "Gap")]:
        f = PROC / f"{task}_features.parquet"
        if f.exists():
            t = pd.read_parquet(f, columns=["target"])["target"]
            mac(f"{key}N", f"{len(t):,}")
            mac(f"{key}Pos", f"{100 * t.mean():.1f}")
        else:
            mac(f"{key}N", "?")
            mac(f"{key}Pos", "?")

    for task, key in TASKS.items():
        for pool, pool_name in POOLS.items():
            d = PRED / task / pool
            sj = d / f"ssl_summary_{task}.json"
            pk = key + ("In" if pool == "indist" else "Gn")
            if not sj.exists():
                for m in ("BestMethod", "BestConfig", "BestComp", "BaseComp", "BaseConfig",
                          "Impr", "FullComp", "DepComp", "DepAUC", "DepFone", "BestSSL",
                          "BestSSLComp"):
                    mac(pk + m, "?")
                continue
            S = json.loads(sj.read_text())
            best, bssl = S["best_model_overall"], S["best_ssl_model"]
            mac(pk + "BestMethod", esc(best["Method"]))
            mac(pk + "BestConfig", esc(best["Model_Config"]))
            mac(pk + "BestComp", f"{best['Composite_Score_mean']:.4f}")
            mac(pk + "BaseComp", f"{S['best_supervised_baseline']['composite_score']:.4f}")
            mac(pk + "BaseConfig", esc(S["best_supervised_baseline"]["config"]))
            mac(pk + "BestSSL", esc(bssl["Method"]))
            mac(pk + "BestSSLConfig", esc(bssl["Model_Config"]))
            mac(pk + "BestSSLComp", f"{bssl['Composite_Score_mean']:.4f}")
            mac(pk + "Impr", f"{S['ssl_improvement_over_best_baseline_pct']:+.2f}")
            mac(pk + "FullComp", f"{S['fully_supervised_upper_bound']['Composite_Score']:.4f}")
            mac(pk + "FullAUC", f"{S['fully_supervised_upper_bound']['ROC-AUC']:.4f}")
            dep = S.get("deployment_model_test_metrics") or {}
            mac(pk + "DepComp", f"{dep.get('Composite_Score', float('nan')):.4f}")
            mac(pk + "DepAUC", f"{dep.get('ROC-AUC', float('nan')):.4f}")
            mac(pk + "DepFone", f"{dep.get('F1-Score', float('nan')):.4f}")
            mac(pk + "NTest", f"{S['training_setup']['n_test_evaluated']:,}")
            mac(pk + "Runs", f"{S['n_training_runs']:,}")

            # best configuration per method table
            rows = []
            for r in S["best_per_method"]:
                rows.append(" & ".join([esc(r["Method"]), esc(r["Model_Config"])] +
                                       [pm(r[f"{m}_mean"], r[f"{m}_std"]) for m in METRICS])
                            + r" \\")
            tab = ("\\begin{table}[H]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
                   "\\resizebox{\\textwidth}{!}{%\n"
                   "\\begin{tabular}{llcccccc}\n\\toprule\nMethod & Best configuration & "
                   "Accuracy & Precision & Recall & F1 & ROC-AUC & Composite\\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"
                   f"\\caption*{{{esc(task.capitalize())} -- unlabeled pool: {pool_name}. "
                   f"1,000 labeled compositions, mean$\\pm$std over 3 seeds, "
                   f"{S['training_setup']['n_test_evaluated']:,} test compositions.}}\n"
                   "\\end{table}\n")
            (GEN / f"table_{task}_{pool}.tex").write_text(tab, encoding="utf-8")

            # pool-size table (mean composite of the best config of each method per size)
            res = pd.read_csv(d / f"ssl_results_{task}.csv")
            ssl = res[~res["Method"].isin(["Baseline", "Self-Supervised (AE)"])]
            piv = (ssl.groupby(["Method", "Config", "N_Unlabeled"])["Composite_Score"].mean()
                   .groupby(["Method", "N_Unlabeled"]).max().unstack())
            base = res[res["Config"] == S["best_supervised_baseline"]["config"]][
                "Composite_Score"].mean()
            cols = list(piv.columns)
            lines = [" & ".join([esc(m)] + [f"{v:.4f}" for v in piv.loc[m]]) + r" \\"
                     for m in piv.index]
            tab2 = ("\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{l" + "c" * len(cols)
                    + "}\n\\toprule\nMethod & " + " & ".join(f"$n_u$={c:,}" for c in cols)
                    + "\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\midrule\nBest supervised "
                    f"baseline & \\multicolumn{{{len(cols)}}}{{c}}{{{base:.4f}}}\\\\\n"
                    "\\bottomrule\n\\end{tabular}\n"
                    f"\\caption*{{Composite score versus number of unlabeled compositions "
                    f"({pool_name}).}}\n\\end{{table}}\n")
            (GEN / f"poolsize_{task}_{pool}.tex").write_text(tab2, encoding="utf-8")

            # budget curve table
            bc = pd.read_csv(d / f"{task}_label_budget_curve.csv")
            g = bc.groupby(["Budget", "Curve"])["Composite_Score"].agg(["mean", "std"]).unstack()
            curves = list(g.columns.levels[1])
            lines = [f"{b:,} & " + " & ".join(pm(g.loc[b, ("mean", c)], g.loc[b, ("std", c)])
                                               for c in curves) + r" \\" for b in g.index]
            tab3 = ("\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{l" + "c" * len(curves)
                    + "}\n\\toprule\nLabeled samples & " + " & ".join(esc(c) for c in curves)
                    + "\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\midrule\nAll training labels"
                    f" ({S['training_setup']['n_train_pool']:,}) & \\multicolumn{{{len(curves)}}}"
                    f"{{c}}{{{S['fully_supervised_upper_bound']['Composite_Score']:.4f} "
                    "(fully supervised)}\\\\\n\\bottomrule\n\\end{tabular}\n"
                    f"\\caption*{{Label-budget study ({pool_name} unlabeled pool).}}\n"
                    "\\end{table}\n")
            (GEN / f"budget_{task}_{pool}.tex").write_text(tab3, encoding="utf-8")

            # model zoo
            zoo = res[res["Method"] == "Baseline"].groupby("Config")[
                ["F1-Score", "ROC-AUC", "Composite_Score"]].agg(["mean", "std"])
            zoo = zoo.sort_values(("Composite_Score", "mean"), ascending=False)
            lines = [esc(c) + " & " + " & ".join(pm(zoo.loc[c, (m, "mean")],
                                                    zoo.loc[c, (m, "std")])
                                                 for m in ["F1-Score", "ROC-AUC",
                                                           "Composite_Score"]) + r" \\"
                     for c in zoo.index]
            tab4 = ("\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n"
                    "Supervised model (1,000 labels) & F1 & ROC-AUC & Composite\\\\\n\\midrule\n"
                    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
                    "\\caption*{Model zoo and Optuna-tuned baselines.}\n\\end{table}\n")
            (GEN / f"zoo_{task}_{pool}.tex").write_text(tab4, encoding="utf-8")

            # figures
            dst = FIG / "results" / task / pool
            dst.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.png"):
                shutil.copy(f, dst / f.name)

    # deployed models (written by pipeline/select_models.py)
    reg_file = ROOT / "app" / "models" / "registry.json"
    reg = json.loads(reg_file.read_text()) if reg_file.exists() else {}
    for task, key in [*TASKS.items(), ("band_gap", "Gap")]:
        r = reg.get(task, {})
        tm = r.get("test_metrics", {})
        mac(key + "SelMethod", esc(r.get("method", "?")))
        mac(key + "SelConfig", esc(r.get("config", "?")))
        mac(key + "SelPool", POOLS.get(r.get("selection_pool", ""), "--"))
        for m, short in [("Composite_Score", "Comp"), ("ROC-AUC", "AUC"), ("F1-Score", "Fone"),
                         ("Accuracy", "Acc"), ("Precision", "Prec"), ("Recall", "Rec")]:
            mac(key + "Sel" + short, f"{tm[m]:.4f}" if m in tm else "?")
    rows = []
    for task, name in [("metallicity", "Metallicity"), ("magnetism", "Magnetism"),
                       ("stability", "Stability"), ("band_gap", "PV band gap")]:
        r = reg.get(task)
        if not r:
            continue
        tm = r["test_metrics"]
        rows.append(f"{name} & {esc(r['method'])} & {esc(r['config'])} & "
                    + " & ".join(f"{tm.get(m, float('nan')):.3f}"
                                 for m in ["Accuracy", "Precision", "Recall", "F1-Score",
                                           "ROC-AUC"]) + r" \\")
    (GEN / "deployed.tex").write_text(
        "\\begin{table}[H]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lllccccc}\n\\toprule\nTask & Method & Configuration & Acc. & Prec. & "
        "Rec. & F1 & ROC-AUC\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}}\n\\caption{Deployed models, re-trained on all training labels and "
        "evaluated once on the held-out test sets.}\\label{tab:deployed}\n\\end{table}\n",
        encoding="utf-8")

    # overview: best SSL vs tuned baseline for every task x pool
    rows = []
    short = {"indist": "same database", "gnome": "GNoME"}
    for task in TASKS:
        for pool in POOLS:
            sj = PRED / task / pool / f"ssl_summary_{task}.json"
            if not sj.exists():
                rows.append(f"{task.capitalize()} & {short[pool]} & "
                            r"\multicolumn{4}{c}{\textit{running}} \\")
                continue
            S = json.loads(sj.read_text())
            b = S["best_ssl_model"]
            rows.append(f"{task.capitalize()} & {short[pool]} & "
                        f"{S['best_supervised_baseline']['composite_score']:.4f} & "
                        f"{esc(b['Method'])} & {b['Composite_Score_mean']:.4f} & "
                        f"{S['ssl_improvement_over_best_baseline_pct']:+.2f}\\% \\\\")
    (GEN / "ssl_overview.tex").write_text(
        "\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{llcccc}\n\\toprule\n"
        "Task & Unlabeled pool & Tuned baseline & Best SSL method & SSL composite & Gain\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Best "
        "semi-supervised method versus the Optuna-tuned supervised baseline (1,000 labels, "
        "composite score, mean of 3 seeds).}\\label{tab:ssl-overview}\n\\end{table}\n",
        encoding="utf-8")

    # band gap
    bj = PRED / "band_gap" / "band_gap_summary.json"
    if bj.exists():
        B = json.loads(bj.read_text())
        ga, ad = B["ga"], B["applicability_domain"]
        for k, v in {
            "GAInitAUC": f"{ga['initial_best_auc']:.4f}", "GAFinalAUC": f"{ga['final_best_auc']:.4f}",
            "GAInitFit": f"{ga['initial_max_fitness']:.4f}",
            "GAFinalFit": f"{ga['final_max_fitness']:.4f}",
            "GAInitMinFit": f"{ga['initial_min_fitness']:.4f}",
            "GAFinalMinFit": f"{ga['final_min_fitness']:.4f}",
            "GAInitN": str(ga["initial_best_n_features"]), "GAFinalN": str(ga["final_n_features"]),
            "GAInitAvgN": f"{ga['initial_avg_n_features']:.1f}",
            "GAFinalAvgN": f"{ga['final_avg_n_features']:.1f}",
            "GAAllCV": f"{ga['all_descriptors_cv_auc']:.4f}",
            "GapAllAUC": f"{B['test_all_descriptors']['ROC-AUC']:.4f}",
            "GapGAAUC": f"{B['test_ga_descriptors']['ROC-AUC']:.4f}",
            "GapAllPR": f"{B['test_all_descriptors']['PR-AUC']:.4f}",
            "GapGAPR": f"{B['test_ga_descriptors']['PR-AUC']:.4f}",
            "GapAllFone": f"{B['test_all_descriptors']['F1-Score']:.4f}",
            "GapGAFone": f"{B['test_ga_descriptors']['F1-Score']:.4f}",
            "GapBestModel": esc(B["model_comparison"][0]["Model"]),
            "ADPCAVar": f"{100 * ad['pca_variance_2pc']:.1f}", "ADPCone": f"{100 * ad['pc1']:.1f}",
            "ADPCtwo": f"{100 * ad['pc2']:.1f}", "ADNTest": f"{ad['n_test']:,}",
            "ADNIn": f"{ad['n_inside']:,}", "ADNOut": f"{ad['n_outside']:,}",
            "ADFracIn": f"{100 * ad['fraction_inside']:.1f}",
            "ADFracOut": f"{100 * (1 - ad['fraction_inside']):.1f}",
            "ADAUCIn": f"{ad['metrics_inside']['ROC-AUC']:.4f}",
            "ADAUCOut": f"{ad['metrics_outside']['ROC-AUC']:.4f}",
            "ADThr": f"{ad['distance_threshold_95pct']:.3f}",
            "GapThr": f"{B['decision_threshold']['value']:.2f}",
            "GapTunedFone": f"{B['test_ga_descriptors_tuned_threshold']['F1-Score']:.4f}",
            "GapTunedRec": f"{B['test_ga_descriptors_tuned_threshold']['Recall']:.4f}",
            "GapTunedPrec": f"{B['test_ga_descriptors_tuned_threshold']['Precision']:.4f}",
            "GapGARec": f"{B['test_ga_descriptors']['Recall']:.4f}",
            "ADFoneInT": f"{ad['metrics_inside_tuned']['F1-Score']:.3f}",
            "ADFoneOutT": f"{ad['metrics_outside_tuned']['F1-Score']:.3f}",
            "ADRecInT": f"{ad['metrics_inside_tuned']['Recall']:.3f}",
            "ADRecOutT": f"{ad['metrics_outside_tuned']['Recall']:.3f}",
            "ADFoneIn": f"{ad['metrics_inside']['F1-Score']:.3f}",
            "ADFoneOut": f"{ad['metrics_outside']['F1-Score']:.3f}",
            "ADRecIn": f"{ad['metrics_inside']['Recall']:.3f}",
            "ADRecOut": f"{ad['metrics_outside']['Recall']:.3f}",
            "ADPRIn": f"{ad['metrics_inside']['PR-AUC']:.3f}",
            "ADPROut": f"{ad['metrics_outside']['PR-AUC']:.3f}",
        }.items():
            mac(k, v)
        cats = ga["descriptor_categories"]
        items = "\n".join(f"  \\item \\textbf{{{esc(k.capitalize())}}} ({v} descriptors)"
                          for k, v in sorted(cats.items(), key=lambda kv: -kv[1]) if v)
        (GEN / "ga_categories.tex").write_text("\\begin{itemize}\n" + items + "\n\\end{itemize}\n",
                                               encoding="utf-8")
        comp = pd.DataFrame(B["model_comparison"])
        lines = [f"{esc(r.Model)} & {r['ROC-AUC']:.4f} & {r['PR-AUC']:.4f} & {r['F1-Score']:.4f}"
                 f" & {r['Training_Time']:.1f}\\,s \\\\" for _, r in comp.iterrows()]
        (GEN / "gap_models.tex").write_text(
            "\\begin{table}[H]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\nModel & "
            "ROC-AUC & PR-AUC & F1 & Training time\\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\caption*{Model comparison on all 142 descriptors "
            "(photovoltaic band gap, test set).}\n\\end{table}\n", encoding="utf-8")
        dst = FIG / "results" / "band_gap"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (PRED / "band_gap").glob("*.png"):
            shutil.copy(f, dst / f.name)

    # unsupervised + self-supervised
    U = PRED / "unsupervised"
    if (U / "unsupervised_summary.json").exists():
        us = json.loads((U / "unsupervised_summary.json").read_text())
        mac("KBest", str(us["clustering"]["best_k"]))
        mac("PCAVarTwo", f"{100 * us['clustering']['pca_explained_2d']:.1f}")
        for t, k in [("stability", "Stab"), ("magnetism", "Mag"), ("metallicity", "Metal"),
                     ("band_gap", "Gap")]:
            r = pd.read_csv(U / f"association_rules_{t}.csv")
            mac(f"Rules{k}", str(len(r)))
            top = r.head(6)
            lines = [f"\\{{{esc(a)}\\}} $\\Rightarrow$ {esc(c)} & {s:.4f} & {cf:.3f} & {lf:.2f}"
                     r" \\" for a, c, s, cf, lf in zip(top.antecedents, top.consequents,
                                                       top.support, top.confidence, top.lift,
                                                       strict=True)]
            (GEN / f"rules_{t}.tex").write_text(
                "\\begin{tabular}{lccc}\n\\toprule\nRule & Support & Confidence & Lift\\\\\n"
                "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n",
                encoding="utf-8")
        dst = FIG / "results" / "unsupervised"
        dst.mkdir(parents=True, exist_ok=True)
        for f in U.glob("*.png"):
            shutil.copy(f, dst / f.name)
    ae = PRED / "self_supervised" / "autoencoder_summary.json"
    if ae.exists():
        a = json.loads(ae.read_text())
        mac("AERtwo", f"{a['final']['val_r2_all']:.3f}")
        mac("AERtwoMasked", f"{a['final']['val_r2_masked_features']:.3f}")

    (GEN / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    (GEN / "values.json").write_text(json.dumps(values, indent=1), encoding="utf-8")
    print(f"wrote {len(macros)} macros and tables to {GEN}")


if __name__ == "__main__":
    main()
