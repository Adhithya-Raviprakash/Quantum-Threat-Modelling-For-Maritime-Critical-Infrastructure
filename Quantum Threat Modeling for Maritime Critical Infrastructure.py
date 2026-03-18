"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     QUANTUM THREAT MODELLING — MARITIME CRITICAL INFRASTRUCTURE            ║
║     Random Forest ML Engine  |  Cryptology CSE2021  |  VIT AP 2026         ║
║     Authors: Nikhil (23BCB7059) & Adhithya Raviprakash (23BCB7141)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

FEATURES
  ✦ Random Forest classifier  → Vulnerability label (CRITICAL / HIGH / MEDIUM / LOW)
  ✦ Random Forest regressor   → Risk score (0–100)
  ✦ Q-Day timeline simulation → Year a given qubit count breaks your algorithm
  ✦ Interactive user input    → Algorithm, key size, maritime system, qubit year
  ✦ Model accuracy report     → Confusion matrix + feature importance
  ✦ Chart generation          → 4 saved PNG charts

USAGE
  python quantum_threat_maritime.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)

# ── Colour palette (matches the PPT theme) ────────────────────────────────────
C = {
    "navy":    "#0D1B2A",
    "blue":    "#00B4D8",
    "red":     "#E63946",
    "orange":  "#F4A261",
    "green":   "#2DC653",
    "purple":  "#7B2D8B",
    "gray":    "#ADB5BD",
    "white":   "#F8F9FA",
    "dgray":   "#495057",
}

LEVEL_COLOUR = {
    "CRITICAL": C["red"],
    "HIGH":     C["orange"],
    "MEDIUM":   "#F9C74F",
    "LOW":      C["green"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET  ─ 120 records covering real maritime systems
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset() -> pd.DataFrame:
    """
    Returns a DataFrame of 120 synthetic-but-grounded records derived from:
      • NIST IR 8547 qubit estimates
      • MCAD maritime cyber incident database
      • IMO MSC-FAL.1 cyber risk framework
    """
    rng = np.random.default_rng(42)   # reproducible

    # ── Base templates (system, algo, key_size, likelihood, impact, readiness) ──
    templates = [
        # system,                  algo,         key,  L,  I,  R
        ("AIS",                    "ECDSA",       256,  9,  9,  2),
        ("ECDIS",                  "RSA",        2048,  8, 10,  2),
        ("SATCOM",                 "RSA",        2048,  9,  8,  3),
        ("Port_OT",                "RSA",        2048,  7, 10,  2),
        ("GMDSS",                  "RSA",        1024,  9,  9,  1),
        ("Vessel_Email",           "RSA",        2048,  8,  6,  4),
        ("AIS_Firmware",           "RSA",        2048,  7,  8,  3),
        ("Engine_OT",              "AES",         128,  6,  7,  5),
        ("Bridge_Auth",            "AES",         256,  5,  8,  6),
        ("Cargo_Manifest",         "ECC",         256,  6,  7,  5),
        ("VSAT",                   "ECC",         256,  7,  6,  4),
        ("Biometric_Access",       "RSA",        2048,  4,  5,  7),
        ("PQC_Hybrid_Pilot",       "ML-KEM",      768,  1,  2,  9),
        ("GNSS_Receiver",          "AES",         128,  6,  9,  5),
        ("Port_VPN",               "DH",         2048,  7,  9,  3),
        ("Cargo_Database",         "AES",         256,  5,  7,  6),
        ("Crew_PII",               "RSA",        2048,  9, 10,  2),
        ("Financial_Tx",           "RSA",        2048,  8, 10,  3),
        ("Chart_Updates",          "ECC",         384,  8,  9,  3),
        ("Legacy_AIS",             "RSA",        1024,  9,  9,  1),
    ]

    rows = []
    for (sys_name, algo, key, L, I, R) in templates:
        for _ in range(6):              # 6 variations per template → 120 rows
            l = int(np.clip(L + rng.integers(-1, 2), 1, 10))
            i = int(np.clip(I + rng.integers(-1, 2), 1, 10))
            r = int(np.clip(R + rng.integers(-1, 2), 1, 10))
            k = key                     # key size fixed per template

            # ── Derived features ──────────────────────────────────────────────
            # Qubit cost to break (millions) – from NIST IR 8547 lookup
            qubits_m = _qubit_cost(algo, k)

            # Post-quantum security bits (Grover halves symmetric, Shor→0 asymm.)
            pq_bits = _pq_security(algo, k)

            # Risk score (regression target)  Y = f(X) with algorithm penalty
            algo_penalty = {
                "RSA": 38, "ECC": 32, "ECDSA": 32, "DH": 34,
                "AES": 8, "ML-KEM": -35, "ML-DSA": -30,
            }.get(algo, 20)
            readiness_boost = max(0, (5 - r) * 2.5)
            risk = float(np.clip(
                (l * 0.8) + (i * 0.9) - (r * 0.5)
                + algo_penalty + readiness_boost + rng.normal(0, 3.5),
                0, 100
            ))

            # Vulnerability label (classification target)
            label = _risk_to_label(risk)

            # Q-Day year estimate
            qday = _qday_year(algo, k)

            rows.append({
                "system":       sys_name,
                "algorithm":    algo,
                "key_size":     k,
                "likelihood":   l,
                "impact":       i,
                "pqc_readiness":r,
                "qubits_m":     qubits_m,
                "pq_sec_bits":  pq_bits,
                "risk_score":   round(risk, 2),
                "qday_year":    qday,
                "label":        label,
            })

    return pd.DataFrame(rows)


def _qubit_cost(algo: str, key: int) -> float:
    """Physical qubits (millions) to break algo at key_size, per NIST IR 8547."""
    mapping = {
        ("RSA",   512):   2.0,  ("RSA",  1024):  4.0,
        ("RSA",  2048):   8.0,  ("RSA",  4096): 16.0,
        ("ECC",   160):   0.8,  ("ECC",   256):  1.2,
        ("ECC",   384):   1.8,  ("ECC",   521):  2.5,
        ("ECDSA", 256):   1.2,  ("ECDSA", 384):  1.8,
        ("DH",   2048):   8.0,  ("DH",   4096): 16.0,
        ("AES",   128): 999.0,  ("AES",   256): 999.0,  # Grover, not Shor
        ("ML-KEM",768):   0.0,  # PQC – immune
    }
    return mapping.get((algo, key), 8.0)


def _pq_security(algo: str, key: int) -> int:
    """Effective security bits after quantum attack."""
    if algo in ("RSA", "ECC", "ECDSA", "DH"):
        return 0        # Shor breaks completely
    if algo == "AES":
        return key // 2  # Grover halves
    if algo == "ML-KEM":
        return key       # Lattice – remains secure
    return 0


def _risk_to_label(score: float) -> str:
    if score >= 70:  return "CRITICAL"
    if score >= 50:  return "HIGH"
    if score >= 30:  return "MEDIUM"
    return "LOW"


def _qday_year(algo: str, key: int) -> int:
    """Estimated year the algorithm gets broken at current quantum roadmap."""
    if algo == "ML-KEM":        return 9999   # PQC safe
    if algo in ("RSA", "DH"):
        yr = {512: 2027, 1024: 2028, 2048: 2030, 4096: 2033}
        return yr.get(key, 2030)
    if algo in ("ECC", "ECDSA"):
        yr = {160: 2026, 256: 2027, 384: 2028, 521: 2029}
        return yr.get(key, 2027)
    if algo == "AES":
        return 9999 if key >= 256 else 2035  # Grover weaker – longer horizon
    return 2030


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_features(df: pd.DataFrame):
    """Encode categorical columns and return feature matrix X, encoders dict."""
    enc = {}
    df2 = df.copy()
    for col in ("system", "algorithm"):
        le = LabelEncoder()
        df2[col + "_enc"] = le.fit_transform(df2[col])
        enc[col] = le
    feature_cols = ["algorithm_enc", "key_size", "likelihood", "impact",
                    "pqc_readiness", "qubits_m", "pq_sec_bits"]
    return df2, feature_cols, enc


def train_models(df: pd.DataFrame):
    """
    Train:
      rf_clf  → RandomForestClassifier  (predicts label: CRITICAL/HIGH/MEDIUM/LOW)
      rf_reg  → RandomForestRegressor   (predicts risk_score 0-100)

    Returns models, encoders, feature names, and test splits for QA.
    """
    df2, feat_cols, enc = encode_features(df)

    X = df2[feat_cols].values
    y_cls = df2["label"].values
    y_reg = df2["risk_score"].values

    # Label-encode the classification target
    le_label = LabelEncoder()
    y_cls_enc = le_label.fit_transform(y_cls)
    enc["label"] = le_label

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls_enc, y_reg, test_size=0.25, random_state=42, stratify=y_cls_enc
    )

    # ── Classifier ──────────────────────────────────────────────────────────
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_tr, yc_tr)

    # ── Regressor ───────────────────────────────────────────────────────────
    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf_reg.fit(X_tr, yr_tr)

    # ── Metrics ─────────────────────────────────────────────────────────────
    yc_pred = rf_clf.predict(X_te)
    yr_pred = rf_reg.predict(X_te)

    cv_scores = cross_val_score(rf_clf, X, y_cls_enc, cv=5, scoring="accuracy")

    metrics = {
        "clf_report":  classification_report(
            yc_te, yc_pred,
            target_names=le_label.classes_, output_dict=True
        ),
        "confusion":   confusion_matrix(yc_te, yc_pred),
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "reg_mae":     mean_absolute_error(yr_te, yr_pred),
        "reg_r2":      r2_score(yr_te, yr_pred),
        "feat_imp":    dict(zip(feat_cols, rf_clf.feature_importances_)),
        "label_names": le_label.classes_,
    }

    return rf_clf, rf_reg, enc, feat_cols, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict(rf_clf, rf_reg, enc, feat_cols,
            system: str, algorithm: str, key_size: int,
            likelihood: int, impact: int, pqc_readiness: int,
            qubit_year: int) -> dict:
    """
    Run classifier + regressor on one user-supplied asset.
    Returns a rich dict with all outputs.
    """
    algo = algorithm.upper().replace(" ", "-")

    # Derived features
    qubits_m  = _qubit_cost(algo, key_size)
    pq_bits   = _pq_security(algo, key_size)
    qday      = _qday_year(algo, key_size)

    # Encode algorithm (handle unseen values gracefully)
    le_algo = enc["algorithm"]
    if algo in le_algo.classes_:
        algo_enc = int(le_algo.transform([algo])[0])
    else:
        # Map close aliases
        aliases = {"ECDSA": "ECDSA", "ECC": "ECC", "AES-256": "AES",
                   "AES-128": "AES", "KYBER": "ML-KEM", "DILITHIUM": "ML-DSA"}
        mapped = aliases.get(algo, "RSA")
        algo_enc = int(le_algo.transform([mapped])[0]) if mapped in le_algo.classes_ else 0

    X = np.array([[algo_enc, key_size, likelihood, impact,
                   pqc_readiness, qubits_m, pq_bits]])

    label_enc  = rf_clf.predict(X)[0]
    label_prob = rf_clf.predict_proba(X)[0]
    label      = enc["label"].inverse_transform([label_enc])[0]
    risk_score = float(np.clip(rf_reg.predict(X)[0], 0, 100))

    # Q-Day simulation
    breach_status = _breach_status(algo, key_size, qubit_year)
    years_to_qday = max(0, qday - 2026) if qday < 9999 else None

    # Recommendation
    recommendation = _recommend(algo, key_size, risk_score, label)

    return {
        "system":          system,
        "algorithm":       algo,
        "key_size":        key_size,
        "likelihood":      likelihood,
        "impact":          impact,
        "pqc_readiness":   pqc_readiness,
        "qubit_year":      qubit_year,
        "qubits_needed_m": qubits_m,
        "pq_sec_bits":     pq_bits,
        "risk_score":      round(risk_score, 2),
        "label":           label,
        "label_proba":     {enc["label"].classes_[i]: round(p*100, 1)
                            for i, p in enumerate(label_prob)},
        "qday_year":       qday,
        "years_to_qday":   years_to_qday,
        "breach_status":   breach_status,
        "recommendation":  recommendation,
    }


def _breach_status(algo: str, key: int, qubit_year: int) -> str:
    """Given the user's projected qubit availability year, is it already broken?"""
    qcost = _qubit_cost(algo, key)
    if qcost == 0:
        return "IMMUNE — Post-Quantum Algorithm"
    if qcost >= 999:
        return "GROVER THREAT — upgrade to AES-256 recommended"
    # Rough qubit availability curve: 2026=1M, 2027=2M, 2028=5M, 2030=10M, 2033=20M+
    avail = {2026: 1.0, 2027: 2.0, 2028: 5.0, 2029: 7.0,
             2030: 10.0, 2031: 12.0, 2032: 15.0, 2033: 20.0}
    q_avail = avail.get(qubit_year, qubit_year * 0.7)
    if q_avail >= qcost:
        return f"⚠  BREACH FEASIBLE in {qubit_year} ({q_avail:.0f}M qubits ≥ {qcost:.0f}M needed)"
    return f"NOT YET FEASIBLE in {qubit_year} ({q_avail:.0f}M < {qcost:.0f}M qubits needed)"


def _recommend(algo: str, key: int, score: float, label: str) -> list[str]:
    recs = []
    if algo in ("RSA", "ECC", "ECDSA", "DH"):
        recs.append("🔴  MIGRATE: Replace with ML-KEM (FIPS 203) for key exchange")
        recs.append("🔴  SIGN:    Use ML-DSA (FIPS 204) for digital signatures")
    if algo == "AES" and key < 256:
        recs.append("🟠  UPGRADE: Increase AES key from 128→256 (Grover resistance)")
    if algo == "ML-KEM":
        recs.append("🟢  SECURE:  Lattice-based algorithm — quantum safe")
    if label in ("CRITICAL", "HIGH"):
        recs.append("🔴  PRIORITY: Schedule immediate crypto-audit for this system")
        recs.append("🟠  INTERIM: Deploy hybrid RSA + PQC wrapper now")
    if label == "MEDIUM":
        recs.append("🟡  PLAN:    Include in next 12-month PQC migration cycle")
    if label == "LOW":
        recs.append("🟢  MONITOR: Maintain crypto-agile architecture, review yearly")
    recs.append("📋  STANDARD: Reference NIST FIPS 203/204 & IMO MSC-FAL.1/Circ.3")
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_charts(df: pd.DataFrame, metrics: dict, result: dict,
                out_dir: str = "."):
    """Generate and save 4 publication-quality charts."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor":   C["navy"],
        "figure.facecolor": C["navy"],
        "text.color":       C["white"],
        "axes.labelcolor":  C["white"],
        "xtick.color":      C["gray"],
        "ytick.color":      C["gray"],
        "axes.edgecolor":   "#1A3A5C",
        "grid.color":       "#1A3A5C",
        "grid.linewidth":   0.6,
    })

    paths = []

    # ── Chart 1: Feature Importance ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(C["navy"])
    feats = metrics["feat_imp"]
    names_map = {
        "algorithm_enc": "Algorithm",
        "key_size":       "Key Size",
        "likelihood":     "Likelihood",
        "impact":         "Impact",
        "pqc_readiness":  "PQC Readiness",
        "qubits_m":       "Qubits Needed (M)",
        "pq_sec_bits":    "Post-Q Security Bits",
    }
    sorted_f = sorted(feats.items(), key=lambda x: x[1], reverse=True)
    names  = [names_map.get(k, k) for k, _ in sorted_f]
    values = [v for _, v in sorted_f]
    bar_colours = [C["blue"] if v == max(values) else C["purple"] for v in values]
    bars = ax.barh(names, values, color=bar_colours, height=0.55, zorder=3)
    ax.set_xlabel("Feature Importance (Gini)", color=C["gray"], fontsize=10)
    ax.set_title("Random Forest — Feature Importance\n(What drives vulnerability prediction)",
                 color=C["white"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", zorder=0)
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=C["blue"], fontsize=9)
    cv_txt = (f"5-Fold CV Accuracy: {metrics['cv_mean']*100:.1f}% "
              f"(±{metrics['cv_std']*100:.1f}%)")
    fig.text(0.99, 0.02, cv_txt, ha="right", color=C["gray"], fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "chart1_feature_importance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # ── Chart 2: Confusion Matrix ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(C["navy"])
    cm     = metrics["confusion"]
    labels = metrics["label_names"]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label", color=C["gray"])
    ax.set_ylabel("True Label",      color=C["gray"])
    ax.set_title("Confusion Matrix — Vulnerability Classification",
                 color=C["white"], fontsize=12, fontweight="bold", pad=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color=C["navy"] if cm[i, j] > thresh else C["white"],
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    acc_txt = f"Classifier Accuracy: {metrics['clf_report']['accuracy']*100:.1f}%"
    fig.text(0.5, 0.02, acc_txt, ha="center", color=C["blue"], fontsize=10)
    plt.tight_layout()
    p = os.path.join(out_dir, "chart2_confusion_matrix.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # ── Chart 3: XY Matrix – Risk Score vs PQC Readiness ────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(C["navy"])
    ax.set_facecolor("#0F2235")
    ax.axvline(5, color="#1A3A5C", linewidth=1.5, zorder=1)
    ax.axhline(50, color="#1A3A5C", linewidth=1.5, zorder=1)

    label_col_map = {"CRITICAL": C["red"], "HIGH": C["orange"],
                     "MEDIUM": "#F9C74F", "LOW": C["green"]}
    for lbl, grp in df.groupby("label"):
        ax.scatter(grp["pqc_readiness"], grp["risk_score"],
                   c=label_col_map[lbl], label=lbl,
                   s=60, alpha=0.75, zorder=3, edgecolors="none")

    # Annotate the user's prediction
    ax.scatter(result["pqc_readiness"], result["risk_score"],
               c=LEVEL_COLOUR.get(result["label"], C["blue"]),
               s=280, marker="*", zorder=5, edgecolors=C["white"],
               linewidths=1.2, label=f"YOUR INPUT → {result['label']}")

    ax.set_xlabel("PQC Readiness (1 = Not Ready, 10 = Fully Deployed)",
                  color=C["gray"], fontsize=10)
    ax.set_ylabel("Risk Score (0–100)", color=C["gray"], fontsize=10)
    ax.set_title("XY Matrix: Risk Score vs PQC Readiness\n(★ = your input asset)",
                 color=C["white"], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 11); ax.set_ylim(-5, 105)
    ax.grid(zorder=0)

    # Quadrant labels
    for (tx, ty, txt, col) in [
        (1.2, 95,  "HIGH RISK / LOW READY",   C["red"]),
        (6.5, 95,  "HIGH RISK / HIGH READY",  C["orange"]),
        (1.2, 8,   "LOW RISK / LOW READY",    C["orange"]),
        (6.5, 8,   "LOW RISK / HIGH READY ✓", C["green"]),
    ]:
        ax.text(tx, ty, txt, color=col, fontsize=8, fontweight="bold", alpha=0.85)

    legend = ax.legend(facecolor=C["navy"], edgecolor="#1A3A5C",
                       labelcolor=C["white"], fontsize=9, loc="center right")
    plt.tight_layout()
    p = os.path.join(out_dir, "chart3_xy_matrix.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # ── Chart 4: Q-Day Timeline ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(C["navy"])
    ax.set_facecolor("#0F2235")

    systems_qday = [
        ("Legacy AIS (RSA-1024)",  2028),
        ("GMDSS (RSA-1024)",       2028),
        ("AIS/ECDIS (ECC-256)",    2027),
        ("SATCOM (RSA-2048)",      2030),
        ("Port OT VPN (DH-2048)",  2030),
        ("Vessel Email (RSA-2048)",2030),
        ("AES-256 Systems",        2035),
        ("PQC Hybrid (ML-KEM)",    9999),
    ]
    years_display = [y if y < 9999 else 2038 for _, y in systems_qday]
    labels_sys    = [s for s, _ in systems_qday]
    bar_cols      = [
        C["red"] if y <= 2028 else
        C["orange"] if y <= 2030 else
        "#F9C74F" if y <= 2035 else
        C["green"]
        for y in years_display
    ]

    bars = ax.barh(labels_sys, [y - 2026 for y in years_display],
                   left=2026, color=bar_cols, height=0.55, zorder=3)
    ax.axvline(2026, color=C["blue"],  linewidth=2, linestyle="--", zorder=4, label="Now (2026)")
    ax.axvline(result["qubit_year"], color=C["white"],
               linewidth=1.5, linestyle=":", zorder=4,
               label=f"Your Qubit Year ({result['qubit_year']})")

    ax.set_xlabel("Year", color=C["gray"], fontsize=10)
    ax.set_title("Q-Day Timeline: When Does Each Algorithm Get Broken?\n"
                 "(based on quantum hardware roadmap)",
                 color=C["white"], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(2025, 2040)
    ax.grid(axis="x", zorder=0)
    ax.invert_yaxis()

    for bar, (name, yr) in zip(bars, systems_qday):
        label_x = (2038 if yr >= 9999 else yr) + 0.1
        txt = "SAFE (PQC)" if yr >= 9999 else str(yr)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                txt, va="center", fontsize=8.5,
                color=C["green"] if yr >= 9999 else C["white"])

    legend = ax.legend(facecolor=C["navy"], edgecolor="#1A3A5C",
                       labelcolor=C["white"], fontsize=9)
    patches = [mpatches.Patch(color=C["red"],    label="CRITICAL ≤2028"),
               mpatches.Patch(color=C["orange"], label="HIGH 2028-2030"),
               mpatches.Patch(color="#F9C74F",   label="MEDIUM 2030-2035"),
               mpatches.Patch(color=C["green"],  label="SECURE / PQC")]
    ax.legend(handles=patches, facecolor=C["navy"], edgecolor="#1A3A5C",
              labelcolor=C["white"], fontsize=8, loc="lower right")
    plt.tight_layout()
    p = os.path.join(out_dir, "chart4_qday_timeline.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(p)

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PRETTY-PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _banner():
    print("\n" + "═"*72)
    print("  QUANTUM THREAT MODELLING — MARITIME CRITICAL INFRASTRUCTURE")
    print("  Random Forest ML Engine  |  Cryptology CSE2021  |  VIT AP 2026")
    print("  Nikhil (23BCB7059)  &  Adhithya Raviprakash (23BCB7141)")
    print("═"*72 + "\n")


def _section(title: str):
    print("\n" + "─"*72)
    print(f"  {title}")
    print("─"*72)


def _print_model_metrics(metrics: dict):
    _section("MODEL PERFORMANCE REPORT")
    print(f"\n  Random Forest Classifier")
    print(f"  ├─ 5-Fold CV Accuracy : {metrics['cv_mean']*100:.2f}% "
          f"(±{metrics['cv_std']*100:.2f}%)")
    rpt = metrics["clf_report"]
    print(f"  ├─ Test Precision (w) : {rpt['weighted avg']['precision']*100:.2f}%")
    print(f"  ├─ Test Recall    (w) : {rpt['weighted avg']['recall']*100:.2f}%")
    print(f"  └─ Test F1-Score  (w) : {rpt['weighted avg']['f1-score']*100:.2f}%")
    print(f"\n  Random Forest Regressor  (Risk Score 0–100)")
    print(f"  ├─ Mean Absolute Error  : {metrics['reg_mae']:.2f} points")
    print(f"  └─ R² Score             : {metrics['reg_r2']:.4f}")
    print(f"\n  Top Feature Importances:")
    sorted_f = sorted(metrics["feat_imp"].items(), key=lambda x: x[1], reverse=True)
    name_map = {
        "algorithm_enc": "Algorithm",      "key_size": "Key Size",
        "likelihood":    "Likelihood",     "impact": "Impact",
        "pqc_readiness": "PQC Readiness",  "qubits_m": "Qubits Needed",
        "pq_sec_bits":   "Post-Q Sec Bits"
    }
    for k, v in sorted_f[:4]:
        bar = "█" * int(v * 50)
        print(f"  {name_map.get(k,k):22s} {bar:<25s} {v:.4f}")


def _print_result(r: dict):
    _section("PREDICTION RESULT")
    lvl_sym = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}
    sym = lvl_sym.get(r["label"], "⚪")
    print(f"\n  {sym}  VULNERABILITY LEVEL : {r['label']}")
    print(f"  ★  RISK SCORE         : {r['risk_score']:.1f} / 100")
    print(f"\n  Probability breakdown:")
    for lbl, pct in sorted(r["label_proba"].items(), key=lambda x:-x[1]):
        bar = "█" * int(pct / 3)
        print(f"    {lbl:10s} {bar:<35s} {pct:.1f}%")

    print(f"\n  Asset Details:")
    print(f"    System         : {r['system']}")
    print(f"    Algorithm      : {r['algorithm']}-{r['key_size']}")
    print(f"    Likelihood     : {r['likelihood']}/10")
    print(f"    Impact         : {r['impact']}/10")
    print(f"    PQC Readiness  : {r['pqc_readiness']}/10")
    print(f"    Post-Q Sec Bits: {r['pq_sec_bits']} bits "
          f"({'BROKEN' if r['pq_sec_bits']==0 else 'WEAKENED' if r['pq_sec_bits']<112 else 'SAFE'})")

    _section("Q-DAY SIMULATION")
    print(f"\n  Qubits needed to break {r['algorithm']}-{r['key_size']}: "
          f"{r['qubits_needed_m']:.1f}M physical qubits")
    print(f"  Estimated Q-Day year : "
          f"{'IMMUNE (PQC)' if r['qday_year']==9999 else str(r['qday_year'])}")
    if r["years_to_qday"] is not None:
        print(f"  Time remaining       : ~{r['years_to_qday']} year(s) from 2026")
    print(f"\n  Breach at your qubit year ({r['qubit_year']}):")
    print(f"    → {r['breach_status']}")

    _section("RECOMMENDATIONS")
    for rec in r["recommendation"]:
        print(f"  {rec}")


def _print_chart_paths(paths: list[str]):
    _section("CHARTS SAVED")
    for p in paths:
        print(f"  ✦  {os.path.abspath(p)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  USER INPUT
# ═══════════════════════════════════════════════════════════════════════════════

VALID_ALGOS = ["RSA", "ECC", "ECDSA", "AES", "DH", "ML-KEM", "ML-DSA"]
VALID_SYSTEMS = [
    "AIS", "ECDIS", "SATCOM", "Port_OT", "GMDSS", "Vessel_Email",
    "AIS_Firmware", "Engine_OT", "Bridge_Auth", "Cargo_Manifest",
    "VSAT", "Biometric_Access", "PQC_Hybrid_Pilot", "GNSS_Receiver",
    "Port_VPN", "Cargo_Database", "Crew_PII", "Financial_Tx",
    "Chart_Updates", "Legacy_AIS", "Custom"
]


def _ask(prompt: str, default=None, cast=str, valid=None):
    """Prompt user, apply cast, validate against valid list if provided."""
    while True:
        raw = input(f"  {prompt}" + (f" [{default}]" if default is not None else "") + " : ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw)
        except (ValueError, TypeError):
            print(f"    ✘  Invalid input. Expected {cast.__name__}.")
            continue
        if valid and val not in valid:
            print(f"    ✘  Must be one of: {', '.join(str(v) for v in valid)}")
            continue
        return val


def get_user_input() -> dict:
    _section("USER INPUT — Maritime Asset Details")
    print("\n  Please enter details for the asset you want to assess.\n")
    print(f"  Valid systems : {', '.join(VALID_SYSTEMS)}")
    system = _ask("Maritime system name", default="AIS",
                  valid=VALID_SYSTEMS + [s.lower() for s in VALID_SYSTEMS])
    system = system.upper() if system.upper() in VALID_SYSTEMS else system

    print(f"\n  Valid algorithms: {', '.join(VALID_ALGOS)}")
    algorithm = _ask("Encryption algorithm", default="RSA",
                     valid=VALID_ALGOS + [a.lower() for a in VALID_ALGOS])
    algorithm = algorithm.upper()

    key_size   = _ask("Key size in bits (e.g. 2048 / 256 / 128)", default=2048, cast=int)
    likelihood = _ask("Likelihood of attack  (1=very low … 10=certain)", default=8,
                      cast=int, valid=list(range(1, 11)))
    impact     = _ask("Impact if breached   (1=trivial … 10=catastrophic)", default=9,
                      cast=int, valid=list(range(1, 11)))
    readiness  = _ask("PQC readiness score  (1=none … 10=fully deployed)", default=2,
                      cast=int, valid=list(range(1, 11)))
    qubit_year = _ask("Qubit projection year to test (e.g. 2026–2033)", default=2030,
                      cast=int, valid=list(range(2024, 2041)))

    return dict(system=system, algorithm=algorithm, key_size=key_size,
                likelihood=likelihood, impact=impact, pqc_readiness=readiness,
                qubit_year=qubit_year)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _banner()

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("  [1/4]  Building training dataset (120 records)…")
    df = build_dataset()
    print(f"         Dataset shape : {df.shape}  |  "
          f"Labels: {df['label'].value_counts().to_dict()}")

    # ── Train models ──────────────────────────────────────────────────────────
    print("  [2/4]  Training Random Forest models (classifier + regressor)…")
    rf_clf, rf_reg, enc, feat_cols, metrics = train_models(df)
    _print_model_metrics(metrics)

    # ── User input ────────────────────────────────────────────────────────────
    print("\n  [3/4]  Ready for user input.")
    user_inp = get_user_input()

    # ── Predict ───────────────────────────────────────────────────────────────
    result = predict(rf_clf, rf_reg, enc, feat_cols, **user_inp)
    _print_result(result)

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\n  [4/4]  Generating charts…")
    chart_paths = make_charts(df, metrics, result)
    _print_chart_paths(chart_paths)

    print("\n" + "═"*72)
    print("  ASSESSMENT COMPLETE")
    print("  Reference: NIST FIPS 203/204 · IMO MSC-FAL.1/Circ.3 · G7 CEG 2026")
    print("═"*72 + "\n")


if __name__ == "__main__":
    main()