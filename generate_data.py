"""
Generate competition dataset from etch_vm simulation.

Train: Chamber_1, lots 1-60 (1500 runs)
Test:  Chamber_2 lots 61-80 (500 runs) + Chamber_1 lots 61-80 (500 runs) = 1000 runs

Total: 4000 runs (2 chambers × 25 wafers × 80 lots)

Outputs:
  data/train.csv           — features + etch_rate (labeled)
  data/test_features.csv   — features only (no etch_rate)
  data/test_answers.csv    — run_id + etch_rate (for scoring)
  data/feature_description.csv — feature name + category + description
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etch_vm.simulator import EtchSimulator, SimulationConfig
from etch_vm.features import FeatureExtractor


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FEATURE_DESCRIPTIONS = {
    # OES line statistics (note: wavelength in nm, species label from sorted order)
    "oes_F_252_mean": ("OES", "SiF 251.6nm emission mean intensity"),
    "oes_F_252_std": ("OES", "SiF 251.6nm emission intensity std"),
    "oes_F_486_mean": ("OES", "H 486.1nm emission mean intensity"),
    "oes_F_486_std": ("OES", "H 486.1nm emission intensity std"),
    "oes_F_656_mean": ("OES", "H-alpha 656.3nm emission mean intensity"),
    "oes_F_656_std": ("OES", "H-alpha 656.3nm emission intensity std"),
    "oes_F_686_mean": ("OES", "F 685.6nm emission mean intensity"),
    "oes_F_686_std": ("OES", "F 685.6nm emission intensity std"),
    "oes_F_704_mean": ("OES", "F 703.7nm emission mean intensity"),
    "oes_F_704_std": ("OES", "F 703.7nm emission intensity std"),
    "oes_F_750_mean": ("OES", "Ar 750.4nm emission mean intensity"),
    "oes_F_750_std": ("OES", "Ar 750.4nm emission intensity std"),
    "oes_F_812_mean": ("OES", "Ar 811.5nm emission mean intensity"),
    "oes_F_812_std": ("OES", "Ar 811.5nm emission intensity std"),
    # OES physics ratios
    "oes_Ar_line_ratio": ("OES_Ratio", "Ar 750nm / Ar 811nm ratio"),
    "oes_F_Ar_ratio": ("OES_Ratio", "F 704nm / Ar 750nm ratio (etchant proxy)"),
    "oes_F_line_ratio": ("OES_Ratio", "F 686nm / F 704nm ratio"),
    "oes_H_F_ratio": ("OES_Ratio", "H 486nm / F 704nm ratio (polymer indicator)"),
    "oes_SiF_F_ratio": ("OES_Ratio", "SiF 252nm / F 704nm ratio (etch product)"),
    "oes_SiF_intensity": ("OES_Ratio", "SiF 251.6nm raw intensity"),
    # RF statistics
    "rf_I_mean": ("RF", "RF current mean"),
    "rf_I_std": ("RF", "RF current std"),
    "rf_V_mean": ("RF", "RF voltage mean"),
    "rf_V_slope": ("RF", "RF voltage linear slope over time"),
    "rf_V_std": ("RF", "RF voltage std"),
    "rf_Z_estimate": ("RF", "Estimated impedance V/I"),
    "rf_phase_mean": ("RF", "RF phase mean (rad)"),
    "rf_phase_slope": ("RF", "RF phase linear slope over time"),
    "rf_phase_std": ("RF", "RF phase std"),
    # Process parameters
    "flow_Ar_mean": ("Process", "Ar gas flow mean (sccm)"),
    "flow_Ar_std": ("Process", "Ar gas flow std"),
    "flow_CF4_mean": ("Process", "CF4 gas flow mean (sccm)"),
    "flow_CF4_std": ("Process", "CF4 gas flow std"),
    "flow_CHF3_mean": ("Process", "CHF3 gas flow mean (sccm)"),
    "flow_CHF3_std": ("Process", "CHF3 gas flow std"),
    "pressure_mean": ("Process", "Chamber pressure mean (mTorr)"),
    "pressure_std": ("Process", "Chamber pressure std"),
    "temp_mean": ("Process", "Wafer temperature mean (deg C)"),
    "temp_std": ("Process", "Wafer temperature std"),
    # Cross-interactions
    "ar_total_ratio": ("Cross", "Ar flow / total flow ratio"),
    "cf4_chf3_ratio": ("Cross", "CF4 / CHF3 flow ratio"),
    "power_per_flow": ("Cross", "RF power / total gas flow"),
    "power_pressure_product": ("Cross", "RF power x pressure interaction"),
    "pressure_flow_product": ("Cross", "Pressure x total flow interaction"),
    "power_squared": ("Cross", "Normalized power squared"),
    # Chamber state proxies
    "is_chamber_2": ("Chamber", "Chamber identity (0=Chamber_1, 1=Chamber_2)"),
    "after_pm": ("Chamber", "Post-PM flag (1 if run is immediately after PM)"),
    "wafer_slot": ("Chamber", "Wafer position within lot (0-24)"),
}


def generate():
    print("Running simulation (4000 runs, seed=42)...")
    cfg = SimulationConfig(
        n_chambers=2,
        n_wafers_per_lot=25,
        n_lots=80,
        metrology_sample_rate=0.12,
        pm_interval=50,
        seed=42,
    )
    sim = EtchSimulator(cfg)
    runs = sim.run_campaign()

    print("Extracting features...")
    fe = FeatureExtractor(feature_noise_std=0.01)
    X, y, feature_names, metrology_mask = fe.extract_batch(runs)

    # Build metadata with per-chamber run counts
    metadata = []
    chamber_run_count = {}  # per-chamber sequential counter
    for run in runs:
        ch = run.chamber_id
        if ch not in chamber_run_count:
            chamber_run_count[ch] = 0
        chamber_run = chamber_run_count[ch]
        chamber_run_count[ch] += 1

        metadata.append({
            "run_id": run.run_id,
            "chamber_id": ch,
            "lot_id": run.wafer_lot_id,
            "wafer_id": run.wafer_id,
            "is_metrology": run.is_metrology_wafer,
            "after_pm": int(run.after_pm),
            "chamber_run": chamber_run,  # per-chamber sequential run number
        })
    meta_df = pd.DataFrame(metadata)

    # Feature dataframe
    feat_df = pd.DataFrame(X, columns=feature_names)
    feat_df["etch_rate"] = y

    # Drop features that leak global run_id or duplicate metadata
    # Replace with per-chamber computed features
    drop_feats = {"run_number", "runs_in_pm_cycle", "aging_quadratic", "after_pm", "wafer_slot"}
    feat_df = feat_df.drop(columns=[c for c in drop_feats if c in feat_df.columns])
    feature_names = [f for f in feature_names if f not in drop_feats]

    # Add per-chamber aging features using chamber_run
    pm_interval = 50
    feat_df["runs_in_pm_cycle"] = (meta_df["chamber_run"] % pm_interval).astype(float)
    feat_df["aging_quadratic"] = ((meta_df["chamber_run"] % pm_interval) ** 2 / pm_interval ** 2).astype(float)
    feat_df["wafer_slot"] = meta_df["wafer_id"].astype(float)
    # after_pm is in meta_df already — don't duplicate in feat_df
    # is_chamber_2 already in features from extractor
    feature_names += ["runs_in_pm_cycle", "aging_quadratic", "wafer_slot"]

    full_df = pd.concat([meta_df, feat_df], axis=1)

    # --- Train/test split ---
    train_mask = (full_df["chamber_id"] == "Chamber_1") & (full_df["lot_id"] < 60)
    test_mask = (
        ((full_df["chamber_id"] == "Chamber_1") & (full_df["lot_id"] >= 60)) |
        ((full_df["chamber_id"] == "Chamber_2") & (full_df["lot_id"] >= 60))
    )

    train_df = full_df[train_mask].reset_index(drop=True)
    test_df = full_df[test_mask].reset_index(drop=True)

    print(f"Train: {len(train_df)} runs (Chamber_1, lots 1-60)")
    print(f"Test:  {len(test_df)} runs (Chamber_1+2, lots 61-80)")

    # Drop chamber_run from output (internal use only)
    # after_pm is both in meta and features — use clean meta version only
    output_meta = ["run_id", "chamber_id", "lot_id", "wafer_id", "is_metrology", "after_pm"]
    feature_names_out = [f for f in feature_names if f not in {"after_pm"}]

    # Save train.csv
    train_cols = output_meta + feature_names_out + ["etch_rate"]
    train_df[train_cols].to_csv(DATA_DIR / "train.csv", index=False)
    print(f"  -> train.csv ({len(train_df)} rows)")

    # Save test_features.csv (no etch_rate)
    test_feature_cols = output_meta + feature_names_out
    test_df[test_feature_cols].to_csv(DATA_DIR / "test_features.csv", index=False)
    print(f"  -> test_features.csv ({len(test_df)} rows)")

    # Save test_answers.csv
    test_df[["run_id", "etch_rate"]].to_csv(DATA_DIR / "test_answers.csv", index=False)
    print(f"  -> test_answers.csv ({len(test_df)} rows)")

    # Save feature_description.csv
    desc_rows = []
    for fname in sorted(feature_names_out):
        cat, desc = FEATURE_DESCRIPTIONS.get(fname, ("Unknown", ""))
        desc_rows.append({"feature": fname, "category": cat, "description": desc})
    pd.DataFrame(desc_rows).to_csv(DATA_DIR / "feature_description.csv", index=False)
    print(f"  -> feature_description.csv ({len(desc_rows)} features)")

    print(f"\nTrain etch_rate: mean={train_df['etch_rate'].mean():.1f} std={train_df['etch_rate'].std():.1f}")
    print(f"Test  etch_rate: mean={test_df['etch_rate'].mean():.1f} std={test_df['etch_rate'].std():.1f}")

    # Quick sanity check
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score

    X_tr = train_df[feature_names_out].values
    y_tr = train_df["etch_rate"].values
    X_te = test_df[feature_names_out].values
    y_te = test_df["etch_rate"].values

    pls = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression(n_components=8))])
    pls.fit(X_tr, y_tr)
    pred = pls.predict(X_te).ravel()
    r2 = r2_score(y_te, pred)
    print(f"\nSanity check — PLS R2: {r2:.4f}")

    if r2 < 0.65:
        print("WARNING: PLS R2 is below 0.65, check feature engineering")
    elif r2 > 0.85:
        print("WARNING: PLS R2 is above 0.85, competition may be too easy")

    print("Done.")


if __name__ == "__main__":
    generate()
