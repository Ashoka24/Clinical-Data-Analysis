"""Exploratory analysis for the synthetic portfolio dataset."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"


def load_data():
    encounters = pd.read_csv(DATA / "encounters.csv", parse_dates=["EncounterDate"])
    labs = pd.read_csv(DATA / "lab_results.csv", parse_dates=["CollectedDate"])
    return encounters, labs


def validate_data(encounters: pd.DataFrame, labs: pd.DataFrame) -> list[str]:
    errors = []
    required_enc = {"EncounterID", "PatientID", "EncounterDate", "Department", "LengthOfStayDays",
                    "SimulatedReadmissionFlag", "TotalCostUSD", "ICUFlag"}
    required_labs = {"EncounterID", "LabTest", "Result", "ReferenceLow", "ReferenceHigh", "AbnormalFlag"}
    if missing := required_enc - set(encounters.columns):
        errors.append(f"Missing encounter columns: {sorted(missing)}")
    if missing := required_labs - set(labs.columns):
        errors.append(f"Missing lab columns: {sorted(missing)}")
    if errors:
        return errors
    if encounters["EncounterID"].isna().any() or encounters["EncounterID"].duplicated().any():
        errors.append("EncounterID values must be present and unique.")
    if encounters[["PatientID", "EncounterDate"]].isna().any().any():
        errors.append("PatientID and EncounterDate cannot be missing.")
    for column in ["LengthOfStayDays", "TotalCostUSD"]:
        if encounters[column].isna().any() or (encounters[column] < 0).any():
            errors.append(f"{column} must be present and non-negative.")
    for column in ["ICUFlag", "SimulatedReadmissionFlag"]:
        if not set(encounters[column].dropna().unique()).issubset({0, 1}):
            errors.append(f"{column} must contain only 0/1 values.")
    if not set(labs["EncounterID"]).issubset(set(encounters["EncounterID"])):
        errors.append("Every lab EncounterID must exist in the encounter table.")
    if labs[["Result", "ReferenceLow", "ReferenceHigh"]].isna().any().any():
        errors.append("Lab results and example bounds cannot be missing.")
    if (labs["ReferenceLow"] > labs["ReferenceHigh"]).any():
        errors.append("Example lower bounds cannot exceed upper bounds.")
    expected = ((labs["Result"] < labs["ReferenceLow"]) | (labs["Result"] > labs["ReferenceHigh"])).astype(int)
    if not expected.equals(labs["AbnormalFlag"].astype(int)):
        errors.append("AbnormalFlag does not match the example bound comparison.")
    return errors


def build_summaries(encounters: pd.DataFrame, labs: pd.DataFrame) -> dict:
    errors = validate_data(encounters, labs)
    if errors:
        raise ValueError("; ".join(errors))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    e = encounters.copy()
    e["Month"] = e["EncounterDate"].dt.to_period("M").dt.to_timestamp()
    quality = {
        "dataset_type": "synthetic demonstration data",
        "encounter_rows": int(len(e)),
        "lab_rows": int(len(labs)),
        "duplicate_encounter_ids": int(e["EncounterID"].duplicated().sum()),
        "orphan_lab_rows": int((~labs["EncounterID"].isin(e["EncounterID"])).sum()),
        "validation_errors": errors,
    }
    (OUTPUT / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    e.groupby("Department", as_index=False).agg(
        Encounters=("EncounterID", "nunique"),
        MeanLOS=("LengthOfStayDays", "mean"),
        MedianLOS=("LengthOfStayDays", "median"),
        MeanCostUSD=("TotalCostUSD", "mean"),
        SimulatedFlagCount=("SimulatedReadmissionFlag", "sum"),
    ).to_csv(OUTPUT / "department_summary.csv", index=False, float_format="%.3f")
    e.groupby("Month", as_index=False).agg(
        Encounters=("EncounterID", "nunique"),
        MeanLOS=("LengthOfStayDays", "mean"),
        TotalCostUSD=("TotalCostUSD", "sum"),
    ).to_csv(OUTPUT / "monthly_summary.csv", index=False, float_format="%.3f")
    lab_summary = labs.groupby(["LabTest", "Unit"], as_index=False).agg(
        TestCount=("Result", "size"), MeanResult=("Result", "mean"),
        MedianResult=("Result", "median"), AbnormalResults=("AbnormalFlag", "sum"))
    lab_summary["AbnormalRatePct"] = 100 * lab_summary["AbnormalResults"] / lab_summary["TestCount"]
    lab_summary.to_csv(OUTPUT / "lab_summary.csv", index=False, float_format="%.3f")
    groups = [g["LengthOfStayDays"].to_numpy() for _, g in e.groupby("SimulatedReadmissionFlag")]
    tests = {}
    if len(groups) == 2 and all(len(g) > 1 for g in groups):
        test = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        tests["welch_t_test_on_simulated_flag"] = {
            "group_0_n": len(groups[0]), "group_1_n": len(groups[1]),
            "t_statistic": float(test.statistic), "p_value": float(test.pvalue),
            "note": "Exploratory only; generated labels and data do not support clinical inference."
        }
    (OUTPUT / "statistical_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    e["LengthOfStayDays"].plot(kind="hist", bins=15, ax=ax)
    ax.set(title="Length of stay (synthetic data)", xlabel="Days", ylabel="Encounters")
    fig.tight_layout()
    fig.savefig(FIGURES / "length_of_stay_distribution.png", dpi=140)
    plt.close(fig)
    summary = {
        "dataset_type": "synthetic demonstration data",
        "encounters": int(e["EncounterID"].nunique()),
        "unique_patients": int(e["PatientID"].nunique()),
        "mean_length_of_stay_days": round(float(e["LengthOfStayDays"].mean()), 2),
        "simulated_readmission_flag_pct": round(float(e["SimulatedReadmissionFlag"].mean() * 100), 2),
        "lab_results": int(len(labs)),
        "lab_abnormal_rate_pct": round(float(labs["AbnormalFlag"].mean() * 100), 2),
        "validation_errors": errors,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    encounters, labs = load_data()
    print(json.dumps(build_summaries(encounters, labs), indent=2))


if __name__ == "__main__":
    main()
