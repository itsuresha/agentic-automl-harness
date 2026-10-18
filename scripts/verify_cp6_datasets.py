"""CP-6 Phase A: verify and preregister external-validation datasets.

This script performs only OpenML metadata checks. It does not download full
datasets, run AIDE traces, or call the OpenAI API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


PRE_REGISTERED = [1590, 1457, 23517, 168350, 168757]

DEV_CORPUS_NAMES = {
    "satellite",
    "amazon-commerce-reviews",
    "christine",
    "credit-approval",
    "credit-g",
    "diabetes",
    "jasmine",
    "kr-vs-kp",
    "letter",
    "mfeat-factors",
    "mfeat-fourier",
    "mfeat-morphological",
    "mfeat-zernike",
    "pendigits",
    "tic-tac-toe",
    "vehicle",
}

OPENML = "https://www.openml.org/api/v1/json"
AMLB_CLASSIFICATION_SUITE_ID = 271


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def qualities_to_dict(raw: Any) -> dict[str, float]:
    if isinstance(raw, dict):
        raw = raw.get("quality", [])
    if isinstance(raw, dict):
        raw = [raw]
    out: dict[str, float] = {}
    for item in raw or []:
        name = item.get("name")
        value = item.get("value")
        if name is None or value in (None, "?"):
            continue
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def fetch_cc18_ids() -> set[int]:
    study = get_json(f"{OPENML}/study/99")["study"]
    data = study.get("data", {})
    ids = data.get("data_id", [])
    if isinstance(ids, str):
        ids = [ids]
    return {int(x) for x in ids}


def fetch_study_dataset_ids(study_id: int) -> set[int]:
    study = get_json(f"{OPENML}/study/{study_id}")["study"]
    data = study.get("data", {})
    ids = data.get("data_id", [])
    if isinstance(ids, str):
        ids = [ids]
    return {int(x) for x in ids}


def fetch_dataset(did: int) -> dict[str, Any]:
    desc = get_json(f"{OPENML}/data/{did}")["data_set_description"]
    q_payload = get_json(f"{OPENML}/data/qualities/{did}")["data_qualities"]
    q = qualities_to_dict(q_payload)
    return {
        "openml_id": did,
        "name": desc.get("name", ""),
        "status": desc.get("status", ""),
        "format": desc.get("format", ""),
        "default_target_attribute": desc.get("default_target_attribute", ""),
        "n_instances": int(q.get("NumberOfInstances", 0)),
        "n_features": int(q.get("NumberOfFeatures", 0)),
        "n_classes": int(q.get("NumberOfClasses", 0)),
        "n_missing_values": int(q.get("NumberOfMissingValues", 0)),
        "minority_class_size": int(q.get("MinorityClassSize", 0)),
        "majority_class_size": int(q.get("MajorityClassSize", 0)),
    }


def rejection_reason(
    meta: dict[str, Any],
    cc18_ids: set[int],
    exclude_ids: set[int],
    require_target: bool = True,
) -> str | None:
    did = int(meta["openml_id"])
    name = str(meta.get("name", "")).lower()
    if did in cc18_ids:
        return "CC-18 member"
    if did in exclude_ids:
        return "already selected/excluded"
    if name in DEV_CORPUS_NAMES:
        return "dev corpus member"
    if str(meta.get("status", "")).lower() != "active":
        return f"not active: {meta.get('status')}"
    if int(meta.get("n_instances", 0)) < 500:
        return f"too small: {meta.get('n_instances')} instances"
    if int(meta.get("n_features", 0)) < 5:
        return f"too few features: {meta.get('n_features')}"
    if int(meta.get("n_classes", 0)) < 2:
        return f"not classification: {meta.get('n_classes')} classes"
    if require_target and not meta.get("default_target_attribute"):
        return "missing default target attribute"
    return None


def verify_candidates(
    candidate_ids: list[int],
    cc18_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for did in candidate_ids:
        try:
            meta = fetch_dataset(did)
            reason = rejection_reason(meta, cc18_ids, set())
            if reason:
                rejected.append({**meta, "reason": reason})
                print(f"  REJECT: {did} ({meta['name']}) — {reason}")
            else:
                verified.append(meta)
                print(
                    f"  VERIFIED: {did} ({meta['name']}) — "
                    f"{meta['n_instances']} inst, {meta['n_features']} feat, "
                    f"{meta['n_classes']} classes"
                )
        except Exception as exc:
            rejected.append(
                {"openml_id": did, "name": "UNKNOWN", "reason": f"OpenML error: {exc}"}
            )
            print(f"  REJECT: {did} — OpenML error: {exc}")
    return verified, rejected


def list_active_datasets(limit: int) -> list[dict[str, Any]]:
    payload = get_json(f"{OPENML}/data/list/status/active/limit/{limit}", timeout=60)
    datasets = payload.get("data", {}).get("dataset", [])
    if isinstance(datasets, dict):
        datasets = [datasets]
    rows: list[dict[str, Any]] = []
    for row in datasets:
        q = qualities_to_dict(row.get("quality", []))
        rows.append(
            {
                "openml_id": int(row["did"]),
                "name": row.get("name", ""),
                "status": row.get("status", ""),
                "format": row.get("format", ""),
                "default_target_attribute": row.get("default_target_attribute", ""),
                "n_instances": int(q.get("NumberOfInstances", 0)),
                "n_features": int(q.get("NumberOfFeatures", 0)),
                "n_classes": int(q.get("NumberOfClasses", 0)),
                "n_missing_values": int(q.get("NumberOfMissingValues", 0)),
                "minority_class_size": int(q.get("MinorityClassSize", 0)),
                "majority_class_size": int(q.get("MajorityClassSize", 0)),
            }
        )
    return rows


def find_replacements(
    n_needed: int,
    cc18_ids: set[int],
    exclude_ids: set[int],
    search_limit: int,
    preferred_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    rows = list_active_datasets(search_limit)
    candidates: list[dict[str, Any]] = []
    for meta in rows:
        if rejection_reason(meta, cc18_ids, exclude_ids, require_target=False) is None:
            candidates.append(meta)

    # Prefer mid-size datasets to control CP-6 API/runtime while preserving
    # diversity. Alternate binary and multiclass where possible.
    candidates.sort(
        key=lambda x: (
            0 if preferred_ids and int(x["openml_id"]) in preferred_ids else 1,
            abs(int(x["n_instances"]) - 5000),
            int(x["n_features"]),
            int(x["openml_id"]),
        )
    )
    binary = [c for c in candidates if int(c["n_classes"]) == 2]
    multi = [c for c in candidates if int(c["n_classes"]) > 2]

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    selected_names: set[str] = set()
    bi = mi = 0
    while len(selected) < n_needed and (bi < len(binary) or mi < len(multi)):
        candidate = None
        if len(selected) % 2 == 0 and bi < len(binary):
            candidate = binary[bi]
            bi += 1
        elif mi < len(multi):
            candidate = multi[mi]
            mi += 1
        elif bi < len(binary):
            candidate = binary[bi]
            bi += 1
        if candidate is None:
            continue
        try:
            full_meta = fetch_dataset(int(candidate["openml_id"]))
        except Exception:
            continue
        if int(full_meta["openml_id"]) in selected_ids:
            continue
        name_key = str(full_meta.get("name", "")).lower()
        if name_key in selected_names:
            continue
        if rejection_reason(full_meta, cc18_ids, exclude_ids | selected_ids) is None:
            selected.append(full_meta)
            selected_ids.add(int(full_meta["openml_id"]))
            selected_names.add(name_key)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-n", type=int, default=6)
    parser.add_argument("--search-limit", type=int, default=5000)
    args = parser.parse_args()

    base = Path(args.output_dir)
    cp6_dir = base / "cp6_results"
    cp6_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CP-6 Phase A: Dataset Verification")
    print("=" * 60)
    print("No AIDE traces and no OpenAI API calls will be made.")

    cc18_ids = fetch_cc18_ids()
    print(f"  CC-18 exclusion set: {len(cc18_ids)} dataset IDs")
    amlb_ids = fetch_study_dataset_ids(AMLB_CLASSIFICATION_SUITE_ID)
    print(
        f"  AMLB classification preferred set: {len(amlb_ids)} dataset IDs "
        f"(suite {AMLB_CLASSIFICATION_SUITE_ID})"
    )

    verified, rejected = verify_candidates(PRE_REGISTERED, cc18_ids)
    exclude_ids = set(PRE_REGISTERED) | {int(d["openml_id"]) for d in verified}

    n_needed = max(0, args.target_n - len(verified))
    replacements = find_replacements(
        n_needed, cc18_ids, exclude_ids, args.search_limit, preferred_ids=amlb_ids
    )
    final_datasets = verified + replacements

    for meta in replacements:
        print(
            f"  REPLACEMENT: {meta['openml_id']} ({meta['name']}) — "
            f"{meta['n_instances']} inst, {meta['n_features']} feat, "
            f"{meta['n_classes']} classes"
        )

    result = {
        "phase": "CP-6 Phase A dataset verification",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pre_registered": PRE_REGISTERED,
        "cc18_suite_id": 99,
        "cc18_dataset_count": len(cc18_ids),
        "replacement_preferred_suite": {
            "openml_study_id": AMLB_CLASSIFICATION_SUITE_ID,
            "alias": "amlb-classification-all",
            "dataset_count": len(amlb_ids),
        },
        "verified": verified,
        "rejected": rejected,
        "replacements": replacements,
        "final_datasets": final_datasets,
        "n_final": len(final_datasets),
        "target_n": args.target_n,
        "notes": [
            "Dataset roster is written before any AIDE traces.",
            "No OpenAI API call is made by this script.",
            "Final roster must be frozen before Phase B.",
        ],
    }
    out = cp6_dir / "dataset_verification.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved {out}")
    print(f"Final roster size: {len(final_datasets)}")


if __name__ == "__main__":
    main()
