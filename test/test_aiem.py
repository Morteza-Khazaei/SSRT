"""CLI test harness comparing AIEM against the NMM3D LUT.

This script provides the same numerical checks as the `test_aiem.ipynb`
notebook, but in a lightweight, automation-friendly Python module. It loads the
40° incidence NMM3D backscatter look-up table, evaluates AIEM for each surface
configuration, and reports goodness-of-fit metrics (RMSE, MAE, bias, Pearson
correlation) for the HH, VV, and HV channels.

Usage
-----
Run from the repository root:

    PYTHONPATH=src MPLCONFIGDIR=/tmp python3 test/test_aiem.py

Optional command-line arguments let you filter ratios, choose a different
incident angle, or point to alternative LUTs. The script exits with a non-zero
status if the LUT cannot be located or no valid comparisons remain after
filtering.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ssrt.surface.aiem import AIEM
from ssrt.utils.util import toLambda

# Default configuration mirrors the notebook
_DEFAULT_LUT = Path("data/NMM3D_LUT_NRCS_40degree.dat")
_DEFAULT_FREQ_GHZ = 5.405
_DEFAULT_INC_DEG = 40.0
_DEFAULT_PHI_DEG = 180.0
_DEFAULT_SURFACE_TYPE = 2  # exponential correlation to match LUT


@dataclass
class Metrics:
    """Container for descriptive statistics between AIEM and NMM3D."""

    count: int
    rmse: float
    mae: float
    bias: float
    corr: float

    def format_row(self, label: str) -> str:  # pragma: no cover - trivial formatting
        return (
            f"{label:<6s} n={self.count:3d}  "
            f"RMSE={self.rmse:5.2f} dB  "
            f"MAE={self.mae:5.2f} dB  "
            f"Bias={self.bias:+5.2f} dB  "
            f"Corr={self.corr:6.3f}"
        )


def _load_lut(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"LUT not found at {path}")
    return np.loadtxt(path)


def _select_angle(table: np.ndarray, incidence_deg: float) -> np.ndarray:
    mask = np.isclose(table[:, 0], incidence_deg)
    return table[mask]


def _isclose(value: float, targets: Sequence[float], tol: float = 1e-6) -> bool:
    return any(math.isclose(value, target, rel_tol=0.0, abs_tol=tol) for target in targets)


def _calc_metrics(model: Iterable[float], reference: Iterable[float]) -> Metrics:
    model_arr = np.asarray(list(model), dtype=float)
    ref_arr = np.asarray(list(reference), dtype=float)
    mask = np.isfinite(model_arr) & np.isfinite(ref_arr)
    if not np.any(mask):
        return Metrics(count=0, rmse=float("nan"), mae=float("nan"), bias=float("nan"), corr=float("nan"))

    diff = model_arr[mask] - ref_arr[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(ref_arr[mask], model_arr[mask])[0, 1]) if mask.sum() > 1 else float("nan")
    return Metrics(count=int(mask.sum()), rmse=rmse, mae=mae, bias=bias, corr=corr)


@dataclass
class ComparisonResult:
    overall: Dict[str, Metrics]
    by_ratio: Dict[float, Dict[str, Metrics]]


def _run_comparison(
    rows: np.ndarray,
    frequency_ghz: float,
    incidence_deg: float,
    phi_deg: float,
    surface_type: int,
    ratios: Sequence[float] | None,
    include_multiple: bool,
) -> ComparisonResult:
    lam = toLambda(frequency_ghz)
    k = 2.0 * math.pi / lam

    overall_model: Dict[str, List[float]] = {pol: [] for pol in ("hh", "vv", "hv")}
    overall_reference: Dict[str, List[float]] = {pol: [] for pol in ("hh", "vv", "hv")}
    grouped_model: Dict[float, Dict[str, List[float]]] = {}
    grouped_reference: Dict[float, Dict[str, List[float]]] = {}

    for row in rows:
        (
            _theta,
            ratio,
            eps_r,
            eps_i,
            rms_norm,
            vv_ref,
            hh_ref,
            hv_ref,
        ) = row

        ratio = float(ratio)
        if ratios and not _isclose(ratio, ratios):
            continue

        sigma = rms_norm * lam
        corr_len = ratio * sigma
        hh_db, vv_db, hv_db, _ = AIEM(
            theta_i=incidence_deg,
            theta_s=incidence_deg,
            phi_s=phi_deg,
            err=float(eps_r),
            eri=float(eps_i),
            itype=surface_type,
            addMultiple=include_multiple,
            frequency_ghz=frequency_ghz,
            k0=k,
            sigma=sigma,
            corr_len=corr_len,
        )

        overall_model["hh"].append(hh_db)
        overall_model["vv"].append(vv_db)
        overall_model["hv"].append(hv_db)

        overall_reference["hh"].append(hh_ref)
        overall_reference["vv"].append(vv_ref)
        overall_reference["hv"].append(hv_ref)

        if ratio not in grouped_model:
            grouped_model[ratio] = {pol: [] for pol in ("hh", "vv", "hv")}
            grouped_reference[ratio] = {pol: [] for pol in ("hh", "vv", "hv")}

        grouped_model[ratio]["hh"].append(hh_db)
        grouped_model[ratio]["vv"].append(vv_db)
        grouped_model[ratio]["hv"].append(hv_db)

        grouped_reference[ratio]["hh"].append(hh_ref)
        grouped_reference[ratio]["vv"].append(vv_ref)
        grouped_reference[ratio]["hv"].append(hv_ref)

    overall_metrics = {
        pol: _calc_metrics(overall_model[pol], overall_reference[pol]) for pol in overall_model
    }

    by_ratio: Dict[float, Dict[str, Metrics]] = {}
    for ratio_value, model_dict in grouped_model.items():
        by_ratio[ratio_value] = {
            pol: _calc_metrics(model_dict[pol], grouped_reference[ratio_value][pol])
            for pol in model_dict
        }

    return ComparisonResult(overall=overall_metrics, by_ratio=by_ratio)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare AIEM predictions with NMM3D LUT data")
    parser.add_argument(
        "--lut",
        type=Path,
        default=_DEFAULT_LUT,
        help=f"Path to LUT file (default: {_DEFAULT_LUT})",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=_DEFAULT_FREQ_GHZ,
        help=f"Radar frequency in GHz (default: {_DEFAULT_FREQ_GHZ})",
    )
    parser.add_argument(
        "--incidence",
        type=float,
        default=_DEFAULT_INC_DEG,
        help=f"Incidence angle in degrees (default: {_DEFAULT_INC_DEG})",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=_DEFAULT_PHI_DEG,
        help=f"Scattering azimuth in degrees (default: {_DEFAULT_PHI_DEG})",
    )
    parser.add_argument(
        "--surface-type",
        type=int,
        default=_DEFAULT_SURFACE_TYPE,
        choices=(1, 2, 3),
        help="AIEM surface correlation type (1=Gaussian, 2=Exponential, 3=1.5 power)",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        help="Optional list of correlation length ratios (ℓ/σ) to evaluate",
    )
    parser.add_argument(
        "--per-ratio",
        action="store_true",
        help="Print metrics broken down by ratio in addition to overall statistics",
    )
    parser.add_argument(
        "--add-multiple",
        action="store_true",
        help="Include the multiple scattering contribution in AIEM evaluations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        table = _load_lut(args.lut)
    except FileNotFoundError as exc:  # pragma: no cover - CLI guard
        parser.error(str(exc))

    rows = _select_angle(table, args.incidence)
    if rows.size == 0:
        parser.error(f"No LUT entries found for incidence angle {args.incidence}")

    result = _run_comparison(
        rows=rows,
        frequency_ghz=args.frequency,
        incidence_deg=args.incidence,
        phi_deg=args.phi,
        surface_type=args.surface_type,
        ratios=args.ratios,
        include_multiple=args.add_multiple,
    )

    overall = result.overall
    print("AIEM vs NMM3D (overall metrics)")
    for pol in ("vv", "hh", "hv"):
        metrics = overall[pol]
        print(metrics.format_row(pol.upper()))

    if args.per_ratio and result.by_ratio:
        print("\nBy-ratio metrics")
        for ratio in sorted(result.by_ratio):
            print(f"\nℓ/σ = {ratio:g}")
            for pol in ("vv", "hh", "hv"):
                print(result.by_ratio[ratio][pol].format_row(pol.upper()))

    total_valid = sum(metrics.count for metrics in overall.values())
    if total_valid == 0:
        parser.error("No finite comparisons available; check LUT values or filters")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
