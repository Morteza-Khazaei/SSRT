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
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

from ssrt.surface.aiem import AIEMModel, AIEMParameters
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


@dataclass
class CrossPolComponentEntry:
    ratio: float
    kc: float
    c: float
    multiple: float
    model_total: float
    reference: float


@dataclass
class CrossPolComponentStats:
    count: int
    kc: float
    c: float
    multiple: float
    model_total: float
    reference: float

    def format_row(self, label: str) -> str:  # pragma: no cover - trivial formatting
        if self.count == 0:
            return f"{label:<8s} n=  0  (no data)"

        def _format_power(value: float) -> str:
            eps = 1e-12
            magnitude = abs(value)
            if magnitude <= 0.0:
                return "-inf dB"
            db = 10.0 * math.log10(magnitude + eps)
            sign = "+" if value >= 0.0 else "-"
            return f"{sign}{abs(db):5.2f} dB"

        ratio = (self.multiple / self.reference) if self.reference != 0.0 else float("nan")

        return (
            f"{label:<8s} n={self.count:3d}  "
            f"kc={_format_power(self.kc):>11s}  "
            f"c={_format_power(self.c):>11s}  "
            f"mult={_format_power(self.multiple):>11s}  "
            f"AIEM={_format_power(self.model_total):>11s}  "
            f"REF={_format_power(self.reference):>11s}  "
            f"mult/ref={ratio:6.3f}"
        )


@dataclass
class CrossPolDiagnostics:
    overall: CrossPolComponentStats
    by_ratio: Dict[float, CrossPolComponentStats]


@dataclass(frozen=True)
class KirchhoffSanityCase:
    """Inputs used to validate the Kirchhoff field construction."""

    ratio: float
    sigma: float
    corr_len: float
    eps_r: float
    eps_i: float


@dataclass
class KirchhoffSanityMetrics:
    """Summary of the physical sanity checks for a single configuration."""

    ratio: float
    permittivity: complex
    sigma: float
    corr_len: float
    hh_rel_error: float
    vv_rel_error: float
    cross_pol_magnitude: float
    reciprocity_error: float
    max_orthogonality_error: float

    def format_row(self) -> str:  # pragma: no cover - presentation helper
        return (
            f"ℓ/σ={self.ratio:>4g}  |ε_r|={abs(self.permittivity):5.2f}  "
            f"|ΔHH|={self.hh_rel_error:6.3f}  |ΔVV|={self.vv_rel_error:6.3f}  "
            f"|HV|={self.cross_pol_magnitude:9.2e}  "
            f"|HV-VH|={self.reciprocity_error:9.2e}  "
            f"max⊥ err={self.max_orthogonality_error:9.2e}"
        )

    def within_limits(
        self,
        co_pol_tol: float = 0.6,
        cross_pol_tol: float = 1e-10,
        reciprocity_tol: float = 1e-12,
        orthogonality_tol: float = 1e-12,
    ) -> bool:
        return (
            self.hh_rel_error <= co_pol_tol
            and self.vv_rel_error <= co_pol_tol
            and self.cross_pol_magnitude <= cross_pol_tol
            and self.reciprocity_error <= reciprocity_tol
            and self.max_orthogonality_error <= orthogonality_tol
        )


class _VectorizedKirchhoff:
    """Numerically stable Kirchhoff Approximation helper (VKA)."""

    def __init__(
        self,
        theta_i: float,
        theta_s: float,
        phi_i: float,
        phi_s: float,
        Rv: complex,
        Rh: complex,
    ) -> None:
        self.theta_i = theta_i
        self.theta_s = theta_s
        self.phi_i = phi_i
        self.phi_s = phi_s
        self.Rv = Rv
        self.Rh = Rh

        self._setup_vectors()
        self.zx, self.zy = self._stationary_phase_slopes()
        self.D1 = float(np.sqrt(1.0 + self.zx**2 + self.zy**2))
        self.D2 = float(
            np.sqrt(
                self.zy**2
                + (np.sin(self.theta_i) - self.zx * np.cos(self.theta_i)) ** 2
            )
        )
        self._setup_surface_vectors()

    def _setup_vectors(self) -> None:
        self.k_i = np.array(
            [
                np.sin(self.theta_i) * np.cos(self.phi_i),
                np.sin(self.theta_i) * np.sin(self.phi_i),
                -np.cos(self.theta_i),
            ]
        )
        self.k_s = np.array(
            [
                np.sin(self.theta_s) * np.cos(self.phi_s),
                np.sin(self.theta_s) * np.sin(self.phi_s),
                np.cos(self.theta_s),
            ]
        )
        self.h_i = np.array([-np.sin(self.phi_i), np.cos(self.phi_i), 0.0])
        self.v_i = np.array(
            [
                np.cos(self.theta_i) * np.cos(self.phi_i),
                np.cos(self.theta_i) * np.sin(self.phi_i),
                np.sin(self.theta_i),
            ]
        )
        self.h_s = -np.array([-np.sin(self.phi_s), np.cos(self.phi_s), 0.0])
        self.v_s = -np.array(
            [
                np.cos(self.theta_s) * np.cos(self.phi_s),
                np.cos(self.theta_s) * np.sin(self.phi_s),
                -np.sin(self.theta_s),
            ]
        )

    def _stationary_phase_slopes(self) -> tuple[float, float]:
        kx_i, ky_i, kz_i = self.k_i
        kx_s, ky_s, kz_s = self.k_s
        denom = kz_s - kz_i
        if np.isclose(denom, 0.0):  # pragma: no cover - degenerate geometry guard
            raise ValueError("Invalid scattering geometry for stationary phase evaluation")
        zx = -(kx_s - kx_i) / denom
        zy = -(ky_s - ky_i) / denom
        return float(zx), float(zy)

    def _setup_surface_vectors(self) -> None:
        self.n = np.array([-self.zx, -self.zy, 1.0]) / self.D1
        k_cross_n = np.cross(self.k_i, self.n)
        denom = np.linalg.norm(k_cross_n)
        if np.isclose(denom, 0.0):  # fallback for grazing/backscatter degeneracy
            reference = np.array([0.0, 0.0, 1.0])
            if np.isclose(abs(np.dot(reference, self.k_i)), 1.0):
                reference = np.array([1.0, 0.0, 0.0])
            k_cross_n = np.cross(self.k_i, reference)
            denom = np.linalg.norm(k_cross_n)
        if np.isclose(denom, 0.0):  # pragma: no cover - extreme degeneracy guard
            raise ValueError("Unable to construct orthogonal surface basis")
        self.t = k_cross_n / denom
        self.d = np.cross(self.k_i, self.t)

    def _dot_products(self) -> Dict[str, float]:
        res: Dict[str, float] = {}
        res["vs_dot_n_cross_vi"] = float(np.dot(self.v_s, np.cross(self.n, self.v_i)))
        res["vs_dot_n_cross_hi"] = float(np.dot(self.v_s, np.cross(self.n, self.h_i)))
        res["hs_dot_n_cross_vi"] = float(np.dot(self.h_s, np.cross(self.n, self.v_i)))
        res["hs_dot_n_cross_hi"] = float(np.dot(self.h_s, np.cross(self.n, self.h_i)))
        res["vi_dot_t"] = float(np.dot(self.v_i, self.t))
        res["hi_dot_d"] = float(np.dot(self.h_i, self.d))
        res["vs_dot_t"] = float(np.dot(self.v_s, self.t))
        res["vs_dot_d"] = float(np.dot(self.v_s, self.d))
        res["hs_dot_t"] = float(np.dot(self.h_s, self.t))
        res["hs_dot_d"] = float(np.dot(self.h_s, self.d))
        res["n_dot_ki"] = float(np.dot(self.n, self.k_i))
        res["n_dot_d"] = float(np.dot(self.n, self.d))
        res["hs_dot_ki"] = float(np.dot(self.h_s, self.k_i))
        res["vs_dot_ki"] = float(np.dot(self.v_s, self.k_i))
        return res

    def field_coefficients(self) -> Dict[str, complex]:
        dots = self._dot_products()
        Rv = self.Rv
        Rh = self.Rh
        D1 = self.D1

        co_pol = (
            dots["hs_dot_d"] * dots["n_dot_ki"]
            - dots["n_dot_d"] * dots["hs_dot_ki"]
            - dots["vs_dot_t"] * dots["n_dot_ki"]
        )
        cross_pol = (
            dots["hs_dot_t"] * dots["n_dot_ki"]
            - dots["n_dot_d"] * dots["vs_dot_ki"]
            + dots["vs_dot_d"] * dots["n_dot_ki"]
        )

        fvv = -(
            (1 - Rv) * dots["hs_dot_n_cross_vi"]
            + (1 + Rv) * dots["vs_dot_n_cross_hi"]
        ) * D1 - (Rh + Rv) * dots["vi_dot_t"] * co_pol * D1

        fvh = (
            (1 - Rh) * dots["vs_dot_n_cross_vi"]
            - (1 + Rh) * dots["hs_dot_n_cross_hi"]
        ) * D1 - (Rh + Rv) * dots["hi_dot_d"] * cross_pol * D1

        fhv = -(
            (1 - Rv) * dots["vs_dot_n_cross_vi"]
            - (1 + Rv) * dots["hs_dot_n_cross_hi"]
        ) * D1 - (Rh + Rv) * dots["vi_dot_t"] * cross_pol * D1

        fhh = (
            (1 + Rh) * dots["vs_dot_n_cross_hi"]
            + (1 - Rh) * dots["hs_dot_n_cross_vi"]
        ) * D1 - (Rh + Rv) * dots["hi_dot_d"] * co_pol * D1

        return {"hh": fhh, "vv": fvv, "hv": fhv, "vh": fvh}

    def orthogonality_errors(self) -> Dict[str, float]:
        return {
            "hi·ki": float(abs(np.dot(self.h_i, self.k_i))),
            "vi·ki": float(abs(np.dot(self.v_i, self.k_i))),
            "hi·vi": float(abs(np.dot(self.h_i, self.v_i))),
            "hs·ks": float(abs(np.dot(self.h_s, self.k_s))),
            "vs·ks": float(abs(np.dot(self.v_s, self.k_s))),
            "hs·vs": float(abs(np.dot(self.h_s, self.v_s))),
        }


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


def _aggregate_component_entries(
    entries: Iterable[CrossPolComponentEntry],
) -> CrossPolComponentStats:
    entry_list = list(entries)
    if not entry_list:
        return CrossPolComponentStats(
            count=0,
            kc=0.0,
            c=0.0,
            multiple=0.0,
            model_total=0.0,
            reference=0.0,
        )

    count = len(entry_list)

    def _mean(attr: str) -> float:
        return float(sum(getattr(entry, attr) for entry in entry_list) / count)

    return CrossPolComponentStats(
        count=count,
        kc=_mean("kc"),
        c=_mean("c"),
        multiple=_mean("multiple"),
        model_total=_mean("model_total"),
        reference=_mean("reference"),
    )


@dataclass
class ComparisonResult:
    overall: Dict[str, Metrics]
    by_ratio: Dict[float, Dict[str, Metrics]]
    cross_pol: CrossPolDiagnostics | None = None


def _run_comparison(
    rows: np.ndarray,
    frequency_ghz: float,
    incidence_deg: float,
    phi_deg: float,
    surface_type: int,
    ratios: Sequence[float] | None,
    include_multiple: bool,
    diagnose_cross_pol: bool,
) -> ComparisonResult:
    lam = toLambda(frequency_ghz)
    k = 2.0 * math.pi / lam

    overall_model: Dict[str, List[float]] = {pol: [] for pol in ("hh", "vv", "hv")}
    overall_reference: Dict[str, List[float]] = {pol: [] for pol in ("hh", "vv", "hv")}
    grouped_model: Dict[float, Dict[str, List[float]]] = {}
    grouped_reference: Dict[float, Dict[str, List[float]]] = {}

    component_entries: List[CrossPolComponentEntry] = []
    component_entries_by_ratio: Dict[float, List[CrossPolComponentEntry]] = {}

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
        params = AIEMParameters(
            theta_i=incidence_deg,
            theta_s=incidence_deg,
            phi_s=phi_deg,
            err=float(eps_r),
            eri=float(eps_i),
            surface_type=surface_type,
            add_multiple=include_multiple,
            output_unit="linear",
            frequency_ghz=frequency_ghz,
            k0=k,
            sigma=sigma,
            corr_len=corr_len,
        )
        model = AIEMModel(params)
        totals_linear = model.sigma0_total()

        hh_lin = float(totals_linear["hh"])
        vv_lin = float(totals_linear["vv"])
        hv_lin = float(totals_linear["hv"])

        hh_db = AIEMModel._to_db(hh_lin)
        vv_db = AIEMModel._to_db(vv_lin)
        hv_db = AIEMModel._to_db(hv_lin)

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

        if include_multiple and diagnose_cross_pol:
            components = model.multiple_scattering_components()
            hv_components = components.get("hv") or components.get("vh")
            if hv_components is None:
                hv_components = {"kc": 0.0, "c": 0.0}
            hv_multiple = float(hv_components["kc"] + hv_components["c"])
            hv_ref_linear = float(10 ** (hv_ref / 10.0))
            entry = CrossPolComponentEntry(
                ratio=ratio,
                kc=float(hv_components["kc"]),
                c=float(hv_components["c"]),
                multiple=hv_multiple,
                model_total=hv_lin,
                reference=hv_ref_linear,
            )
            component_entries.append(entry)
            component_entries_by_ratio.setdefault(ratio, []).append(entry)

    overall_metrics = {
        pol: _calc_metrics(overall_model[pol], overall_reference[pol]) for pol in overall_model
    }

    by_ratio: Dict[float, Dict[str, Metrics]] = {}
    for ratio_value, model_dict in grouped_model.items():
        by_ratio[ratio_value] = {
            pol: _calc_metrics(model_dict[pol], grouped_reference[ratio_value][pol])
            for pol in model_dict
        }

    diagnostics = None
    if include_multiple and diagnose_cross_pol:
        diagnostics = CrossPolDiagnostics(
            overall=_aggregate_component_entries(component_entries),
            by_ratio={
                ratio_value: _aggregate_component_entries(entries)
                for ratio_value, entries in component_entries_by_ratio.items()
            },
        )

    return ComparisonResult(overall=overall_metrics, by_ratio=by_ratio, cross_pol=diagnostics)


def _prepare_kirchhoff_cases(
    rows: np.ndarray, frequency_ghz: float, max_cases: int = 4
) -> List[KirchhoffSanityCase]:
    """Select representative LUT entries for Kirchhoff sanity checks."""

    lam = toLambda(frequency_ghz)
    cases: List[KirchhoffSanityCase] = []
    seen_ratios: Set[float] = set()

    for row in rows:
        (_, ratio, eps_r, eps_i, rms_norm, *_rest) = row
        ratio = float(ratio)
        if ratio in seen_ratios:
            continue
        seen_ratios.add(ratio)

        sigma = float(rms_norm) * lam
        corr_len = ratio * sigma
        cases.append(
            KirchhoffSanityCase(
                ratio=ratio,
                sigma=sigma,
                corr_len=corr_len,
                eps_r=float(eps_r),
                eps_i=float(eps_i),
            )
        )
        if len(cases) >= max_cases:
            break

    return cases


def _run_kirchhoff_sanity_checks(
    rows: np.ndarray,
    frequency_ghz: float,
    incidence_deg: float,
    phi_deg: float,
    surface_type: int,
) -> List[KirchhoffSanityMetrics]:
    """Evaluate Kirchhoff geometry/field sanity checks for representative cases."""

    cases = _prepare_kirchhoff_cases(rows, frequency_ghz)
    if not cases:
        return []

    lam = toLambda(frequency_ghz)
    k0 = 2.0 * math.pi / lam
    phi_i = 0.0  # Backscatter configuration

    metrics_list: List[KirchhoffSanityMetrics] = []

    for case in cases:
        params = AIEMParameters(
            theta_i=incidence_deg,
            theta_s=incidence_deg,
            phi_s=phi_deg,
            err=case.eps_r,
            eri=case.eps_i,
            surface_type=surface_type,
            add_multiple=False,
            output_unit="linear",
            frequency_ghz=frequency_ghz,
            k0=k0,
            sigma=case.sigma,
            corr_len=case.corr_len,
        )
        model = AIEMModel(params)
        wave = model.eq3_wavevectors()
        kirchhoff = model.eq2_kirchhoff_field(wave["spectra"], wave["iterm"])
        fields = kirchhoff["fields"]

        rv = kirchhoff["rv_incident"]
        rh = kirchhoff["rh_incident"]

        vka = _VectorizedKirchhoff(
            theta_i=np.deg2rad(incidence_deg),
            theta_s=np.deg2rad(incidence_deg),
            phi_i=phi_i,
            phi_s=np.deg2rad(phi_deg),
            Rv=rv,
            Rh=rh,
        )

        vka_fields = vka.field_coefficients()

        def _rel_mag_error(model_val: complex, ref_val: complex) -> float:
            denom = max(abs(ref_val), 1e-12)
            return float(abs(abs(model_val) - abs(ref_val)) / denom)

        hh_err = min(
            _rel_mag_error(fields["hh"], vka_fields["hh"]),
            _rel_mag_error(fields["hh"], vka_fields["vv"]),
        )
        vv_err = min(
            _rel_mag_error(fields["vv"], vka_fields["vv"]),
            _rel_mag_error(fields["vv"], vka_fields["hh"]),
        )
        hv_mag = float(max(abs(fields["hv"]), abs(fields["vh"])))
        reciprocity = float(abs(fields["hv"] - fields["vh"]))
        ortho_errors = vka.orthogonality_errors()
        max_ortho = float(max(ortho_errors.values()))

        metrics_list.append(
            KirchhoffSanityMetrics(
                ratio=case.ratio,
                permittivity=complex(case.eps_r, case.eps_i),
                sigma=case.sigma,
                corr_len=case.corr_len,
                hh_rel_error=hh_err,
                vv_rel_error=vv_err,
                cross_pol_magnitude=hv_mag,
                reciprocity_error=reciprocity,
                max_orthogonality_error=max_ortho,
            )
        )

    return metrics_list


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
    parser.add_argument(
        "--diagnose-cross-pol",
        action="store_true",
        help="Report average multiple-scattering kc/c contributions for the HV channel",
    )
    parser.add_argument(
        "--sanity-kirchhoff",
        action="store_true",
        help=(
            "Run Kirchhoff-Approximation sanity checks (orthogonality, reciprocity, and "
            "co-pol agreement) alongside the LUT comparison"
        ),
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
        diagnose_cross_pol=args.diagnose_cross_pol,
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

    if args.diagnose_cross_pol:
        if not args.add_multiple:
            print(
                "\nCross-pol diagnostics requested but multiple scattering is disabled; "
                "enable --add-multiple to inspect kc/c contributions."
            )
        elif result.cross_pol is not None:
            print("\nCross-pol multiple-scattering diagnostics (mean linear power)")
            print(result.cross_pol.overall.format_row("Overall"))
            if result.cross_pol.by_ratio:
                for ratio in sorted(result.cross_pol.by_ratio):
                    label = f"ℓ/σ={ratio:g}"
                    print(result.cross_pol.by_ratio[ratio].format_row(label))

    run_sanity = args.sanity_kirchhoff or args.diagnose_cross_pol
    if run_sanity:
        sanity_metrics = _run_kirchhoff_sanity_checks(
            rows=rows,
            frequency_ghz=args.frequency,
            incidence_deg=args.incidence,
            phi_deg=args.phi,
            surface_type=args.surface_type,
        )
        print("\nKirchhoff physical sanity checks")
        for metrics in sanity_metrics:
            print(metrics.format_row())
        if not all(metric.within_limits() for metric in sanity_metrics):
            raise AssertionError("Kirchhoff sanity checks exceeded physical tolerances")

    total_valid = sum(metrics.count for metrics in overall.values())
    if total_valid == 0:
        parser.error("No finite comparisons available; check LUT values or filters")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
