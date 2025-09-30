"""Multiple-scattering contribution for the AIEM model.

This module is a direct translation of the MATLAB `aiem_ms_gpu` routine and the
supporting integral expressions listed in `reference/AIEM_Complete_Equations_FULL_v2.md`.
It provides a CPU implementation that evaluates the kc and complementary blocks
for the requested polarisations and returns linear-power contributions that can
be added to the single-scatter result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Geometry:
    theta_i: float
    theta_s: float
    phi_i: float
    phi_s: float


@dataclass(frozen=True)
class Physics:
    k: float
    er: complex


@dataclass(frozen=True)
class Surface:
    type: str
    ks: float
    kl: float
    k: float
    sigma: float
    corr_len: float


@dataclass(frozen=True)
class Quadrature:
    U: np.ndarray
    V: np.ndarray
    wu: np.ndarray
    wv: np.ndarray
    Nmax: int


def compute_multiple_scattering(
    theta_i: float,
    theta_s: float,
    phi_i: float,
    phi_s: float,
    er: complex,
    k0: float,
    sigma: float,
    corr_len: float,
    surface_label: str,
    polarisations: Sequence[str] = ("hh", "vv", "hv", "vh"),
    n_points: int = 129,
    nmax: int = 8,
) -> Dict[str, float]:
    """Evaluate multiple-scattering contributions for the requested polarisations."""

    geom = Geometry(theta_i=theta_i, theta_s=theta_s, phi_i=phi_i, phi_s=phi_s)
    phys = Physics(k=k0, er=er)
    ks = k0 * sigma
    kl = k0 * corr_len
    surf = Surface(
        type=surface_label.lower(),
        ks=ks,
        kl=kl,
        k=phys.k,
        sigma=sigma,
        corr_len=corr_len,
    )
    quad = _build_quadrature(surf, n_points=n_points, nmax=nmax)

    integrator = _MultipleScatteringIntegrator(geom, phys, surf, quad, polarisations)
    return integrator.compute()


def _build_quadrature(surf: Surface, n_points: int, nmax: int) -> Quadrature:
    corr = max(surf.corr_len, 1e-6)
    umax = 5.0 / corr
    grid = np.linspace(-umax, umax, n_points, dtype=float)
    U, V = np.meshgrid(grid, grid, indexing="ij")

    if len(grid) > 1:
        du = grid[1] - grid[0]
    else:
        du = 0.0
    dv = du
    wu = np.full_like(grid, du)
    wv = np.full_like(grid, dv)
    if len(grid) > 1:
        wu[0] = wu[-1] = 0.5 * du
        wv[0] = wv[-1] = 0.5 * dv

    return Quadrature(U=U, V=V, wu=wu, wv=wv, Nmax=nmax)


class _MultipleScatteringIntegrator:
    """Internal helper translating the MATLAB multiple-scattering routine."""

    def __init__(
        self,
        geom: Geometry,
        phys: Physics,
        surf: Surface,
        quad: Quadrature,
        polarisations: Sequence[str],
    ) -> None:
        self.geom = geom
        self.phys = phys
        self.surf = surf
        self.quad = quad
        self.pols = tuple(p.lower() for p in polarisations)
        self._wn = _make_Wn_provider(surf)
        self._geom_data = _prepare_geometry_terms(geom, phys)

    def compute(self) -> Dict[str, float]:
        U = self.quad.U
        V = self.quad.V
        wu = self.quad.wu
        wv = self.quad.wv
        k = self.phys.k
        er = self.phys.er

        # Preserve the evanescent behaviour of the spectral components by allowing
        # complex propagation constants.  Clamping the radicand to zero collapses
        # the imaginary branch and explodes the Fresnel prefactors for cross-pol
        # channels near the critical angle.
        q1 = np.sqrt((k**2 - (U**2 + V**2)) + 0.0j)
        q2 = np.sqrt(er * k**2 - (U**2 + V**2))

        qmin = 1e-6
        rad = (np.abs(q1) > qmin) | (np.abs(q2) > qmin)
        W2D = np.outer(wu, wv)

        results: Dict[str, float] = {pol: 0.0 for pol in self.pols}
        hv_value: float | None = None

        for pol in self.pols:
            if pol in {"hv", "vh"} and hv_value is not None:
                results[pol] = hv_value
                continue

            integrand_kc, integrand_c = _assemble_integrands(
                U,
                V,
                q1,
                q2,
                k,
                er,
                self._geom_data,
                self.surf,
                self._wn,
                self.quad.Nmax,
                pol,
            )

            if pol in {"hh", "vv"}:
                Ikc = np.real(integrand_kc) * rad
                Ic = np.real(integrand_c) * rad
                val = (
                    (k**2 / (8.0 * np.pi)) * np.sum(Ikc * W2D)
                    + (k**2 / (64.0 * np.pi)) * np.sum(Ic * W2D)
                )
                results[pol] = max(float(np.real(val)), 0.0)

            elif pol in {"hv", "vh"}:
                Ikc = np.real(integrand_kc) * rad  # ADD THIS
                Ic = np.real(integrand_c) * rad
                val = (
                    (k**2 / (8.0 * np.pi)) * np.sum(Ikc * W2D) +   # kc term
                    (k**2 / (64.0 * np.pi)) * np.sum(Ic * W2D)     # c term
                )
                hv_value = max(float(np.real(val)), 0.0)
                results["hv"] = hv_value
                results["vh"] = hv_value

        return results


def _prepare_geometry_terms(geom: Geometry, phys: Physics) -> Dict[str, float]:
    sin_theta_i = math.sin(geom.theta_i)
    cos_theta_i = math.cos(geom.theta_i)
    sin_theta_s = math.sin(geom.theta_s)
    cos_theta_s = math.cos(geom.theta_s)
    sin_phi_i = math.sin(geom.phi_i)
    cos_phi_i = math.cos(geom.phi_i)
    sin_phi_s = math.sin(geom.phi_s)
    cos_phi_s = math.cos(geom.phi_s)

    k = phys.k
    return {
        "theta_i": geom.theta_i,
        "theta_s": geom.theta_s,
        "phi_i": geom.phi_i,
        "phi_s": geom.phi_s,
        "sin_theta_s": sin_theta_s,
        "cos_theta_s": cos_theta_s,
        "sin_phi_s": sin_phi_s,
        "cos_phi_s": cos_phi_s,
        "z_x": sin_theta_i * cos_phi_i,
        "z_y": sin_theta_i * sin_phi_i,
        "zp_x": sin_theta_s * cos_phi_s,
        "zp_y": sin_theta_s * sin_phi_s,
        "kx": k * sin_theta_i * cos_phi_i,
        "ky": k * sin_theta_i * sin_phi_i,
        "kz": k * cos_theta_i,
        "ksx": k * sin_theta_s * cos_phi_s,
        "ksy": k * sin_theta_s * sin_phi_s,
        "ksz": k * cos_theta_s,
    }


def _make_Wn_provider(surf: Surface) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:
    sigma2 = (surf.ks / surf.k) ** 2
    corr_len = surf.corr_len

    if surf.type == "gauss":
        def provider(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
            order = max(n, 1)
            factor = corr_len**2 / order
            exp_arg = -(corr_len**2 / (4.0 * order)) * (u**2 + v**2)
            return (factor / (4.0 * np.pi)) * np.exp(exp_arg)
    else:
        def provider(u: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
            order = max(n, 1)
            rho = corr_len / order
            denom = 1.0 + (rho**2) * (u**2 + v**2)
            return (rho**2) * denom ** (-1.5)

    return provider


def _assemble_integrands(
    U: np.ndarray,
    V: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    k: float,
    er: complex,
    geom_data: Dict[str, float],
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
    pol: str,
) -> tuple[np.ndarray, np.ndarray]:
    propagators = _build_propagators(U, V, q1, q2, k, er, geom_data, pol)

    K1 = _build_gkc1(U, V, geom_data, q1, surf, wn_provider, Nmax)
    K2 = _build_gkc2(U, V, geom_data, q1, surf, wn_provider, Nmax)
    K3 = _build_gkc3(U, V, geom_data, q1, surf, wn_provider, Nmax)

    C1 = _build_gc_block1(U, V, geom_data, q1, q1, surf, wn_provider, Nmax)
    C2 = _build_gc_block2(U, V, geom_data, q1, q1, surf, wn_provider, Nmax)

    Int_kc = (
        np.conjugate(propagators["Fp"]) * propagators["Fp"] * K1
        + np.conjugate(propagators["Fm"]) * propagators["Fm"] * K2
        + np.conjugate(propagators["Gp"]) * propagators["Gp"] * K3
    )

    Int_c = np.zeros_like(U, dtype=np.complex128)
    P = propagators
    Int_c += (P["Fp"] * np.conjugate(P["Fp"])) * C1["gc1"]
    Int_c += (P["Fp"] * np.conjugate(P["Fm"])) * C1["gc2"]
    Int_c += (P["Fm"] * np.conjugate(P["Fp"])) * C1["gc3"]
    Int_c += (P["Fm"] * np.conjugate(P["Fm"])) * C1["gc4"]
    Int_c += (P["Gp"] * np.conjugate(P["Gp"])) * C1["gc5"]
    Int_c += (P["Gp"] * np.conjugate(P["Gm"])) * C1["gc6"]
    Int_c += (P["Gm"] * np.conjugate(P["Gp"])) * C1["gc7"]
    Int_c += (P["Gm"] * np.conjugate(P["Gm"])) * C1["gc8"]

    Int_c += (P["Fp"] * np.conjugate(P["Gp"])) * C2["gc9"]
    Int_c += (P["Fp"] * np.conjugate(P["Gm"])) * C2["gc10"]
    Int_c += (P["Fm"] * np.conjugate(P["Gp"])) * C2["gc11"]
    Int_c += (P["Fm"] * np.conjugate(P["Gm"])) * C2["gc12"]
    Int_c += (P["Gp"] * np.conjugate(P["Fp"])) * C2["gc13"]
    Int_c += (P["Gm"] * np.conjugate(P["Fp"])) * C2["gc14"]

    return Int_kc, Int_c

def _build_propagators(
    U: np.ndarray,
    V: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    k: float,
    er: complex,
    geom_data: Dict[str, float],
    pol: str,
) -> Dict[str, np.ndarray]:
    cos_phi, sin_phi = _spectral_angles(U, V)
    C_air = _compute_C_coeffs(q1, geom_data, cos_phi, sin_phi, U, V)
    C_soil = _compute_C_coeffs(q2, geom_data, cos_phi, sin_phi, U, V)
    B_air = _compute_B_coeffs(q1, geom_data, cos_phi, sin_phi, U, V)
    B_soil = _compute_B_coeffs(q2, geom_data, cos_phi, sin_phi, U, V)

    Rh, Rv = _fresnel_coeffs(er, q1, q2)
    R = 0.5 * (Rv - Rh)
    mu_r = 1.0
    u_r = 1.0
    inv_q1 = _safe_inverse(q1)
    inv_q2 = _safe_inverse(q2)

    pol = pol.lower()
    if pol not in {"hh", "vv", "hv", "vh"}:
        raise ValueError(f"Unsupported polarisation '{pol}' for multiple scattering")

    def _components(coeff_air, coeff_soil, refl_primary, refl_secondary, cross_refl):
        if pol in {"hh", "vv"}:
            coeff = coeff_air
            coeff_t = coeff_soil
        else:
            coeff = coeff_air
            coeff_t = coeff_soil
        return coeff, coeff_t

    if pol == "vv":
        coeff, coeff_t = C_air, C_soil
        Fp_plus = (
            -(1 - Rv) * (1 + Rv) * inv_q1 * coeff["C1"]
            + (1 - Rv) * (1 - Rv) * inv_q1 * coeff["C2"]
            + (1 - Rv) * (1 + Rv) * inv_q1 * coeff["C3"]
            + (1 + Rv) * (1 - Rv) * inv_q1 * coeff["C4"]
            + (1 + Rv) * (1 + Rv) * inv_q1 * coeff["C5"]
            + (1 + Rv) * (1 - Rv) * inv_q1 * coeff["C6"]
        )
        Gp_plus = (
            (1 + Rv) * (1 + Rv) * u_r * inv_q2 * coeff_t["C1"]
            - (1 + Rv) * (1 - Rv) * inv_q2 * coeff_t["C2"]
            - (1 + Rv * er) * (1 + Rv) * inv_q2 * coeff_t["C3"]
            - (1 - Rv) * er * (1 - Rv) * inv_q2 * coeff_t["C4"]
            - (1 - Rv) * (1 + Rv) * inv_q2 * coeff_t["C5"]
            - (1 - Rv) * (1 - Rv) * u_r * inv_q2 * coeff_t["C6"]
        )
    elif pol == "hh":
        coeff, coeff_t = C_air, C_soil
        Fp_plus = (
            (1 - Rh) * (1 + Rh) * inv_q1 * coeff["C1"]
            - (1 - Rh) * (1 - Rh) * inv_q1 * coeff["C2"]
            - (1 - Rh) * (1 + Rh) * inv_q1 * coeff["C3"]
            - (1 + Rh) * (1 - Rh) * inv_q1 * coeff["C4"]
            - (1 + Rh) * (1 + Rh) * inv_q1 * coeff["C5"]
            - (1 + Rh) * (1 - Rh) * inv_q1 * coeff["C6"]
        )
        Gp_plus = (
            -(1 + Rh) * (1 + Rh) * er * inv_q2 * coeff_t["C1"]
            + (1 + Rh) * (1 - Rh) * inv_q2 * coeff_t["C2"]
            + (1 + Rh) * (1 + Rh) * u_r * inv_q2 * coeff_t["C3"]
            + (1 - Rh) * (1 - Rh) * u_r * inv_q2 * coeff_t["C4"]
            + (1 - Rh) * (1 + Rh) * inv_q2 * coeff_t["C5"]
            + (1 - Rh) * (1 - Rh) * er * inv_q2 * coeff_t["C6"]
        )
    else:  # cross-pol
        coeff, coeff_t = B_air, B_soil
        Fp_plus = (
            (1 - R) * (1 + R) * inv_q1 * coeff["B1"]
            - (1 - R) * (1 - R) * inv_q1 * coeff["B2"]
            - (1 - R) * (1 + R) * inv_q1 * coeff["B3"]
            + (1 + R) * (1 - R) * inv_q1 * coeff["B4"]
            + (1 + R) * (1 + R) * inv_q1 * coeff["B5"]
            + (1 + R) * (1 - R) * inv_q1 * coeff["B6"]
        )
        Gp_plus = (
            -(1 + R) * (1 + R) * mu_r * inv_q2 * coeff_t["B1"]
            + (1 + R) * (1 - R) * inv_q2 * coeff_t["B2"]
            + (1 + R * er) * (1 + R) * inv_q2 * coeff_t["B3"]
            - (1 - R) * er * (1 - R) * inv_q2 * coeff_t["B4"]
            - (1 - R) * (1 + R) * inv_q2 * coeff_t["B5"]
            - (1 - R) * (1 - R) * mu_r * inv_q2 * coeff_t["B6"]
        )

    Fm_plus, Gm_plus = _compute_downward_propagators(
        U,
        V,
        q1,
        q2,
        k,
        er,
        geom_data,
        pol,
        cos_phi,
        sin_phi,
    )

    return {"Fp": Fp_plus, "Fm": Fm_plus, "Gp": Gp_plus, "Gm": Gm_plus}


def _compute_downward_propagators(
    U: np.ndarray,
    V: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    k: float,
    er: complex,
    geom_data: Dict[str, float],
    pol: str,
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Rh, Rv = _fresnel_coeffs(er, -q1, -q2)
    R = 0.5 * (Rv - Rh)
    mu_r = 1.0
    u_r = 1.0
    inv_q1 = _safe_inverse(-q1)
    inv_q2 = _safe_inverse(-q2)

    C_air = _compute_C_coeffs(-q1, geom_data, cos_phi, sin_phi, U, V)
    C_soil = _compute_C_coeffs(-q2, geom_data, cos_phi, sin_phi, U, V)
    B_air = _compute_B_coeffs(-q1, geom_data, cos_phi, sin_phi, U, V)
    B_soil = _compute_B_coeffs(-q2, geom_data, cos_phi, sin_phi, U, V)

    if pol == "vv":
        coeff = C_air
        coeff_t = C_soil
        Fm = (
            -(1 - Rv) * (1 + Rv) * inv_q1 * coeff["C1"]
            + (1 - Rv) * (1 - Rv) * inv_q1 * coeff["C2"]
            + (1 - Rv) * (1 + Rv) * inv_q1 * coeff["C3"]
            + (1 + Rv) * (1 - Rv) * inv_q1 * coeff["C4"]
            + (1 + Rv) * (1 + Rv) * inv_q1 * coeff["C5"]
            + (1 + Rv) * (1 - Rv) * inv_q1 * coeff["C6"]
        )
        Gm = (
            (1 + Rv) * (1 + Rv) * u_r * inv_q2 * coeff_t["C1"]
            - (1 + Rv) * (1 - Rv) * inv_q2 * coeff_t["C2"]
            - (1 + Rv * er) * (1 + Rv) * inv_q2 * coeff_t["C3"]
            - (1 - Rv) * er * (1 - Rv) * inv_q2 * coeff_t["C4"]
            - (1 - Rv) * (1 + Rv) * inv_q2 * coeff_t["C5"]
            - (1 - Rv) * (1 - Rv) * u_r * inv_q2 * coeff_t["C6"]
        )
    elif pol == "hh":
        coeff = C_air
        coeff_t = C_soil
        Fm = (
            (1 - Rh) * (1 + Rh) * inv_q1 * coeff["C1"]
            - (1 - Rh) * (1 - Rh) * inv_q1 * coeff["C2"]
            - (1 - Rh) * (1 + Rh) * inv_q1 * coeff["C3"]
            - (1 + Rh) * (1 - Rh) * inv_q1 * coeff["C4"]
            - (1 + Rh) * (1 + Rh) * inv_q1 * coeff["C5"]
            - (1 + Rh) * (1 - Rh) * inv_q1 * coeff["C6"]
        )
        Gm = (
            -(1 + Rh) * (1 + Rh) * er * inv_q2 * coeff_t["C1"]
            + (1 + Rh) * (1 - Rh) * inv_q2 * coeff_t["C2"]
            + (1 + Rh) * (1 + Rh) * inv_q2 * coeff_t["C3"]
            + (1 - Rh) * (1 - Rh) * inv_q2 * coeff_t["C4"]
            + (1 - Rh) * (1 + Rh) * inv_q2 * coeff_t["C5"]
            + (1 - Rh) * (1 - Rh) * er * inv_q2 * coeff_t["C6"]
        )
    else:
        coeff = B_air
        coeff_t = B_soil
        Fm = (
            (1 - R) * (1 + R) * inv_q1 * coeff["B1"]
            - (1 - R) * (1 - R) * inv_q1 * coeff["B2"]
            - (1 - R) * (1 + R) * inv_q1 * coeff["B3"]
            + (1 + R) * (1 - R) * inv_q1 * coeff["B4"]
            + (1 + R) * (1 + R) * inv_q1 * coeff["B5"]
            + (1 + R) * (1 - R) * inv_q1 * coeff["B6"]
        )
        Gm = (
            -(1 + R) * (1 + R) * mu_r * inv_q2 * coeff_t["B1"]
            + (1 + R) * (1 - R) * inv_q2 * coeff_t["B2"]
            + (1 + R * er) * (1 + R) * inv_q2 * coeff_t["B3"]
            - (1 - R) * er * (1 - R) * inv_q2 * coeff_t["B4"]
            - (1 - R) * (1 + R) * inv_q2 * coeff_t["B5"]
            - (1 - R) * (1 - R) * mu_r * inv_q2 * coeff_t["B6"]
        )

    return Fm, Gm

def _spectral_angles(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rho = np.hypot(U, V)
    cos_phi = np.ones_like(U)
    sin_phi = np.zeros_like(V)
    mask = rho > 0
    cos_phi[mask] = U[mask] / rho[mask]
    sin_phi[mask] = V[mask] / rho[mask]
    return cos_phi, sin_phi


def _compute_C_coeffs(
    q: np.ndarray,
    geom: Dict[str, float],
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> Dict[str, np.ndarray]:
    cos_phi_s = geom["cos_phi_s"]
    sin_phi_s = geom["sin_phi_s"]
    cos_theta = geom["cos_theta_s"]
    sin_theta = geom["sin_theta_s"]
    z_x = geom["z_x"]
    z_y = geom["z_y"]
    zp_x = geom["zp_x"]
    zp_y = geom["zp_y"]

    q = np.asarray(q)
    cos_phi = np.asarray(cos_phi)
    sin_phi = np.asarray(sin_phi)
    U = np.asarray(U)
    V = np.asarray(V)

    C1 = -cos_phi_s * (-cos_phi - z_x * zp_x * cos_phi - z_x * zp_y * sin_phi) + sin_phi_s * (
        sin_phi + zp_x * z_y * cos_phi + z_y * zp_y * sin_phi
    )

    C2 = -cos_phi_s * (
        -q * cos_theta * cos_phi
        - U * z_x * cos_theta
        - V * zp_y * cos_theta * cos_phi
        - q * zp_x * sin_theta
        - U * z_x * zp_x * sin_theta
        - V * z_x * zp_y * sin_theta
        - V * z_x * cos_theta * sin_phi
        + V * zp_x * cos_theta * sin_phi
    ) + sin_phi_s * (
        U * z_y * cos_theta * cos_phi
        - U * zp_y * cos_theta
        + U * zp_x * z_y * sin_theta
        + q * zp_y * sin_theta
        + V * z_y * zp_y * sin_theta
        + q * cos_theta * sin_phi
        + U * zp_x * cos_theta * sin_phi
        + V * z_y * cos_theta * sin_phi
    )

    C3 = cos_phi_s * (
        U * zp_x * cos_theta * cos_phi
        - q * z_x * zp_x * cos_theta * cos_phi
        - U * sin_theta
        + q * z_x * sin_theta
        + U * zp_y * cos_theta * sin_phi
        - q * z_x * zp_y * cos_theta * sin_phi
    ) + sin_phi_s * (
        V * zp_x * cos_theta * cos_phi
        - q * zp_x * z_y * cos_theta * cos_phi
        - V * sin_theta
        + q * z_y * sin_theta
        + V * zp_y * cos_theta * sin_phi
        - q * z_y * zp_y * cos_theta * sin_phi
    )

    C4 = sin_theta * (
        -z_x * cos_theta * cos_phi
        - z_x * zp_x * sin_theta
        - z_y * zp_y * sin_theta
        - z_y * cos_theta * sin_phi
    ) - cos_theta * cos_phi_s * (
        -cos_theta * cos_phi
        - z_y * zp_y * cos_theta * cos_phi
        - zp_x * sin_theta
        + zp_x * z_y * cos_theta * sin_phi
    ) - cos_theta * sin_phi_s * (
        z_x * zp_y * cos_theta * cos_phi
        - zp_y * sin_theta
        - cos_theta * sin_phi
        - z_x * zp_x * cos_theta * sin_phi
    )

    C5 = sin_theta * (
        q * z_x * cos_phi
        + U * z_x * zp_x * cos_phi
        + V * zp_x * z_y * cos_phi
        + q * z_y * sin_phi
        + U * z_x * zp_y * sin_phi
        + V * z_y * zp_y * sin_phi
    ) - cos_theta * cos_phi_s * (
        q * cos_phi
        + U * zp_x * cos_phi
        + V * z_y * cos_phi
        - U * z_y * sin_phi
        + U * zp_y * sin_phi
    ) - cos_theta * sin_phi_s * (
        -V * z_x * cos_phi
        + V * zp_x * cos_phi
        + q * sin_phi
        + U * z_x * sin_phi
        + V * zp_y * sin_phi
    )

    C6 = sin_theta * (
        V * z_x * zp_y * cos_phi
        - U * z_y * zp_y * cos_phi
        - V * z_x * zp_x * sin_phi
        + U * zp_x * z_y * sin_phi
    ) + cos_theta * cos_phi_s * (
        -V * zp_y * cos_phi
        + q * z_y * zp_y * cos_phi
        + V * zp_x * sin_phi
        - q * zp_x * z_y * sin_phi
    ) + cos_theta * sin_phi_s * (
        U * zp_y * cos_phi
        - q * z_x * zp_y * cos_phi
        - U * zp_x * sin_phi
        + q * z_x * zp_x * sin_phi
    )

    return {"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6": C6}


def _compute_B_coeffs(
    q: np.ndarray,
    geom: Dict[str, float],
    cos_phi: np.ndarray,
    sin_phi: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> Dict[str, np.ndarray]:
    cos_phi_s = geom["cos_phi_s"]
    sin_phi_s = geom["sin_phi_s"]
    cos_theta = geom["cos_theta_s"]
    sin_theta = geom["sin_theta_s"]
    z_x = geom["z_x"]
    z_y = geom["z_y"]
    zp_x = geom["zp_x"]
    zp_y = geom["zp_y"]

    q = np.asarray(q)
    cos_phi = np.asarray(cos_phi)
    sin_phi = np.asarray(sin_phi)
    U = np.asarray(U)
    V = np.asarray(V)

    B1 = sin_theta * (-z_y * cos_phi + z_x * sin_phi) - cos_theta * cos_phi_s * (
        zp_x * z_y * cos_phi + sin_phi + z_y * zp_y * sin_phi
    ) - cos_theta * sin_phi_s * (
        -cos_phi - z_x * zp_x * cos_phi - z_x * zp_y * sin_phi
    )

    B2 = sin_theta * (
        -q * z_y * cos_theta * cos_phi
        - U * z_x * zp_y * cos_theta * cos_phi
        - V * z_y * zp_y * cos_theta * cos_phi
        - q * zp_x * z_y * sin_theta
        + q * z_x * zp_y * sin_theta
        + q * z_x * cos_theta * sin_phi
        + U * z_x * zp_x * cos_theta * sin_phi
        + V * zp_x * z_y * cos_theta * sin_phi
    ) - cos_theta * cos_phi_s * (
        U * z_y * cos_theta * cos_phi
        - U * zp_y * cos_theta * cos_phi
        + U * zp_x * z_y * sin_theta
        + q * zp_y * sin_theta
        + V * z_y * zp_y * sin_theta
        + q * cos_theta * sin_phi
        + U * zp_x * cos_theta * sin_phi
        + V * z_y * cos_theta * sin_phi
    ) - cos_theta * sin_phi_s * (
        -q * cos_theta * cos_phi
        - U * z_x * cos_theta * cos_phi
        - V * zp_y * cos_theta * cos_phi
        - q * zp_x * sin_theta
        - U * z_x * zp_x * sin_theta
        - V * z_x * zp_y * sin_theta
        - V * z_x * cos_theta * sin_phi
        + V * zp_x * cos_theta * sin_phi
    )

    B3 = sin_theta * (
        V * z_x * zp_x * cos_theta * cos_phi
        - U * z_y * zp_x * cos_theta * cos_phi
        - V * z_x * sin_theta
        + U * z_y * sin_theta
        + V * z_x * zp_y * cos_theta * sin_phi
        - U * z_y * zp_y * cos_theta * sin_phi
    ) + cos_theta * cos_phi_s * (
        -V * zp_x * cos_theta * cos_phi
        + q * zp_x * z_y * cos_theta * cos_phi
        + V * sin_theta
        - q * z_y * sin_theta
        - V * zp_y * cos_theta * sin_phi
        + q * z_y * zp_y * cos_theta * sin_phi
    ) + cos_theta * sin_phi_s * (
        U * zp_x * cos_theta * cos_phi
        - q * z_x * zp_x * cos_theta * cos_phi
        - U * sin_theta
        + q * z_x * sin_theta
        + U * zp_y * cos_theta * sin_phi
        - q * z_x * zp_y * cos_theta * sin_phi
    )

    B4 = -cos_phi_s * (
        z_x * zp_y * cos_theta * cos_phi
        - zp_y * sin_theta
        - cos_theta * sin_phi
        - z_x * zp_x * cos_theta * sin_phi
    ) + sin_phi_s * (
        -cos_theta * cos_phi
        - z_y * zp_y * cos_theta * cos_phi
        - zp_x * sin_theta
        + zp_x * z_y * cos_theta * sin_phi
    )

    B5 = -cos_phi_s * (
        -V * z_x * cos_phi
        + V * zp_x * cos_phi
        + q * sin_phi
        + U * z_x * sin_phi
        - V * zp_y * sin_phi
    ) + sin_phi_s * (
        q * cos_phi
        + U * zp_x * cos_phi
        + V * z_y * cos_phi
        - U * z_y * sin_phi
        + U * zp_y * sin_phi
    )

    B6 = cos_phi_s * (
        U * zp_y * cos_phi
        - q * z_x * zp_y * cos_phi
        - U * zp_x * sin_phi
        + q * z_x * zp_x * sin_phi
    ) + sin_phi_s * (
        V * zp_y * cos_phi
        - q * z_y * zp_y * cos_phi
        - V * zp_x * sin_phi
        + q * zp_x * z_y * sin_phi
    )

    return {"B1": B1, "B2": B2, "B3": B3, "B4": B4, "B5": B5, "B6": B6}

_SAFE_EPS = 1e-8


def _safe_div(a: np.ndarray | complex | float, b: np.ndarray | complex | float) -> np.ndarray:
    b = np.asarray(b) + 0.0
    mask = np.abs(b) < _SAFE_EPS
    b = b + mask * (_SAFE_EPS + 1j * _SAFE_EPS)
    return np.asarray(a) / b


def _safe_inverse(x: np.ndarray) -> np.ndarray:
    return _safe_div(1.0, x)


def _fresnel_coeffs(er: complex, q1: np.ndarray, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Rh = _safe_div(q1 - q2, q1 + q2)
    Rv = _safe_div(er * q1 - q2, er * q1 + q2)
    return Rh, Rv

def _series_sum(
    coeff: np.ndarray | float,
    arg_x: np.ndarray | float,
    arg_y: np.ndarray | float,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> np.ndarray:
    coeff_arr = np.asarray(coeff, dtype=np.complex128)
    if np.isscalar(arg_x):
        arg_x_arr = np.full_like(coeff_arr, float(arg_x))
    else:
        arg_x_arr = np.asarray(arg_x, dtype=float)
    if np.isscalar(arg_y):
        arg_y_arr = np.full_like(coeff_arr, float(arg_y))
    else:
        arg_y_arr = np.asarray(arg_y, dtype=float)

    result = np.zeros_like(coeff_arr, dtype=np.complex128)
    for n in range(1, Nmax + 1):
        Wn = wn_provider(arg_x_arr, arg_y_arr, n)
        result += (np.power(coeff_arr, n) / math.factorial(n)) * Wn
    return result

def _build_gkc1(
    U: np.ndarray,
    V: np.ndarray,
    geom: Dict[str, float],
    q: np.ndarray,
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> np.ndarray:
    sigma2 = (surf.ks / surf.k) ** 2
    kz = geom["kz"]
    ksz = geom["ksz"]
    kx = geom["kx"]
    ky = geom["ky"]
    ksx = geom["ksx"]
    ksy = geom["ksy"]

    expo = np.exp(
        -sigma2 * (ksz**2 + kz**2 + ksz * kz + q**2 - q * ksz + q * kz)
    )
    a_m = sigma2 * (kz + q) * (ksz + kz)
    a_n = sigma2 * (ksz - q) * (ksz + kz)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    return expo * sum_m * sum_n


def _build_gkc2(
    U: np.ndarray,
    V: np.ndarray,
    geom: Dict[str, float],
    q: np.ndarray,
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> np.ndarray:
    sigma2 = (surf.ks / surf.k) ** 2
    kz = geom["kz"]
    ksz = geom["ksz"]
    kx = geom["kx"]
    ky = geom["ky"]
    ksx = geom["ksx"]
    ksy = geom["ksy"]

    expo = np.exp(
        -sigma2 * (ksz**2 + kz**2 + ksz * kz + q**2 - q * ksz + q * kz)
    )
    a_m = sigma2 * (kz + q) * (ksz + kz)
    a_n = -sigma2 * (ksz - q) * (kz + q)
    sum_m = _series_sum(a_m, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    return expo * sum_m * sum_n


def _build_gkc3(
    U: np.ndarray,
    V: np.ndarray,
    geom: Dict[str, float],
    q: np.ndarray,
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> np.ndarray:
    sigma2 = (surf.ks / surf.k) ** 2
    kz = geom["kz"]
    ksz = geom["ksz"]
    kx = geom["kx"]
    ky = geom["ky"]
    ksx = geom["ksx"]
    ksy = geom["ksy"]

    expo = np.exp(
        -sigma2 * (ksz**2 + kz**2 + ksz * kz + q**2 - q * ksz + q * kz)
    )
    a_m = -sigma2 * (ksz - q) * (kz + q)
    a_n = sigma2 * (ksz - q) * (ksz + kz)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    return expo * sum_m * sum_n

def _build_gc_block1(
    U: np.ndarray,
    V: np.ndarray,
    geom: Dict[str, float],
    q: np.ndarray,
    qp: np.ndarray,
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> Dict[str, np.ndarray]:
    sigma2 = (surf.ks / surf.k) ** 2
    kz = geom["kz"]
    ksz = geom["ksz"]
    kx = geom["kx"]
    ky = geom["ky"]
    ksx = geom["ksx"]
    ksy = geom["ksy"]

    def expo(q_, qp_):
        return np.exp(-sigma2 * (ksz**2 + kz**2 + q_**2 + qp_**2 - (ksz - kz) * (q_ + qp_)))

    # gc1 (A4)
    a_n = sigma2 * (ksz - q) * (ksz - qp)
    a_m = sigma2 * (kz + q) * (kz + qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc1 = expo(q, qp) * sum_n * sum_m

    # gc2 (A5)
    a_n = sigma2 * (ksz - q) * (kz + qp)
    a_m = sigma2 * (kz + q) * (ksz - qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc2 = expo(q, qp) * sum_n * sum_m

    # gc3 (A6)
    a_n = sigma2 * (ksz - q) * (kz + qp)
    a_m = sigma2 * (kz + q) * (kz + qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc3 = expo(q, qp) * sum_n * sum_m

    # gc4 (A7)
    a_n = sigma2 * (ksz - q) * (kz + qp)
    a_m = -sigma2 * (ksz - q) * (kz + q)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc4 = expo(q, qp) * sum_n * sum_m

    # gc5 (A8)
    a_n = sigma2 * (kz + q) * (kz + qp)
    a_m = -sigma2 * (ksz - q) * (kz + q)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, ksx + U, ksy + V, wn_provider, Nmax)
    gc5 = expo(q, qp) * sum_n * sum_m

    # gc6 (A9)
    a_n = sigma2 * (ksz - q) * (ksz - qp)
    a_m = sigma2 * (kz + q) * (ksz - qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc6 = expo(q, qp) * sum_n * sum_m

    # gc7 (A10)
    a_n = sigma2 * (ksz - q) * (ksz - qp)
    a_m = -sigma2 * (ksz - q) * (kz + q)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc7 = expo(q, qp) * sum_n * sum_m

    # gc8 (A11)
    a_n = sigma2 * (kz + q) * (ksz - qp)
    a_m = -sigma2 * (ksz - q) * (kz + q)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, ksx + U, ksy + V, wn_provider, Nmax)
    gc8 = expo(q, qp) * sum_n * sum_m

    return {
        "gc1": gc1,
        "gc2": gc2,
        "gc3": gc3,
        "gc4": gc4,
        "gc5": gc5,
        "gc6": gc6,
        "gc7": gc7,
        "gc8": gc8,
    }


def _build_gc_block2(
    U: np.ndarray,
    V: np.ndarray,
    geom: Dict[str, float],
    q: np.ndarray,
    qp: np.ndarray,
    surf: Surface,
    wn_provider: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    Nmax: int,
) -> Dict[str, np.ndarray]:
    sigma2 = (surf.ks / surf.k) ** 2
    kz = geom["kz"]
    ksz = geom["ksz"]
    kx = geom["kx"]
    ky = geom["ky"]
    ksx = geom["ksx"]
    ksy = geom["ksy"]

    def expo(q_, qp_):
        return np.exp(-sigma2 * (ksz**2 + kz**2 + q_**2 + qp_**2 - (ksz - kz) * (q_ + qp_)))

    # gc9 (A12)
    a_n = sigma2 * (kz + q) * (ksz - qp)
    a_m = sigma2 * (kz + q) * (kz + qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc9 = expo(q, qp) * sum_n * sum_m

    # gc10 (A13)
    a_n = sigma2 * (kz + q) * (ksz - qp)
    a_m = -sigma2 * (ksz - qp) * (kz + qp)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc10 = expo(q, qp) * sum_n * sum_m

    # gc11 (A14)
    a_n = sigma2 * (kz + q) * (kz + qp)
    a_m = -sigma2 * (ksz - qp) * (kz + qp)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, ksx + U, ksy + V, wn_provider, Nmax)
    gc11 = expo(q, qp) * sum_n * sum_m

    # gc12 (A15)
    a_n = sigma2 * (ksz - q) * (ksz - qp)
    a_m = sigma2 * (ksz - q) * (kz + qp)
    sum_n = _series_sum(a_n, ksx + U, ksy + V, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc12 = expo(q, qp) * sum_n * sum_m

    # gc13 (A16)
    a_n = sigma2 * (ksz - q) * (ksz - qp)
    a_m = -sigma2 * (ksz - qp) * (kz + qp)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, kx + U, ky + V, wn_provider, Nmax)
    gc13 = expo(q, qp) * sum_n * sum_m

    # gc14 (A17)
    a_n = sigma2 * (ksz - q) * (kz + qp)
    a_m = -sigma2 * (ksz - qp) * (kz + qp)
    sum_n = _series_sum(a_n, kx - ksx, ky - ksy, wn_provider, Nmax)
    sum_m = _series_sum(a_m, ksx + U, ksy + V, wn_provider, Nmax)
    gc14 = expo(q, qp) * sum_n * sum_m

    return {
        "gc9": gc9,
        "gc10": gc10,
        "gc11": gc11,
        "gc12": gc12,
        "gc13": gc13,
        "gc14": gc14,
    }
