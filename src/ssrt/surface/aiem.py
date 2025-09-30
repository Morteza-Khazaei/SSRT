"""Advanced Integral Equation Model (AIEM) implementation.

This module translates the MATLAB reference implementation of AIEM into
Python. The port keeps the mathematical structure intact while adopting an
object-oriented organisation for clarity and maintainability. The translation
also follows Pythonic style guidelines (PEP 8, descriptive naming, limited
mutability) and annotates the major sections of the algorithm for readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math

import numpy as np
from scipy.special import gammaln, kv

from .aiem_ms import compute_multiple_scattering


@dataclass(frozen=True)
class AIEMParameters:
    """Input arguments required by the AIEM solver."""

    theta_i: float
    theta_s: float
    phi_s: float
    err: float
    eri: float
    surface_type: int
    add_multiple: bool = False
    output_unit: str = "dB"
    frequency_ghz: float | None = None
    k0: float | None = None
    sigma: float | None = None
    corr_len: float | None = None

    def __post_init__(self) -> None:
        if self.sigma is None:
            raise ValueError("sigma (surface RMS height in metres) must be provided")
        if self.corr_len is None:
            raise ValueError("corr_len (surface correlation length in metres) must be provided")


@dataclass(frozen=True)
class AIEMResult:
    """Backscatter for the four polarisation channels."""

    hh: float
    vv: float
    hv: float
    vh: float


@dataclass
class SingleScatteringBreakdown:
    """Container for Eq. (8) single-scattering contributions."""

    sigma_k: Dict[str, float]
    sigma_kc_plus_sigma_c: Dict[str, float]
    total: Dict[str, float]




class AIEMModel:
    """Object-oriented wrapper around the AIEM forward model."""

    def __init__(self, params: AIEMParameters) -> None:
        self.params = params

        # Medium properties and observation geometry (radians)
        self.er: complex = complex(params.err, params.eri)
        self.ur: float = 1.0
        self.theta_i: float = np.deg2rad(params.theta_i)
        self.theta_s: float = np.deg2rad(params.theta_s)
        self.phi_s: float = np.deg2rad(params.phi_s)
        self.phi_i: float = 0.0  # kept at zero for backscatter scenarios

        # Trigonometric helpers reused throughout the solver
        self.si = np.sin(self.theta_i)
        self.cs = np.cos(self.theta_i)
        self.sis = np.sin(self.theta_s)
        self.css = np.cos(self.theta_s)
        self.sfs = np.sin(self.phi_s)
        self.csfs = np.cos(self.phi_s)

        self.si2 = self.si**2
        self.sis2 = self.sis**2
        self.cs2 = self.cs**2
        self.css2 = self.css**2

        # Physical wavenumber and dimensional roughness descriptors
        self.k0 = self._compute_wavenumber(params)

        self.sigma = params.sigma
        self.corr_len = params.corr_len

        # Roughness parameters (dimensionless, built from k0)
        self.ks = self.k0 * self.sigma
        self.ks2 = self.ks**2
        self.kl = self.k0 * self.corr_len
        self.kl2 = self.kl**2

        self._single_cache: SingleScatteringBreakdown | None = None
        self._multiple_cache: Dict[str, float] | None = None
        self._multiple_components_cache: Dict[str, Dict[str, float]] | None = None
        self._eq10_cache: Dict[str, np.ndarray] | None = None
        self._wavevector_cache: Dict[str, np.ndarray | int | float] | None = None
        self._kirchhoff_cache: Dict[str, Dict[str, float | complex]] | None = None
        self._eq4_cache: Dict[str, Dict[str, tuple[complex, complex]]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self) -> AIEMResult:
        """Execute the AIEM model and return four-polarisation NRCS in dB."""

        totals = self.sigma0_total()
        formatted = self._format_output(totals)
        return AIEMResult(
            hh=formatted["hh"],
            vv=formatted["vv"],
            hv=formatted["hv"],
            vh=formatted["vh"],
        )

    # ------------------------------------------------------------------
    # Public API (Eqs. 5–11)
    # ------------------------------------------------------------------
    def sigma0_total(self) -> Dict[str, float]:
        """Eq. (7): Compute total NRCS as σ⁰₍qp₎ = σ₍qp₎^(s) + σ₍qp₎^(m)."""

        surface_type = self.params.surface_type
        if surface_type not in (1, 2, 3):
            raise ValueError(
                "surface_type must be 1 (Gaussian), 2 (Exponential), or 3 (1.5 power)."
            )

        single = self.sigma0_single()
        totals = dict(single.total)

        if self.params.add_multiple:
            multiple = self.sigma0_multiple()
            for pol in totals:
                totals[pol] += multiple.get(pol, 0.0)

        return totals

    def sigma0_single(self) -> SingleScatteringBreakdown:
        """Eqs. (8–10): Evaluate the single-scattering NRCS expansion."""

        wave = self.eq3_wavevectors()
        kirchhoff = self.eq2_kirchhoff_field(wave["spectra"], wave["iterm"])

        series = self.eq9_single_explicit(
            spectra=wave["spectra"],
            iterm=wave["iterm"],
            rv_incident=kirchhoff["rv_incident"],
            rh_incident=kirchhoff["rh_incident"],
            rhv_incident=kirchhoff["rhv_incident"],
            fvv=kirchhoff["fields"]["vv"],
            fhh=kirchhoff["fields"]["hh"],
            fhv=kirchhoff["fields"]["hv"],
            fvh=kirchhoff["fields"]["vh"],
        )

        totals = {
            pol: series[pol] + kirchhoff["sigma_k"][pol]
            for pol in ("hh", "vv", "hv", "vh")
        }

        breakdown = SingleScatteringBreakdown(
            sigma_k=kirchhoff["sigma_k"],
            sigma_kc_plus_sigma_c=series,
            total=totals,
        )
        self._single_cache = breakdown
        return breakdown

    def sigma0_multiple(self) -> Dict[str, float]:
        """Eqs. (11–14): Assemble multiple-scattering NRCS contributions."""

        surface_label = "gauss" if self.params.surface_type == 1 else "exp"
        pols = ("hh", "vv", "hv", "vh")
        contributions, components = compute_multiple_scattering(
            theta_i=self.theta_i,
            theta_s=self.theta_s,
            phi_i=self.phi_i,
            phi_s=self.phi_s,
            er=self.er,
            k0=self.k0,
            sigma=self.sigma,
            corr_len=self.corr_len,
            surface_label=surface_label,
            polarisations=pols,
        )
        self._multiple_cache = contributions
        self._multiple_components_cache = components
        return contributions

    def multiple_scattering_components(self) -> Dict[str, Dict[str, float]]:
        """Return cached multiple-scattering kc/c components (linear power)."""

        if self._multiple_cache is None or self._multiple_components_cache is None:
            self.sigma0_multiple()

        assert self._multiple_components_cache is not None  # narrow type for mypy
        return {
            pol: comps.copy() for pol, comps in self._multiple_components_cache.items()
        }

    def _compute_wavenumber(self, params: AIEMParameters) -> float:
        """Determine the physical wavenumber from frequency or explicit input."""

        if params.frequency_ghz is not None:
            if params.frequency_ghz <= 0.0:
                raise ValueError("frequency_ghz must be positive when provided")
            c0 = 299_792_458.0  # speed of light (m/s)
            lam = c0 / (params.frequency_ghz * 1e9)
            k0 = 2.0 * math.pi / lam
        elif params.k0 is not None:
            k0 = params.k0
        else:
            k0 = 1.0

        if params.add_multiple and params.frequency_ghz is None and params.k0 is None:
            raise ValueError(
                "frequency_ghz or k0 must be provided when multiple scattering is enabled"
            )
        if k0 <= 0.0:
            raise ValueError("k0 must be positive")

        return k0

    def set_propagation_branch(self, z: float, zp: float) -> str:
        """Eq. (5): Return propagation branch '+' (upward) or '-' (downward)."""

        return "+" if z > zp else "-"

    # ------------------------------------------------------------------
    # Internal core (Eq. 1 onwards)
    # ------------------------------------------------------------------

    def eq1_scattered_field_decomp(self) -> Dict[str, Dict[str, complex]]:
        """Eq. (1): Provide the decomposition E₍qp₎ˢ = E₍qp₎ᵏ + E₍qp₎ᶜ."""

        if self._single_cache is None:
            self.sigma0_single()

        fields = {}
        complementary: Dict[str, np.ndarray] = {}
        if self._kirchhoff_cache is not None:
            fields = {
                pol: self._kirchhoff_cache["fields"][pol] for pol in self._kirchhoff_cache["fields"]
            }
        if self._eq10_cache is not None:
            complementary = {pol: values.copy() for pol, values in self._eq10_cache.items()}

        return {"E_k": fields, "E_c_series": complementary}

    def eq2_kirchhoff_field(
        self, spectra: np.ndarray, iterm: int
    ) -> Dict[str, Dict[str, complex | float]]:
        """Eq. (2): Kirchhoff field coefficients and transition reflection terms."""

        if self._kirchhoff_cache is not None:
            return self._kirchhoff_cache

        si = self.si
        cs = self.cs
        sis = self.sis
        css = self.css
        sfs = self.sfs
        csfs = self.csfs
        si2 = self.si2
        sis2 = self.sis2
        cs2 = self.cs2
        css2 = self.css2
        ks = self.ks
        er = self.er
        ur = self.ur

        stem = np.sqrt(er * ur - si2)
        rvi = (er * cs - stem) / (er * cs + stem)
        rhi = (ur * cs - stem) / (ur * cs + stem)
        rvhi = (rvi - rhi) / 2.0

        csl = np.sqrt(1.0 + cs * css - si * sis * csfs) / np.sqrt(2.0)
        sil = np.sqrt(max(1.0 - csl**2, 0.0))
        steml = np.sqrt(er * ur - sil**2)
        rvl = (er * csl - steml) / (er * csl + steml)
        rhl = (ur * csl - steml) / (ur * csl + steml)

        rv0 = (np.sqrt(er) - 1.0) / (np.sqrt(er) + 1.0)
        rh0 = -(np.sqrt(er) - 1.0) / (np.sqrt(er) + 1.0)

        root_term = np.sqrt(er - si2)
        Ftv = 8.0 * (rv0**2) * si2 * (cs + root_term) / (cs * root_term)
        Fth = -8.0 * (rh0**2) * si2 * (cs + root_term) / (cs * root_term)

        st0v = 1.0 / abs(1.0 + 8.0 * rv0 / (cs * Ftv)) ** 2
        st0h = 1.0 / abs(1.0 + 8.0 * rv0 / (cs * Fth)) ** 2

        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        temp1 = 1.0
        ks_cs_sq = (ks * cs) ** 2
        for n in range(1, iterm + 1):
            temp1 *= 1.0 / n
            ladder = ks_cs_sq**n
            factor = temp1 * ladder * spectra[n - 1]
            sum1 += factor

            term_v = abs(Ftv + 2.0 ** (n + 2.0) * rv0 / (cs * np.exp(ks_cs_sq))) ** 2
            term_h = abs(Fth + 2.0 ** (n + 2.0) * rv0 / (cs * np.exp(-ks_cs_sq))) ** 2
            sum2 += factor * term_v
            sum3 += factor * term_h

        stv = abs(Ftv) ** 2 * sum1 / sum2 if sum2 != 0.0 else 0.0
        sth = abs(Fth) ** 2 * sum1 / sum3 if sum3 != 0.0 else 0.0

        tfv = max(1.0 - stv / st0v, 0.0)
        tfh = max(1.0 - sth / st0h, 0.0)

        rvtran = rvi + (rvl - rvi) * tfv
        rhtran = rhi + (rhl - rhi) * tfh
        rvhtran = (rvtran - rhtran) / 2.0

        rv = rvtran
        rh = rhtran
        rhv = rvhtran

        zxx = -(sis * csfs - si) / (css + cs)
        zyy = -(sis * sfs) / (css + cs)
        d2 = np.sqrt((zxx * cs - si) ** 2 + zyy**2)

        hsnv = -(cs * csfs + si * (zxx * csfs + zyy * sfs))
        vsnh = css * csfs - zxx * sis
        hsnh = -sfs
        vsnv = zyy * cs * sis + css * (zyy * csfs * si - (cs + zxx * si) * sfs)
        hsnt = (
            (-(cs2 + si2) * sfs * (-si + cs * zxx)
             + csfs * (cs + si * zxx) * zyy
             + si * sfs * (zyy**2))
            / d2
        )
        hsnd = (
            (-(cs + si * zxx) * (-csfs * si + cs * csfs * zxx + cs * sfs * zyy))
            / d2
        )
        vsnt = (
            (
                (cs2 + si2) * (-si + cs * zxx) * (csfs * css - sis * zxx)
                + css * sfs * (cs + si * zxx) * zyy
                - (csfs * css * si + cs * sis) * (zyy**2)
            )
            / d2
        )
        vsnd = (
            -(cs + si * zxx)
            * (si * sis * zyy - css * (si * sfs - cs * sfs * zxx + cs * csfs * zyy))
            / d2
        )

        fhh = (
            (1.0 - rh) * hsnv
            + (1.0 + rh) * vsnh
            - (hsnt + vsnd) * (rh + rv) * (zyy / d2)
        )
        fvv = -(
            (1.0 - rv) * hsnv
            + (1.0 + rv) * vsnh
        ) + (hsnt + vsnd) * (rh + rv) * (zyy / d2)
        fhv = (
            -(1.0 + rv) * hsnh
            + (1.0 - rv) * vsnv
            + (hsnd - vsnt) * (rh + rv) * (zyy / d2)
        )
        fvh = (
            -(1.0 + rh) * hsnh
            + (1.0 - rh) * vsnv
            + (hsnd - vsnt) * (rh + rv) * (zyy / d2)
        )

        fields = {"vv": fvv, "hh": fhh, "hv": fhv, "vh": fvh}
        sigma_k = self.eq6_mean_scattered_power(fields, spectra, iterm)

        cache = {
            "fields": fields,
            "sigma_k": sigma_k,
            "rv_incident": rvi,
            "rh_incident": rhi,
            "rhv_incident": rvhi,
            "rv_transition": rvtran,
            "rh_transition": rhtran,
            "rhv_transition": rvhtran,
        }
        self._kirchhoff_cache = cache
        return cache

    def eq3_wavevectors(self) -> Dict[str, np.ndarray | int | float]:
        """Eq. (3): Build spectral wavevector terms and W⁽ⁿ⁾(·) spectrum."""

        if self._wavevector_cache is not None:
            return self._wavevector_cache

        surface_type = self.params.surface_type
        ks2 = self.ks2
        cs = self.cs
        css = self.css
        sis = self.sis
        sfs = self.sfs
        csfs = self.csfs
        si = self.si
        phi_i = self.phi_i
        kl = self.kl
        kl2 = self.kl2

        tolerance = 1.0e-16
        iterm = 1
        temp_old = 0.0
        temp = ks2 * (cs + css) ** 2
        while abs(temp - temp_old) > tolerance:
            temp_old = temp
            iterm += 1
            temp = temp_old * ks2 * (cs + css) ** 2 / iterm
            if iterm > 500:
                break

        n_terms = max(iterm, 100)
        spectra = np.zeros(n_terms, dtype=np.float64)

        delta_phi_s = sis * csfs - si * np.cos(phi_i)
        delta_phi_i = sis * sfs - si * np.sin(phi_i)
        K = kl * np.sqrt(delta_phi_s**2 + delta_phi_i**2)

        for n in range(1, iterm + 1):
            fn = float(n)
            if surface_type == 1:
                spectra[n - 1] = kl2 * np.exp(-(K**2) / (4.0 * fn)) / (2.0 * fn)
            elif surface_type == 2:
                spectra[n - 1] = (kl / fn) ** 2 * (1.0 + (K / fn) ** 2) ** (-1.5)
            else:
                e = 1.5 * fn - 1.0
                y = 1.5 * fn
                if np.isclose(K, 0.0):
                    spectra[n - 1] = kl2 / (3.0 * fn - 2.0)
                else:
                    m = 1.5 * fn - 1.0
                    log_bessel = np.log(kv(-m, K))
                    out = kl2 * (K / 2.0) ** e
                    spectra[n - 1] = out * np.exp(log_bessel - gammaln(y))

        cache = {
            "spectra": spectra,
            "iterm": iterm,
            "n_terms": n_terms,
            "K": K,
            "delta_phi_s": delta_phi_s,
            "delta_phi_i": delta_phi_i,
        }
        self._wavevector_cache = cache
        return cache

    def eq4_complementary_field(self) -> Dict[str, Dict[str, tuple[complex, complex]]]:
        """Eq. (4): Cached complementary field propagator terms (F±, G±)."""

        if self._eq4_cache is None:
            self.sigma0_single()
        return self._eq4_cache

    def eq6_mean_scattered_power(
        self, fields: Dict[str, complex], spectra: np.ndarray, iterm: int
    ) -> Dict[str, float]:
        """Eq. (6): Mean scattered power for Kirchhoff contribution."""

        series_sum = 0.0
        temp = 1.0
        for n in range(1, iterm + 1):
            temp *= self.ks2 * (self.cs + self.css) ** 2 / n
            series_sum += temp * spectra[n - 1]

        expk = np.exp(-self.ks2 * (self.css + self.cs) ** 2) * series_sum
        return {
            pol: float(0.5 * expk * abs(value) ** 2)
            for pol, value in fields.items()
        }

    def eq10_I_n(self, qp: str, n: int) -> complex:
        """Eq. (10): Return cached Iⁿ₍qp₎ term (1-indexed n)."""

        if self._eq10_cache is None:
            raise RuntimeError("Eq. 10 cache unavailable; call eq9_single_explicit first.")

        qp_lower = qp.lower()
        if qp_lower not in self._eq10_cache:
            raise KeyError(f"Unknown polarisation '{qp}'.")
        index = n - 1
        values = self._eq10_cache[qp_lower]
        if index < 0 or index >= len(values):
            raise IndexError(f"Order n={n} outside cached range (1..{len(values)}).")
        return values[index]

    def sigma_kc_l(self, l: int) -> Dict[str, float]:
        """Eq. (12): Placeholder for σ₍qp₎^{kc_l}(m) cross terms."""

        raise NotImplementedError("sigma_kc_l is delegated to the multiple-scattering solver.")

    def sigma_c_i(self, i: int) -> Dict[str, float]:
        """Eq. (13a): Placeholder for σ₍qp₎^{c_i}(m) complementary terms."""

        raise NotImplementedError("sigma_c_i is delegated to the multiple-scattering solver.")

    def sigma_c_j(self, j: int) -> Dict[str, float]:
        """Eq. (13b): Placeholder for σ₍qp₎^{c_j}(m) complementary terms."""

        raise NotImplementedError("sigma_c_j is delegated to the multiple-scattering solver.")

    def eq14_arg_mapping(
        self, u: float, v: float, q: float
    ) -> Tuple[float, float, float]:
        """Eq. (14): Map (u, v, q) ↔ (u′, v′, q′) arguments for Appendix kernels."""

        return (-u, -v, q)

    def eq15_crosspol_slightly_rough(self) -> float:
        """Eqs. (15a–15b): Cross-pol corrections for slightly rough surfaces."""

        raise NotImplementedError("Eq. 15 cross-polar formulation not yet ported.")

    def eq16_spm_crosspol(self) -> float:
        """Eq. (16): Small-perturbation-model limit for cross-polarisation."""

        raise NotImplementedError("Eq. 16 SPM cross-pol limit not yet ported.")

    def eq17_pec_limit(self) -> float:
        """Eq. (17): Perfect electric conductor limiting behaviour."""

        raise NotImplementedError("Eq. 17 PEC limit not yet ported.")

    # ------------------------------------------------------------------
    # Appendix helpers (A–C blocks)
    # ------------------------------------------------------------------

    def g_kc_l(self, u: float, v: float, q: float, l: int) -> complex:
        """Appendix A (A1–A3): Placeholder for g^{kc_l}(u, v, q)."""

        raise NotImplementedError("g_kc_l is provided by the dedicated multiple-scattering module.")

    def g_c_i(self, u: float, v: float, q: float, qp: float, i: int) -> complex:
        """Appendix A (A4–A11): Placeholder for g^{c_i}(u, v, q, q′)."""

        raise NotImplementedError("g_c_i is provided by the dedicated multiple-scattering module.")

    def g_c_j(self, up: float, vp: float, q: float, qp: float, j: int) -> complex:
        """Appendix A (A12–A17): Placeholder for g^{c_j}(u′, v′, q, q′)."""

        raise NotImplementedError("g_c_j is provided by the dedicated multiple-scattering module.")

    def F_plus(
        self,
        pol: str,
        u: float,
        v: float,
        q: float,
        q_slp: float,
        q_fix: float,
        reflection: complex,
    ) -> complex:
        """Appendix B (B1/B3/B5): Upward F⁺_{qp} helper mapped to field kernels."""

        pol_lower = pol.lower()
        if pol_lower == "vv":
            return self._favv(u, v, q, q_slp, q_fix, reflection)
        if pol_lower == "hh":
            return self._fahh(u, v, q, q_slp, q_fix, reflection)
        if pol_lower == "hv":
            return self._fahv(u, v, q, q_slp, q_fix, reflection)
        if pol_lower == "vh":
            return self._favh(u, v, q, q_slp, q_fix, reflection)
        raise KeyError(f"Unsupported polarisation '{pol}'.")

    def G_plus(
        self,
        pol: str,
        u: float,
        v: float,
        q: float,
        q_slp: float,
        q_fix: float,
        reflection: complex,
        er: complex,
    ) -> complex:
        """Appendix B (B2/B4/B6): Upward G⁺_{qp} helper mapped to field kernels."""

        pol_lower = pol.lower()
        if pol_lower == "vv":
            return self._fbvv(u, v, q, q_slp, q_fix, reflection, er)
        if pol_lower == "hh":
            return self._fbhh(u, v, q, q_slp, q_fix, reflection, er)
        if pol_lower == "hv":
            return self._fbhv(u, v, q, q_slp, q_fix, reflection, er)
        if pol_lower == "vh":
            return self._fbvh(u, v, q, q_slp, q_fix, reflection, er)
        raise KeyError(f"Unsupported polarisation '{pol}'.")

    def C_coeffs(
        self, u: float, v: float, q: float, q_slp: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Appendix C: Return (C1..C6) coefficients for complementary kernels."""

        zx, zy, zxp, zyp = self._common_terms(u, v, q_slp)
        return self._compute_c_terms(u, v, q, zx, zy, zxp, zyp)

    def B_coeffs(
        self, u: float, v: float, q: float, q_slp: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Appendix C: Return (B1..B6) coefficients for cross-polar kernels."""

        zx, zy, zxp, zyp = self._common_terms(u, v, q_slp)
        return self._compute_b_terms(u, v, q, zx, zy, zxp, zyp)

    def eq9_single_explicit(
        self,
        spectra: np.ndarray,
        iterm: int,
        rv_incident: complex,
        rh_incident: complex,
        rhv_incident: complex,
        fvv: complex,
        fhh: complex,
        fhv: complex,
        fvh: complex,
    ) -> Dict[str, float]:
        """Eq. (9): Explicit series for σ₍qp₎^(s) with Eq. (10) integrals cached."""

        rv = rv_incident
        rh = rh_incident
        rhv = rhv_incident

        qq = self.cs
        qqt = np.sqrt(self.er - self.si2)
        qqs = self.css
        qqts = np.sqrt(self.er - self.sis2)

        qq1 = qq
        qq2 = qqs
        qq3 = qqt
        qq4 = qqts
        qq5 = qqt
        qq6 = qqts

        Fvaupi = self._favv(-self.si, 0.0, qq1, qq1, qq, rv) * self._expal(qq1)
        Fvadni = self._favv(-self.si, 0.0, -qq1, -qq1, qq, rv) * self._expal(-qq1)
        Fvaups = self._favv(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rv) * self._expal(qq2)
        Fvadns = self._favv(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rv) * self._expal(-qq2)
        Fvbupi = self._fbvv(-self.si, 0.0, qq3, qq5, qqt, rv, self.er) * self._expal(qq5)
        Fvbdni = self._fbvv(-self.si, 0.0, -qq3, -qq5, qqt, rv, self.er) * self._expal(-qq5)
        Fvbups = self._fbvv(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rv, self.er) * self._expal(qq6)
        Fvbdns = self._fbvv(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rv, self.er) * self._expal(-qq6)

        Fhaupi = self._fahh(-self.si, 0.0, qq1, qq1, qq, rh) * self._expal(qq1)
        Fhadni = self._fahh(-self.si, 0.0, -qq1, -qq1, qq, rh) * self._expal(-qq1)
        Fhaups = self._fahh(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rh) * self._expal(qq2)
        Fhadns = self._fahh(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rh) * self._expal(-qq2)
        Fhbupi = self._fbhh(-self.si, 0.0, qq3, qq5, qqt, rh, self.er) * self._expal(qq5)
        Fhbdni = self._fbhh(-self.si, 0.0, -qq3, -qq5, qqt, rh, self.er) * self._expal(-qq5)
        Fhbups = self._fbhh(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rh, self.er) * self._expal(qq6)
        Fhbdns = self._fbhh(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rh, self.er) * self._expal(-qq6)

        Fhvaupi = self._fahv(-self.si, 0.0, qq1, qq1, qq, rhv) * self._expal(qq1)
        Fhvadni = self._fahv(-self.si, 0.0, -qq1, -qq1, qq, rhv) * self._expal(-qq1)
        Fhvaups = self._fahv(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rhv) * self._expal(qq2)
        Fhvadns = self._fahv(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rhv) * self._expal(-qq2)
        Fhvbupi = self._fbhv(-self.si, 0.0, qq3, qq5, qqt, rhv, self.er) * self._expal(qq5)
        Fhvbdni = self._fbhv(-self.si, 0.0, -qq3, -qq5, qqt, rhv, self.er) * self._expal(-qq5)
        Fhvbups = self._fbhv(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rhv, self.er) * self._expal(qq6)
        Fhvbdns = self._fbhv(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rhv, self.er) * self._expal(-qq6)

        Fvhaupi = self._favh(-self.si, 0.0, qq1, qq1, qq, rhv) * self._expal(qq1)
        Fvhadni = self._favh(-self.si, 0.0, -qq1, -qq1, qq, rhv) * self._expal(-qq1)
        Fvhaups = self._favh(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rhv) * self._expal(qq2)
        Fvhadns = self._favh(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rhv) * self._expal(-qq2)
        Fvhbupi = self._fbvh(-self.si, 0.0, qq3, qq5, qqt, rhv, self.er) * self._expal(qq5)
        Fvhbdni = self._fbvh(-self.si, 0.0, -qq3, -qq5, qqt, rhv, self.er) * self._expal(-qq5)
        Fvhbups = self._fbvh(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rhv, self.er) * self._expal(qq6)
        Fvhbdns = self._fbvh(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rhv, self.er) * self._expal(-qq6)

        Ivv = np.zeros(iterm, dtype=complex)
        Ihh = np.zeros_like(Ivv)
        Ihv = np.zeros_like(Ivv)
        Ivh = np.zeros_like(Ivv)

        base_factor = np.exp(-self.ks2 * self.cs * self.css)
        sum_base = self.cs + self.css

        for n in range(1, iterm + 1):
            fn = float(n)
            pref = sum_base**fn * base_factor

            Ivv[n - 1] = (
                pref * fvv
                + 0.25
                * (
                    Fvaupi * (self.css - qq1) ** fn
                    + Fvadni * (self.css + qq1) ** fn
                    + Fvaups * (self.cs + qq2) ** fn
                    + Fvadns * (self.cs - qq2) ** fn
                    + Fvbupi * (self.css - qq5) ** fn
                    + Fvbdni * (self.css + qq5) ** fn
                    + Fvbups * (self.cs + qq6) ** fn
                    + Fvbdns * (self.cs - qq6) ** fn
                )
            )

            Ihh[n - 1] = (
                pref * fhh
                + 0.25
                * (
                    Fhaupi * (self.css - qq1) ** fn
                    + Fhadni * (self.css + qq1) ** fn
                    + Fhaups * (self.cs + qq2) ** fn
                    + Fhadns * (self.cs - qq2) ** fn
                    + Fhbupi * (self.css - qq5) ** fn
                    + Fhbdni * (self.css + qq5) ** fn
                    + Fhbups * (self.cs + qq6) ** fn
                    + Fhbdns * (self.cs - qq6) ** fn
                )
            )

            Ihv[n - 1] = (
                pref * fhv
                + 0.25
                * (
                    Fhvaupi * (self.css - qq1) ** fn
                    + Fhvadni * (self.css + qq1) ** fn
                    + Fhvaups * (self.cs + qq2) ** fn
                    + Fhvadns * (self.cs - qq2) ** fn
                    + Fhvbupi * (self.css - qq5) ** fn
                    + Fhvbdni * (self.css + qq5) ** fn
                    + Fhvbups * (self.cs + qq6) ** fn
                    + Fhvbdns * (self.cs - qq6) ** fn
                )
            )

            Ivh[n - 1] = (
                pref * fvh
                + 0.25
                * (
                    Fvhaupi * (self.css - qq1) ** fn
                    + Fvhadni * (self.css + qq1) ** fn
                    + Fvhaups * (self.cs + qq2) ** fn
                    + Fvhadns * (self.cs - qq2) ** fn
                    + Fvhbupi * (self.css - qq5) ** fn
                    + Fvhbdni * (self.css + qq5) ** fn
                    + Fvhbups * (self.cs + qq6) ** fn
                    + Fvhbdns * (self.cs - qq6) ** fn
                )
            )

        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        sum4 = 0.0
        temp = 1.0
        for n in range(1, iterm + 1):
            temp *= self.ks2 / n
            spectrum = spectra[n - 1]
            sum1 += temp * (Ivv[n - 1] * np.conjugate(Ivv[n - 1])) * spectrum
            sum2 += temp * (Ihh[n - 1] * np.conjugate(Ihh[n - 1])) * spectrum
            sum3 += temp * (Ihv[n - 1] * np.conjugate(Ihv[n - 1])) * spectrum
            sum4 += temp * (Ivh[n - 1] * np.conjugate(Ivh[n - 1])) * spectrum

        ft_factor_copol = 1.0 / (2.0 * np.pi) ** 2
        ft_factor_xpol = 1.0 / (2.0 * np.pi)
        attenuation = np.exp(-self.ks2 * (self.cs2 + self.css2))

        single_scatter = {
            "vv": float(np.real(ft_factor_copol * 0.5 * attenuation * sum1)),
            "hh": float(np.real(ft_factor_copol * 0.5 * attenuation * sum2)),
            "hv": float(np.real(ft_factor_xpol * 0.5 * attenuation * sum3)),
            "vh": float(np.real(ft_factor_xpol * 0.5 * attenuation * sum4)),
        }

        for pol in single_scatter:
            single_scatter[pol] = max(single_scatter[pol], 0.0)

        self._eq10_cache = {
            "vv": Ivv.copy(),
            "hh": Ihh.copy(),
            "hv": Ihv.copy(),
            "vh": Ivh.copy(),
        }

        self._eq4_cache = {
            "vv": {
                "F_plus": (Fvaupi, Fvaups),
                "F_minus": (Fvadni, Fvadns),
                "G_plus": (Fvbupi, Fvbups),
                "G_minus": (Fvbdni, Fvbdns),
            },
            "hh": {
                "F_plus": (Fhaupi, Fhaups),
                "F_minus": (Fhadni, Fhadns),
                "G_plus": (Fhbupi, Fhbups),
                "G_minus": (Fhbdni, Fhbdns),
            },
            "hv": {
                "F_plus": (Fhvaupi, Fhvaups),
                "F_minus": (Fhvadni, Fhvadns),
                "G_plus": (Fhvbupi, Fhvbups),
                "G_minus": (Fhvbdni, Fhvbdns),
            },
            "vh": {
                "F_plus": (Fvhaupi, Fvhaups),
                "F_minus": (Fvhadni, Fvhadns),
                "G_plus": (Fvhbupi, Fvhbups),
                "G_minus": (Fvhbdni, Fvhbdns),
            },
        }

        return single_scatter

    def _common_terms(self, u, v, qslp):
        kxu = self.si + u
        ksxu = self.sis * self.csfs + u
        kyv = v
        ksyv = self.sis * self.sfs + v

        denom1 = self.css - qslp
        if abs(np.real(denom1)) < 1e-12:
            zx = 0.0
            zy = 0.0
        else:
            zx = -ksxu / denom1
            zy = -ksyv / denom1

        denom2 = self.cs + qslp
        if abs(np.real(denom2)) < 1e-12:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / denom2
            zyp = kyv / denom2

        return zx, zy, zxp, zyp

    def _compute_c_terms(self, u, v, q, zx, zy, zxp, zyp):
        c1 = -self.csfs * (-1.0 - zx * zxp) + self.sfs * zxp * zy
        c2 = -self.csfs * (
            -self.cs * q
            - self.cs * u * zx
            - q * self.si * zxp
            - self.si * u * zx * zxp
            - self.cs * v * zyp
            - self.si * v * zx * zyp
        ) + self.sfs * (
            self.cs * u * zy
            + self.si * u * zxp * zy
            + q * self.si * zyp
            - self.cs * u * zyp
            + self.si * v * zy * zyp
        )
        c3 = -self.csfs * (
            self.si * u
            - q * self.si * zx
            - self.cs * u * zxp
            + self.cs * q * zx * zxp
        ) + self.sfs * (
            -self.si * v
            + self.cs * v * zxp
            + q * self.si * zy
            - self.cs * q * zxp * zy
        )
        c4 = (
            -self.css * self.sfs * (-self.si * zyp + self.cs * zx * zyp)
            - self.csfs * self.css * (-self.cs - self.si * zxp - self.cs * zy * zyp)
            + self.sis * (-self.cs * zx - self.si * zx * zxp - self.si * zy * zyp)
        )
        c5 = (
            -self.css * self.sfs * (-v * zx + v * zxp)
            - self.csfs * self.css * (q + u * zxp + v * zy)
            + self.sis * (q * zx + u * zx * zxp + v * zxp * zy)
        )
        c6 = (
            -self.css * self.sfs * (-u * zyp + q * zx * zyp)
            - self.csfs * self.css * (v * zyp - q * zy * zyp)
            + self.sis * (v * zx * zyp - u * zy * zyp)
        )
        return c1, c2, c3, c4, c5, c6

    def _compute_b_terms(self, u, v, q, zx, zy, zxp, zyp):
        b1 = -self.css * self.sfs * (-1.0 - zx * zxp) - self.sis * zy - self.csfs * self.css * zxp * zy
        b2 = -self.css * self.sfs * (
            -self.cs * q
            - self.cs * u * zx
            - q * self.si * zxp
            - self.si * u * zx * zxp
            - self.cs * v * zyp
            - self.si * v * zx * zyp
        ) + self.sis * (
            -self.cs * q * zy
            - q * self.si * zxp * zy
            + q * self.si * zx * zyp
            - self.cs * u * zx * zyp
            - self.cs * v * zy * zyp
        ) - self.csfs * self.css * (
            self.cs * u * zy
            + self.si * u * zxp * zy
            + q * self.si * zyp
            - self.cs * u * zyp
            + self.si * v * zy * zyp
        )
        b3 = -self.css * self.sfs * (
            self.si * u
            - q * self.si * zx
            - self.cs * u * zxp
            + self.cs * q * zx * zxp
        ) - self.csfs * self.css * (
            -self.si * v
            + self.cs * v * zxp
            + q * self.si * zy
            - self.cs * q * zxp * zy
        ) + self.sis * (
            -self.si * v * zx
            + self.cs * v * zx * zxp
            + self.si * u * zy
            - self.cs * u * zxp * zy
        )
        b4 = -self.csfs * (-self.si * zyp + self.cs * zx * zyp) + self.sfs * (-self.cs - self.si * zxp - self.cs * zy * zyp)
        b5 = -self.csfs * (-v * zx + v * zxp) + self.sfs * (q + u * zxp + v * zy)
        b6 = -self.csfs * (-u * zyp + q * zx * zyp) + self.sfs * (v * zyp - q * zy * zyp)
        return b1, b2, b3, b4, b5, b6

    def _fahh(self, u, v, q, qslp, qfix, rh):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        c1, c2, c3, c4, c5, c6 = self._compute_c_terms(u, v, q, zx, zy, zxp, zyp)
        rph = 1.0 + rh
        rmh = 1.0 - rh
        ah = rph / qfix
        bh = rmh / qfix
        return -bh * (-rph * c1 + rmh * c2 + rph * c3) - ah * (rmh * c4 + rph * c5 + rmh * c6)

    def _favv(self, u, v, q, qslp, qfix, rv):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        c1, c2, c3, c4, c5, c6 = self._compute_c_terms(u, v, q, zx, zy, zxp, zyp)
        rpv = 1.0 + rv
        rmv = 1.0 - rv
        av = rpv / qfix
        bv = rmv / qfix
        return bv * (-rpv * c1 + rmv * c2 + rpv * c3) + av * (rmv * c4 + rpv * c5 + rmv * c6)

    def _fahv(self, u, v, q, qslp, qfix, rhv):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        b1, b2, b3, b4, b5, b6 = self._compute_b_terms(u, v, q, zx, zy, zxp, zyp)
        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix
        return b * (rp * b1 - rm * b2 - rp * b3) + a * (rm * b4 + rp * b5 + rm * b6)

    def _favh(self, u, v, q, qslp, qfix, rhv):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        b1, b2, b3, b4, b5, b6 = self._compute_b_terms(u, v, q, zx, zy, zxp, zyp)
        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix
        return b * (rp * b4 + rm * b5 + rp * b6) - a * (-rm * b1 + rp * b2 + rm * b3)

    def _fbhh(self, u, v, q, qslp, qfix, rh, er):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        c1, c2, c3, c4, c5, c6 = self._compute_c_terms(u, v, q, zx, zy, zxp, zyp)
        rph = 1.0 + rh
        rmh = 1.0 - rh
        ah = rph / qfix
        bh = rmh / qfix
        return ah * (-rph * c1 * er + rmh * c2 + rph * c3) + bh * (rmh * c4 + rph * c5 + rmh * c6 / er)

    def _fbvv(self, u, v, q, qslp, qfix, rv, er):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        c1, c2, c3, c4, c5, c6 = self._compute_c_terms(u, v, q, zx, zy, zxp, zyp)
        rpv = 1.0 + rv
        rmv = 1.0 - rv
        av = rpv / qfix
        bv = rmv / qfix
        return av * (rpv * c1 - rmv * c2 - rpv * c3 / er) - bv * (rmv * c4 * er + rpv * c5 + rmv * c6)

    def _fbhv(self, u, v, q, qslp, qfix, rhv, er):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        b1, b2, b3, b4, b5, b6 = self._compute_b_terms(u, v, q, zx, zy, zxp, zyp)
        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix
        return a * (-rp * b1 + rm * b2 + rp * b3 / er) - b * (rm * b4 * er + rp * b5 + rm * b6)

    def _fbvh(self, u, v, q, qslp, qfix, rhv, er):
        zx, zy, zxp, zyp = self._common_terms(u, v, qslp)
        b1, b2, b3, b4, b5, b6 = self._compute_b_terms(u, v, q, zx, zy, zxp, zyp)
        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix
        return -a * (rp * b4 + rm * b5 + rp * b6 / er) + b * (-rm * b1 * er + rp * b2 + rm * b3)

    def _expal(self, q):
        return np.exp(-self.ks2 * (q**2 - q * (self.css - self.cs)))

    def _format_output(self, totals: Dict[str, float]) -> Dict[str, float]:
        unit = self.params.output_unit.lower()
        if unit == "db":
            return {pol: self._to_db(value) for pol, value in totals.items()}
        if unit == "linear":
            return totals
        raise ValueError("output_unit must be 'dB' or 'linear'")

    @staticmethod
    def _to_db(power: float) -> float:
        return 10.0 * np.log10(power + np.finfo(float).eps)


def AIEM(
    theta_i: float,
    theta_s: float,
    phi_s: float,
    err: float,
    eri: float,
    itype: int,
    *,
    addMultiple: bool = False,
    output_unit: str = "dB",
    frequency_ghz: float | None = None,
    k0: float | None = None,
    sigma: float | None = None,
    corr_len: float | None = None,
) -> Tuple[float, float, float, float]:
    """Convenience wrapper for the historical MATLAB driver.

    Provide the surface roughness via the dimensional descriptors ``sigma``
    (surface RMS height, metres) and ``corr_len`` (correlation length, metres).
    The solver rebuilds the dimensionless parameters internally using the
    physical wavenumber derived from ``frequency_ghz`` or ``k0``.
    """

    params = AIEMParameters(
        theta_i=theta_i,
        theta_s=theta_s,
        phi_s=phi_s,
        err=err,
        eri=eri,
        surface_type=itype,
        add_multiple=addMultiple,
        output_unit=output_unit,
        frequency_ghz=frequency_ghz,
        k0=k0,
        sigma=sigma,
        corr_len=corr_len,
    )
    result = AIEMModel(params).compute()
    return result.hh, result.vv, result.hv, result.vh


__all__ = ["AIEM", "AIEMModel", "AIEMParameters", "AIEMResult"]
