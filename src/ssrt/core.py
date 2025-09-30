import numpy as np
from .utils.fresnel import ReflTransm_PlanarBoundary
from .utils.util import toDB, toPower, toLambda


class S2RTR:
    """Single-scattering radiative transfer (SSRT) canopy + soil model.

    Parameters
    ----------
    frq_GHz : float
        Frequency in GHz.
    theta_i, theta_s : float
        Incidence and scattering angles in degrees.
    phi_i, phi_s : float
        Azimuth angles (deg) of incident and scattered directions.
    s : float
        Soil RMS height (m).
    cl : float
        Soil correlation length (m).
    eps2 : complex or array-like
        Complex permittivity of the canopy layer.
    eps3 : complex or array-like
        Complex permittivity of the soil.
    a : float
        Single-scattering albedo (0 < a < 0.2).
    kappa_e : float
        Extinction coefficient of the canopy (Np/m).
    d : float
        Canopy thickness (m).
    acftype : str
        Surface autocorrelation type (``'exp'``, ``'gauss'``, ``'pow'``).
    RT_models : dict
        Mapping describing soil (`'RT_s'`) and canopy (`'RT_c'`) models.

    Examples
    --------
    >>> ssrt = S2RTR(
    ...     frq_GHz=5.405,
    ...     theta_i=40.0,
    ...     theta_s=40.0,
    ...     phi_i=0.0,
    ...     phi_s=180.0,
    ...     s=0.02,
    ...     cl=0.1,
    ...     eps2=12.0 + 3.0j,
    ...     eps3=5.0 + 1.0j,
    ...     a=0.08,
    ...     kappa_e=0.5,
    ...     d=0.3,
    ...     acftype='exp',
    ...     RT_models={'RT_s': 'I2EM', 'RT_c': 'Diff'},
    ... )  # doctest: +SKIP
    """

    def __init__(self, frq_GHz, theta_i, theta_s, phi_i, phi_s, s, cl, eps2, eps3, a, kappa_e, d, acftype, RT_models):
        """
        Initializes the S2RTR class with the given parameters.

        Parameters:
            f : float
                Frequency in GHz.
            theta_i : float
                Incidence angle in degrees.
            theta_s : float
                Scattering angle in degrees.
            phi_i : float
                Azimuth angle of the incident wave in degrees.
            phi_s : float
                Azimuth angle of the scattered wave in degrees.
            s : float
                RMS height of the ground surface (m).
            cl : float
                Correlation length of the surface (m).
            eps2 : complex
                Dielectric constant of the canopy layer.
            eps3 : complex
                Dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            acftype : str
                Type of autocorrelation function ('Gaussian', 'Exponential', 'Spherical').
            RT_s : str
                Surface scattering model ('AIEM', 'PRISM1').
            RT_c : str
                Radiative transfer model ('Diff', 'Spec').
            get_sig_ground : bool
                If True, returns the backscatter coefficients of the ground surface.

        Examples
        --------
        >>> rt = S2RTR(
        ...     frq_GHz=5.405,
        ...     theta_i=40.0,
        ...     theta_s=40.0,
        ...     phi_i=0.0,
        ...     phi_s=180.0,
        ...     s=0.02,
        ...     cl=0.1,
        ...     eps2=12.0 + 3.0j,
        ...     eps3=5.0 + 1.0j,
        ...     a=0.08,
        ...     kappa_e=0.5,
        ...     d=0.3,
        ...     acftype='exp',
        ...     RT_models={'RT_s': 'I2EM', 'RT_c': 'Diff'},
        ... )  # doctest: +SKIP
        """
        
        self.f = frq_GHz
        self.theta_i = theta_i
        self.theta_s = theta_s
        self.phi_i = phi_i
        self.phi_s = phi_s
        self.s = s
        self.cl = cl
        # Air dielectric constant
        self.eps = 1. 
        self.eps2 = eps2
        self.eps3 = eps3
        self.eps_ratio = eps3 / eps2
        self.a = a
        self.kappa_e = kappa_e
        self.d = d
        self.acftype = acftype
        self.RT_s = RT_models['RT_s']  # Surface scattering model
        self.RT_c = RT_models['RT_c']  # Radiative transfer model

        
        self.__check__()


    def __check__(self):
        """
        Checks if the input parameters are valid.
        
        Raises:
            ValueError: If any of the parameters are invalid.
        """
        assert self.a < 0.2, "albedo must be < 0.2"
        assert self.s < 0.5, "RMS height must be < 0.5"
        assert self.cl > 0, "Correlation length must be > 0"
        assert self.kappa_e > 0, "Extinction coefficient must be > 0"
        assert self.d > 0, "Thickness of the Rayleigh layer must be > 0"
        assert self.RT_s in ['AIEM', 'PRISM1', 'Dubois95', 'SMART', 'SPM3D', 'I2EM'], "RT_s must be 'AIEM', 'PRISM1', 'Dubois95', 'SMART', 'SPM3D', or 'I2EM'"
        assert self.RT_c in ['Diff', 'Spec'], "RT_c must be 'Diff' or 'Spec'"
        assert self.acftype in ['gauss', 'exp', 'pow'], "acftype must be 'Gaussian', 'Exponential', or 'Power-law 1.5'"
        assert self.theta_i >= 0, "Incidence angle must be >= 0"
        assert self.theta_i <= 90, "Incidence angle must be <= 90"
        assert self.theta_s >= 0, "Scattering angle must be >= 0"
        assert self.theta_s <= 90, "Scattering angle must be <= 90"
        assert self.phi_i >= 0, "Azimuth angle of the incident wave must be >= 0"
        assert self.phi_i <= 360, "Azimuth angle of the incident wave must be <= 360"
        assert self.phi_s >= 0, "Azimuth angle of the scattered wave must be >= 0"
        assert self.phi_s <= 360, "Azimuth angle of the scattered wave must be <= 360"
        assert self.f > 0, "Frequency must be > 0"
        assert self.s > 0, "RMS height must be > 0"
        assert self.cl > 0, "Correlation length must be > 0"


    def calc_sigma(self, todB=True):
        """
        Computes the backscatter coefficients for the given parameters.

        Returns:
            dict: A dictionary containing the backscatter coefficients in dB for 
                  'vv', 'hh', 'hv', and 'vh' polarizations.

        Examples
        --------
        >>> rt = S2RTR(
        ...     frq_GHz=5.405,
        ...     theta_i=40.0,
        ...     theta_s=40.0,
        ...     phi_i=0.0,
        ...     phi_s=180.0,
        ...     s=0.02,
        ...     cl=0.1,
        ...     eps2=12.0 + 3.0j,
        ...     eps3=5.0 + 1.0j,
        ...     a=0.08,
        ...     kappa_e=0.5,
        ...     d=0.3,
        ...     acftype='exp',
        ...     RT_models={'RT_s': 'I2EM', 'RT_c': 'Diff'},
        ... )
        >>> rt.calc_sigma()  # doctest: +SKIP
        """

        pol_list = ['vv', 'hh', 'hv', 'vh']
        
        # --- Call the radiative transfer model ---
        if self.RT_c == 'Diff':
            sig_s = {}
            # --- Call the AIEM model ---
            if self.RT_s == 'AIEM':
                from .surface.aiem import AIEM

                lambda_m = toLambda(self.f)
                k = 2.0 * np.pi / lambda_m
                surface_map = {'gauss': 1, 'exp': 2, 'pow': 3}
                itype = surface_map.get(self.acftype, 1)
                phi_rel = (self.phi_s - self.phi_i) % 360
                eps_surface = np.atleast_1d(self.eps3)[0]

                hh_db, vv_db, hv_db, vh_db = AIEM(
                    theta_i=self.theta_i,
                    theta_s=self.theta_s,
                    phi_s=phi_rel,
                    err=float(np.real(eps_surface)),
                    eri=float(np.imag(eps_surface)),
                    itype=itype,
                    addMultiple=False,
                    frequency_ghz=float(self.f),
                    k0=k,
                    sigma=float(self.s),
                    corr_len=float(self.cl),
                )
                sig_s = {
                    'vv': toPower(vv_db),
                    'hh': toPower(hh_db),
                    'hv': toPower(hv_db),
                    'vh': toPower(vh_db),
                }
            
            # --- Call the PRISM1 model ---
            if self.RT_s == 'PRISM1':
                from .surface.prism1 import PRISM1
                prism0 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps3, s=self.s)
                sig_s_full = prism0.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            if self.RT_s == 'Dubois95':
                from .surface.dubois95 import Dubois95
                db95 = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps3, s=self.s)
                sig_s_full = db95.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            if self.RT_s == 'SMART':
                from .surface.smart import SMART
                smart = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps3)
                sig_s_full = smart.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            if self.RT_s == 'SPM3D':
                from .surface.spm import SPM3D
                spm = SPM3D(fr=self.f, sig=self.s, L=self.cl, thi=self.theta_i, eps=self.eps3)
                sig_s_full = spm.calc_sigma(todB=False)
                sig_s = dict(zip(pol_list, sig_s_full))
            if self.RT_s == 'I2EM':
                from .surface.i2em import I2EM_Bistat_model

                sp_map = {'exp': 1, 'gauss': 2, 'pow': 3}
                sp = sp_map.get(self.acftype, 1)
                xx = 1.5 if sp == 3 else 0.0
                phi_rel = (self.phi_s - self.phi_i) % 360
                eps_surface = float(np.real(np.atleast_1d(self.eps3)[0]))

                vv_db, hh_db, hv_db, vh_db = I2EM_Bistat_model(
                    fr=self.f,
                    sig=self.s,
                    L=self.cl,
                    thi=self.theta_i,
                    ths=self.theta_s,
                    phs=phi_rel,
                    er=eps_surface,
                    sp=sp,
                    xx=xx
                )
                sig_s = {
                    'vv': toPower(vv_db),
                    'hh': toPower(hh_db),
                    'hv': toPower(hv_db),
                    'vh': toPower(vh_db)
                }
            # else:
            #     raise ValueError("RT_s must be 'AIEM' or 'PRISM1'")
            
            # --- Call the Rayleigh model ---
            sig_t = self.__S2RTR_DiffuseUB_FullPol(sig_s, self.eps2, self.a, self.kappa_e, self.d, self.theta_i, todB=todB)

            if todB:
                # Convert to dB
                sig_s_dB = {pq: toDB(sig_s[pq]) for pq in pol_list}
                return sig_s_dB, sig_s_dB, sig_t
            else:
                return sig_s, sig_s, sig_t

        elif self.RT_c == 'Spec':
            
            sig_0_top = {}
            sig_0_bot = {}
            
            # --- Call the AIEM model ---
            if self.RT_s == 'AIEM':
                from .surface.aiem import AIEM

                lambda_m = toLambda(self.f)
                k = 2.0 * np.pi / lambda_m
                surface_map = {'gauss': 1, 'exp': 2, 'pow': 3}
                itype = surface_map.get(self.acftype, 1)
                phi_rel = (self.phi_s - self.phi_i) % 360

                eps_top = np.atleast_1d(self.eps2)[0]
                hh_top_db, vv_top_db, hv_top_db, vh_top_db = AIEM(
                    theta_i=self.theta_i,
                    theta_s=self.theta_s,
                    phi_s=phi_rel,
                    err=float(np.real(eps_top)),
                    eri=float(np.imag(eps_top)),
                    itype=itype,
                    addMultiple=False,
                    frequency_ghz=float(self.f),
                    k0=k,
                    sigma=float(self.s),
                    corr_len=float(self.cl),
                )
                sig_0_top = {
                    'vv': toPower(vv_top_db),
                    'hh': toPower(hh_top_db),
                    'hv': toPower(hv_top_db),
                    'vh': toPower(vh_top_db),
                }

                eps_bot = np.atleast_1d(self.eps_ratio)[0]
                hh_bot_db, vv_bot_db, hv_bot_db, vh_bot_db = AIEM(
                    theta_i=self.theta_i,
                    theta_s=self.theta_s,
                    phi_s=phi_rel,
                    err=float(np.real(eps_bot)),
                    eri=float(np.imag(eps_bot)),
                    itype=itype,
                    addMultiple=False,
                    frequency_ghz=float(self.f),
                    k0=k,
                    sigma=float(self.s),
                    corr_len=float(self.cl),
                )
                sig_0_bot = {
                    'vv': toPower(vv_bot_db),
                    'hh': toPower(hh_bot_db),
                    'hv': toPower(hv_bot_db),
                    'vh': toPower(vh_bot_db),
                }
            
            # --- Call the PRISM1 model ---
            elif self.RT_s == 'PRISM1':
                
                
                # --- Import the PRISM1 model ---
                from .surface.prism1 import PRISM1
                # --- Call the PRISM1 model for the Rayleigh layer ---
                prism0 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps2, s=self.s)
                sig_0_top_full = prism0.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the PRISM1 model for the ground surface ---
                prism1 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps_ratio, s=self.s)
                sig_0_bot_full = prism1.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            elif self.RT_s == 'SPM3D':
                from .surface.spm import SPM3D

                spm_top = SPM3D(fr=self.f, sig=self.s, L=self.cl, thi=self.theta_i, eps=self.eps2)
                sig_0_top_full = spm_top.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))

                spm_bot = SPM3D(fr=self.f, sig=self.s, L=self.cl, thi=self.theta_i, eps=self.eps_ratio)
                sig_0_bot_full = spm_bot.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            
            elif self.RT_s == 'Dubois95':

                # --- Import the Dubois95 model ---
                from .surface.dubois95 import Dubois95
                # --- Call the Dubois95 model for the Rayleigh layer ---
                db95_top = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps2, s=self.s)
                sig_0_top_full = db95_top.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the Dubois95 model for the ground surface ---
                db95_bot = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps_ratio, s=self.s)
                sig_0_bot_full = db95_bot.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            
            elif self.RT_s == 'SMART':
                from .surface.smart import SMART
                # --- Call the SMART model for the Rayleigh layer ---
                smart_top = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps2)
                sig_0_top_full = smart_top.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the SMART model for the ground surface ---
                smart_bot = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps_ratio)
                sig_0_bot_full = smart_bot.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            elif self.RT_s == 'I2EM':
                from .surface.i2em import I2EM_Bistat_model

                sp_map = {'exp': 1, 'gauss': 2, 'pow': 3}
                sp = sp_map.get(self.acftype, 1)
                xx = 1.5 if sp == 3 else 0.0
                phi_rel = (self.phi_s - self.phi_i) % 360
                eps_top = float(np.real(np.atleast_1d(self.eps2)[0]))
                eps_bottom = float(np.real(np.atleast_1d(self.eps_ratio)[0]))

                vv_db, hh_db, hv_db, vh_db = I2EM_Bistat_model(
                    fr=self.f,
                    sig=self.s,
                    L=self.cl,
                    thi=self.theta_i,
                    ths=self.theta_s,
                    phs=phi_rel,
                    er=eps_top,
                    sp=sp,
                    xx=xx
                )
                sig_0_top = {
                    'vv': toPower(vv_db),
                    'hh': toPower(hh_db),
                    'hv': toPower(hv_db),
                    'vh': toPower(vh_db)
                }

                vv_db, hh_db, hv_db, vh_db = I2EM_Bistat_model(
                    fr=self.f,
                    sig=self.s,
                    L=self.cl,
                    thi=self.theta_i,
                    ths=self.theta_s,
                    phs=phi_rel,
                    er=eps_bottom,
                    sp=sp,
                    xx=xx
                )
                sig_0_bot = {
                    'vv': toPower(vv_db),
                    'hh': toPower(hh_db),
                    'hv': toPower(hv_db),
                    'vh': toPower(vh_db)
                }
            else:
                raise ValueError("RT_s must be 'AIEM' or 'PRISM1'")
            
            # --- Call the Rayleigh model ---
            sig_t = self.__S2RTR_SpecularUB_FullPol(
                sig_0_top, sig_0_bot, self.eps,self.eps2, self.eps3, self.a, self.kappa_e, self.d, self.theta_i, todB=todB)
            
            if todB:
                # Convert to dB
                sig_12_dB = {pq: toDB(sig_0_top[pq]) for pq in pol_list}
                sig_23_dB = {pq: toDB(sig_0_bot[pq]) for pq in pol_list}
                return sig_12_dB, sig_23_dB, sig_t
            else:
                return sig_0_top, sig_0_bot, sig_t
        else:
            raise ValueError("RT_c must be 'Diff' or 'Spec'")


    def __S2RTR_DiffuseUB_FullPol(self, sig_s, eps, a, kappa_e, d, theta_i, n=2, todB=True):
        """
        Computes σ₀ for all polarizations (vv, hh, hv, vh) for a weakly
        scattering Rayleigh layer with a diffuse upper boundary.
        
        Parameters:
            sig_s : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            eps : complex
                Complex dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            theta_i : float
                Incidence angle in degrees.
            n : int
                1 if the scatterer was incohirent, 2 if coherent.
            todB : bool
                If True, returns the results in dB. If False, returns in Power.
        
        Returns:
            A dictionary of σ₀ in dB for:
                'vv', 'hh', 'hv', 'vh'
        """
        theta_rad = np.radians(theta_i)
        cos_theta = np.cos(theta_rad)
        tau = (kappa_e * d) / cos_theta
        T = np.exp(-tau)
        T2 = T ** 2
        kappa_s = a * kappa_e

        # --- Reflectivity (Fresnel) ---
        _, _, gammah, gammav, *_ = ReflTransm_PlanarBoundary(1, eps, theta_i)
        
        # --- Scattering coefficient ---
        sigma_0_db = {}

        # Apply generalized equation
        sigma_0_vv = (
            T2 * sig_s['vv']
            + 0.75 * a * cos_theta * (1 - T2) * (1 + gammav**2 * T2)
            + 3 * n * kappa_s * d * gammav * T2
        )
        sigma_0_hh = (
            T2 * sig_s['hh']
            + 0.75 * a * cos_theta * (1 - T2) * (1 + gammah**2 * T2)
            + 3 * n * kappa_s * d * gammah * T2
        )
        sigma_0_hv = T2 * sig_s['hv']
        sigma_0_vh = T2 * sig_s['vh']
        
        # Convert to dB
        if todB:
            sigma_0_db['vv'] = toDB(sigma_0_vv)
            sigma_0_db['hh'] = toDB(sigma_0_hh)
            sigma_0_db['hv'] = toDB(sigma_0_hv)
            sigma_0_db['vh'] = toDB(sigma_0_vh)
        else:
            sigma_0_db['vv'] = sigma_0_vv
            sigma_0_db['hh'] = sigma_0_hh
            sigma_0_db['hv'] = sigma_0_hv
            sigma_0_db['vh'] = sigma_0_vh


        return sigma_0_db


    def __S2RTR_SpecularUB_FullPol(self, sig_0_top, sig_0_bot, eps, eps2, eps3, a, kappa_e, d, theta_i, n=2, todB=True):
        """
        Computes sigma_0 for all polarizations (vv, hh, hv, vh) for a weakly scattering
        Rayleigh layer with distinct upper boundary using PRISM models.

        Parameters:
            sig_0_top : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            sig_0_bot : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            eps2 : complex
                Complex dielectric constant of the Rayleigh layer.
            eps3 : complex
                Complex dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            theta_i : float
                Incidence angle in degrees.
            n : int
                1 if the scatterer was incoherent, 2 if coherent.
            todB : bool
                If True, returns the results in dB. If False, returns in Power.

        Returns:
            Dictionary with σ₀ in dB for polarizations: 'vv', 'hh', 'hv', 'vh'.
        """
        theta_rad = np.radians(theta_i)
        sin_theta = np.sin(theta_rad)

        # Transmission angle inside Rayleigh layer
        thetapr_rad = np.arcsin(np.sqrt(1 / eps2.real) * sin_theta)
        thetapr_deg = np.degrees(thetapr_rad)
        costhetapr = np.sqrt(1 - (1 / eps2.real) * sin_theta**2)

        # Scattering coefficient
        kappa_s = a * kappa_e

        # Transmissivity
        tau = kappa_e * d / costhetapr
        T = np.exp(-tau)
        T2 = T ** 2

        # Reflectivity and transmissivity of top boundary (air -> layer)
        _, _, _, _, _, _, Th_12, Tv_12 = ReflTransm_PlanarBoundary(eps, eps2, theta_i)

        # Reflectivity of bottom boundary (layer -> ground)
        _, _, gammah_23, gammav_23, *_ = ReflTransm_PlanarBoundary(eps2, eps3, thetapr_deg)

        # Compute σ₀ for all polarization combinations
        sigma_0 = {}

        # Generalized version of Eq. 11.79
        sigma_0_vv = (Tv_12**2 * (T2 * sig_0_bot['vv'] + 0.75*a * costhetapr * (1 - T2) * 
                 (1 + gammav_23**2 * T2) + 3 * n * kappa_s * d * gammav_23 * T2) + 
                 sig_0_top['vv'])
    
        sigma_0_hh = (Th_12**2 * (T2 * sig_0_bot['hh'] + 0.75*a * costhetapr * (1 - T2) * 
                    (1 + gammah_23**2 * T2) + 3 * n * kappa_s * d * gammah_23 * T**2) + 
                    sig_0_top['hh'])
        
        sigma_0_vh = (Tv_12 * Th_12 * (T2 * sig_0_bot['vh'] + T2 * sig_0_top['vh']))
        
        sigma_0_hv = (Tv_12 * Th_12 * (T2 * sig_0_bot['hv'] + T2 * sig_0_top['hv']))
        
        # Convert to dB
        if todB:
            sigma_0['vv'] = toDB(sigma_0_vv)
            sigma_0['hh'] = toDB(sigma_0_hh)
            sigma_0['hv'] = toDB(sigma_0_hv)
            sigma_0['vh'] = toDB(sigma_0_vh)
        else:
            sigma_0['vv'] = sigma_0_vv
            sigma_0['hh'] = sigma_0_hh
            sigma_0['hv'] = sigma_0_hv
            sigma_0['vh'] = sigma_0_vh

        return sigma_0
