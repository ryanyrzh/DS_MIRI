#!/usr/bin/env python3
"""Parallel NIRSpec TLUSTY SMDS fits (optional Bagpipes galaxy joint fit).

Fitting routine follows organized_notebooks/01_spectral_fitting.ipynb.
Plots extend model curves to 12 um so the MIRI-band prediction is visible.
For JADES-GS-z14-0, the combined MIRI/LRS x1d (program ID 8544) is stitched onto the plot.
Fits are run on both NIRSpec alone and NIRSpec+MIRI.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize

plt.rcParams.update({"font.size": 14})


class _HandlerLineOverFill(HandlerBase):
    """Legend handle: gray band with spectrum line drawn through it."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans,
    ):
        fill_color, line_color, line_lw = orig_handle
        band = Rectangle(
            (-xdescent, -ydescent), width, height,
            facecolor=fill_color, edgecolor="none", transform=trans,
        )
        midline = Line2D(
            [-xdescent, -xdescent + width],
            [-ydescent + 0.5 * height, -ydescent + 0.5 * height],
            color=line_color, lw=line_lw, transform=trans,
        )
        return [band, midline]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GALAXY_UNIT_SCALE = 1e21
SMDS_UNIT_SCALE = 1.0
COSMO_SMDS = dict(H0=72.0, Om=0.27)
COSMO_GALAXY = dict(H0=70.0, Om=0.3)
C_CGS = 2.99792458e10
# SMDS-only uses a tighter upper wall than the joint fit
LOGM_DS_MIN = 5.0
LOGM_DS_MAX_SMDS = 7.95
LOGM_DS_MAX_JOINT = 8.5
LOGM_GAL_MIN, LOGM_GAL_MAX = 6.0, 11.0
PLOT_LAM_MAX_UM = 12.0
NIRSPEC_R = 100.0  # prism resolving power for the galaxy kernel
N_PLOT = 2000

TEMPLATE_GRIDS = {
    "T3": {
        "mass_1e5": np.array([1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70, 90], float),
        "files": [
            "T3M1_0e5.13", "T3M2_0e5.13", "T3M3_0e5.13", "T3M4_0e5.13",
            "T3M5_0e5.13", "T3M6_0e5.13", "T3M7_0e5.13", "T3M1_0e6.13",
            "T3M2_0e6.13", "T3M3_0e6.13", "T3M4_0e6.13", "T3M5_0e6.13",
            "T3M6_0e6.13", "T3M7_0e6.13", "T3M9_0e6.13",
        ],
    },
    "AC": {
        "mass_1e5": np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90], float
        ),
        "files": [
            "T3M1_0e5.13", "T3M2_0e5.13", "T3M3_0e5.13", "T3M4_0e5.13",
            "T3M5_0e5.13", "T3M6_0e5.13", "T3M7_0e5.13", "T3M8_0e5.13",
            "T3M9_0e5.13", "T3M1_0e6.13", "ACM2_00e+06.13", "ACM3_00e+06.13",
            "ACM4_00e+06.13", "ACM5_00e+06.13", "ACM6_00e+06.13", "ACM7_00e+06.13",
            "ACM8_00e+06.13", "ACM9_00e+06.13",
        ],
    },
}
TEMPLATE_SET = "T3"

_M_ANCHOR = np.array([1e5, 1e6, 1e7])
_LOGM_ANCHOR = np.log10(_M_ANCHOR)
_LOGR_ANCHOR = np.log10(np.array([5.83e14, 9.12e14, 1.45e15]))
_LOGG_ANCHOR = np.array([1.591, 2.202, 2.8])
_LOGT_ANCHOR = np.log10(np.array([14000.0, 19000.0, 27000.0]))
_interp_logR = interp1d(_LOGM_ANCHOR, _LOGR_ANCHOR, kind="linear", fill_value="extrapolate")
_interp_logg = interp1d(_LOGM_ANCHOR, _LOGG_ANCHOR, kind="linear", fill_value="extrapolate")
_interp_logT = interp1d(_LOGM_ANCHOR, _LOGT_ANCHOR, kind="linear", fill_value="extrapolate")

REPO_ROOT = Path("/home/yzhan629/DS_MIRI")
TEMPLATE_DIR = REPO_ROOT / "tlusty_templates"
CSV_DIR = REPO_ROOT / "data" / "nirspec_spectra_csv"
# Paper-extracted 1D spectra for the two z~14 candidates
EXTRACTED_CSV_DIR = REPO_ROOT / "data" / "nirspec_candidates_extracted_spectra"
OUTDIR = REPO_ROOT / "production" / "nirspec_candidates"
PLOT_DIR = OUTDIR / "plots"
NPZ_DIR = OUTDIR / "npz"
# Combined MIRI x1d for the one candidate that has MIRI spectroscopy
MIRI_X1D_BY_LABEL = {
    "JADES_GS_z14_0": REPO_ROOT / "data" / "jw08544_obsAll_t001_miri_p750l_x1d.fits",
}

# Four Ilie+2025 NIRSpec DS candidates
TARGETS = [
    dict(
        name="JADES-GS-z14-0",
        label="JADES_GS_z14_0",
        root="jades-gds10-v4",
        srcid=183348,
        z=14.1793,
        z_lit=14.18,
        csv_name="JADES_GS_z14_0_spec1D.csv",
        csv_dir=EXTRACTED_CSV_DIR,
    ),
    dict(
        name="JADES-GS-z14-1",
        label="JADES_GS_z14_1",
        root="jades-gds10-v4",
        srcid=20018044,
        z=14.0801,
        z_lit=13.90,
        csv_name="JADES_GS_z14_1_spec1D.csv",
        csv_dir=EXTRACTED_CSV_DIR,
    ),
    dict(
        name="JADES-GS-z13",
        label="JADES_GS_z13",
        root="gds-udeep-v4",
        srcid=20128771,
        z=13.117,
        z_lit=13.20,
        csv_name="JADES_GS_z13_nirspec.csv",
    ),
    dict(
        name="JADES-GS-z11",
        label="JADES_GS_z11",
        root="gds-deep-v4",
        srcid=14220,
        z=11.5936,
        z_lit=11.58,
        csv_name="JADES_GS_z11_nirspec.csv",
    ),
]


# ---------------------------------------------------------------------------
# Distance / units
# ---------------------------------------------------------------------------
def luminosity_distance_cm(z, H0=70.0, Om=0.3):
    # Flat LCDM luminosity distance in cm. z=0 returns 0.
    z = float(z)
    if z <= 0:
        return 0.0
    H0_si = H0 * 1e5 / 3.085677581e24
    Ol = 1.0 - Om

    def inv_e(zp):
        return 1.0 / np.sqrt(Om * (1.0 + zp) ** 3 + Ol)

    chi = quad(inv_e, 0.0, z, epsabs=0, epsrel=1e-6)[0]
    return (1.0 + z) * (C_CGS / H0_si) * chi


def nJy_to_flambda_1e21(nJy, wavelength_um):
    # nJy -> f_lambda [erg/s/cm^2/A] * 1e21
    jy = np.asarray(nJy, float) * 1e-9
    f_nu = jy * 1e-23
    lam_cm = np.asarray(wavelength_um, float) * 1e-4
    f_per_cm = f_nu * (C_CGS / lam_cm ** 2)
    f_per_A = f_per_cm * 1e-8
    return f_per_A * 1e21


def jy_to_flambda_1e21(jy, wavelength_um):
    # Jy -> f_lambda [erg/s/cm^2/A] * 1e21 (same units as NIRSpec CSVs)
    f_nu = np.asarray(jy, float) * 1e-23
    lam_cm = np.asarray(wavelength_um, float) * 1e-4
    f_per_cm = f_nu * (C_CGS / lam_cm ** 2)
    return f_per_cm * 1e-8 * 1e21


def load_miri_x1d(path, lam_min_um=5.0, lam_max_um=PLOT_LAM_MAX_UM):
    # MIRI L3/Helton x1d EXTRACT1D: WAVELENGTH [um], FLUX/FLUX_ERROR [Jy]
    from astropy.io import fits

    path = Path(path)
    if not path.is_file():
        path = REPO_ROOT / path
    with fits.open(path) as hdul:
        spectrum = hdul["EXTRACT1D"].data
        wavelength_um = np.asarray(spectrum["WAVELENGTH"], float)
        flux_jy = np.asarray(spectrum["FLUX"], float)
        err_jy = np.asarray(spectrum["FLUX_ERROR"], float)
    flux = jy_to_flambda_1e21(flux_jy, wavelength_um)
    err = jy_to_flambda_1e21(err_jy, wavelength_um)
    # x1d often stores long -> short; sort blue -> red for plotting
    order = np.argsort(wavelength_um)
    wavelength_um = wavelength_um[order]
    flux = flux[order]
    err = err[order]
    good = (
        np.isfinite(wavelength_um)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (err > 0)
        & (wavelength_um >= lam_min_um)
        & (wavelength_um <= lam_max_um)
    )
    return wavelength_um[good], flux[good], err[good]


# ---------------------------------------------------------------------------
# TLUSTY / SMDS
# ---------------------------------------------------------------------------
def fortran_float(value):
    return float(str(value).strip().replace("D", "E").replace("d", "e"))


def load_tlusty(path):
    raw = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    freq = np.array([fortran_float(x) for x in raw.iloc[:, 0]])
    flux = np.array([fortran_float(x) for x in raw.iloc[:, 1]])
    return freq, flux


def radius_cm(mass_msun):
    return float(10.0 ** _interp_logR(np.log10(mass_msun)))


def log_g(mass_msun):
    return float(_interp_logg(np.log10(mass_msun)))


def teff(mass_msun):
    return float(10.0 ** _interp_logT(np.log10(mass_msun)))


def redshift_tlusty(freq, surf_flux, z, radius, mu=1.0, lam_max_um=30.0):
    lam_um = (C_CGS / freq) * (1.0 + z) * 1e4
    keep = np.isfinite(lam_um) & np.isfinite(surf_flux) & (lam_um > 0) & (lam_um <= lam_max_um)
    lam_um = lam_um[keep]
    surf = surf_flux[keep]
    order = np.argsort(lam_um)
    lam_um = lam_um[order]
    surf = surf[order]
    keep_unique = np.concatenate([[True], np.diff(lam_um) > 0])
    lam_um = lam_um[keep_unique]
    surf = surf[keep_unique]

    if z == 0:
        f_nJy = 4.0 * np.pi * surf * 1e32
    else:
        dl = luminosity_distance_cm(z, **COSMO_SMDS)
        f_nJy = 4.0 * np.pi * surf * (1.0 + z) * radius ** 2 / dl ** 2 * 1e32
    f_nJy *= mu
    f_lam = nJy_to_flambda_1e21(f_nJy, lam_um)
    lya = 0.1216 * (1.0 + z)
    f_lam = np.where(lam_um > lya, f_lam, 0.0)
    return lam_um, f_lam


def nearest_template(mass_msun, template_set=None):
    grid = TEMPLATE_GRIDS[template_set or TEMPLATE_SET]
    mass_1e5 = mass_msun / 1e5
    idx = int(np.argmin(np.abs(grid["mass_1e5"] - mass_1e5)))
    m_grid = float(grid["mass_1e5"][idx])
    return {
        "file": grid["files"][idx],
        "mass_1e5": m_grid,
        "mu": mass_1e5 / m_grid,
        "path": TEMPLATE_DIR / grid["files"][idx],
    }


def spectrum_from_mass_z(mass_msun, z, obs_lambda_um, template_set=None,
                         min_um=0.5, max_um=30.0):
    meta = nearest_template(mass_msun, template_set)
    freq, surf = load_tlusty(meta["path"])
    r_cm = radius_cm(mass_msun)
    lam, fl = redshift_tlusty(freq, surf, z, r_cm, mu=meta["mu"], lam_max_um=max_um)
    sel = (lam >= min_um) & (lam <= max_um)
    interp = interp1d(lam[sel], fl[sel], kind="linear", bounds_error=False, fill_value="extrapolate")
    meta.update({
        "R_cm": r_cm, "z": z, "mass_msun": float(mass_msun),
        "logg": log_g(mass_msun), "Teff": teff(mass_msun),
    })
    return np.asarray(interp(np.asarray(obs_lambda_um, float)), float), meta


# ---------------------------------------------------------------------------
# BAGPIPES galaxy (imported lazily so SMDS-only runs without bagpipes)
# ---------------------------------------------------------------------------
_bagpipes_ready = False


def _ensure_bagpipes():
    global _bagpipes_ready
    if _bagpipes_ready:
        return
    import bagpipes as pipes  # noqa: F401
    import bagpipes.config as cfg

    cfg.max_redshift = 30
    for flag in ("apply_igm", "use_igm", "include_igm"):
        if hasattr(cfg, flag):
            setattr(cfg, flag, False)
    # Keep module refs for galaxy_spectrum
    globals()["pipes"] = pipes
    _bagpipes_ready = True


def constant_R_grid(lam_min_A, lam_max_A, R, pix_per_fwhm=3.0):
    dln = (1.0 / R) / pix_per_fwhm
    grid = np.exp(np.arange(np.log(lam_min_A), np.log(lam_max_A) + dln, dln))
    return grid, dln


def galaxy_spectrum(
    z, logM, age, tau, Z, Av, logU, fesc,
    *,
    R=100.0,
    lam_min_A=0.6e4,
    lam_max_A=12.0e4,
    lya_cap_factor=3.0,
    lya_window_kms=600.0,
    apply_igm=True,
    veldisp_kms=80.0,
    eta_birthcloud=3.0,
):
    from scipy.ndimage import gaussian_filter1d

    _ensure_bagpipes()
    dln = (1.0 / R) / 3.0
    pad = 1.05
    lam_rest = np.exp(np.arange(
        np.log((lam_min_A / (1.0 + z)) / pad),
        np.log((lam_max_A / (1.0 + z)) * pad) + dln,
        dln,
    ))
    lam_obs, dln_obs = constant_R_grid(lam_min_A, lam_max_A, R)

    comp = {
        "redshift": 0.0,
        "t_bc": 0.01,
        "veldisp": veldisp_kms,
        "exponential": {"age": age, "tau": tau, "massformed": logM, "metallicity": Z},
        "dust": {"type": "Calzetti", "Av": Av, "eta": eta_birthcloud},
        "nebular": {"logU": logU, "f_esc": fesc},
    }
    spec = np.asarray(pipes.model_galaxy(comp, spec_wavs=lam_rest).spectrum)
    lam_rf, L_rf = spec[:, 0], spec[:, 1]
    lam_of = lam_rf * (1.0 + z)
    L_of = L_rf / (1.0 + z)
    f_L = np.interp(lam_obs, lam_of, L_of, left=0.0, right=0.0)
    flux = f_L / (4.0 * np.pi * luminosity_distance_cm(z, **COSMO_GALAXY) ** 2)

    lya = 1215.67 * (1.0 + z)
    if apply_igm:
        flux = np.where(lam_obs >= lya, flux, 0.0)

    c_kms = 2.99792458e5
    ln_lam = np.log(lam_obs)
    core = np.abs(ln_lam - np.log(lya)) <= (lya_window_kms / c_kms)
    if np.any(core):
        shoulder = (
            (ln_lam >= np.log(lya) + 1000.0 / c_kms)
            & (ln_lam <= np.log(lya) + 5000.0 / c_kms)
        )
        if not np.any(shoulder):
            shoulder = (lam_obs >= 1.01 * lya) & (lam_obs <= 1.03 * lya)
        cont = np.median(flux[shoulder]) if np.any(shoulder) else np.median(flux)
        cont = max(float(cont), 1e-40)
        flux = flux.copy()
        flux[core] = np.minimum(flux[core], lya_cap_factor * cont)

    sigma_pix = ((1.0 / R) / 2.355) / dln_obs
    return lam_obs, gaussian_filter1d(flux, sigma_pix, mode="nearest")


def model_flux_at_obs_lambda(obs_lambda_um, params, **kwargs):
    lam_A, flux = galaxy_spectrum(**params, **kwargs)
    order = np.argsort(lam_A)
    interp = interp1d(
        lam_A[order] / 1e4, flux[order], kind="linear",
        bounds_error=False, fill_value=0.0,
    )
    return interp(np.asarray(obs_lambda_um, float))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_smds_mass(
    obs_lambda_um, obs_flux, z,
    obs_err=None,
    x0_logM=6.2,
    lya_buffer=1.10,
    logm_min=LOGM_DS_MIN,
    logm_max=LOGM_DS_MAX_SMDS,
    template_set=None,
    nm_options=None,
):
    if obs_err is None:
        inv_var = None
    else:
        err = np.asarray(obs_err, float)
        inv_var = 1.0 / np.maximum(err, 1e-30) ** 2

    def loss(x):
        logM = float(x[0])
        # Soft wall: outside [logm_min, logm_max], add 1e4 * (distance past the wall)^2
        pen = 0.0
        if logM < logm_min:
            pen += (logm_min - logM) ** 2 * 1e4
        if logM > logm_max:
            pen += (logM - logm_max) ** 2 * 1e4
        try:
            model, _ = spectrum_from_mass_z(10.0 ** logM, z, obs_lambda_um, template_set)
        except Exception:
            return 1e50
        # Ignore wavelengths blueward of 1.10 * Lyman-alpha
        mask = obs_lambda_um >= lya_buffer * 0.1216 * (1.0 + z)
        if not np.any(mask):
            return 1e50
        resid = (model * SMDS_UNIT_SCALE)[mask] - obs_flux[mask]
        if inv_var is None:
            return float(np.mean(resid * resid) + pen)
        return float(np.mean(resid * resid * inv_var[mask]) + pen)

    res = minimize(
        loss, np.array([x0_logM]), method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4, **(nm_options or {})},
    )
    model, meta = spectrum_from_mass_z(10.0 ** res.x[0], z, obs_lambda_um, template_set)
    smds_flux = model * SMDS_UNIT_SCALE
    return {
        "result": res,
        "log10M_ds": float(res.x[0]),
        "smds_meta": meta,
        "smds_flux": smds_flux,
        "galaxy_flux": None,
        "total_flux": smds_flux,
        "galaxy": None,
    }


def _to_vec(params, keys, log_keys):
    return np.array([
        np.log10(params[k]) if k in log_keys else float(params[k]) for k in keys
    ])


def _from_vec(x, base, keys, log_keys):
    out = dict(base)
    for k, v in zip(keys, x):
        out[k] = float(10.0 ** v) if k in log_keys else float(v)
    return out


def fit_joint(
    obs_lambda_um, obs_flux, base_galaxy, z,
    obs_err=None,
    gal_keys=("logM", "age", "tau", "Z", "Av", "logU", "fesc"),
    gal_log_keys=("age", "tau", "Z"),
    x0_logM_ds=6.0,
    lya_buffer=1.10,
    template_set=None,
    nm_options=None,
    **spec_kwargs,
):
    _ensure_bagpipes()
    gal_keys = tuple(gal_keys)
    gal_log_keys = set(gal_log_keys)
    x0 = np.concatenate([_to_vec(base_galaxy, gal_keys, gal_log_keys), [x0_logM_ds]])
    if obs_err is None:
        inv_var = None
    else:
        err = np.asarray(obs_err, float)
        inv_var = 1.0 / np.maximum(err, 1e-30) ** 2

    def loss(x):
        logM_ds = float(x[-1])
        pen = 0.0
        if logM_ds < LOGM_DS_MIN:
            pen += (LOGM_DS_MIN - logM_ds) ** 2 * 1e4
        if logM_ds > LOGM_DS_MAX_JOINT:
            pen += (logM_ds - LOGM_DS_MAX_JOINT) ** 2 * 1e4

        gal = _from_vec(x[:-1], base_galaxy, gal_keys, gal_log_keys)
        gal["z"] = float(z)
        logM_gal = float(gal["logM"])
        if logM_gal < LOGM_GAL_MIN:
            pen += (LOGM_GAL_MIN - logM_gal) ** 2 * 1e4
        if logM_gal > LOGM_GAL_MAX:
            pen += (logM_gal - LOGM_GAL_MAX) ** 2 * 1e4

        fesc = float(gal["fesc"])
        if fesc < 0.0:
            pen += (0.0 - fesc) ** 2 * 1e4
        if fesc > 1.0:
            pen += (fesc - 1.0) ** 2 * 1e4

        try:
            g = model_flux_at_obs_lambda(obs_lambda_um, gal, **spec_kwargs) * GALAXY_UNIT_SCALE
            s, _ = spectrum_from_mass_z(10.0 ** logM_ds, z, obs_lambda_um, template_set)
        except Exception:
            return 1e50
        mask = obs_lambda_um >= lya_buffer * 0.1216 * (1.0 + z)
        resid = (g + s * SMDS_UNIT_SCALE)[mask] - obs_flux[mask]
        if inv_var is None:
            return float(np.mean(resid * resid) + pen)
        return float(np.mean(resid * resid * inv_var[mask]) + pen)

    res = minimize(
        loss, x0, method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-6, **(nm_options or {})},
    )
    gal = _from_vec(res.x[:-1], base_galaxy, gal_keys, gal_log_keys)
    gal["z"] = float(z)
    logM_ds = float(res.x[-1])
    g = model_flux_at_obs_lambda(obs_lambda_um, gal, **spec_kwargs) * GALAXY_UNIT_SCALE
    s, meta = spectrum_from_mass_z(10.0 ** logM_ds, z, obs_lambda_um, template_set)
    return {
        "result": res,
        "galaxy": gal,
        "log10M_ds": logM_ds,
        "smds_meta": meta,
        "galaxy_flux": g,
        "smds_flux": s * SMDS_UNIT_SCALE,
        "total_flux": g + s * SMDS_UNIT_SCALE,
    }


def evaluate_models_on_grid(out, z, lam_grid_um, mode, **spec_kwargs):
    # Re-evaluate best-fit model(s) on an extended wavelength grid for plotting
    s, _ = spectrum_from_mass_z(10.0 ** out["log10M_ds"], z, lam_grid_um)
    s = s * SMDS_UNIT_SCALE
    if mode == "joint":
        gal = dict(out["galaxy"])
        gal["z"] = float(z)
        g = model_flux_at_obs_lambda(lam_grid_um, gal, **spec_kwargs) * GALAXY_UNIT_SCALE
        return g, s, g + s
    return None, s, s


def sfr_exp(logM, age_gyr, tau_gyr):
    M = 10.0 ** np.asarray(logM, float)
    a = np.asarray(age_gyr, float) / np.asarray(tau_gyr, float)
    sfr_per_gyr = M * np.exp(-a) / (np.asarray(tau_gyr, float) * (-np.expm1(-a)))
    return sfr_per_gyr / 1e9


def sfr_delayed(logM, age_gyr, tau_gyr):
    M = 10.0 ** np.asarray(logM, float)
    age = np.asarray(age_gyr, float)
    tau = np.asarray(tau_gyr, float)
    a = age / tau
    ea = np.exp(-a)
    sfr_per_gyr = M * (age / tau ** 2) * ea / (1.0 - (1.0 + a) * ea)
    return sfr_per_gyr / 1e9


def load_spectrum_csv(path):
    # Returns wavelength [um], flux and Flux_Error in f_lambda * 1e21 units.
    path = Path(path)
    if not path.is_file():
        path = REPO_ROOT / path
    df = pd.read_csv(path)
    wave_col = "Wavelength" if "Wavelength" in df.columns else df.columns[0]
    flux_col = " Flux" if " Flux" in df.columns else "Flux"
    lam = df[wave_col].to_numpy(float)
    flux = df[flux_col].to_numpy(float)
    err_col = None
    for name in ("Flux_Error", " Flux_Error", "flux_error", "err"):
        if name in df.columns:
            err_col = name
            break
    if err_col is None:
        raise ValueError(f"no Flux_Error column in {path}")
    err = df[err_col].to_numpy(float)
    return lam, flux, err


# ---------------------------------------------------------------------------
# Per-source worker
# ---------------------------------------------------------------------------
def fit_one_source(job):
    """Fit a single candidate. `job` is a plain dict (pickle-safe)."""
    label = job["label"]
    mode = job["mode"]
    t0 = time.time()
    # Keep BLAS single-threaded inside each worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        lam, flux, err = load_spectrum_csv(job["spectrum_csv"])
        good = np.isfinite(lam) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
        lam, flux, err = lam[good], flux[good], err[good]
        if lam.size == 0:
            raise ValueError("no finite pixels with err>0")
        z = float(job["z"])
        R = float(job["R"])
        nm_options = {"maxiter": int(job["maxiter"])}

        # Galaxy / plot grid spans data blue edge through MIRI 12 um
        lam_min_A = float(np.min(lam) * 1e4 * 0.95)
        lam_max_A = PLOT_LAM_MAX_UM * 1e4
        spec_kwargs = dict(
            R=R,
            lam_min_A=lam_min_A,
            lam_max_A=lam_max_A,
            apply_igm=True,
            lya_cap_factor=1.0,
            lya_window_kms=600.0,
            veldisp_kms=80.0,
        )

        if mode == "joint":
            base = dict(
                z=z, logM=7.8, age=0.03, tau=0.03,
                Z=0.01, Av=0.15, logU=-2.5, fesc=0.0,
            )
            out = fit_joint(
                lam, flux, base, z,
                obs_err=err,
                x0_logM_ds=6.0,
                lya_buffer=1.10,
                nm_options=nm_options,
                **spec_kwargs,
            )
            tag = "joint_fit"
        else:
            out = fit_smds_mass(
                lam, flux, z,
                obs_err=err,
                x0_logM=6.2,
                lya_buffer=1.10,
                nm_options=nm_options,
            )
            tag = "smds_fit"

        # Extended wavelength grid for the long-wavelength prediction plot
        lam_plot = np.linspace(float(np.min(lam)), PLOT_LAM_MAX_UM, N_PLOT)
        g_plot, s_plot, t_plot = evaluate_models_on_grid(
            out, z, lam_plot, mode, **spec_kwargs,
        )

        # MIRI stitch
        miri_lam = miri_flux = miri_err = None
        out_nm = None
        g_plot_nm = s_plot_nm = t_plot_nm = None
        miri_x1d = job.get("miri_x1d") or ""
        if miri_x1d:
            miri_lam, miri_flux, miri_err = load_miri_x1d(miri_x1d)
            stitch = miri_lam > float(np.max(lam))
            miri_lam, miri_flux, miri_err = (
                miri_lam[stitch], miri_flux[stitch], miri_err[stitch],
            )
            if miri_lam.size == 0:
                miri_lam = miri_flux = miri_err = None
            else:
                # Concatenate NIRSpec + MIRI for the combined-wavelength fit
                lam_nm = np.concatenate([lam, miri_lam])
                flux_nm = np.concatenate([flux, miri_flux])
                err_nm = np.concatenate([err, miri_err])
                if mode == "joint":
                    base_nm = dict(
                        z=z, logM=7.8, age=0.03, tau=0.03,
                        Z=0.01, Av=0.15, logU=-2.5, fesc=0.0,
                    )
                    out_nm = fit_joint(
                        lam_nm, flux_nm, base_nm, z,
                        obs_err=err_nm,
                        x0_logM_ds=6.0,
                        lya_buffer=1.10,
                        nm_options=nm_options,
                        **spec_kwargs,
                    )
                else:
                    out_nm = fit_smds_mass(
                        lam_nm, flux_nm, z,
                        obs_err=err_nm,
                        x0_logM=6.2,
                        lya_buffer=1.10,
                        nm_options=nm_options,
                    )
                g_plot_nm, s_plot_nm, t_plot_nm = evaluate_models_on_grid(
                    out_nm, z, lam_plot, mode, **spec_kwargs,
                )

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = str(PLOT_DIR / f"{label}_{tag}.png")
        lya_cut = 1.10 * 0.1216 * (1.0 + z)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.fill_between(
            lam, flux - err, flux + err,
            facecolor="0.8", edgecolor="none", linewidth=0,
        )
        ax.plot(lam, flux, lw=0.8, color="0.35")
        if miri_lam is not None:
            ax.fill_between(
                miri_lam, miri_flux - miri_err, miri_flux + miri_err,
                facecolor="0.8", edgecolor="none", linewidth=0,
            )
            ax.plot(miri_lam, miri_flux, lw=0.8, color="0.35")

        if mode == "joint":
            ax.plot(
                lam_plot, g_plot, lw=1, ls="-.", color="deepskyblue",
                label="Galaxy (NIRSpec)",
            )
            ax.plot(
                lam_plot, s_plot, lw=1, ls="-.", color="darkviolet",
                label="SMDS (NIRSpec)",
            )
            ax.plot(
                lam_plot, t_plot, lw=1.2, ls="-", color="coral",
                label="Total (NIRSpec)",
            )
            if t_plot_nm is not None:
                # Combined-fit total only (components clutter the panel)
                ax.plot(
                    lam_plot, t_plot_nm, lw=1.4, ls="--", color="limegreen",
                    label="Total (NIRSpec+MIRI)",
                )
        else:
            ax.plot(
                lam_plot, s_plot, lw=1.2, ls="-", color="darkviolet",
                label="SMDS (NIRSpec)",
            )
            if s_plot_nm is not None:
                ax.plot(
                    lam_plot, s_plot_nm, lw=1.4, ls="--", color="limegreen",
                    label="SMDS (NIRSpec+MIRI)",
                )
        ax.axvline(lya_cut, ls=":", color="k", alpha=0.5, label="fit boundary")
        ax.axvline(0.1640 * (1 + z), ls=":", color="c", alpha=0.5, label=r"He II 1640")
        ax.set_xlim(float(np.min(lam)), PLOT_LAM_MAX_UM)
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel(r"$f_\lambda \times 10^{21}$")
        logm_n = float(out["log10M_ds"])
        logm_nm = float(out_nm["log10M_ds"]) if out_nm is not None else None
        if logm_nm is not None and f"{logm_n:.2f}" != f"{logm_nm:.2f}":
            mass_title = rf"$\log M_{{\rm DS}}={logm_n:.2f}$, ${logm_nm:.2f}$"
        else:
            mass_title = rf"$\log M_{{\rm DS}}={logm_n:.2f}$"
        ax.set_title(rf"{job['name']}  $z={z:.3f}$  {mass_title}")
        fit_peak = float(np.nanmax(t_plot))
        if t_plot_nm is not None:
            fit_peak = max(fit_peak, float(np.nanmax(t_plot_nm)))
        if miri_flux is not None and miri_flux.size:
            # Keep MIRI emission features inside the frame
            fit_peak = max(fit_peak, float(np.nanpercentile(miri_flux, 99)))
        y_hi = fit_peak * 1.5
        ax.set_ylim(-0.3 * y_hi, y_hi)
        # Single legend entry: gray fill with spectrum line through it
        obs_proxy = ("0.8", "0.35", 0.8)  # fill, line color, line width
        model_handles, model_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=[obs_proxy] + model_handles,
            labels=[r"Observed $\pm 1\sigma$"] + model_labels,
            handler_map={tuple: _HandlerLineOverFill()},
            fontsize=11,
            ncol=1,
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = str(NPZ_DIR / f"{label}_{tag}.npz")
        save_kwargs = dict(
            lam=lam,
            flux=flux,
            err=err,
            lam_plot=lam_plot,
            smds_flux=out["smds_flux"],
            smds_plot=s_plot,
            total_plot=t_plot,
            z=z,
            R=R,
            log10M_ds=out["log10M_ds"],
            mode=mode,
        )
        if mode == "joint":
            save_kwargs["galaxy_flux"] = out["galaxy_flux"]
            save_kwargs["total_flux"] = out["total_flux"]
            save_kwargs["galaxy_plot"] = g_plot
        if out_nm is not None:
            save_kwargs["log10M_ds_nirspec_miri"] = out_nm["log10M_ds"]
            save_kwargs["smds_plot_nirspec_miri"] = s_plot_nm
            save_kwargs["total_plot_nirspec_miri"] = t_plot_nm
            if mode == "joint":
                save_kwargs["galaxy_plot_nirspec_miri"] = g_plot_nm
        np.savez_compressed(npz_path, **save_kwargs)

        row = {
            "label": label,
            "name": job["name"],
            "root": job["root"],
            "srcid": int(job["srcid"]),
            "z": z,
            "z_lit": float(job["z_lit"]),
            "mode": mode,
            "R": R,
            "success": bool(out["result"].success),
            "mse": float(out["result"].fun),
            "nfev": int(getattr(out["result"], "nfev", -1)),
            "nit": int(getattr(out["result"], "nit", -1)),
            "log10M_ds": float(out["log10M_ds"]),
            "log10M_ds_nirspec_miri": (
                float(out_nm["log10M_ds"]) if out_nm is not None else np.nan
            ),
            "mse_nirspec_miri": (
                float(out_nm["result"].fun) if out_nm is not None else np.nan
            ),
            "success_nirspec_miri": (
                bool(out_nm["result"].success) if out_nm is not None else False
            ),
            "message": str(out["result"].message),
            "runtime_sec": float(time.time() - t0),
            "error": "",
            "plot_path": plot_path,
            "npz_path": npz_path,
        }
        if mode == "joint":
            for key in ("logM", "age", "tau", "Z", "Av", "logU", "fesc"):
                row[f"gal_{key}"] = out["galaxy"][key]
                row[f"gal_{key}_nirspec_miri"] = (
                    out_nm["galaxy"][key] if out_nm is not None else np.nan
                )
            row["sfr_exp"] = float(sfr_exp(
                out["galaxy"]["logM"], out["galaxy"]["age"], out["galaxy"]["tau"],
            ))
            row["sfr_delayed"] = float(sfr_delayed(
                out["galaxy"]["logM"], out["galaxy"]["age"], out["galaxy"]["tau"],
            ))

        nm_msg = ""
        if out_nm is not None:
            nm_msg = (
                f" log10M_ds_NM={out_nm['log10M_ds']:.3f}"
                f" mse_NM={out_nm['result'].fun:.4g}"
            )
        print(
            f"[done] {label} mode={mode} success={row['success']} mse={row['mse']:.4g} "
            f"log10M_ds={row['log10M_ds']:.3f}{nm_msg} t={row['runtime_sec']:.1f}s",
            flush=True,
        )
        return row
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"[fail] {label} mode={mode}: {err}\n{traceback.format_exc()}", flush=True)
        row = {
            "label": label,
            "name": job["name"],
            "root": job["root"],
            "srcid": int(job["srcid"]),
            "z": float(job["z"]),
            "z_lit": float(job["z_lit"]),
            "mode": mode,
            "R": float(job["R"]),
            "success": False,
            "mse": np.nan,
            "nfev": -1,
            "nit": -1,
            "log10M_ds": np.nan,
            "log10M_ds_nirspec_miri": np.nan,
            "mse_nirspec_miri": np.nan,
            "success_nirspec_miri": False,
            "message": "",
            "runtime_sec": float(time.time() - t0),
            "error": err,
            "plot_path": "",
            "npz_path": "",
        }
        if mode == "joint":
            for key in ("logM", "age", "tau", "Z", "Av", "logU", "fesc"):
                row[f"gal_{key}"] = np.nan
                row[f"gal_{key}_nirspec_miri"] = np.nan
            row["sfr_exp"] = np.nan
            row["sfr_delayed"] = np.nan
        return row


# ---------------------------------------------------------------------------
# Catalog + CLI
# ---------------------------------------------------------------------------
def build_jobs(args):
    wanted = set(args.source) if args.source else None
    jobs = []
    for target in TARGETS:
        if wanted and target["label"] not in wanted and target["name"] not in wanted:
            continue
        # z14-0/1 use paper-extracted CSVs; others use DJA nirspec_spectra_csv
        csv_path = Path(target.get("csv_dir", CSV_DIR)) / target["csv_name"]
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing spectrum CSV: {csv_path}")
        miri_path = MIRI_X1D_BY_LABEL.get(target["label"])
        jobs.append({
            "label": target["label"],
            "name": target["name"],
            "root": target["root"],
            "srcid": target["srcid"],
            "z": target["z"],
            "z_lit": target["z_lit"],
            "R": NIRSPEC_R,
            "spectrum_csv": str(csv_path),
            "miri_x1d": str(miri_path) if miri_path is not None else "",
            "maxiter": args.maxiter,
            "mode": args.mode,
        })
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("smds", "joint"), default="smds",
        help="smds = TLUSTY mass only; joint = Bagpipes galaxy + SMDS (default: smds)",
    )
    parser.add_argument(
        "--source", action="append", default=[],
        help="Optional label/name filter (e.g. JADES_GS_z11); repeatable",
    )
    parser.add_argument(
        "--maxiter", type=int, default=3000,
        help="Nelder-Mead maxiter per fit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    jobs = build_jobs(args)
    results_csv = OUTDIR / f"nirspec_candidates_{args.mode}_results.csv"

    print(f"mode      = {args.mode}")
    print(f"n_targets = {len(jobs)}")
    print(f"outdir    = {OUTDIR}")
    print(f"n_workers = {n_workers}  (SLURM_CPUS_PER_TASK)")
    print(f"maxiter   = {args.maxiter}")
    for job in jobs:
        print(f"  {job['label']}  z={job['z']:.4f}  {job['spectrum_csv']}")
    sys.stdout.flush()

    t0 = time.time()
    rows = []
    if n_workers == 1 or len(jobs) <= 1:
        for job in jobs:
            rows.append(fit_one_source(job))
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs))) as pool:
            futures = {pool.submit(fit_one_source, job): job["label"] for job in jobs}
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    print(f"[pool] {label} raised: {exc}", flush=True)
                    rows.append({
                        "label": label, "mode": args.mode, "success": False,
                        "error": str(exc), "mse": np.nan, "log10M_ds": np.nan,
                        "runtime_sec": np.nan,
                    })

    results = pd.DataFrame(rows)
    if "label" in results.columns:
        results = results.sort_values("label").reset_index(drop=True)
    results.to_csv(results_csv, index=False)

    n_ok = int(results["success"].fillna(False).sum()) if "success" in results else 0
    print(f"\nFinished {len(results)} sources ({n_ok} success) in {time.time() - t0:.1f}s")
    print(f"Wrote {results_csv}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
