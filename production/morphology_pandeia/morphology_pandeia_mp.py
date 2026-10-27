#!/usr/bin/env python3
"""Parallel Monte Carlo Sérsic morphology fits (notebook 02_morphology_pandeia).

One worker process per MC draw. Galaxies run sequentially so peak memory
stays bounded (Pandeia is heavy). Logic matches the organized notebook.
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
import shutil

matplotlib.use("Agg")
# TeX only if latex is on PATH (cluster sbatch loads texlive; local runs often do not).
_USE_TEX = shutil.which("latex") is not None
matplotlib.rcParams["text.usetex"] = _USE_TEX
if _USE_TEX:
    matplotlib.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/yzhan629/DS_MIRI")
OUTDIR = REPO_ROOT / "production" / "morphology_pandeia"
PLOT_DIR = OUTDIR / "plots"
NPZ_DIR = OUTDIR / "npz"
RESULTS_CSV = OUTDIR / "morphology_pandeia_results.csv"
DATA_DIR = REPO_ROOT / "data"
TEMPLATE_DIR = REPO_ROOT / "tlusty_templates"

# Cosmology for TLUSTY distance (matches SMDS_Morphology_Fitting_MC_mse.ipynb).
C_CGS = 3.0e10
H0_KM_S_MPC = 72.0
OMEGA_M = 0.27
OMEGA_L = 0.73
# c/H0 in cm: H0 = 7.2e6 cm/s/Mpc in the draft, then * 3.086e24 cm/Mpc.
D_H_CM = (C_CGS / (H0_KM_S_MPC * 1e5 / 3.086e24))

FILTER = "f200w"
PIXEL_SCALE = {"sw": 0.031, "lw": 0.063, "miri": 0.11}  # arcsec / pixel
PSF_FWHM = {
    # NIRCam
    "f090w": 0.034, "f115w": 0.040, "f150w": 0.050, "f200w": 0.066,
    "f277w": 0.091, "f356w": 0.116, "f444w": 0.145,
    # MIRI imager (approx. diffraction FWHM, arcsec)
    "f560w": 0.18, "f770w": 0.24, "f1000w": 0.32, "f1130w": 0.36,
    "f1280w": 0.41, "f1500w": 0.48, "f1800w": 0.58, "f2100w": 0.67,
    "f2550w": 0.82,
}
LW_FILTERS = {
    "f250m", "f277w", "f300m", "f322w2", "f335m", "f356w", "f360m",
    "f410m", "f430m", "f444w", "f460m", "f480m",
}
MIRI_FILTERS = {
    "f560w", "f770w", "f1000w", "f1130w", "f1280w",
    "f1500w", "f1800w", "f2100w", "f2550w",
}
# Approx. Pandeia instrument bandpasses [um] for SED overlap checks.
MIRI_FILTER_BANDPASS = {
    "f560w": (5.0, 6.5),
    "f770w": (6.0, 9.0),
    "f1000w": (8.5, 11.5),
    "f1130w": (10.5, 12.5),
    "f1280w": (11.5, 14.0),
    "f1500w": (12.5, 18.0),
    "f1800w": (15.5, 21.0),
    "f2100w": (18.5, 24.5),
    "f2550w": (22.0, 29.0),
}

# Catalog of downloaded MIRI Level-3 i2d stamps (status==ok rows used).
MIRI_IMAG_CSV = DATA_DIR / "miri_dja_imag_ready.csv"
MIRI_SPEC_CSV = DATA_DIR / "miri_dja_spectra_ready.csv"

_SIM_ERR_LOGGED = False

CATALOG = {
    "GS-z14-0":     dict(z=14.18, AV=0.13, re_pc=260, logM=8.84, logxi=-4.76, logZ=-0.75, field="JADES-GS"),
    "GS-z14-1":     dict(z=13.90, AV=0.20, re_pc=160, logM=8.00, logxi=-4.15, logZ=-1.10, field="JADES-GS"),
    "GS-z13-0":     dict(z=13.20, AV=0.05, re_pc=52,  logM=7.95, logxi=-5.68, logZ=-1.69, field="JADES-GS"),
    "UNCOVER-z13":  dict(z=13.08, AV=0.04, re_pc=309, logM=8.13, logxi=-4.41, logZ=-1.57, field="UNCOVER"),
    "GS-z12-0":     dict(z=12.63, AV=0.05, re_pc=144, logM=7.64, logxi=-4.49, logZ=-1.44, field="JADES-GS"),
    "GHZ2":         dict(z=12.34, AV=0.04, re_pc=105, logM=9.05, logxi=-6.27, logZ=-1.40, field="GLASS"),
    "UNCOVER-z12":  dict(z=12.39, AV=0.19, re_pc=426, logM=8.35, logxi=-3.67, logZ=-1.34, field="UNCOVER"),
    "GS-z11-0":     dict(z=11.58, AV=0.18, re_pc=77,  logM=8.67, logxi=-5.50, logZ=-1.87, field="JADES-GS"),
    "Maisie":       dict(z=11.44, AV=0.07, re_pc=340, logM=8.40, logxi=-4.35, logZ=-1.50, field="CEERS"),
    "CEERS2-588":   dict(z=11.04, AV=0.10, re_pc=477, logM=8.99, logxi=-4.49, logZ=-0.84, field="CEERS"),
    "GN-z11":       dict(z=10.60, AV=0.17, re_pc=64,  logM=8.73, logxi=-5.75, logZ=-0.92, field="JADES-GN"),
    "GS-z10-0":     dict(z=10.38, AV=0.05, re_pc=62,  logM=7.58, logxi=-5.16, logZ=-1.91, field="JADES-GS"),
    "UNCOVER-37126": dict(z=10.25, AV=0.18, re_pc=426, logM=8.16, logxi=-3.51, logZ=-1.50, field="UNCOVER"),
    "MACS0647-JD":  dict(z=10.17, AV=0.01, re_pc=70,  logM=7.50, logxi=-5.67, logZ=-0.90, field="MACS0647"),
    "CEERS2-7929":  dict(z=10.10, AV=0.14, re_pc=520, logM=8.50, logxi=-3.78, logZ=-1.50, field="CEERS"),
}

DA_KPC = {
    10.10: 1006e3, 10.17: 1002e3, 10.25: 1000e3, 10.38: 997e3,
    10.60: 991e3, 11.04: 979e3, 11.44: 969e3, 11.58: 966e3,
    12.34: 949e3, 12.39: 948e3, 12.63: 943e3, 13.08: 933e3,
    13.20: 931e3, 13.90: 918e3, 14.18: 913e3,
}

JADES_MORPH = {
    "JZ11": dict(
        z=11.37, template="T3M6_0e5.13", R_cm=8.35e14, mu=1.0,
        ngroup=5, nint=1, nexp=35, theta0=0.10,
        profile_csv="JZ11_rad_prof_data.csv",
        err_anchors=dict(
            y0_fixed=3,
            lower_tail=[0.124, 0.067, 0.042, 0.0172, 0.0011],
            upper_tail=[0.154, 0.093, 0.0664, 0.0388, 0.0195],
        ),
    ),
    "JZ13": dict(
        z=13.17, template="T3M4_0e5.13", R_cm=7.632e14, mu=1.0,
        ngroup=5, nint=1, nexp=35, theta0=0.017,
        profile_csv="JZ13_rad_prof_data.csv",
        err_anchors=dict(
            y0_fixed=2,
            lower_tail=[0.18, 0.086, 0.038, 0.0067, 0.0011, 0.0018],
            upper_tail=[0.22, 0.12, 0.068, 0.034, 0.024, 0.026],
        ),
    ),
    "JZ14_0": dict(
        z=14.44, template="ACM2_00e+06.13", R_cm=1.01e15, mu=1.2,
        ngroup=5, nint=1, nexp=35, theta0=0.02,
        profile_csv="JZ14_0_rad_prof_data.csv",
        err_anchors=dict(
            y0_fixed=6,
            lower_tail=[0.097, 0.076, 0.0855],
            upper_tail=[0.116, 0.096, 0.108],
            floor=0.006,
        ),
    ),
    "Maisies": dict(
        z=11.4, template="T3M9_0e5.13", R_cm=8.935e14, mu=1.0,
        ngroup=7, nint=1, nexp=19, theta0=0.05,
        profile_csv=None,
        err_anchors=None,
    ),
}

# Set in main() / worker init from CLI; workers inherit via job dict.
USE_PANDEIA = False
_PANDEIA_READY = False


def setup_pandeia(repo_root=None):
    """Point env at refdata/PSFs and import Pandeia. Call once per process."""
    global USE_PANDEIA, _PANDEIA_READY, build_default_calc, perform_calculation
    if _PANDEIA_READY:
        return
    root = Path(repo_root or REPO_ROOT)
    refdata = root / "pandeia_data-2026.2-jwst"
    psf_dir = root / "pandeia_psfs-2026.2-jwst"
    if not (refdata / "VERSION_DATA").is_file():
        raise RuntimeError(f"Missing Pandeia refdata at {refdata}")
    if not (psf_dir / "VERSION_PSF").is_file():
        raise RuntimeError(f"Missing Pandeia PSFs at {psf_dir}")
    os.environ["pandeia_refdata"] = str(refdata)
    os.environ["PSF_DIR"] = str(psf_dir)
    from pandeia.engine.calc_utils import build_default_calc as _bdc
    from pandeia.engine.perform_calculation import perform_calculation as _pc
    build_default_calc = _bdc
    perform_calculation = _pc
    USE_PANDEIA = True
    _PANDEIA_READY = True


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def angular_diameter_distance_pc(z):
    """Flat LCDM D_A in pc (same cosmology as luminosity_distance_cm)."""
    z = float(z)
    if z in DA_KPC:
        return float(DA_KPC[z])
    # D_A = D_H * chi / (1+z); D_H_CM is c/H0 in cm.
    from scipy.integrate import quad

    def inv_e(zp):
        return 1.0 / np.sqrt(OMEGA_M * (1.0 + zp) ** 3 + OMEGA_L)

    if z <= 0:
        return 0.0
    chi = quad(inv_e, 0.0, z, epsabs=0, epsrel=1e-6)[0]
    d_a_cm = D_H_CM * chi / (1.0 + z)
    return d_a_cm / 3.086e18  # cm → pc


def pc_to_arcsec(r_pc, z):
    # Small-angle: physical size = angle_rad * distance.
    da_pc = angular_diameter_distance_pc(z)
    kpc_per_arcsec = da_pc * np.pi / (180 * 3600)
    return (r_pc / 1000.0) / kpc_per_arcsec


def arcsec_to_pc(theta, z):
    da_pc = angular_diameter_distance_pc(z)
    kpc_per_arcsec = da_pc * np.pi / (180 * 3600)
    return theta * kpc_per_arcsec * 1000


def sersic_1d(r, I_e, r_e, n):
    # b_n so half the light falls inside r_e.
    b_n = 1.9992 * n - 0.3271
    return I_e * np.exp(-b_n * ((r / r_e) ** (1.0 / n) - 1.0))


def luminosity_distance_cm(z):
    # Flat LCDM DL = (1+z) * int_0^z c dz' / H(z'), matching the morphology draft.
    from scipy.integrate import quad

    z = float(z)
    if z <= 0:
        return 0.0

    def inv_e(zp):
        return 1.0 / np.sqrt(OMEGA_M * (1.0 + zp) ** 3 + OMEGA_L)

    chi = quad(inv_e, 0.0, z, epsabs=0, epsrel=1e-6)[0]
    return (1.0 + z) * D_H_CM * chi


def fortran_float(value):
    # TLUSTY writes 1.23D+04; Python float() only accepts E.
    return float(str(value).strip().replace("D", "E").replace("d", "e"))


def load_tlusty(path):
    """Frequency [Hz] and TLUSTY surface-flux columns."""
    raw = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    freq = np.array([fortran_float(x) for x in raw.iloc[:, 0]])
    flux = np.array([fortran_float(x) for x in raw.iloc[:, 1]])
    return freq, flux


def redshift_tlusty_mjy(freq, surf_flux, z, radius_cm, mu=1.0):
    """Observed wavelength [um] and flux [mJy].

    Matches shftspecTLUmJy in SMDS_Morphology_Fitting_MC_mse.ipynb:
    F_obs = 4 pi F_surf (1+z) R^2 / D_L^2 * 1e26 * mu, zero shortward of LyA.
    """
    wave_um = (C_CGS / np.asarray(freq, float)) * (1.0 + z) * 1e4
    dl_cm = luminosity_distance_cm(z)
    flux_mJy = (
        4.0 * np.pi * np.asarray(surf_flux, float)
        * (1.0 + z) * float(radius_cm) ** 2 / dl_cm ** 2
        * 1e26 * float(mu)
    )
    # IGM: kill light blueward of observed-frame Lyman-alpha.
    flux_mJy = np.where(wave_um < (1.0 + z) * 0.1216, 0.0, flux_mJy)
    order = np.argsort(wave_um)
    wave_um = wave_um[order]
    flux_mJy = flux_mJy[order]
    # Drop duplicate wavelengths so Pandeia interpolation stays stable.
    keep = np.concatenate([[True], np.diff(wave_um) > 0])
    mask = keep & np.isfinite(wave_um) & np.isfinite(flux_mJy) & (wave_um > 0)
    return wave_um[mask], flux_mJy[mask]


def load_jades_sed(name, sed_path=None):
    """Two-column (um, mJy) override, else redshifted TLUSTY from JADES_MORPH."""
    if sed_path is not None:
        data = np.loadtxt(sed_path)
        return data[:, 0], data[:, 1], f"file:{sed_path}"
    cfg = JADES_MORPH[name]
    path = TEMPLATE_DIR / cfg["template"]
    if not path.is_file():
        raise FileNotFoundError(f"TLUSTY template not found: {path}")
    freq, surf = load_tlusty(path)
    wave, flux = redshift_tlusty_mjy(
        freq, surf, cfg["z"], cfg["R_cm"], mu=cfg["mu"],
    )
    return wave, flux, f"tlusty:{cfg['template']}"


def mock_spectrum(z, Muv=-20.0, beta=-2.3, wave_max_um=28.0):
    """UV power law plus Lyman break. Used when no real SED is available.

    Rest-frame grid extends red enough that observed-frame coverage reaches
    wave_max_um (MIRI F2550W).
    """
    from astropy.cosmology import Planck18

    z = float(z)

    wave_rest_max_A = max(5000.0, float(wave_max_um) / (1.0 + z) * 1e4)
    n_wave = max(800, int(wave_rest_max_A / 5.0))
    wave_rest = np.linspace(800.0, wave_rest_max_A, n_wave)
    f_rest = (wave_rest / 1500.0) ** beta
    f_rest[wave_rest < 912] *= 0.001
    f_rest *= 1.0 / (1.0 + np.exp(-(wave_rest - 1216.0) / 20.0))
    dl_cm = Planck18.luminosity_distance(z).to("cm").value
    Lnu_1500 = 10 ** (-0.4 * (Muv - 51.6))
    fnu_1500 = Lnu_1500 / (4 * np.pi * dl_cm ** 2 * (1 + z))
    flam_1500 = fnu_1500 * 3e10 / (1500e-8) ** 2
    f_rest *= flam_1500 / np.interp(1500, wave_rest, f_rest)
    wave_um = wave_rest * (1 + z) / 1e4
    flam_obs = f_rest / (1 + z)
    fnu = flam_obs * (wave_rest * (1 + z) * 1e-8) ** 2 / 3e10
    flux_mJy = fnu * 1e26
    mask = (wave_um > 0.6) & (wave_um < float(wave_max_um))
    return wave_um[mask], flux_mJy[mask]


def sed_overlaps_filter(wave_um, filter_name):
    """True if SED wavelengths overlap the approx instrument bandpass."""
    band = MIRI_FILTER_BANDPASS.get(str(filter_name).lower())
    if band is None or wave_um is None or len(wave_um) == 0:
        return True  # unknown filter: do not block
    w = np.asarray(wave_um, float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return False
    return bool((np.nanmin(w) < band[1]) and (np.nanmax(w) > band[0]))


def load_miri_spectrum_csv(path):
    """MIRI x1d export CSV → (wavelength [um], flux [mJy]).

    find_miri_spec_sources.ipynb writes Wavelength [um] and Flux as
    f_λ [erg/s/cm²/Å] × 1e21 (not Jy). Convert to mJy for Pandeia.
    """
    raw = pd.read_csv(path)
    # Column names may have leading spaces from the export notebook.
    cols = {c.strip().lower(): c for c in raw.columns}
    wave = raw[cols["wavelength"]].to_numpy(float)
    flux_1e21 = raw[cols["flux"]].to_numpy(float)
    order = np.argsort(wave)
    wave = wave[order]
    flux_1e21 = flux_1e21[order]
    # Inverse of jy_to_flambda_1e21: f_λ → f_ν → Jy → mJy.
    f_per_A = flux_1e21 * 1e-21
    f_per_cm = f_per_A * 1e8
    lam_cm = wave * 1e-4
    f_nu = f_per_cm * (lam_cm ** 2) / C_CGS  # erg/s/cm^2/Hz
    flux_mJy = np.maximum(f_nu / 1e-23 * 1e3, 0.0)
    keep = np.isfinite(wave) & np.isfinite(flux_mJy) & (wave > 0)
    return wave[keep], flux_mJy[keep]


# ---------------------------------------------------------------------------
# Image / radial profile
# ---------------------------------------------------------------------------
def _pixel_scale_for(filter_name, instrument="nircam"):
    filt = filter_name.lower()
    if instrument == "miri" or filt in MIRI_FILTERS:
        return PIXEL_SCALE["miri"]
    return PIXEL_SCALE["lw"] if filt in LW_FILTERS else PIXEL_SCALE["sw"]


def build_pandeia_calc(wave_um, flux_mJy, theta_arcsec, n_sersic, *,
                       filter_name="f200w", instrument="nircam",
                       ngroup=None, nint=None, nexp=None,
                       readout_pattern=None, nircam_filter=None):
    # Backward-compatible alias used by the JADES path.
    if nircam_filter is not None:
        filter_name = nircam_filter
    filter_name = filter_name.lower()
    instrument = instrument.lower()

    if instrument == "miri":
        calc = build_default_calc("jwst", "miri", "imaging")
        ngroup = 100 if ngroup is None else ngroup
        nint = 1 if nint is None else nint
        nexp = 4 if nexp is None else nexp
        readout_pattern = "fastr1" if readout_pattern is None else readout_pattern
    else:
        # NIRCam: SW vs LW mode from filter set.
        mode = "lw_imaging" if filter_name in LW_FILTERS else "sw_imaging"
        calc = build_default_calc("jwst", "nircam", mode)
        ngroup = 10 if ngroup is None else ngroup
        nint = 1 if nint is None else nint
        nexp = 1 if nexp is None else nexp
        readout_pattern = "deep8" if readout_pattern is None else readout_pattern

    src = calc["scene"][0]
    src["spectrum"]["sed"] = {
        "sed_type": "input",
        "spectrum": [np.asarray(wave_um).tolist(), np.asarray(flux_mJy).tolist()],
        "unit": "fnu",
    }
    src["spectrum"]["normalization"] = {"type": "none"}
    src["shape"] = {
        "geometry": "sersic",
        "major": float(theta_arcsec),
        "minor": float(theta_arcsec),
        "sersic_index": float(n_sersic),
        "position_angle": 0.0,
        "norm_method": "integ_infinity",
        "surf_area_units": "arcsec^2",
    }
    calc["configuration"]["instrument"]["filter"] = filter_name
    det = calc["configuration"]["detector"]
    det["ngroup"] = ngroup
    det["nint"] = nint
    det["nexp"] = nexp
    det["readout_pattern"] = readout_pattern
    calc["strategy"]["background_subtraction"] = True
    return calc


def azimuthal_average(image, pixel_scale, annulus_width=0.03, max_radius=0.5,
                      center=None, err_image=None):
    ny, nx = image.shape
    if center is None:
        cy, cx = ny / 2.0, nx / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])
    y, x = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * pixel_scale
    edges = np.arange(0, int(max_radius / annulus_width) + 1) * annulus_width
    centers, prof, err = [], [], []
    for i in range(len(edges) - 1):
        mask = (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(image)
        pix = image[mask]
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        if len(pix):
            prof.append(np.mean(pix))
            if err_image is not None:
                e_pix = err_image[mask]
                e_pix = e_pix[np.isfinite(e_pix)]
                # Propagate mean error of finite ERR pixels in the annulus.
                err.append(float(np.mean(e_pix) / np.sqrt(len(e_pix))) if len(e_pix) else 0.0)
            else:
                err.append(np.std(pix) / np.sqrt(len(pix)))
        else:
            prof.append(0.0)
            err.append(0.0)
    return np.array(centers), np.array(prof), np.array(err)


def radial_profile_at(image, center, radii_arcsec, pixel_scale):
    """Mean in bins centered on the observed radii."""
    x, y = np.indices(image.shape)
    r_pix = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    d_pix = np.asarray(radii_arcsec) / pixel_scale
    out = np.zeros(len(d_pix))
    for i, d in enumerate(d_pix):
        if i == 0:
            half = (d_pix[i + 1] - d) / 2 if len(d_pix) > 1 else d
            mask = r_pix < d + half
        elif i == len(d_pix) - 1:
            lo = 0.5 * (d_pix[i - 1] + d)
            hi = d + 0.5 * (d - d_pix[i - 1])
            mask = (r_pix >= lo) & (r_pix < hi)
        else:
            lo = 0.5 * (d_pix[i - 1] + d)
            hi = 0.5 * (d + d_pix[i + 1])
            mask = (r_pix >= lo) & (r_pix < hi)
        out[i] = image[mask].mean() if np.any(mask) else 0.0
    return out


def simulate_profile(wave_um, flux_mJy, theta, n, radii_arcsec=None, *,
                     filter_name="f200w", instrument="nircam",
                     annulus_width=0.03, max_radius=0.5,
                     ngroup=None, nint=None, nexp=None,
                     readout_pattern=None, use_pandeia=False,
                     nircam_filter=None):
    if nircam_filter is not None:
        filter_name = nircam_filter
    filter_name = filter_name.lower()
    instrument = instrument.lower()

    if use_pandeia:
        if not _PANDEIA_READY:
            setup_pandeia()
        calc = build_pandeia_calc(
            wave_um, flux_mJy, theta, n,
            filter_name=filter_name, instrument=instrument,
            ngroup=ngroup, nint=nint, nexp=nexp,
            readout_pattern=readout_pattern,
        )
        result = perform_calculation(calc)
        image = result["2d"]["snr"]
        pix = _pixel_scale_for(filter_name, instrument)
        if radii_arcsec is None:
            return azimuthal_average(image, pix, annulus_width, max_radius)
        center = np.unravel_index(np.nanargmax(image), image.shape)
        prof = radial_profile_at(image, center, radii_arcsec, pix)
        return np.asarray(radii_arcsec), prof, np.zeros_like(prof)

    # Mock: blur size and PSF in quadrature (exact for Gaussians).
    default_fwhm = 0.32 if instrument == "miri" else 0.066
    fwhm = PSF_FWHM.get(filter_name, default_fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    r_eff = np.sqrt(theta ** 2 + sigma ** 2)
    if radii_arcsec is None:
        r = np.arange(annulus_width / 2, max_radius, annulus_width)
    else:
        r = np.asarray(radii_arcsec, float)
    prof = sersic_1d(np.maximum(r, 1e-4), 1.0, max(r_eff, 1e-4), n)
    err = 1e-4 + 0.005 * (r / max(max_radius, 1e-6))
    return r, prof, err


# ---------------------------------------------------------------------------
# Observed profiles
# ---------------------------------------------------------------------------
def load_radial_csv(path):
    data = pd.read_csv(path, header=None)
    return data.iloc[:, 0].to_numpy(float), data.iloc[:, 1].to_numpy(float)


def load_miri_imag_catalog(path=None):
    """Rows with status==ok and a local FITS path from miri_dja_imag_ready.csv."""
    csv_path = Path(path) if path else MIRI_IMAG_CSV
    cat = pd.read_csv(csv_path)
    ok = cat[cat["status"].astype(str).str.lower() == "ok"].copy()
    ok = ok[ok["local_fits"].notna() & (ok["local_fits"].astype(str).str.len() > 0)]
    # Stable source name used as CLI --names key and output stem.
    ok["name"] = ok["root"].astype(str) + "_" + ok["srcid"].astype(str)
    return ok.reset_index(drop=True)


def _resolve_repo_path(rel_or_abs):
    path = Path(rel_or_abs)
    if path.is_file():
        return path
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"File not found: {rel_or_abs}")


def extract_miri_radial_profile(fits_path, ra_deg, dec_deg, *,
                                stamp_half_arcsec=2.5, annulus_width=None,
                                max_radius=1.5, bg_r_min=1.5, bg_r_max=2.5,
                                refine_peak_arcsec=0.3):
    """Cutout around (ra, dec) and return (r, profile, err, meta).

    Background = median in [bg_r_min, bg_r_max]. Profile is background-subtracted
    SCI; errors from ERR annulus means when present.
    """
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u

    fits_path = _resolve_repo_path(fits_path)
    with fits.open(fits_path) as hdul:
        sci_hdu = hdul["SCI"]
        sci = np.asarray(sci_hdu.data, float)
        err = np.asarray(hdul["ERR"].data, float) if "ERR" in hdul else None
        header = sci_hdu.header
        wcs = WCS(header)

    # Pixel scale from CDELT (deg) → arcsec; fall back to MIRI default.
    if header.get("CDELT1") is not None:
        pix_scale = abs(float(header["CDELT1"])) * 3600.0
    elif header.get("CD1_1") is not None:
        pix_scale = abs(float(header["CD1_1"])) * 3600.0
    else:
        pix_scale = PIXEL_SCALE["miri"]

    if annulus_width is None:
        annulus_width = pix_scale  # one pixel wide rings

    coord = SkyCoord(float(ra_deg), float(dec_deg), unit="deg")
    x0, y0 = wcs.world_to_pixel(coord)
    # world_to_pixel returns (x, y); numpy images are indexed [row=y, col=x].
    cy_cat, cx_cat = float(y0), float(x0)

    half = int(np.ceil(stamp_half_arcsec / pix_scale))
    y_lo = max(0, int(np.floor(cy_cat)) - half)
    y_hi = min(sci.shape[0], int(np.ceil(cy_cat)) + half + 1)
    x_lo = max(0, int(np.floor(cx_cat)) - half)
    x_hi = min(sci.shape[1], int(np.ceil(cx_cat)) + half + 1)
    stamp = sci[y_lo:y_hi, x_lo:x_hi]
    err_stamp = err[y_lo:y_hi, x_lo:x_hi] if err is not None else None
    cy = cy_cat - y_lo
    cx = cx_cat - x_lo

    # Optional: snap center to local peak within refine_peak_arcsec.
    if refine_peak_arcsec and refine_peak_arcsec > 0:
        yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
        r_pix = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        search = r_pix <= (refine_peak_arcsec / pix_scale)
        if np.any(search & np.isfinite(stamp)):
            masked = np.where(search, stamp, -np.inf)
            peak = np.unravel_index(np.nanargmax(masked), stamp.shape)
            cy, cx = float(peak[0]), float(peak[1])

    # Local background from outer annulus in the stamp.
    yy, xx = np.mgrid[0:stamp.shape[0], 0:stamp.shape[1]]
    r_arc = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * pix_scale
    bg_mask = (r_arc >= bg_r_min) & (r_arc < bg_r_max) & np.isfinite(stamp)
    bg = float(np.nanmedian(stamp[bg_mask])) if np.any(bg_mask) else 0.0
    stamp_sub = stamp - bg

    r, prof, e = azimuthal_average(
        stamp_sub, pix_scale, annulus_width=annulus_width, max_radius=max_radius,
        center=(cy, cx), err_image=err_stamp,
    )
    # Drop empty outer bins and enforce a small error floor for MC sampling.
    finite = np.isfinite(prof) & (e >= 0)
    r, prof, e = r[finite], prof[finite], e[finite]
    e = np.maximum(e, 1e-4 * max(np.nanmax(np.abs(prof)), 1e-6))
    meta = dict(
        pix_scale=pix_scale, background=bg, center_yx=(cy, cx),
        fits_path=str(fits_path),
    )
    return r, prof, e, meta


def load_miri_sed(row, sed_path=None, prefer_spectrum=True, filter_name=None):
    """SED for one MIRI imaging source.

    Priority: --sed file → matched MIRI x1d CSV (if prefer_spectrum and it
    overlaps the imaging filter) → mock_spectrum(z) covering MIRI bands.
    """
    filt = str(filter_name or row.get("filter", "")).lower()
    z = float(row["z"])

    def _mock():
        wave, flux = mock_spectrum(z, wave_max_um=28.0)
        return wave, flux, f"mock_spectrum:z={z:.3f}"

    if sed_path is not None:
        data = np.loadtxt(sed_path)
        wave, flux = data[:, 0], data[:, 1]
        if sed_overlaps_filter(wave, filt):
            return wave, flux, f"file:{sed_path}"
        print(
            f"[sed] file {sed_path} does not overlap {filt} "
            f"([{np.nanmin(wave):.2f},{np.nanmax(wave):.2f}] um); using mock",
            flush=True,
        )
        return _mock()

    if prefer_spectrum and MIRI_SPEC_CSV.is_file():
        spec_cat = pd.read_csv(MIRI_SPEC_CSV)
        hit = spec_cat[
            (spec_cat["root"].astype(str) == str(row["root"]))
            & (spec_cat["srcid"].astype(str) == str(row["srcid"]))
            & (spec_cat["status"].astype(str).str.lower() == "ok")
        ]
        if len(hit) and pd.notna(hit.iloc[0].get("spectrum_csv")):
            csv_rel = hit.iloc[0]["spectrum_csv"]
            try:
                wave, flux = load_miri_spectrum_csv(_resolve_repo_path(csv_rel))
                if len(wave) >= 5 and np.nanmax(flux) > 0:
                    if sed_overlaps_filter(wave, filt):
                        return wave, flux, f"miri_x1d:{csv_rel}"
                    print(
                        f"[sed] x1d {csv_rel} does not overlap {filt} "
                        f"([{np.nanmin(wave):.2f},{np.nanmax(wave):.2f}] um); "
                        f"using mock",
                        flush=True,
                    )
            except FileNotFoundError:
                pass

    return _mock()


def errors_from_anchors(y, anchors, floor=0.008):
    if anchors is None:
        return np.full_like(y, floor), np.full_like(y, floor)
    n_fixed = anchors["y0_fixed"]
    floor = anchors.get("floor", floor)
    lower = np.empty_like(y)
    upper = np.empty_like(y)
    lower[:n_fixed] = floor
    upper[:n_fixed] = floor
    lower[n_fixed:] = y[n_fixed:] - np.asarray(anchors["lower_tail"])
    upper[n_fixed:] = np.asarray(anchors["upper_tail"]) - y[n_fixed:]
    return lower, upper


def sampler_sigma(lower_err):
    """Draft MC width: sigma = 2 * lower_err / 3.92 (95% half-width)."""
    return 2.0 * np.asarray(lower_err, float) / 3.92


def mock_observed_profile(name, info, annulus_width=0.03, max_radius=0.4):
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
    theta = pc_to_arcsec(info["re_pc"], info["z"])
    n_true = float(rng.uniform(0.5, 2.0))
    sigma = PSF_FWHM["f200w"] / (2 * np.sqrt(2 * np.log(2)))
    r = np.arange(annulus_width / 2, max_radius, annulus_width)
    prof = sersic_1d(r, 1.0, np.sqrt(theta ** 2 + sigma ** 2), n_true)
    snr = np.maximum(30.0 * prof / prof.max(), 1.0)
    err = prof / snr
    noisy = np.maximum(prof + rng.normal(0, err), 0)
    return r, noisy, err, dict(theta_true=theta, n_true=n_true)


# ---------------------------------------------------------------------------
# Loss + one MC draw (top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def profile_mse(params, wave, flux, r_obs, prof_obs, sim_kwargs):
    global _SIM_ERR_LOGGED
    theta, n = params
    if not (0.001 < theta < 0.5 and 0.1 < n < 8):
        return 1e10
    try:
        _, sim, _ = simulate_profile(wave, flux, theta, n, radii_arcsec=r_obs, **sim_kwargs)
    except Exception as exc:
        # Log once per process; Nelder–Mead would otherwise spam every eval.
        if not _SIM_ERR_LOGGED:
            print(
                f"[sim] {type(exc).__name__}: {exc}",
                flush=True,
            )
            _SIM_ERR_LOGGED = True
        return 1e10
    # Require fully finite profiles; partial NaNs make mean(...) → NaN.
    if (not np.all(np.isfinite(sim)) or np.nanmax(sim) <= 0
            or not np.all(np.isfinite(prof_obs)) or np.nanmax(prof_obs) <= 0):
        return 1e10
    sim_n = sim / np.nanmax(sim)
    obs_n = prof_obs / np.nanmax(prof_obs)
    mse = float(np.mean((obs_n - sim_n) ** 2))
    return mse if np.isfinite(mse) else 1e10


def _one_mc_draw(job):
    """Run one Nelder–Mead restart. Must be top-level for pickling."""
    # Keep BLAS single-threaded inside each worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if job.get("use_pandeia"):
        setup_pandeia(job.get("repo_root"))

    rng = np.random.default_rng(job["seed"])
    sample = np.maximum(rng.normal(job["prof_obs"], job["err_obs"]), 0)
    n0 = float(np.clip(rng.normal(*job["n_prior"]), 0.3, 6.0))
    th0 = job["theta_init"] * float(rng.uniform(0.5, 1.5))
    sim_kwargs = dict(job["sim_kwargs"])
    sim_kwargs["use_pandeia"] = bool(job.get("use_pandeia"))

    res = minimize(
        profile_mse, [th0, n0],
        args=(job["wave"], job["flux"], job["r_obs"], sample, sim_kwargs),
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-8, "adaptive": True},
    )
    return dict(
        draw=int(job["draw"]),
        theta=float(res.x[0]),
        n=float(res.x[1]),
        mse=float(res.fun),
        success=bool(res.success),
    )


def fit_profile_mp(wave, flux, r_obs, prof_obs, err_obs, *, z, re_pc,
                   n_mc=75, n_prior=(1.0, 0.5), theta_init=None, seed=0,
                   sim_kwargs=None, n_workers=1, use_pandeia=False):
    """Monte Carlo fit with one process per draw."""
    sim_kwargs = sim_kwargs or {}
    if theta_init is None:
        theta_init = max(pc_to_arcsec(re_pc, z), 0.005)

    base = dict(
        wave=np.asarray(wave, float),
        flux=np.asarray(flux, float),
        r_obs=np.asarray(r_obs, float),
        prof_obs=np.asarray(prof_obs, float),
        err_obs=np.asarray(err_obs, float),
        n_prior=tuple(n_prior),
        theta_init=float(theta_init),
        sim_kwargs=dict(sim_kwargs),
        use_pandeia=bool(use_pandeia),
        repo_root=str(REPO_ROOT),
    )
    jobs = []
    for draw in range(n_mc):
        jobs.append({**base, "draw": draw, "seed": int(seed) + draw + 1})

    draws = []
    workers = max(1, int(n_workers))
    if workers == 1 or n_mc <= 1:
        for job in jobs:
            draws.append(_one_mc_draw(job))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, n_mc)) as pool:
            futures = {pool.submit(_one_mc_draw, job): job["draw"] for job in jobs}
            for fut in as_completed(futures):
                draw_id = futures[fut]
                try:
                    draws.append(fut.result())
                except Exception as exc:
                    print(f"[pool] draw {draw_id} raised: {exc}", flush=True)
                    draws.append(dict(
                        draw=draw_id, theta=np.nan, n=np.nan, mse=1e10, success=False,
                    ))

    draws = sorted(draws, key=lambda d: d["draw"])
    theta_all = np.array([d["theta"] for d in draws], float)
    n_all = np.array([d["n"] for d in draws], float)
    mse_all = np.array([d["mse"] for d in draws], float)
    if not np.any(np.isfinite(mse_all)):
        raise RuntimeError("all MC draws returned non-finite MSE")
    best = int(np.nanargmin(mse_all))
    return dict(
        theta=theta_all, n=n_all, mse=mse_all,
        theta_med=float(np.nanmedian(theta_all)), theta_std=float(np.nanstd(theta_all)),
        n_med=float(np.nanmedian(n_all)), n_std=float(np.nanstd(n_all)),
        theta_best=float(theta_all[best]), n_best=float(n_all[best]),
        re_pc_med=float(arcsec_to_pc(np.nanmedian(theta_all), z)),
    )


# ---------------------------------------------------------------------------
# Galaxy drivers
# ---------------------------------------------------------------------------
def fit_one_blue_monster(name, n_mc, n_prior, n_workers, use_pandeia, seed=0):
    t0 = time.time()
    info = CATALOG[name]
    wave, flux = mock_spectrum(
        info["z"], Muv=-20.0 - 0.5 * (info["logM"] - 8.5), beta=-2.3,
    )
    r, prof, err, truth = mock_observed_profile(name, info)
    out = fit_profile_mp(
        wave, flux, r, prof, err,
        z=info["z"], re_pc=info["re_pc"],
        n_mc=n_mc, n_prior=n_prior, seed=seed,
        n_workers=n_workers, use_pandeia=use_pandeia,
    )
    out["name"] = name
    out["mode"] = "blue_monster"
    out["z"] = info["z"]
    out["re_pc_catalog"] = info["re_pc"]
    out["truth"] = truth
    out["r"] = r
    out["profile"] = prof
    out["err"] = err
    out["runtime_sec"] = float(time.time() - t0)
    out["error"] = ""
    print(
        f"[done] {name}: theta={out['theta_med']:.4f}±{out['theta_std']:.4f} arcsec  "
        f"n={out['n_med']:.2f}±{out['n_std']:.2f}  t={out['runtime_sec']:.1f}s",
        flush=True,
    )
    return out


def fit_one_jades(name, n_mc, n_prior, n_workers, use_pandeia, sed_path=None, seed=0):
    t0 = time.time()
    cfg = JADES_MORPH[name]
    if cfg["profile_csv"] is None:
        raise ValueError(f"{name}: no profile_csv in JADES_MORPH")
    csv_path = Path(cfg["profile_csv"])
    if not csv_path.is_file():
        csv_path = DATA_DIR / cfg["profile_csv"]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Radial profile CSV not found: {cfg['profile_csv']}")

    r, prof = load_radial_csv(csv_path)
    lower, _upper = errors_from_anchors(prof, cfg["err_anchors"])
    err = sampler_sigma(lower)

    # Default: redshifted TLUSTY from JADES_MORPH; --sed overrides with (um, mJy).
    wave, flux, sed_label = load_jades_sed(name, sed_path=sed_path)
    print(
        f"[sed] {name}: {sed_label}  n_wave={len(wave)}  "
        f"F200W window peak={np.nanmax(flux[(wave > 1.5) & (wave < 2.5)]):.4g} mJy",
        flush=True,
    )

    out = fit_profile_mp(
        wave, flux, r, prof, err,
        z=cfg["z"], re_pc=100,
        n_mc=n_mc, n_prior=n_prior, theta_init=cfg["theta0"], seed=seed,
        sim_kwargs=dict(
            filter_name="f200w", instrument="nircam",
            ngroup=cfg["ngroup"], nint=cfg["nint"], nexp=cfg["nexp"],
        ),
        n_workers=n_workers, use_pandeia=use_pandeia,
    )
    out["name"] = name
    out["mode"] = "jades"
    out["z"] = cfg["z"]
    out["re_pc_catalog"] = np.nan
    out["truth"] = None
    out["r"] = r
    out["profile"] = prof
    out["err"] = err
    out["sed"] = sed_label
    out["runtime_sec"] = float(time.time() - t0)
    out["error"] = ""
    print(
        f"[done] {name}: theta={out['theta_med']:.4f}±{out['theta_std']:.4f} arcsec  "
        f"n={out['n_med']:.2f}±{out['n_std']:.2f}  t={out['runtime_sec']:.1f}s",
        flush=True,
    )
    return out


def fit_one_miri(name, n_mc, n_prior, n_workers, use_pandeia, sed_path=None,
                 seed=0, catalog=None, prefer_spectrum=True):
    """Fit one MIRI imaging source: FITS cutout profile + MIRI Pandeia/mock."""
    t0 = time.time()
    cat = catalog if catalog is not None else load_miri_imag_catalog()
    rows = cat[cat["name"] == name]
    if len(rows) == 0:
        raise KeyError(f"{name} not in MIRI imaging catalog (status=ok)")
    row = rows.iloc[0]

    filt = str(row["filter"]).lower()
    z = float(row["z"])
    r, prof, err, meta = extract_miri_radial_profile(
        row["local_fits"], float(row["ra"]), float(row["dec"]),
    )
    if len(r) < 3:
        raise RuntimeError(f"{name}: radial profile too short ({len(r)} bins)")

    wave, flux, sed_label = load_miri_sed(
        row, sed_path=sed_path, prefer_spectrum=prefer_spectrum, filter_name=filt,
    )
    print(
        f"[sed] {name}: {sed_label}  filter={filt}  n_wave={len(wave)}  "
        f"wave=[{np.nanmin(wave):.2f},{np.nanmax(wave):.2f}] um  "
        f"pix={meta['pix_scale']:.4f}\"/px  n_r={len(r)}",
        flush=True,
    )

    if use_pandeia and not sed_overlaps_filter(wave, filt):
        band = MIRI_FILTER_BANDPASS.get(filt, ("?", "?"))
        raise RuntimeError(
            f"{name}: SED [{np.nanmin(wave):.2f},{np.nanmax(wave):.2f}] um "
            f"does not overlap {filt} bandpass [{band[0]},{band[1]}] um"
        )

    # Theta init ~ MIRI PSF scale; physical re_pc only for reporting fallback.
    theta_init = max(PSF_FWHM.get(filt, 0.32) * 0.5, 0.05)
    out = fit_profile_mp(
        wave, flux, r, prof, err,
        z=z, re_pc=100,
        n_mc=n_mc, n_prior=n_prior, theta_init=theta_init, seed=seed,
        sim_kwargs=dict(
            filter_name=filt, instrument="miri",
            annulus_width=meta["pix_scale"], max_radius=float(np.nanmax(r)),
        ),
        n_workers=n_workers, use_pandeia=use_pandeia,
    )
    out["name"] = name
    out["mode"] = "miri"
    out["z"] = z
    out["filter"] = filt
    out["re_pc_catalog"] = np.nan
    out["truth"] = None
    out["r"] = r
    out["profile"] = prof
    out["err"] = err
    out["sed"] = sed_label
    out["runtime_sec"] = float(time.time() - t0)
    out["error"] = ""
    print(
        f"[done] {name}: theta={out['theta_med']:.4f}±{out['theta_std']:.4f} arcsec  "
        f"n={out['n_med']:.2f}±{out['n_std']:.2f}  t={out['runtime_sec']:.1f}s",
        flush=True,
    )
    return out


def save_galaxy_result(out):
    """Write npz + summary plot for one galaxy. Plot failure is non-fatal."""
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    name = out["name"]
    npz_path = NPZ_DIR / f"{name}_morph.npz"
    truth = out.get("truth") or {}
    np.savez_compressed(
        npz_path,
        theta=out["theta"], n=out["n"], mse=out["mse"],
        r=out["r"], profile=out["profile"], err=out["err"],
        theta_med=out["theta_med"], theta_std=out["theta_std"],
        n_med=out["n_med"], n_std=out["n_std"],
        theta_best=out["theta_best"], n_best=out["n_best"],
        re_pc_med=out["re_pc_med"], z=out["z"],
        theta_true=truth.get("theta_true", np.nan),
        n_true=truth.get("n_true", np.nan),
    )

    try:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        ax = axes[0]
        ax.errorbar(
            out["r"], out["profile"] / np.nanmax(out["profile"]),
            yerr=out["err"] / np.nanmax(out["profile"]),
            fmt="o", ms=3, color="0.3", label="observed",
        )
        ax.set_xlabel(r"$r$ [arcsec]")
        ax.set_ylabel(r"normalized SB")
        ax.set_title(name)
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(out["theta"], out["n"], c=out["mse"], s=12, cmap="viridis_r")
        ax.axvline(out["theta_med"], color="C1", ls="--", lw=1)
        ax.axhline(out["n_med"], color="C1", ls="--", lw=1)
        ax.set_xlabel(r"$\theta_e$ [arcsec]")
        ax.set_ylabel(r"$n$")
        ax.set_title(rf"MC draws ($n_{{\rm med}}={out['n_med']:.2f}$)")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{name}_morph.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plot] {name}: {type(exc).__name__}: {exc}", flush=True)
        plt.close("all")
    return npz_path


def result_row(out):
    return {
        "name": out["name"],
        "mode": out.get("mode", ""),
        "z": out.get("z", np.nan),
        "filter": out.get("filter", ""),
        "theta_med": out.get("theta_med", np.nan),
        "theta_std": out.get("theta_std", np.nan),
        "n_med": out.get("n_med", np.nan),
        "n_std": out.get("n_std", np.nan),
        "theta_best": out.get("theta_best", np.nan),
        "n_best": out.get("n_best", np.nan),
        "re_pc_med": out.get("re_pc_med", np.nan),
        "re_pc_catalog": out.get("re_pc_catalog", np.nan),
        "mse_best": float(np.nanmin(out["mse"])) if "mse" in out else np.nan,
        "sed": out.get("sed", ""),
        "runtime_sec": out.get("runtime_sec", np.nan),
        "error": out.get("error", ""),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("blue_monsters", "jades", "miri"), default="blue_monsters",
        help="Mock blue monsters, JADES radial CSVs, or MIRI i2d cutouts",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Subset of galaxy names (default: full catalog for the mode). "
             "For miri use root_srcid, e.g. uncover-v4_26185",
    )
    parser.add_argument("--n-mc", type=int, default=75, help="Monte Carlo draws per galaxy")
    parser.add_argument(
        "--n-prior-mean", type=float, default=None,
        help="Sérsic-index prior mean (default: 1.0 blue / 4.0 jades / 1.0 miri)",
    )
    parser.add_argument(
        "--n-prior-std", type=float, default=None,
        help="Sérsic-index prior std (default: 0.5 blue / 0.033 jades / 0.5 miri)",
    )
    parser.add_argument(
        "--use-pandeia", action="store_true",
        help="Call Pandeia (needs refdata under REPO_ROOT). Default is analytic mock.",
    )
    parser.add_argument(
        "--sed", type=Path, default=None,
        help="Optional two-column SED file (um, mJy); jades default TLUSTY, "
             "miri default matched x1d or mock_spectrum",
    )
    parser.add_argument(
        "--no-spectrum-match", action="store_true",
        help="For --mode miri: skip miri_dja_spectra_ready.csv and use mock_spectrum",
    )
    parser.add_argument(
        "--catalog", type=Path, default=None,
        help="For --mode miri: override imaging catalog CSV (default: data/miri_dja_imag_ready.csv)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    # Worker count from Slurm (#SBATCH --cpus-per-task).
    n_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    miri_cat = None
    if args.mode == "blue_monsters":
        names = list(args.names or CATALOG)
        n_prior = (
            args.n_prior_mean if args.n_prior_mean is not None else 1.0,
            args.n_prior_std if args.n_prior_std is not None else 0.5,
        )
    elif args.mode == "jades":
        names = list(args.names or [k for k, v in JADES_MORPH.items() if v["profile_csv"]])
        n_prior = (
            args.n_prior_mean if args.n_prior_mean is not None else 4.0,
            args.n_prior_std if args.n_prior_std is not None else 0.033,
        )
    else:
        miri_cat = load_miri_imag_catalog(args.catalog)
        names = list(args.names or miri_cat["name"].tolist())
        n_prior = (
            args.n_prior_mean if args.n_prior_mean is not None else 1.0,
            args.n_prior_std if args.n_prior_std is not None else 0.5,
        )

    if args.use_pandeia:
        setup_pandeia()

    print(f"mode      = {args.mode}")
    print(f"names     = {names}")
    print(f"n_mc      = {args.n_mc}")
    print(f"n_prior   = {n_prior}")
    print(f"use_pandeia = {args.use_pandeia}")
    print(f"n_workers = {n_workers}  (SLURM_CPUS_PER_TASK; one process per MC draw)")
    print(f"outdir    = {OUTDIR}")
    sys.stdout.flush()

    t0 = time.time()
    rows = []
    for name in names:
        try:
            if args.mode == "blue_monsters":
                out = fit_one_blue_monster(
                    name, args.n_mc, n_prior, n_workers, args.use_pandeia, seed=args.seed,
                )
            elif args.mode == "jades":
                out = fit_one_jades(
                    name, args.n_mc, n_prior, n_workers, args.use_pandeia,
                    sed_path=args.sed, seed=args.seed,
                )
            else:
                out = fit_one_miri(
                    name, args.n_mc, n_prior, n_workers, args.use_pandeia,
                    sed_path=args.sed, seed=args.seed, catalog=miri_cat,
                    prefer_spectrum=not args.no_spectrum_match,
                )
            save_galaxy_result(out)
            rows.append(result_row(out))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {name}: {err}\n{traceback.format_exc()}", flush=True)
            rows.append({
                "name": name, "mode": args.mode, "z": np.nan, "filter": "",
                "theta_med": np.nan, "theta_std": np.nan,
                "n_med": np.nan, "n_std": np.nan,
                "theta_best": np.nan, "n_best": np.nan,
                "re_pc_med": np.nan, "re_pc_catalog": np.nan,
                "mse_best": np.nan, "sed": "", "runtime_sec": np.nan, "error": err,
            })
        # Checkpoint after each galaxy so a crash does not lose earlier work.
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nFinished {len(results)} galaxies in {time.time() - t0:.1f}s")
    print(f"Wrote {RESULTS_CSV}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
