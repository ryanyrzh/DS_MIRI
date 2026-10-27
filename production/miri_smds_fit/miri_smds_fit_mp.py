#!/usr/bin/env python3
"""Parallel MIRI SMDS-only Nelder-Mead fits (no galaxy component).

One worker process per catalog source. Logic matches miri_spectral_fitting.ipynb
`fit_smds_mass`, with inverse-variance weighting as in the joint production script.
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
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants (same as the notebook / joint fit, minus galaxy)
# ---------------------------------------------------------------------------
SMDS_UNIT_SCALE = 1.0
COSMO_SMDS = dict(H0=72.0, Om=0.27)
C_CGS = 2.99792458e10
# Mass-only fit uses a tighter upper wall than the joint fit (8.5)
LOGM_DS_MIN, LOGM_DS_MAX = 5.0, 7.95
# MIRI LRS reliable window; drop noisy band edges before fitting
LAM_FIT_MIN_UM, LAM_FIT_MAX_UM = 5.0, 12.0

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
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CATALOG = DATA_DIR / "miri_dja_spectra_ready.csv"
OUTDIR = REPO_ROOT / "production" / "miri_smds_fit"
PLOT_DIR = OUTDIR / "plots"
NPZ_DIR = OUTDIR / "npz"
RESULTS_CSV = OUTDIR / "miri_smds_fit_results.csv"


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
                         min_um=1.0, max_um=30.0):
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
# Fitting (SMDS mass only)
# ---------------------------------------------------------------------------
def fit_smds_mass(
    obs_lambda_um, obs_flux, z,
    obs_err=None,
    x0_logM=6.2,
    lya_buffer=1.10,
    logm_min=LOGM_DS_MIN,
    logm_max=LOGM_DS_MAX,
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
        "total_flux": smds_flux,
    }


def miri_resolving_power(instrument_name):
    name = str(instrument_name).upper()
    if "IFU" in name:
        return 1500.0
    return 100.0


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
# Per-source worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------
def fit_one_source(job):
    """Fit a single catalog row. `job` is a plain dict (pickle-safe)."""
    label = job["label"]
    t0 = time.time()
    # Keep BLAS single-threaded inside each worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        lam, flux, err = load_spectrum_csv(job["spectrum_csv"])
        # Finite flux/err and clean MIRI LRS window (drop edge fringing / thermal noise)
        good = (
            np.isfinite(lam) & np.isfinite(flux)
            & np.isfinite(err) & (err > 0)
            & (lam >= LAM_FIT_MIN_UM) & (lam <= LAM_FIT_MAX_UM)
        )
        lam, flux, err = lam[good], flux[good], err[good]
        if lam.size == 0:
            raise ValueError(
                f"no finite pixels with err>0 in [{LAM_FIT_MIN_UM}, {LAM_FIT_MAX_UM}] um"
            )
        z = float(job["z"])
        R = float(job["R"])
        nm_options = {"maxiter": int(job["maxiter"])}
        out = fit_smds_mass(
            lam, flux, z,
            obs_err=err,
            x0_logM=6.2,
            lya_buffer=1.10,
            nm_options=nm_options,
        )

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = str(PLOT_DIR / f"{label}_smds_fit.png")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.fill_between(
            lam, flux - err, flux + err,
            facecolor="0.6", alpha=0.25, edgecolor="none", linewidth=0,
            label=r"Observed $\pm 1\sigma$",
        )
        ax.plot(lam, flux, lw=1, color="0.35", label="Observed MIRI")
        ax.plot(lam, out["smds_flux"], lw=1, ls="--", color="limegreen",
                label="SMDS")
        ax.set_xlabel(r"Wavelength [$\mu$m]")
        ax.set_ylabel(r"$f_\lambda \times 10^{21}$")
        ax.set_title(rf"{label}  $z={z:.3f}$")
        # Spectra with coverage blueward of 8 um are usually faint; clip
        # nonphysical red-end spikes so the continuum is readable
        if np.any(lam < 8.0):
            ax.set_ylim(-5.0, 5.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        NPZ_DIR.mkdir(parents=True, exist_ok=True)
        npz_path = str(NPZ_DIR / f"{label}_smds_fit.npz")
        np.savez_compressed(
            npz_path,
            lam=lam,
            flux=flux,
            err=err,
            smds_flux=out["smds_flux"],
            total_flux=out["total_flux"],
            z=z,
            R=R,
            log10M_ds=out["log10M_ds"],
        )

        row = {
            "label": label,
            "root": job["root"],
            "srcid": int(job["srcid"]),
            "z": z,
            "instrument": job["instrument"],
            "R": R,
            "success": bool(out["result"].success),
            "mse": float(out["result"].fun),
            "nfev": int(getattr(out["result"], "nfev", -1)),
            "nit": int(getattr(out["result"], "nit", -1)),
            "log10M_ds": float(out["log10M_ds"]),
            "message": str(out["result"].message),
            "runtime_sec": float(time.time() - t0),
            "error": "",
        }
        print(
            f"[done] {label} success={row['success']} mse={row['mse']:.4g} "
            f"log10M_ds={row['log10M_ds']:.3f} t={row['runtime_sec']:.1f}s",
            flush=True,
        )
        return row
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"[fail] {label}: {err}\n{traceback.format_exc()}", flush=True)
        return {
            "label": label,
            "root": job["root"],
            "srcid": int(job["srcid"]),
            "z": float(job["z"]),
            "instrument": job["instrument"],
            "R": float(job["R"]),
            "success": False,
            "mse": np.nan,
            "nfev": -1,
            "nit": -1,
            "log10M_ds": np.nan,
            "message": "",
            "runtime_sec": float(time.time() - t0),
            "error": err,
        }


# ---------------------------------------------------------------------------
# Catalog + CLI
# ---------------------------------------------------------------------------
def load_ready_catalog(catalog_path, max_sources=None, source_filter=None):
    cat = pd.read_csv(catalog_path)
    cat = cat[cat["status"] == "ok"].copy()
    cat = cat[cat["spectrum_csv"].notna()].copy()
    if source_filter:
        keep = set((str(root), int(srcid)) for root, srcid in source_filter)
        cat = cat[cat.apply(lambda r: (str(r["root"]), int(r["srcid"])) in keep, axis=1)]
    if max_sources is not None:
        cat = cat.head(int(max_sources))
    return cat.reset_index(drop=True)


def parse_source_filter(items):
    # CLI form: root:srcid  (repeatable)
    out = []
    for item in items or []:
        root, srcid = item.rsplit(":", 1)
        out.append((root, int(srcid)))
    return out


def build_jobs(catalog, args):
    jobs = []
    for _, row in catalog.iterrows():
        label = f"{row['root']}_{row['srcid']}"
        jobs.append({
            "label": label,
            "root": str(row["root"]),
            "srcid": int(row["srcid"]),
            "z": float(row["z"]),
            "instrument": str(row["miri_instrument_name"]),
            "R": miri_resolving_power(row["miri_instrument_name"]),
            "spectrum_csv": str(row["spectrum_csv"]),
            "maxiter": args.maxiter,
        })
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG,
                        help=f"Ready catalog CSV (default: {DEFAULT_CATALOG})")
    parser.add_argument("--max-sources", type=int, default=None,
                        help="Cap number of status==ok rows")
    parser.add_argument("--source", action="append", default=[],
                        help="Optional root:srcid filter; repeatable")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="Nelder-Mead maxiter per SMDS-only fit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.catalog = Path(args.catalog)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    # Worker count comes only from Slurm (#SBATCH --cpus-per-task).
    n_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    source_filter = parse_source_filter(args.source)
    catalog = load_ready_catalog(args.catalog, args.max_sources, source_filter)
    jobs = build_jobs(catalog, args)

    print(f"catalog   = {args.catalog}  ({len(jobs)} sources)")
    print(f"outdir    = {OUTDIR}")
    print(f"n_workers = {n_workers}  (SLURM_CPUS_PER_TASK)")
    print(f"maxiter   = {args.maxiter}")
    print(catalog[["root", "srcid", "z", "miri_instrument_name", "spectrum_csv"]].to_string(index=False))
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
                        "label": label, "success": False, "error": str(exc),
                        "mse": np.nan, "log10M_ds": np.nan, "runtime_sec": np.nan,
                    })

    results = pd.DataFrame(rows)
    if "label" in results.columns:
        results = results.sort_values("label").reset_index(drop=True)
    results.to_csv(RESULTS_CSV, index=False)

    n_ok = int(results["success"].fillna(False).sum()) if "success" in results else 0
    print(f"\nFinished {len(results)} sources ({n_ok} success) in {time.time() - t0:.1f}s")
    print(f"Wrote {RESULTS_CSV}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
