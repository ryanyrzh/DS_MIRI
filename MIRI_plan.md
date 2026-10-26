# MIRI SMDS plan

Same objects and redshifts as notebooks 01–03 (z ~ 10–14). MIRI sees their rest-frame optical light. He II 1640 stays in NIRSpec.

At these redshifts a galaxy has a Balmer break near rest 3650 Å and often strong Hβ, [O III], and Hα. A dark-star atmosphere is a smooth hot continuum, so those features are weak.

Filter placement:

- z ≈ 10: F560W on [O III]/Hβ, F770W on Hα.
- z ≈ 14: F560W on the Balmer break, F770W on [O III]/Hβ, F1000W on Hα.

## Steps

1. **Sample.** Start from the notebook 02 list (JADES z ~ 11–14 and the Ferrara blue-monster sample). Keep rows with public MIRI imaging. MRS spectra are optional.
2. **Spectra (notebook 01).** Reuse the TLUSTY mass grid, the BAGPIPES galaxy, and the three fits (galaxy, dark star, both). Hold redshift fixed. Score models on MIRI filter fluxes, plus the NIRSpec spectrum when one exists. A candidate improves the error when the dark star is included.
3. **Images (notebook 02).** One filter first: F560W (sharpest PSF; sits on the break at z ~ 14). Pandeia instrument is MIRI, pixel scale 0.11 arcsec. Fit radius on the real profile, and run a point-source case for a bare dark star.
4. **Break and lines (replaces notebook 03).** Same continuum-versus-dip style:
   - Break: blue side of rest 3650 Å versus F560W at z ~ 14 (NIRCam F444W is the blue side).
   - Lines: excess in the filter that contains Hα or [O III] relative to the neighbor filter.
   - Dark-star-like: weak break and weak line excess.
5. **Rank.** Keep a source if the dark-star term helps, the MIRI size is small, and the break and line scores are weak. Columns: name, z, best dark-star mass, error improvement, radius, break score, line score.
