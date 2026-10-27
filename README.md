This dark star repo is built on codes authored by Shafaat Mahmud and Jillian Paulin, whose original notebooks for finding DS candidates in NIRSpec and NIRCam can be found in `original_notebooks/`. Those are now organized into a 3-step pipeline - spectral fitting, morphology fitting, He II 1640 significance - laid out in `organized_notebooks/`. 

Data needed to run the above notebooks have already been included in the repo, in `data/` and `tlusty_templates/`. MIRI data required by the new project can be downloaded by running `find_miri_imag_sources.ipynb` and `find_miri_spec_sources.ipynb`. Pandeia data and PSFs also need to be separately downloaded.

`production/` contains parallelization scripts for the pipeline, geared toward MIRI.