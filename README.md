This repository contains the following files that go along with the paper:

HATS-19b, HATS-20b, HATS-21b: three transiting hot-Saturns from the HATSouth
survey (Bhatti et al. 2016)

## Jupyter notebook

- `hats19to21.ipynb`: contains code to reproduce some of the plots and perform
  random forest regression on the observed planet radii given several other
  planet features

## Python modules

- `hatlc.py`: to read new-format HAT LCs
- `oldhatlc.py`: to read older-format HAT LCs (like those from http://hatnet.org
  or http://hatsouth.org)
- `periodbase.py`: some functions to run period-finding algorithms
- `glsp.py`: a parallelized Generalized-Lomb-Scargle implementation
- `varbase.py`: some functions to calculate variability metrics and manipulate
  LCs
- `plotbase.py`: some functions to plot LCs
- `fortney2k7.py`: planet models from <a
  href="http://adsabs.harvard.edu/abs/2007ApJ...659.1661F">Fortney et
  al. (2007)</a> in an importable Python dict for convenience

## Python pickles:

- `saturns-regressor.pkl`: a pickle of the best `RandomForestRegressor` instance
  chosen by cross-validation for Saturn-class planets
- `jupiters-regressor.pkl`: a pickle of the best `RandomForestRegressor`
  instance chosen by cross-validation for Jupiter-class planets
- `highmass-regressor.pkl`: a pickle of the best `RandomForestRegressor`
  instance chosen by cross-validation for high-mass planets

## Data files

The following data files should be included along with this repository:

- `planet-mass-radius-with-errors.csv`: CSV file containing transiting planet
  info from http://exoplanets.org, last updated on 2016-04-12.
- `HATS-19b-hatlc.csv.gz`: the HATSouth light curve for HATS-19b.
- `HATS-20b-old-hatlc.csv.gz`: the HATSouth light curve for HATS-20b.
- `HATS-21b-old-hatlc.csv.gz`: the HATSouth light curve for HATS-21b.
