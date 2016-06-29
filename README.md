This repository contains the following files that go along with the paper:

HATS-19b, HATS-20b, HATS-21b: three transiting hot-Saturns from the HATSouth
survey (Bhatti et al. 2016)

## Jupyter notebook

- `hats19to21.ipynb`: contains code to reproduce some of the plots and perform
  random forest regression on the observed planet radii given several other
  planet features

This notebook requires the following modules:

- numpy (http://www.numpy.org/)
- scipy (http://scipy.org/scipylib/index.html)
- pyfits (or astropy.fits; http://www.stsci.edu/institute/software_hardware/pyfits)
- matplotlib (http://matplotlib.org/)
- Jupyter and IPython (http://ipython.org/)
- scikit-learn (http://www.scikit-learn.org)

## Python modules

- `hatlc.py`: to read new-format HAT LCs
- `oldhatlc.py`: to read older-format HAT LCs (like those from http://hatnet.org
  or http://hatsouth.org)
- `periodbase.py`: some functions to run period-finding algorithms
- `glsp.py`: a Generalized-Lomb-Scargle implementation (Zechmeister+ 2009)
- `varbase.py`: some functions to calculate variability metrics and manipulate
  LCs
- `plotbase.py`: some functions to plot LCs and LSPs
- `fortney2k7.py`: planet models from <a
  href="http://adsabs.harvard.edu/abs/2007ApJ...659.1661F">Fortney et
  al. (2007)</a> in an importable Python dict for convenience

## Python pickles

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
- `HATS-19b-RVs.txt`: radial velocities and bisector spans for HATS-19b
- `HATS-20b-RVs.txt`: radial velocities and bisector spans for HATS-20b
- `HATS-21b-RVs.txt`: radial velocities and bisector spans for HATS-21b
- `HATS-19-astraluxsur-bestframe.fits`: the best combined frame from Astra Lux Sur lucky imaging camera for HATS-19
- `HATS-20-astraluxsur-bestframe.fits`: the best combined frame from Astra Lux Sur lucky imaging camera for HATS-20
- `HATS-19-astraluxsur-contrastcurve.txt`: the 5-sigma contrast curve for HATS-19
- `HATS-20-astraluxsur-contrastcurve.txt`: the 5-sigma contrast curve for HATS-20
