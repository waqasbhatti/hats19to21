#!/usr/bin/env python

'''
lcmath.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2015

Contains various useful tools for calculating various things related to
lightcurves (like phasing, sigma-clipping, etc.)

'''

import logging


import multiprocessing as mp


import numpy as np

from scipy.spatial import cKDTree as kdtree
from scipy.signal import medfilt
from scipy.linalg import lstsq
from scipy.stats import sigmaclip as stats_sigmaclip
from scipy.optimize import curve_fit

import scipy.stats
import numpy.random as nprand

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.lcmath' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


# DEBUG mode
DEBUG = False

####################
## SIGMA-CLIPPING ##
####################

def sigclip_magseries(times, mags, maxsig=4.0):
    '''
    This sigmaclips a magnitude timeseries given a maxsig value in one shot.
    The median is used as the central value of the mags array.

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    # calculate median and stdev
    mag_median = np.median(mags[finiteind])
    mag_stdev = np.std(mags[finiteind])

    # if a maxsig is provided, then do the sigma-clip
    if maxsig:

        # do the oneshot sigma clip
        excludeind = (np.abs(finite_mags - mag_median)) < (maxsig*mag_stdev)

        final_mags = finite_mags[excludeind]
        final_times = finite_times[excludeind]

    # otherwise, just pass through the finite times and magnitudes
    else:

        final_mags = finite_mags
        final_times = finite_times

    return {'sctimes':final_times,
            'scmags':final_mags,
            'sigclip':maxsig,
            'magmedian':mag_median,
            'magstdev':mag_stdev}



#################
## PHASING LCS ##
#################

def phase_magseries(times, mags, period, epoch, wrap=True, sort=True):
    '''
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
        )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))

    return outdict



def phase_magseries_with_errs(times, mags, errs, period, epoch,
                              wrap=True, sort=True):
    '''
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.

    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
        )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'errs':finite_errs,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]
        outdict['errs'] = outdict['errs'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))
        outdict['errs'] = np.concatenate((outdict['errs'],
                                          outdict['errs']))

    return outdict



#################
## BINNING LCs ##
#################

def time_bin_magseries(times, mags, binsize=540.0):
    '''
    This bins the given mag timeseries in time using the binsize given. binsize
    is in seconds.

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and len(times) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    # convert binsize in seconds to JD units
    binsizejd = binsize/(86400.0)
    nbins = int(np.ceil((np.nanmax(finite_times) -
                         np.nanmin(finite_times))/binsizejd) + 1)

    minjd = np.nanmin(finite_times)
    jdbins = [(minjd + x*binsizejd) for x in range(nbins)]

    # make a KD-tree on the JDs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_times])
    jdtree = kdtree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:
        # find all bin indices close to within binsizejd of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                              binsizejd/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_timeseries_indices and
            len(bin_indices)) > 0:
            binned_finite_timeseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_timeseries_indices = [np.array(x) for x in
                                        binned_finite_timeseries_indices]

    collected_binned_mags['jdbins_indices'] = binned_finite_timeseries_indices
    collected_binned_mags['jdbins'] = jdbins
    collected_binned_mags['nbins'] = len(binned_finite_timeseries_indices)

    # collect the finite_times
    binned_jd = np.array([np.median(finite_times[x])
                          for x in binned_finite_timeseries_indices])
    collected_binned_mags['binnedtimes'] = binned_jd
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_timeseries_indices])
        )

    return collected_binned_mags



def time_bin_magseries_with_errs(times, mags, errs, binsize=540.0):
    '''
    This bins the given mag timeseries in time using the binsize given. binsize
    is in seconds.

    '''

    # check if the input arrays are ok
    if not(times.shape and mags.shape and errs.shape and
           len(times) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times) & np.isfinite(errs)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    # convert binsize in seconds to JD units
    binsizejd = binsize/(86400.0)
    nbins = int(np.ceil((np.nanmax(finite_times) -
                         np.nanmin(finite_times))/binsizejd) + 1)

    minjd = np.nanmin(finite_times)
    jdbins = [(minjd + x*binsizejd) for x in range(nbins)]

    # make a KD-tree on the JDs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_times])
    jdtree = kdtree(time_coords)
    binned_finite_timeseries_indices = []

    collected_binned_mags = {}

    for jd in jdbins:
        # find all bin indices close to within binsizejd of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = jdtree.query_ball_point(np.array([jd,1.0]),
                                              binsizejd/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_timeseries_indices and
            len(bin_indices)) > 0:
            binned_finite_timeseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_timeseries_indices = [np.array(x) for x in
                                        binned_finite_timeseries_indices]

    collected_binned_mags['jdbins_indices'] = binned_finite_timeseries_indices
    collected_binned_mags['jdbins'] = jdbins
    collected_binned_mags['nbins'] = len(binned_finite_timeseries_indices)

    # collect the finite_times
    binned_jd = np.array([np.median(finite_times[x])
                          for x in binned_finite_timeseries_indices])
    collected_binned_mags['binnedtimes'] = binned_jd
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_timeseries_indices])
        )

    # FIXME: calculate the error in the median-binned magnitude correctly
    # for now, just take the median of the errors in this bin
    collected_binned_mags['binnederrs'] = (
        np.array([np.median(finite_errs[x])
                  for x in binned_finite_timeseries_indices])
        )


    return collected_binned_mags



def phase_bin_magseries(phases, mags, binsize=0.005):
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

    '''

    # check if the input arrays are ok
    if not(phases.shape and mags.shape and len(phases) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and phases
    finiteind = np.isfinite(mags) & np.isfinite(phases)
    finite_phases = phases[finiteind]
    finite_mags = mags[finiteind]

    nbins = int(np.ceil((np.nanmax(finite_phases) -
                         np.nanmin(finite_phases))/binsize) + 1)

    minphase = np.nanmin(finite_phases)
    phasebins = [(minphase + x*binsize) for x in range(nbins)]

    # make a KD-tree on the PHASEs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_phases])
    phasetree = kdtree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:
        # find all bin indices close to within binsize of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                              binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices)) > 0:
            binned_finite_phaseseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_phaseseries_indices = [np.array(x) for x in
                                         binned_finite_phaseseries_indices]

    collected_binned_mags['phasebins_indices'] = (
        binned_finite_phaseseries_indices
    )
    collected_binned_mags['phasebins'] = phasebins
    collected_binned_mags['nbins'] = len(binned_finite_phaseseries_indices)

    # collect the finite_phases
    binned_phase = np.array([np.median(finite_phases[x])
                          for x in binned_finite_phaseseries_indices])
    collected_binned_mags['binnedphases'] = binned_phase
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_phaseseries_indices])
        )


    return collected_binned_mags



def phase_bin_magseries_with_errs(phases, mags, errs, binsize=0.005):
    '''
    This bins a magnitude timeseries in phase using the binsize (in phase)
    provided.

    '''

    # check if the input arrays are ok
    if not(phases.shape and mags.shape and len(phases) > 10 and len(mags) > 10):

        LOGERROR("input time/mag arrays don't have enough elements")
        return

    # find all the finite values of the magnitudes and phases
    finiteind = np.isfinite(mags) & np.isfinite(phases) & np.isfinite(errs)
    finite_phases = phases[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    nbins = int(np.ceil((np.nanmax(finite_phases) -
                         np.nanmin(finite_phases))/binsize) + 1)

    minphase = np.nanmin(finite_phases)
    phasebins = [(minphase + x*binsize) for x in range(nbins)]

    # make a KD-tree on the PHASEs so we can do fast distance calculations.  we
    # need to add a bogus y coord to make this a problem that KD-trees can
    # solve.
    time_coords = np.array([[x,1.0] for x in finite_phases])
    phasetree = kdtree(time_coords)
    binned_finite_phaseseries_indices = []

    collected_binned_mags = {}

    for phase in phasebins:
        # find all bin indices close to within binsize of this point
        # using the kdtree query. we use the p-norm = 1 (I think this
        # means straight-up pairwise distance? FIXME: check this)
        bin_indices = phasetree.query_ball_point(np.array([phase,1.0]),
                                              binsize/2.0, p=1.0)

        # if the bin_indices have already been collected, then we're
        # done with this bin, move to the next one. if they haven't,
        # then this is the start of a new bin.
        if (bin_indices not in binned_finite_phaseseries_indices and
            len(bin_indices)) > 0:
            binned_finite_phaseseries_indices.append(bin_indices)

    # convert to ndarrays
    binned_finite_phaseseries_indices = [np.array(x) for x in
                                         binned_finite_phaseseries_indices]

    collected_binned_mags['phasebins_indices'] = (
        binned_finite_phaseseries_indices
    )
    collected_binned_mags['phasebins'] = phasebins
    collected_binned_mags['nbins'] = len(binned_finite_phaseseries_indices)

    # collect the finite_phases
    binned_phase = np.array([np.median(finite_phases[x])
                          for x in binned_finite_phaseseries_indices])
    collected_binned_mags['binnedphases'] = binned_phase
    collected_binned_mags['binsize'] = binsize

    # median bin the magnitudes according to the calculated indices
    collected_binned_mags['binnedmags'] = (
        np.array([np.median(finite_mags[x])
                  for x in binned_finite_phaseseries_indices])
        )
    collected_binned_mags['binnederrs'] = (
        np.array([np.median(finite_errs[x])
                  for x in binned_finite_phaseseries_indices])
        )


    return collected_binned_mags



###################
## EPD FUNCTIONS ##
###################


def epd_diffmags(coeff, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag):
    '''
    This calculates the difference in mags after EPD coefficients are
    calculated.

    final EPD mags = median(magseries) + epd_diffmags()

    '''

    return -(coeff[0]*fsv**2. +
             coeff[1]*fsv +
             coeff[2]*fdv**2. +
             coeff[3]*fdv +
             coeff[4]*fkv**2. +
             coeff[5]*fkv +
             coeff[6] +
             coeff[7]*fsv*fdv +
             coeff[8]*fsv*fkv +
             coeff[9]*fdv*fkv +
             coeff[10]*np.sin(2*np.pi*xcc) +
             coeff[11]*np.cos(2*np.pi*xcc) +
             coeff[12]*np.sin(2*np.pi*ycc) +
             coeff[13]*np.cos(2*np.pi*ycc) +
             coeff[14]*np.sin(4*np.pi*xcc) +
             coeff[15]*np.cos(4*np.pi*xcc) +
             coeff[16]*np.sin(4*np.pi*ycc) +
             coeff[17]*np.cos(4*np.pi*ycc) +
             coeff[18]*bgv +
             coeff[19]*bge -
             mag)


def epd_magseries(mag, fsv, fdv, fkv, xcc, ycc, bgv, bge,
                  smooth=21, sigmaclip=3.0):
    '''
    Detrends a magnitude series given in mag using accompanying values of S in
    fsv, D in fdv, K in fkv, x coords in xcc, y coords in ycc, background in
    bgv, and background error in bge. smooth is used to set a smoothing
    parameter for the fit function.

    This returns EPD mag corrections. To convert RMx to EPx, do:

    EPx = RMx + correction

    '''

    # find all the finite values of the magnitude
    finiteind = np.isfinite(mag)

    # calculate median and stdev
    mag_median = np.median(mag[finiteind])
    mag_stdev = np.nanstd(mag)

    # if we're supposed to sigma clip, do so
    if sigmaclip:
        excludeind = abs(mag - mag_median) < sigmaclip*mag_stdev
        finalind = finiteind & excludeind
    else:
        finalind = finiteind

    final_mag = mag[finalind]
    final_len = len(final_mag)

    if DEBUG:
        print('final epd fit mag len = %s' % final_len)

    # smooth the signal
    smoothedmag = medfilt(final_mag, smooth)

    # make the linear equation matrix
    epdmatrix = np.c_[fsv[finalind]**2.0,
                      fsv[finalind],
                      fdv[finalind]**2.0,
                      fdv[finalind],
                      fkv[finalind]**2.0,
                      fkv[finalind],
                      np.ones(final_len),
                      fsv[finalind]*fdv[finalind],
                      fsv[finalind]*fkv[finalind],
                      fdv[finalind]*fkv[finalind],
                      np.sin(2*np.pi*xcc[finalind]),
                      np.cos(2*np.pi*xcc[finalind]),
                      np.sin(2*np.pi*ycc[finalind]),
                      np.cos(2*np.pi*ycc[finalind]),
                      np.sin(4*np.pi*xcc[finalind]),
                      np.cos(4*np.pi*xcc[finalind]),
                      np.sin(4*np.pi*ycc[finalind]),
                      np.cos(4*np.pi*ycc[finalind]),
                      bgv[finalind],
                      bge[finalind]]

    # solve the equation epdmatrix * x = smoothedmag
    # return the EPD differential mags if the solution succeeds
    try:

        coeffs, residuals, rank, singulars = lstsq(epdmatrix, smoothedmag)

        if DEBUG:
            print('coeffs = %s, residuals = %s' % (coeffs, residuals))

        return epd_diffmags(coeffs, fsv, fdv, fkv, xcc, ycc, bgv, bge, mag)

    # if the solution fails, return nothing
    except Exception as e:

        LOGEXCEPTION('%sZ: EPD solution did not converge! Error was: %s' %
                     (datetime.utcnow().isoformat(), e))
        return None
