#!/usr/bin/env python

'''
periodbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Dec 2015

Contains various useful tools for period finding.


'''


from multiprocessing import Pool
import ctypes
import logging
from datetime import datetime
from traceback import format_exc

import numpy as np

# import these to avoid lookup overhead
from numpy import nan as npnan, sum as npsum, abs as npabs, \
    roll as nproll, isfinite as npisfinite, std as npstd, \
    sign as npsign, sqrt as npsqrt, median as npmedian, \
    array as nparray, percentile as nppercentile, \
    polyfit as nppolyfit, var as npvar, max as npmax, min as npmin, \
    log10 as nplog10, arange as nparange, pi as MPI, floor as npfloor, \
    argsort as npargsort, cos as npcos, sin as npsin, tan as nptan, \
    where as npwhere, linspace as nplinspace, \
    zeros_like as npzeros_like, full_like as npfull_like, \
    arctan as nparctan, nanargmax as npnanargmax

from scipy.signal import lombscargle


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.periodbase' % parent_name)

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


###################
## LOCAL IMPORTS ##
###################

from lcmath import phase_magseries, sigclip_magseries, time_bin_magseries

from glsp import generalized_lsp_value as glspval, \
    generalized_lsp_value_notau as glspvalnt


####################################
## BINNED DWORETSKY STRING LENGTH ##
####################################

def dwsl_value(times, mags, errs, period):
    '''String length value for a single period.

    Note: mags should be modified mags:

    mod_mags = (mags - np.min(mags))/(2.0*(np.max(mags) - np.min(mags))) - 0.25

    times, mags, errs should NOT have any nans

    '''

    phase = (times - times[0])/period - npfloor((times - times[0])/period)
    psort = npargsort(phase)

    # sort by phase order
    stimes = times[psort]
    sphase = phase[psort]
    smags = mags[psort]
    serrs = errs[psort]

    ndet = len(stimes)

    epsilon = 2.0*npmedian(serrs)
    deltal = (
        0.34*(epsilon - 0.5*epsilon*epsilon)*(ndet - npsqrt(10.0/epsilon))
    )
    keep_threshold1 = 1.6 + 1.2*deltal

    l = 0.212*ndet
    sigl = ndet/37.5
    keep_threshold2 = l + 4.0*sigl

    # calculate the string length


    return periodogramvalue



def dwsl_periodogram(times,
                     mags,
                     errs,
                     frequencies,
                     sigclip=None,
                     timebinsec=None,
                     nworkers=4):
    '''
    This runs the loops for the DWSL calculation.

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = ferrs

        if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

            # get the frequencies to use
            startf = 1.0/endp
            endf = 1.0/startp
            omegas = 2*np.pi*np.arange(startf, endf, stepsize)

            # map to parallel workers
            pool = Pool(nworkers)

            tasks = [(stimes, smags, serrs, x) for x in omegas]
            lsp = pool.map(glspfunc, tasks)

            pool.close()
            pool.join()
            del pool

            lsp = np.array(lsp)
            periods = 2.0*np.pi/omegas

            # find the nbestpeaks for the periodogram: 1. sort the lsp array by
            # highest value first 2. go down the values until we find five
            # values that are separated by at least periodepsilon in period
            bestperiodind = npnanargmax(lsp)

            sortedlspind = np.argsort(lsp)[::-1]
            sortedlspperiods = periods[sortedlspind]
            sortedlspvals = lsp[sortedlspind]

            prevbestlspval = sortedlspvals[0]
            # now get the nbestpeaks
            nbestperiods, nbestlspvals, peakcount = (
                [periods[bestperiodind]],
                [lsp[bestperiodind]],
                1
            )
            prevperiod = sortedlspperiods[0]

            # find the best nbestpeaks in the lsp and their periods
            for period, lspval in zip(sortedlspperiods, sortedlspvals):

                if peakcount == nbestpeaks:
                    break
                perioddiff = abs(period - prevperiod)
                bestperiodsdiff = [abs(period - x) for x in nbestperiods]

                # print('prevperiod = %s, thisperiod = %s, '
                #       'perioddiff = %s, peakcount = %s' %
                #       (prevperiod, period, perioddiff, peakcount))

                # this ensures that this period is different from the last
                # period and from all the other existing best periods by
                # periodepsilon to make sure we jump to an entire different peak
                # in the periodogram
                if (perioddiff > periodepsilon and
                    all(x > periodepsilon for x in bestperiodsdiff)):
                    nbestperiods.append(period)
                    nbestlspvals.append(lspval)
                    peakcount = peakcount + 1

                prevperiod = period


            return {'bestperiod':periods[bestperiodind],
                    'bestlspval':lsp[bestperiodind],
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':nbestlspvals,
                    'nbestperiods':nbestperiods,
                    'lspvals':lsp,
                    'omegas':omegas,
                    'periods':periods}

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'omegas':None,
                    'periods':None}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'omegas':None,
                'periods':None}

###################################
## PHASE DISPERSION MINIMIZATION ##
###################################


##########################
## ANALYSIS of VARIANCE ##
##########################




##############################
## GENERALIZED LOMB-SCARGLE ##
##############################

def glsp_worker(task):
    '''
    This is a worker to wrap the scipy lombscargle function.

    '''

    try:
        return glspval(*task)
    except Exception as e:
        return npnan



def glsp_worker_notau(task):
    '''
    This is a worker to wrap the scipy lombscargle function.

    '''

    try:
        return glspvalnt(*task)
    except Exception as e:
        return npnan


def pgen_lsp(
        times,
        mags,
        errs,
        startp,
        endp,
        nbestpeaks=5,
        periodepsilon=0.1, # 0.1
        stepsize=1.0e-4,
        nworkers=4,
        sigclip=30.0,
        glspfunc=glsp_worker,
):
    '''
    This calculates the generalized LSP value for a single frequency omega given
    times, mags, errors. Uses the algorithm from Zechmeister and Kurster (2009).

    '''

    # get rid of nans first
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    ftimes = times[find]
    fmags = mags[find]
    ferrs = errs[find]

    if len(ftimes) > 9 and len(fmags) > 9 and len(ferrs) > 9:

        # get the median and stdev = 1.483 x MAD
        median_mag = np.median(fmags)
        stddev_mag = (np.median(np.abs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (np.abs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = ferrs

        if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

            # get the frequencies to use
            startf = 1.0/endp
            endf = 1.0/startp
            omegas = 2*np.pi*np.arange(startf, endf, stepsize)

            # map to parallel workers
            pool = Pool(nworkers)

            tasks = [(stimes, smags, serrs, x) for x in omegas]
            lsp = pool.map(glspfunc, tasks)

            pool.close()
            pool.join()
            del pool

            lsp = np.array(lsp)
            periods = 2.0*np.pi/omegas

            # find the nbestpeaks for the periodogram: 1. sort the lsp array by
            # highest value first 2. go down the values until we find five
            # values that are separated by at least periodepsilon in period
            bestperiodind = npnanargmax(lsp)

            sortedlspind = np.argsort(lsp)[::-1]
            sortedlspperiods = periods[sortedlspind]
            sortedlspvals = lsp[sortedlspind]

            prevbestlspval = sortedlspvals[0]
            # now get the nbestpeaks
            nbestperiods, nbestlspvals, peakcount = (
                [periods[bestperiodind]],
                [lsp[bestperiodind]],
                1
            )
            prevperiod = sortedlspperiods[0]

            # find the best nbestpeaks in the lsp and their periods
            for period, lspval in zip(sortedlspperiods, sortedlspvals):

                if peakcount == nbestpeaks:
                    break
                perioddiff = abs(period - prevperiod)
                bestperiodsdiff = [abs(period - x) for x in nbestperiods]

                # print('prevperiod = %s, thisperiod = %s, '
                #       'perioddiff = %s, peakcount = %s' %
                #       (prevperiod, period, perioddiff, peakcount))

                # this ensures that this period is different from the last
                # period and from all the other existing best periods by
                # periodepsilon to make sure we jump to an entire different peak
                # in the periodogram
                if (perioddiff > periodepsilon and
                    all(x > periodepsilon for x in bestperiodsdiff)):
                    nbestperiods.append(period)
                    nbestlspvals.append(lspval)
                    peakcount = peakcount + 1

                prevperiod = period


            return {'bestperiod':periods[bestperiodind],
                    'bestlspval':lsp[bestperiodind],
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':nbestlspvals,
                    'nbestperiods':nbestperiods,
                    'lspvals':lsp,
                    'omegas':omegas,
                    'periods':periods}

        else:

            LOGERROR('no good detections for these times and mags, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'omegas':None,
                    'periods':None}
    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'omegas':None,
                'periods':None}



##############################
## LOMB-SCARGLE PERIODOGRAM ##
##############################

def townsend_lombscargle_value(times, mags, omega):
    '''
    This calculates the periodogram value for each omega (= 2*pi*f). Mags must
    be normalized to zero with variance scaled to unity.

    '''
    cos_omegat = npcos(omega*times)
    sin_omegat = npsin(omega*times)

    xc = npsum(mags*cos_omegat)
    xs = npsum(mags*sin_omegat)

    cc = npsum(cos_omegat*cos_omegat)
    ss = npsum(sin_omegat*sin_omegat)

    cs = npsum(cos_omegat*sin_omegat)

    tau = nparctan(2*cs/(cc - ss))/(2*omega)

    ctau = npcos(omega*tau)
    stau = npsin(omega*tau)

    leftsumtop = (ctau*xc + stau*xs)*(ctau*xc + stau*xs)
    leftsumbot = ctau*ctau*cc + 2.0*ctau*stau*cs + stau*stau*ss
    leftsum = leftsumtop/leftsumbot

    rightsumtop = (ctau*xs - stau*xc)*(ctau*xs - stau*xc)
    rightsumbot = ctau*ctau*ss - 2.0*ctau*stau*cs + stau*stau*cc
    rightsum = rightsumtop/rightsumbot

    pval = 0.5*(leftsum + rightsum)

    return pval


def townsend_lombscargle_wrapper(task):
    '''
    This wraps the function above for use with mp.Pool.

    task[0] = times
    task[1] = mags
    task[2] = omega

    '''

    try:
        return townsend_lombscargle_value(*task)

    # if the LSP calculation fails for this omega, return a npnan
    except Exception as e:
        return npnan



def parallel_townsend_lsp(times, mags, startp, endp,
                          stepsize=1.0e-4,
                          nworkers=4):
    '''
    This calculates the Lomb-Scargle periodogram for the frequencies
    corresponding to the period interval (startp, endp) using a frequency step
    size of stepsize cycles/day. This uses the algorithm in Townsend 2010.

    '''

    # make sure there are no nans anywhere
    finiteind = np.isfinite(times) & np.isfinite(mags)
    ftimes, fmags = times[finiteind], mags[finiteind]

    # renormalize the mags to zero and scale them so that the variance = 1
    nmags = (fmags - np.median(fmags))/np.std(fmags)

    startf = 1.0/endp
    endf = 1.0/startp
    omegas = 2*np.pi*np.arange(startf, endf, stepsize)

    # parallel map the lsp calculations
    pool = Pool(nworkers)

    tasks = [(ftimes, nmags, x) for x in omegas]
    lsp = pool.map(townsend_lombscargle_wrapper, tasks)

    pool.close()
    pool.join()

    return np.array(omegas), np.array(lsp)



def parallel_townsend_lsp_sharedarray(times, mags, startp, endp,
                                      stepsize=1.0e-4,
                                      nworkers=16):
    '''
    This is a version of the above which uses shared ctypes arrays for the times
    and mags arrays so as not to copy them to each worker process.

    TODO: we'll need to pass a single argument to the worker so make a 2D array
    and wrap the worker function with partial?

    '''

########################
## SCIPY LOMB-SCARGLE ##
########################

def parallel_scipylsp_worker(task):
    '''
    This is a worker to wrap the scipy lombscargle function.

    '''

    try:
        return lombscargle(*task)
    except Exception as e:
        return npnan



def scipylsp_parallel(times,
                      mags,
                      errs, # ignored but for consistent API
                      startp,
                      endp,
                      nbestpeaks=5,
                      periodepsilon=0.1, # 0.1
                      stepsize=1.0e-4,
                      nworkers=4,
                      sigclip=None,
                      timebin=None):
    '''
    This uses the LSP function from the scipy library, which is fast as hell. We
    try to make it faster by running LSP for sections of the omegas array in
    parallel.

    '''

    # make sure there are no nans anywhere
    finiteind = np.isfinite(times) & np.isfinite(mags)
    ftimes, fmags = times[finiteind], mags[finiteind]

    if len(ftimes) > 0 and len(fmags) > 0:

        # sigclip the lightcurve if asked to do so
        if sigclip:
            sigclipped = sigclip_magseries(ftimes,
                                           fmags,
                                           maxsig=sigclip)
            worktimes = sigclipped['sctimes']
            workmags = sigclipped['scmags']
            LOGINFO('ndet after sigclipping = %s' % len(worktimes))

        else:
            worktimes = ftimes
            workmags = fmags

        # bin the lightcurve if asked to do so
        if timebin:

            binned = time_bin_magseries(worktimes, workmags, binsize=timebin)
            worktimes = binned['binnedtimes']
            workmags = binned['binnedmags']

        # renormalize the working mags to zero and scale them so that the
        # variance = 1 for use with our LSP functions
        normmags = (workmags - np.median(workmags))/np.std(workmags)

        startf = 1.0/endp
        endf = 1.0/startp
        omegas = 2*np.pi*np.arange(startf, endf, stepsize)

        # partition the omegas array by nworkers
        tasks = []
        chunksize = int(float(len(omegas))/nworkers) + 1
        tasks = [omegas[x*chunksize:x*chunksize+chunksize]
                 for x in range(nworkers)]

        # map to parallel workers
        pool = Pool(nworkers)

        tasks = [(worktimes, normmags, x) for x in tasks]
        lsp = pool.map(parallel_scipylsp_worker, tasks)

        pool.close()
        pool.join()

        lsp = np.concatenate(lsp)
        periods = 2.0*np.pi/omegas

        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # highest value first 2. go down the values until we find five values
        # that are separated by at least periodepsilon in period
        bestperiodind = npnanargmax(lsp)

        sortedlspind = np.argsort(lsp)[::-1]
        sortedlspperiods = periods[sortedlspind]
        sortedlspvals = lsp[sortedlspind]

        prevbestlspval = sortedlspvals[0]
        # now get the nbestpeaks
        nbestperiods, nbestlspvals, peakcount = (
            [periods[bestperiodind]],
            [lsp[bestperiodind]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval in zip(sortedlspperiods, sortedlspvals):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last period
            # and from all the other existing best periods by periodepsilon to
            # make sure we jump to an entire different peak in the periodogram
            if (perioddiff > periodepsilon and
                all(x > periodepsilon for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                peakcount = peakcount + 1

            prevperiod = period


        return {'bestperiod':periods[bestperiodind],
                'bestlspval':lsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'omegas':omegas,
                'periods':periods}

    else:

        LOGERROR('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None}
