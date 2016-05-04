#!/usr/bin/env python

'''
plotbase.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Feb 2016
License: MIT.

Contains various useful functions for plotting light curves and associated data.


'''
import os
import os.path
import cPickle as pickle

import numpy as np
from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs

import matplotlib.pyplot as plt

import logging
from datetime import datetime
from traceback import format_exc


#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.plotbase' % parent_name)

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

from lcmath import phase_magseries, phase_magseries_with_errs, \
    phase_bin_magseries, phase_bin_magseries_with_errs, \
    time_bin_magseries, time_bin_magseries_with_errs

from varbase import spline_fit_magseries

#########################
## SIMPLE LIGHT CURVES ##
#########################

def plot_mag_series(times,
                    mags,
                    errs=None,
                    outfile=None,
                    sigclip=30.0,
                    timebin=None,
                    yrange=None):
    '''This plots a magnitude time series.

    If outfile is none, then plots to matplotlib interactive window. If outfile
    is a string denoting a filename, uses that to write a png/eps/pdf figure.

    timebin is either a float indicating binsize in seconds, or None indicating
    no time-binning is required.

    '''

    if errs is not None:

        # remove nans
        find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
        ftimes, fmags, ferrs = times[find], mags[find], errs[find]

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

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

    else:

        # remove nans
        find = npisfinite(times) & npisfinite(mags)
        ftimes, fmags, ferrs = times[find], mags[find], None

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = None

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = None

    # now we proceed to binning
    if timebin and errs is not None:

        binned = time_bin_magseries_with_errs(stimes, smags, serrs,
                                              binsize=timebin)
        btimes, bmags, berrs = (binned['binnedtimes'],
                                binned['binnedmags'],
                                binned['binnederrs'])

    elif timebin and errs is None:

        binned = time_bin_magseries(stimes, smags,
                                    binsize=timebin)
        btimes, bmags, berrs = binned['binnedtimes'], binned['binnedmags'], None

    else:

        btimes, bmags, berrs = stimes, smags, serrs


    # finally, proceed with plotting
    fig = plt.figure()
    fig.set_size_inches(7.5,4.8)

    plt.errorbar(btimes, bmags, fmt='go', yerr=berrs,
                 markersize=2.0, markeredgewidth=0.0, ecolor='grey',
                 capsize=0)

    # make a grid
    plt.grid(color='#a9a9a9',
             alpha=0.9,
             zorder=0,
             linewidth=1.0,
             linestyle=':')

    # fix the ticks to use no offsets
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # get the yrange
    if yrange and isinstance(yrange,list) and len(yrange) == 2:
        ymin, ymax = yrange
    else:
        ymin, ymax = plt.ylim()
    plt.ylim(ymax,ymin)

    plt.xlim(npmin(btimes) - 0.001*npmin(btimes),
             npmax(btimes) + 0.001*npmin(btimes))

    plt.xlabel('time [JD]')
    plt.ylabel('magnitude')

    if outfile and isinstance(outfile, str):

        plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return os.path.abspath(outfile)

    else:

        plt.show()
        plt.close()
        return



#########################
## PHASED LIGHT CURVES ##
#########################

def plot_phased_mag_series(times,
                           mags,
                           period,
                           errs=None,
                           epoch='min',
                           outfile=None,
                           sigclip=30.0,
                           phasewrap=True,
                           phasesort=True,
                           phasebin=None,
                           plotphaselim=[-0.8,0.8],
                           yrange=None):
    '''This plots a phased magnitude time series using the period provided.

    If epoch is None, uses the min(times) as the epoch.

    If epoch is a string 'min', then fits a cubic spline to the phased light
    curve using min(times), finds the magnitude minimum from the fitted light
    curve, then uses the corresponding time value as the epoch.

    If epoch is a float, then uses that directly to phase the light curve and as
    the epoch of the phased mag series plot.

    If outfile is none, then plots to matplotlib interactive window. If outfile
    is a string denoting a filename, uses that to write a png/eps/pdf figure.

    '''

    if errs is not None:

        # remove nans
        find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
        ftimes, fmags, ferrs = times[find], mags[find], errs[find]

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

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

    else:

        # remove nans
        find = npisfinite(times) & npisfinite(mags)
        ftimes, fmags, ferrs = times[find], mags[find], None

        # get the median and stdev = 1.483 x MAD
        median_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - median_mag))) * 1.483

        # sigclip next
        if sigclip:

            sigind = (npabs(fmags - median_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = None

            LOGINFO('sigclip = %s: before = %s observations, '
                    'after = %s observations' %
                    (sigclip, len(times), len(stimes)))

        else:

            stimes = ftimes
            smags = fmags
            serrs = None


    # figure out the epoch, if it's None, use the min of the time
    if epoch is None:

        epoch = npmin(stimes)

    # if the epoch is 'min', then fit a spline to the light curve phased
    # using the min of the time, find the fit mag minimum and use the time for
    # that as the epoch
    elif isinstance(epoch,str) and epoch == 'min':

        spfit = spline_fit_magseries(stimes, smags, serrs, period)
        epoch = spfit['fitepoch']


    # now phase (and optionally, phase bin the light curve)
    if errs is not None:

        # phase the magseries
        phasedlc = phase_magseries_with_errs(stimes,
                                             smags,
                                             serrs,
                                             period,
                                             epoch,
                                             wrap=phasewrap,
                                             sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = phasedlc['errs']

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries_with_errs(plotphase,
                                                        plotmags,
                                                        ploterrs,
                                                        binsize=phasebin)
            plotphase = binphasedlc['binnedphases']
            plotmags = binphasedlc['binnedmags']
            ploterrs = binphasedlc['binnederrs']

    else:

        # phase the magseries
        phasedlc = phase_magseries(stimes,
                                   smags,
                                   period,
                                   epoch,
                                   wrap=phasewrap,
                                   sort=phasesort)
        plotphase = phasedlc['phase']
        plotmags = phasedlc['mags']
        ploterrs = None

        # if we're supposed to bin the phases, do so
        if phasebin:

            binphasedlc = phase_bin_magseries(plotphase,
                                              plotmags,
                                              binsize=phasebin)
            plotphase = binphasedlc['binnedphases']
            plotmags = binphasedlc['binnedmags']
            ploterrs = None


    # finally, make the plots

    # initialize the plot
    fig = plt.figure()
    fig.set_size_inches(7.5,4.8)

    plt.errorbar(plotphase, plotmags, fmt='bo', yerr=ploterrs,
                 markersize=2.0, markeredgewidth=0.0, ecolor='#B2BEB5',
                 capsize=0)

    # make a grid
    plt.grid(color='#a9a9a9',
             alpha=0.9,
             zorder=0,
             linewidth=1.0,
             linestyle=':')

    # make lines for phase 0.0, 0.5, and -0.5
    plt.axvline(0.0,alpha=0.9,linestyle='dashed',color='g')
    plt.axvline(-0.5,alpha=0.9,linestyle='dashed',color='g')
    plt.axvline(0.5,alpha=0.9,linestyle='dashed',color='g')

    # fix the ticks to use no offsets
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    # get the yrange
    if yrange and isinstance(yrange,list) and len(yrange) == 2:
        ymin, ymax = yrange
    else:
        ymin, ymax = plt.ylim()
    plt.ylim(ymax,ymin)

    # set the x axis limit
    if not plotphaselim:
        plot_xlim = plt.xlim()
        plt.xlim((npmin(plotphase)-0.1,
                  npmax(plotphase)+0.1))
    else:
        plt.xlim((plotphaselim[0],plotphaselim[1]))

    # set up the labels
    plt.xlabel('phase')
    plt.ylabel('magnitude')
    plt.title('using period: %.6f d and epoch: %.6f' % (period, epoch))

    # make the figure
    if outfile and isinstance(outfile, str):

        plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return os.path.abspath(outfile)

    else:

        plt.show()
        plt.close()
        return



##################
## PERIODOGRAMS ##
##################

def plot_periodbase_lsp(lspinfo, outfile=None):
    '''Makes a plot of the L-S periodogram obtained from periodbase functions.

    If lspinfo is a dictionary, uses the information directly. If it's a
    filename string ending with .pkl, then this assumes it's a periodbase LSP
    pickle and loads the corresponding info from it.

    '''

    # get the lspinfo from a pickle file transparently
    if isinstance(lspinfo,str) and os.path.exists(lspinfo):
        LOGINFO('loading LSP info from pickle %s' % lspinfo)
        with open(lspinfo,'rb') as infd:
            lspinfo = pickle.load(infd)

    # get the things to plot out of the data
    periods = lspinfo['periods']
    lspvals = lspinfo['lspvals']
    bestperiod = lspinfo['bestperiod']

    # make the LSP plot on the first subplot
    plt.plot(periods,lspvals)

    plt.xscale('log',basex=10)
    plt.xlabel('Period [days]')
    plt.ylabel('LSP power')
    plottitle = 'best period = %.6f d' % bestperiod
    plt.title(plottitle)

    # show the best five peaks on the plot
    for bestperiod, bestpeak in zip(lspinfo['nbestperiods'],
                                    lspinfo['nbestlspvals']):

        plt.annotate('%.6f' % bestperiod,
                     xy=(bestperiod, bestpeak), xycoords='data',
                     xytext=(0.0,25.0), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->"),fontsize='x-small')

    # make a grid
    plt.grid(color='#a9a9a9',
             alpha=0.9,
             zorder=0,
             linewidth=1.0,
             linestyle=':')

    # make the figure
    if outfile and isinstance(outfile, str):

        plt.savefig(outfile,bbox_inches='tight')
        plt.close()
        return os.path.abspath(outfile)

    else:

        plt.show()
        plt.close()
        return
