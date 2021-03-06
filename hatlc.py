#!/usr/bin/env python
'''
hatlc.py - Waqas Bhatti (wbhatti@astro.princeton.edu) - Jan 2016
License: MIT - see LICENSE for the full text.

This contains functions to read HAT sqlite ("sqlitecurves") and CSV light curves
generated by the new HAT data server.

The most probably useful functions in this module are:

read_and_filter_sqlitecurve(lcfile, columns=None, sqlfilters=None,
                            raiseonfail=False):
    This reads the sqlitecurve and optionally filters it, returns an lcdict.

    Returns columns requested in columns. If None, then returns all columns
    present in the latest columnlist in the lightcurve. See COLUMNDEFS for the
    full list of HAT LC columns.

    If sqlfilters is not None, it must be a list of text sql filters that apply
    to the columns in the lightcurve.

    This returns an lcdict with an added 'lcfiltersql' key that indicates what
    the parsed SQL filter string was.

read_csvlc(lcfile):
    This reads the HAT data server producd CSV light curve into an lcdict.

    lcfile is the HAT gzipped CSV LC (with a .hatlc.csv.gz extension)

'''



####################
## SYSTEM IMPORTS ##
####################

import os.path
import gzip
import logging
from datetime import datetime
from traceback import format_exc
import subprocess
import re

import numpy as np
from numpy import nan

import sqlite3 as sql

# try using simplejson if that's available; it's faster than stdlib json
try:
    import simplejson as json
except:
    import json


#################
## DEFINITIONS ##
#################

# LC column definitions
# the first elem is the column description, the second is the format to use when
# writing a CSV LC column, the third is the type to use when parsing a CSV LC
# column
COLUMNDEFS = {
    # TIME
    'rjd':('time of observation in Reduced Julian date (JD = 2400000.0 + RJD)',
           '%.7f',
           float),
    'bjd':(('time of observation in Baryocentric Julian date '
            '(note: this is BJD_UTC, not BJD_TDB)'),
           '%.7f',
           float),
    # FRAME METADATA
    'net':('network of telescopes observing this target',
           '%s',
           str),
    'stf':('station ID of the telescope observing this target',
           '%i',
           int),
    'cfn':('camera frame serial number',
           '%i',
           int),
    'cfs':('camera subframe id',
           '%s',
           str),
    'ccd':('camera CCD position number',
           '%i',
           int),
    'prj':('project ID of this observation',
           '%s',
           str),
    'fld':('observed field name',
           '%s',
           str),
    'frt':('image frame type (flat, object, etc.)',
           '%s',
           str),
    # FILTER CONFIG
    'flt':('filter ID from the filters table',
           '%i',
           int),
    'flv':('filter version used',
           '%i',
           int),
    # CAMERA CONFIG
    'cid':('camera ID ',
           '%i',
           int),
    'cvn':('camera version',
           '%i',
           int),
    'cbv':('camera bias-frame version',
           '%i',
           int),
    'cdv':('camera dark-frame version',
           '%i',
           int),
    'cfv':('camera flat-frame version',
           '%i',
           int),
    'exp':('exposure time for this observation in seconds',
           '%.3f',
           float),
    # TELESCOPE CONFIG
    'tid':('telescope ID',
           '%i',
           int),
    'tvn':('telescope version',
           '%i',
           int),
    'tfs':('telescope focus setting',
           '%i',
           int),
    'ttt':('telescope tube temperature [deg]',
           '%.3f',
           float),
    'tms':('telescope mount state (tracking, drizzling, etc.)',
           '%s',
           str),
    'tmi':('telescope mount ID',
           '%i',
           int),
    'tmv':('telescope mount version',
           '%i',
           int),
    'tgs':('telescope guider status (MGen)',
           '%s',
           str),
    # ENVIRONMENT
    'mph':('moon phase at this observation',
           '%.2f',
           float),
    'iha':('hour angle of object at this observation',
           '%.3f',
           float),
    'izd':('zenith distance of object at this observation',
           '%.3f',
           float),
    # APERTURE PHOTOMETRY METADATA
    'xcc':('x coordinate on CCD chip',
           '%.3f',
           float),
    'ycc':('y coordinate on CCD chip',
           '%.3f',
           float),
    'bgv':('sky background measurement around object in ADU',
           '%.3f',
           float),
    'bge':('error in sky background measurement in ADU',
           '%.3f',
           float),
    'fsv':('source extraction S parameter (the PSF spatial RMS)',
           '%.5f',
           float),
    'fdv':('source extraction D parameter (the PSF spatial ellipticity in xy)',
           '%.5f',
           float),
    'fkv':('source extraction K parameter (the PSF spatial diagonal ellipticity)',
           '%.5f',
           float),
    # APERTURE PHOTOMETRY COLUMNS (NOTE: these are per aperture)
    'aim':('aperture photometry raw instrumental magnitude in aperture %s',
            '%.5f',
            float),
    'aie':('aperture photometry raw instrumental mag error in aperture %s',
            '%.5f',
            float),
    'aiq':('aperture photometry raw instrumental mag quality flag for aperture %s',
            '%s',
            str),
    'arm':('aperture photometry fit magnitude in aperture %s',
            '%.5f',
            float),
    'aep':('aperture photometry EPD magnitude in aperture %s',
            '%.5f',
            float),
    'atf':('aperture photometry TFA magnitude in aperture %s',
            '%.5f',
            float),
    # PSF FIT PHOTOMETRY METADATA
    'psv':('PSF fit S parameter (the PSF spatial RMS)',
           '%.5f',
           float),
    'pdv':('PSF fit D parameter (the PSF spatial ellipticity in xy)',
           '%.5f',
           float),
    'pkv':('PSF fit K parameter (the PSF spatial diagonal ellipticity)',
           '%.5f',
           float),
    'ppx':('PSF fit number of pixels used for fit',
           '%i',
           int),
    'psn':('PSF fit signal-to-noise ratio',
           '%.3f',
           float),
    'pch':('PSF fit chi-squared value',
           '%.5f',
           float),
    # PSF FIT PHOTOMETRY COLUMNS
    'psim':('PSF fit instrumental raw magnitude',
            '%.5f',
            float),
    'psie':('PSF fit instrumental raw magnitude error',
            '%.5f',
            float),
    'psiq':('PSF fit instrumental raw magnitude quality flag',
            '%s',
            str),
    'psrm':('PSF fit final magnitude after mag-fit',
            '%.5f',
            float),
    'psrr':('PSF fit residual',
            '%.5f',
            float),
    'psrn':('PSF fit number of sources used',
            '%i',
            int),
    'psep':('PSF fit EPD magnitude',
            '%.5f',
            float),
    'pstf':('PSF fit TFA magnitude',
            '%.5f',
            float),
    # IMAGE SUBTRACTION PHOTOMETRY METADATA
    'xic':('x coordinate on CCD chip after image-subtraction frame warp',
           '%.3f',
           float),
    'yic':('y coordinate on CCD chip after image-subtraction frame warp',
           '%.3f',
           float),
    # IMAGE SUBTRACTION PHOTOMETRY COLUMNS
    'irm':('image subtraction fit magnitude in aperture %s',
            '%.5f',
            float),
    'ire':('image subtraction fit magnitude error in aperture %s',
            '%.5f',
            float),
    'irq':('image subtraction fit magnitude quality flag for aperture %s',
            '%s',
            str),
    'iep':('image subtraction EPD magnitude in aperture %s',
            '%.5f',
            float),
    'itf':('image subtraction TFA magnitude in aperture %s',
            '%.5f',
            float),
}


LC_MAG_COLUMNS = ('aim','arm','aep','atf',
                  'psim','psrm','psep','pstf',
                  'irm','iep','itf')

LC_ERR_COLUMNS = ('aie','psie','ire')

LC_FLAG_COLUMNS = ('aiq','psiq','irq')

# used to validate the filter string
# http://www.sqlite.org/lang_keywords.html
SQLITE_ALLOWED_WORDS = ['and','between','in',
                        'is','isnull','like','not',
                        'notnull','null','or',
                        '=','<','>','<=','>=','!=','%']



#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.hatlc' % parent_name)

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



#######################
## UTILITY FUNCTIONS ##
#######################

# this is from Tornado's source (MIT License):
# http://www.tornadoweb.org/en/stable/_modules/tornado/escape.html#squeeze
def squeeze(value):
    """Replace all sequences of whitespace chars with a single space."""
    return re.sub(r"[\x00-\x20]+", " ", value).strip()



########################################
## SQLITECURVE COMPRESSIION FUNCTIONS ##
########################################

def compress_sqlitecurve(sqlitecurve, force=True):
    '''
    This just compresses the sqlitecurve in gzip format.

    '''

    if force:
        cmd = 'gzip -f %s' % sqlitecurve
    else:
        cmd = 'gzip %s' % sqlitecurve

    try:
        procout = subprocess.check_output(cmd, shell=True)
        return '%s.gz' % sqlitecurve
    except subprocess.CalledProcessError:
        LOGERROR('could not compress %s' % sqlitecurve)
        return None



def uncompress_sqlitecurve(sqlitecurve):
    '''
    This just uncompresses the sqlitecurve in gzip format.

    '''

    cmd = 'gunzip %s' % sqlitecurve

    try:
        procout = subprocess.check_output(cmd, shell=True)
        return sqlitecurve.replace('.gz','')
    except subprocess.CalledProcessError:
        LOGERROR('could not uncompress %s' % sqlite)
        return None



###################################
## READING SQLITECURVE FUNCTIONS ##
###################################

def validate_sqlitecurve_filters(filterstring, lccolumns):
    '''This validates the sqlitecurve filter string.

    This MUST be valid SQL but not contain any commands.

    '''

    # first, lowercase, then squeeze to single spaces
    stringelems = squeeze(filterstring).lower()

    # replace shady characters
    stringelems = filterstring.replace('(','')
    stringelems = stringelems.replace(')','')
    stringelems = stringelems.replace(',','')
    stringelems = stringelems.replace("'",'"')
    stringelems = stringelems.replace('\n',' ')
    stringelems = stringelems.replace('\t',' ')
    stringelems = squeeze(stringelems)

    # split into words
    stringelems = stringelems.split(' ')
    stringelems = [x.strip() for x in stringelems]

    # get rid of all numbers
    stringwords = []
    for x in stringelems:
        try:
            floatcheck = float(x)
        except ValueError as e:
            stringwords.append(x)

    # get rid of everything within quotes
    stringwords2 = []
    for x in stringwords:
        if not(x.startswith('"') and x.endswith('"')):
            stringwords2.append(x)
    stringwords2 = [x for x in stringwords2 if len(x) > 0]

    # check the filterstring words against the allowed words
    wordset = set(stringwords2)

    # generate the allowed word set for these LC columns
    allowedwords = SQLITE_ALLOWED_WORDS + lccolumns
    checkset = set(allowedwords)

    validatecheck = list(wordset - checkset)

    # if there are words left over, then this filter string is suspicious
    if len(validatecheck) > 0:

        # check if validatecheck contains an elem with % in it
        LOGWARNING("provided SQL filter string '%s' "
                   "contains non-allowed keywords" % filterstring)
        return None

    else:
        return filterstring



def read_and_filter_sqlitecurve(lcfile,
                                columns=None,
                                sqlfilters=None,
                                raiseonfail=False):
    '''This reads the sqlitecurve and optionally filters it.

    Returns columns requested in columns. If None, then returns all columns
    present in the latest columnlist in the lightcurve.

    If sqlfilters is not None, it must be a list of text sql filters that apply
    to the columns in the lightcurve.

    This returns an lcdict with an added 'lcfiltersql' key that indicates what
    the parsed SQL filter string was.

    '''

    # we're proceeding with reading the LC...
    try:

        # if this file is a gzipped sqlite3 db, then gunzip it
        if '.gz' in lcfile[-4:]:
            lcf = uncompress_sqlitecurve(lcfile)
        else:
            lcf = lcfile

        db = sql.connect(lcf)
        cur = db.cursor()

        # get the objectinfo from the sqlitecurve
        query = ("select * from objectinfo")
        cur.execute(query)
        objectinfo = cur.fetchone()

        # get the lcinfo from the sqlitecurve
        query = ("select * from lcinfo "
                 "order by version desc limit 1")
        cur.execute(query)
        lcinfo = cur.fetchone()

        (lcversion, lcdatarelease, lccols, lcsortcol,
         lcapertures, lcbestaperture,
         objinfocols, objidcol,
         lcunixtime, lcgitrev, lccomment) = lcinfo

        # load the JSON dicts
        lcapertures = json.loads(lcapertures)
        lcbestaperture = json.loads(lcbestaperture)

        objectinfokeys = objinfocols.split(',')
        objectinfodict = {x:y for (x,y) in zip(objectinfokeys, objectinfo)}
        objectid = objectinfodict[objidcol]

        # need to generate the objectinfo dict and the objectid from the lcinfo
        # columns

        # get the filters from the sqlitecurve
        query = ("select * from filters")
        cur.execute(query)
        filterinfo = cur.fetchall()

        # validate the requested columns
        if columns and all([x in lccols.split(',') for x in columns]):
            LOGINFO('retrieving columns %s' % columns)
            proceed = True
        elif columns is None:
            LOGINFO('retrieving all latest columns')
            columns = lccols.split(',')
            proceed = True
        else:
            proceed = False

        # bail out if there's a problem and tell the user what happened
        if not proceed:
            # recompress the lightcurve at the end
            if '.gz' in lcfile[-4:] and lcf:
                dcf = compress_sqlitecurve(lcf)
            return None, "requested columns are invalid"

        # create the lcdict with the object, lc, and filter info
        lcdict = {'objectid':objectid,
                  'objectinfo':objectinfodict,
                  'objectinfokeys':objectinfokeys,
                  'lcversion':lcversion,
                  'datarelease':lcdatarelease,
                  'columns':columns,
                  'lcsortcol':lcsortcol,
                  'lcapertures':lcapertures,
                  'lcbestaperture':lcbestaperture,
                  'lastupdated':lcunixtime,
                  'lcserver':lcgitrev,
                  'comment':lccomment,
                  'filters':filterinfo}

        # validate the SQL filters for this LC
        if ((sqlfilters is not None) and
            (isinstance(sqlfilters,str) or isinstance(sqlfilters,unicode))):

            # give the validator the sqlfilters string and a list of lccols in
            # the lightcurve
            validatedfilters = validate_sqlitecurve_filters(sqlfilters,
                                                            lccols.split(','))
            if validatedfilters is not None:
                LOGINFO('filtering LC using: %s' % validatedfilters)
                filtersok = True
            else:
                filtersok = False
        else:
            LOGINFO('no LC filters specified')
            validatedfilters = None
            filtersok = None

        # now read all the required columns in the order indicated

        # we use the validated SQL filter string here
        if validatedfilters is not None:

            query = ("select {columns} from lightcurve where {sqlfilter} "
                     "order by {sortcol} asc").format(
                         columns=','.join(columns), # columns is always a list
                         sqlfilter=validatedfilters,
                         sortcol=lcsortcol
                     )
            lcdict['lcfiltersql'] = validatedfilters

        else:
            query = ("select %s from lightcurve order by %s asc") % (
                ','.join(columns),
                lcsortcol
            )

        cur.execute(query)
        lightcurve = cur.fetchall()

        if lightcurve and len(lightcurve) > 0:

            lightcurve = zip(*lightcurve)
            lcdict.update({x:y for (x,y) in zip(lcdict['columns'],
                                                lightcurve)})
            lcok = True

            # update the ndet after filtering
            lcdict['objectinfo']['ndet'] = len(lightcurve[0])

        else:
            LOGWARNING('LC for %s has no detections' % lcdict['objectid'])

            # fill the lightcurve with empty lists to indicate that it is empty
            lcdict.update({x:y for (x,y) in
                           zip(lcdict['columns'],
                               [[] for x in lcdict['columns']])})
            lcok = False

        # generate the returned lcdict and status message
        if filtersok is True and lcok:
            statusmsg = 'SQL filters OK, LC OK'
        elif filtersok is None and lcok:
            statusmsg = 'no SQL filters, LC OK'
        elif filtersok is False and lcok:
            statusmsg = 'SQL filters invalid, LC OK'
        else:
            statusmsg = 'LC retrieval failed'

        returnval = (lcdict, statusmsg)

        # recompress the lightcurve at the end
        if '.gz' in lcfile[-4:] and lcf:
            dcf = compress_sqlitecurve(lcf)

    except Exception as e:

        LOGEXCEPTION('could not open sqlitecurve %s' % lcfile)
        returnval = (None, 'error while reading lightcurve file')

        # recompress the lightcurve at the end
        if '.gz' in lcfile[-4:] and lcf:
            dcf = compress_sqlitecurve(lcf)

        if raiseonfail:
            raise

    return returnval



#############################
## READING CSV LIGHTCURVES ##
#############################

def smartcast(castee, caster, subval=None):
    '''
    This just tries to apply the caster function to castee.

    Returns None on failure.

    '''

    try:
        return caster(castee)
    except Exception as e:
        if caster is float or caster is int:
            return nan
        elif caster is str:
            return ''
        else:
            return subval



# these are the keys used in the metadata section of the CSV LC
METAKEYS = {'objectid':str,
            'hatid':str,
            'twomassid':str,
            'ucac4id':str,
            'network':str,
            'stations':str,
            'ndet':int,
            'ra':float,
            'decl':float,
            'pmra':float,
            'pmra_err':float,
            'pmdecl':float,
            'pmdecl_err':float,
            'jmag':float,
            'hmag':float,
            'kmag':float,
            'bmag':float,
            'vmag':float,
            'sdssg':float,
            'sdssr':float,
            'sdssi':float}



def parse_csv_header(header):
    '''
    This parses the CSV header from the CSV HAT sqlitecurve.

    Returns a dict that can be used to update an existing lcdict with the
    relevant metadata info needed to form a full LC.

    '''

    # first, break into lines
    headerlines = header.split('\n')
    headerlines = [x.lstrip('# ') for x in headerlines]

    # next, find the indices of the metadata sections
    objectstart = headerlines.index('OBJECT')
    metadatastart = headerlines.index('METADATA')
    camfilterstart = headerlines.index('CAMFILTERS')
    photaperturestart = headerlines.index('PHOTAPERTURES')
    columnstart = headerlines.index('COLUMNS')
    lcstart = headerlines.index('LIGHTCURVE')

    # get the lines for the header sections
    objectinfo = headerlines[objectstart+1:metadatastart-1]
    metadatainfo = headerlines[metadatastart+1:camfilterstart-1]
    camfilterinfo = headerlines[camfilterstart+1:photaperturestart-1]
    photapertureinfo = headerlines[photaperturestart+1:columnstart-1]
    columninfo = headerlines[columnstart+1:lcstart-1]

    # parse the header sections and insert the appropriate key-val pairs into
    # the lcdict
    metadict = {'objectinfo':{}}

    # first, the objectinfo section
    objectinfo = [x.split(';') for x in objectinfo]

    for elem in objectinfo:
        for kvelem in elem:
            key, val = kvelem.split(' = ',1)
            metadict['objectinfo'][key.strip()] = (
                smartcast(val, METAKEYS[key.strip()])
                )

    # the objectid belongs at the top level
    metadict['objectid'] = metadict['objectinfo']['objectid'][:]
    del metadict['objectinfo']['objectid']

    # get the lightcurve metadata
    metadatainfo = [x.split(';') for x in metadatainfo]
    for elem in metadatainfo:
        for kvelem in elem:

            try:
                key, val = kvelem.split(' = ',1)

                # get the lcbestaperture into a dict again
                if key.strip() == 'lcbestaperture':
                    val = json.loads(val)

                # get the lcversion and datarelease as integers
                if key.strip() in ('datarelease', 'lcversion'):
                    val = int(val)

                # get the lastupdated as a float
                if key.strip() == 'lastupdated':
                    val = float(val)

                # put the key-val into the dict
                metadict[key.strip()] = val

            except Exception as e:

                LOGWARNING('could not understand header element "%s",'
                           ' skipped.' % kvelem)


    # get the camera filters
    metadict['filters'] = []
    for row in camfilterinfo:
        filterid, filtername, filterdesc = row.split(' - ')
        metadict['filters'].append((int(filterid),
                                    filtername,
                                    filterdesc))

    # get the photometric apertures
    metadict['lcapertures'] = {}
    for row in photapertureinfo:
        apnum, appix = row.split(' - ')
        appix = float(appix.rstrip(' px'))
        metadict['lcapertures'][apnum.strip()] = appix

    # get the columns
    metadict['columns'] = []

    for row in columninfo:
        colnum, colname, coldesc = row.split(' - ')
        metadict['columns'].append(colname)

    return metadict



def read_csvlc(lcfile):
    '''
    This reads the HAT data server producd CSV light curve into a lcdict.

    lcfile is the HAT gzipped CSV LC (with a .hatlc.csv.gz extension)

    '''

    # read in the file and split by lines
    if '.gz' in os.path.basename(lcfile):
        LOGINFO('reading gzipped HATLC: %s' % lcfile)
        infd = gzip.open(lcfile,'rb')
    else:
        LOGINFO('reading HATLC: %s' % lcfile)
        infd = open(lcfile,'rb')

    lctext = infd.read()
    infd.close()

    # figure out the header and get the LC columns
    lcstart = lctext.index('# LIGHTCURVE\n')
    lcheader = lctext[:lcstart+12]
    lccolumns = lctext[lcstart+13:].split('\n')
    lccolumns = [x for x in lccolumns if len(x) > 0]

    # initialize the lcdict and parse the CSV header
    lcdict = parse_csv_header(lcheader)

    # tranpose the LC rows into columns
    lccolumns = [x.split(',') for x in lccolumns]
    lccolumns = zip(*lccolumns)

    # write the columns to the dict
    for colind, col in enumerate(lcdict['columns']):

        if (col.split('_')[0] in LC_MAG_COLUMNS or
            col.split('_')[0] in LC_ERR_COLUMNS or
            col.split('_')[0] in LC_FLAG_COLUMNS):
            lcdict[col] = np.array([smartcast(x,
                                              COLUMNDEFS[col.split('_')[0]][2])
                                    for x in lccolumns[colind]])

        elif col in COLUMNDEFS:
            lcdict[col] = np.array([smartcast(x,COLUMNDEFS[col][2])
                                    for x in lccolumns[colind]])

        else:
            LOGWARNING('lcdict col %s has no formatter available' % col)
            continue

    return lcdict
