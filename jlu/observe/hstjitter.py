# HST Roll angles 
#
# WIC June 22 2011

# do some checking on the commanded and actual pointing of images in
# an HST association. This implements in an automated way the process
# I used to do by hand when making Phase II's for NICMOS and later
# WFC3/IR. 

# The jitter files contain a good deal of interesting information
# about the telescope pointing. Much of the code here was put into
# place to allow flexibility in the keywords the user cares about, and
# to produce nicely-formatted output lists. If all you want to know
# from this source code is the way the commanded roll-angle is
# calculated, search for the string "QQQ" in your editor.

import os
import pyfits

import numpy as N
import pylab as P

class hstroll(object):
    """Class hstroll: 

    Gather information from *jif.fits and *flt.fits
    Estimate commanded and actual roll angle for HST observations.

    Class "hstroll" gathers the roll information of interest. Most of
    the information needed to assess pointing is in the *jif fits
    logfile, but the one missing keyword (ORIENTAT) required to assess
    the commanded roll is in the science frame FITS header, so we need
    to poll that too. Pointing information is stored in a central
    dictionary keyed on the science frames listed in the jitter file,
    with derived information updated by the methods (allows flexibility
    in the choice of header keywords of interest).

    In the absence of a nice way to flexibly read mixed-format text
    input, class "hstroll" writes out to two files: *.dat only writes
    out the fields determined by the lists self.Keys2Write and
    self.Fmts2Write while *.info contains all the self.KeysOfInterest
    as well as the individual science exposure names. The latter
    should be user-definable, I haven't got round to making that
    happen yet. WIC Jun 23 2011
    """

    def __init__(self,JifFits = None, DirSci = None, OutLisData=None, \
                     DirOut = 'Jitters', OrientTyp = 'ima'):

        # couple of control variables
        self.Verbose = True
        self.JifFits = JifFits

        # where do we find the science frames?
        self.DirSci = DirSci

        # where do the outputs go?
        self.DirOut = DirOut
        if not os.access(self.DirOut,os.R_OK):
            os.mkdir(self.DirOut)

        # hdulist headers from the jitter file
        self.LHeaders = []

        # pointing information is read into a dictionary with key
        # given by the exposure file name (will want to poll the
        # header keyword of the _flt.fits files to perform a check on
        # the ORIENTAT keywords, this makes it more transparent.)
        self.DJitter = {}

        # header keywords we are likely to want to plot that are not
        # essential for estimating the calculated roll or reading in
        # the measured roll angle. "hstroll" will have no problem if
        # this is omitted, but "showroll" will not work.
        self.KeysToPlot = ['ALTITUDE']
        
        # interesting header keywords we might care about, but which
        # are not required for plotting or commanded-roll
        # calculation. At the moment these are passed straight out to
        # the self.OutLisInfo file for other routines to handle.
        self.KeysOfInterest = ['ROOTNAME', \
                                   'DEC_AVG','RA_AVG', \
                                   'V2_RMS', 'V3_RMS', \
                                   'V2_P2P','V3_P2P']

        # keys of interest from each primary science header
        self.KeysPriInterest = ['TARGNAME', \
                                    'POSTARG1', 'POSTARG2', \
                                    'FILTER', 'EXPTIME']

        # look for input
        self.OrientTyp = OrientTyp
        
        # set up default output file if none indicated
        if not OutLisData:
            self.OutLisData = self.JifFits.split('_ji')[0]+'_pointings.dat'
        else:
            self.OutLisData = OutLisData

        # set up the info file
        self.OutLisInfo = self.OutLisData.split('.')[0]+'.info'

        # List of keys that will be written to the pointing data
        # file. Should be all numeric for convenient loading by other
        # routines.
        self.Keys2Write = ['ROLL_AVG', \
                               'CommandedRoll','CommandedAPT', \
                               'ORIENTAT', 'BETA_2', \
                               'ALTITUDE','ROUTTIME']

        # formats for the above keys.
        self.Fmts2Write = ['%9.4f', \
                               '%9.4f', '%9.4f', \
                               '%9.4f', '%7.2f', \
                               '%7.2f','%10.4f']

        # for the more general "info" file, construct the list of
        # keywords and formats to output once the info-dictionary has
        # been built.
        self.Keys2Info = []
        self.Fmts2Info = []

    def LoadJitterFits(self, InFits = None):

        """load the jitter fits file into an HDUlist containing headers only."""

        if not InFits:
            InFits = self.JifFits

        if not os.access(InFits,os.R_OK):
            if self.Verbose:
                print "LoadJitterFits WARN - cannot load jitter file %s." \
                    % (InFits)
            return

        # we only care about the headers
        hdulist = pyfits.open(InFits)
        hdulist.close()

        self.LHeaders = hdulist

    def MakePointingDict(self):
        """For each frame listed in the *.jif jitter file, construct
        dictionary with pointing information. Loop through frames
        takes place here."""
        
        # parse the type of fits extension chosen for science frames.
        if not self.OrientTyp in ['ima','cal','flt']:
            if self.Verbose:
                print "MakePointingDict WARN - sciframe _%s not valid. Defaulting to _%s." % (self.OrientTyp, 'ima')
            self.OrientTyp = 'ima'


        if len(self.LHeaders) < 1:
            if self.Verbose:
                print "ReadLHeaders WARN - header list LHeaders empty."
            return

        NHeaders = len(self.LHeaders)
        self.DJitter = {}  # initialize jitter pointing info

        for iframe in range(1,NHeaders):
            ThisHeader = self.LHeaders[iframe].header

            # find the exposure stem - this will be our dictionary key
            # for the frame. NB the jitter file entry has format
            # 'ib4101goj' while the science frame has format
            # 'ib4101goq' so the final char must be replaced.
            ThisExp = self.SXPAR(ThisHeader,'EXPNAME',None)
            ThisExp = ThisExp[0:-1]+'q'  # replace final character

            if not ThisExp:
                continue

            # if this exposure is already duplicated in the HDUList,
            # something's wrong. Warn and continue.
            if ThisExp in self.DJitter.keys():
                if self.Verbose:
                    print "ReadLHeaders WARN - ignoring duplicates of %s in %s hdulist." % (ThisExp, self.JifFits)
                continue

            # pass the calculated roll angle and the +Y - V3 offset to
            # the pointing-info dictionary
            self.DJitter[ThisExp] = {}
            self.DJitter[ThisExp]['ROLL_AVG'] = \
                self.SXPAR(ThisHeader,'ROLL_AVG',-999.9)
            self.DJitter[ThisExp]['BETA_2'] = \
                self.SXPAR(ThisHeader,'BETA_2',-999.9)

            # put in a placeholder for the commanded roll and readout
            # time, which will be searched for in another method
            self.DJitter[ThisExp]['ORIENTAT'] = -999.9            
            self.DJitter[ThisExp]['CommandedRoll'] = -999.9
            self.DJitter[ThisExp]['CommandedAPT'] = -999.9
            self.DJitter[ThisExp]['ROUTTIME'] = -999.9

            # add the science frame searched for commanded pointing ingo
            self.DJitter[ThisExp]['sciframe'] = \
                ThisExp+'_'+self.OrientTyp+'.fits'
            
            # keywords that we are likely to want for plotting
            for PlotKey in self.KeysToPlot:
                self.DJitter[ThisExp][PlotKey] = \
                    self.SXPAR(ThisHeader,PlotKey,'None')

            # pass thru those other keywords we might care about
            for ExtraKey in self.KeysOfInterest:
                self.DJitter[ThisExp][ExtraKey] = \
                    self.SXPAR(ThisHeader,ExtraKey,'None')

    def EstCommandedRoll(self):

        """With pointing dictionary self.DJitter in place, estimate
        the commanded roll-angle for each frame by polling the FITS
        header of the 1st extension of the science frame for the
        relevant keyword - usually ORIENTAT. Also read in ROUTTIME as
        a proxy for the observation time."""

        if not self.DJitter:
            if self.Verbose:
                print "EstCommandedRoll WARN - jitter-dictionary not yet populated."
            return

        # science-file directory checking
        if self.DirSci:
            if not os_access(self.DirSci, os.R_OK):
                self.DirSci = None
        if not self.DirSci:
            self.DirSci = os.getcwd()

        if self.Verbose:
            print "EstCommandedRoll INFO - searching %s/ for sciframes" \
                % (self.DirSci)
            print "EstCommandedRoll INFO - Examining _%s frames..." \
                % (self.OrientTyp)            
            
       # what exposures do we have...
        ExpNames = self.DJitter.keys()
        for Frame in ExpNames:
            LThis = self.DJitter[Frame]  # save typoes
            
            if not 'sciframe' in LThis:
                continue

            ThisFits = self.DirSci+'/'+LThis['sciframe']
            
            if not os.access(ThisFits,os.R_OK):
                continue

            # science ext header
            ThisHeader = pyfits.getheader(ThisFits,1)
            
            # pass the ORIENTAT keyword to the pointing dictionary
            ThisORIENTAT = self.SXPAR(ThisHeader,'ORIENTAT',-888.8)
            ThisROUTTIME = self.SXPAR(ThisHeader,'ROUTTIME',-888.8)
            
            # QQQ calculate the commanded roll and its APT equivalent,
            # recording both as 0 <= theta < 360. 
            CommandedRoll = ThisORIENTAT - self.DJitter[Frame]['BETA_2']
            CommandedRoll = CommandedRoll % 360.0
            CommandedAPT  = ( CommandedRoll + 180.0 ) % 360.0

            # pass the angles up to the pointing dictionary
            self.DJitter[Frame]['ORIENTAT'] = ThisORIENTAT
            self.DJitter[Frame]['CommandedRoll'] = CommandedRoll
            self.DJitter[Frame]['CommandedAPT'] = CommandedAPT
            self.DJitter[Frame]['ROUTTIME'] = ThisROUTTIME

            # pass thru primary-header ancilliary keys of interest
            ThisPrimary = pyfits.getheader(ThisFits,0)
            for PriKey in self.KeysPriInterest:
                self.DJitter[Frame][PriKey] = \
                    self.SXPAR(ThisPrimary,PriKey,'None')
                
        if self.Verbose:
            print "EstCommandedRoll INFO - ... done."

    # save typing try... except... every time a fits header is
    # polled... Implements IDLASTRO's sxpar.pro
    def SXPAR(self,hdr=[], Keyword = None, AltVal = ''):

        """Return value of "Keyword" in input pyfits header "hdr", returning
        "AltVal" if the keyword is missing."""
        try:
            retval = hdr[Keyword]
        except:
            retval = AltVal

        return retval

    # output the pointing information
    def WritePointingData(self):

        """Writes a subset of the pointing data to *pointings.dat
        file. Keywords written out are given in comment string at the
        head of the file.

        Currently the columns are, in order: Measured roll, Commanded Roll,
        Commanded APT ORIENT, ORIENTAT, BETA_2, ALTITUDE, ROUTTIME"""

        # stick on the output directory name
        if self.DirOut:
            self.OutLisData = self.DirOut+'/'+self.OutLisData

        if os.access(self.OutLisData,os.W_OK):
            os.remove(self.OutLisData)
            
        if self.Verbose:
            print "WritePointingData INFO - writing pointing data to %s" \
                % (self.OutLisData)

        # making a properly-formatted set of column-headings is too
        # much hassle, just bung them into comments...
        fObj = open(self.OutLisData,'w')
    
        # create comment string from self.Keys2Write
        StrCommen = ' '.join(["%s" % en for en in self.Keys2Write])
        fObj.write("# %s \n" % (StrCommen))

        # also produce comment string with the target, filter and exptime
        try:
            FirstFrame = self.DJitter.keys()[0]
            StrID = "# %s %s %s" % (self.DJitter[FirstFrame]['TARGNAME'], \
                                        self.DJitter[FirstFrame]['FILTER'], \
                                        self.DJitter[FirstFrame]['EXPTIME'])
            fObj.write("%s \n" % (StrID))
        except:
            dummy = 1 

        FrameList = sorted(self.DJitter.keys())
        for Frame in FrameList: 
            outstring = self.DictListToString(self.DJitter[Frame])

            fObj.write("%s \n" % (outstring))
    
        fObj.close()
    
    def WritePointingInfo(self):

        """Writes all keywords specified as being interesting to
        *pointings.info file"""

        if self.DirOut:
            self.OutLisInfo = self.DirOut+'/'+self.OutLisInfo

        if os.access(self.OutLisInfo,os.W_OK):
            os.remove(self.OutLisInfo)
            
        if self.Verbose:
            print "WritePointingInfo INFO - writing pointing info to %s" \
                % (self.OutLisInfo)

        # build list of keywords to write out from the various lists
        # previously defined.
        self.OutKeyListBuild()
        
        # write the comment string
        fObjInfo = open(self.OutLisInfo,'w')
        StrCommen = ' '.join(["%s" % en for en in self.Keys2Info])
        fObjInfo.write("# %s \n" % (StrCommen))

        FrameList = sorted(self.DJitter.keys())
        for ThisFrame in FrameList:
            
            # ensure format list is built - ThisFrame is optional
            # here.  OutKeyListFormats does nothing if the format list
            # already same len as keyword list.
            self.OutKeyListFormats(ThisFrame)

            outstring = self.DictListToString(self.DJitter[ThisFrame], \
                                                  self.Keys2Info, \
                                                  self.Fmts2Info)

            fObjInfo.write("%s \n" % (outstring))

        fObjInfo.close()
        
    def OutKeyListBuild(self):

        """utility method to build list of keys to write to output and
        remove duplicates. Also constructs format string using sensible
        defaults."""

        self.Keys2Info = ['sciframe']
        self.Fmts2Info = ['%18s']

        # this rather clunky way of doing things also removes any
        # duplicates in the input lists of keys to write.
        for KeyOut in self.Keys2Write:
            if not KeyOut in self.Keys2Info:
                self.Keys2Info.append(KeyOut)

        for KeyPlot in self.KeysToPlot:
            if not KeyPlot in self.Keys2Info:
                self.Keys2Info.append(KeyPlot)

        for KeyInterest in self.KeysOfInterest:
            if not KeyInterest in self.Keys2Info:
                self.Keys2Info.append(KeyInterest)

        for KeyPrimary in self.KeysPriInterest:
            if not KeyPrimary in self.Keys2Info:
                self.Keys2Info.append(KeyPrimary)

    # construct format string using format string if already present,
    # or construct one using the value of the keyword to discriminate
    # string from float
    def OutKeyListFormats(self,ThisFrame = None):

        """Sets format strings for the full list of interesting header
        keywords and does some rudimentary parsing into
        floats/strings."""

        # don't do anything if the list is already populated
        if len(self.Fmts2Info) == len(self.Keys2Info):
            return

        # if the frame of interest is not specified, just use the
        # first in the dictionary
        if not ThisFrame:
            ThisFrame = self.DJitter.keys()[0]

        for iKey in range(len(self.Fmts2Info),len(self.Keys2Info)):
            InKey = self.Keys2Info[iKey]
            if InKey in self.Keys2Write:
                iFormat = self.Keys2Write.index(InKey)
                self.Fmts2Info.append(self.Fmts2Write[iFormat])
                continue

            # construct format string using the value of the keyword
            try:
                dummy = 1.0 * self.DJitter[ThisFrame][InKey]
                self.Fmts2Info.append("%9.4f")
            except:
                self.Fmts2Info.append("%9s")
            
#        if self.Verbose:
#            print "OutKeyListFormats INFO - key list:", self.Keys2Info
#            print "OutKeyListFormats INFO - fmt list:", self.Fmts2Info
                
    def DictListToString(self,Entry = [], \
                             Keys2WriteIn = [], Fmts2WriteIn = [], \
                             DBG = False):

        """Create output string from dictionary list. If entry is
        missing, substitute with a badval string of the same length as
        the required format. Take the list of keys to write as input,
        defaults to self.Keys2Write if not set."""
        
        if len(Entry) < 1:
            return ''

        # Only accept input keys and fmts lists if they are the same
        # nonzero length.
        HasInputs = False
        if len(Keys2WriteIn) > 1 and len(Fmts2WriteIn):
            if len(Keys2WriteIn) == len(Fmts2WriteIn):
                HasInputs = True
            else:
                if self.Verbose:
                    print "DictListToString WARN - input keys and fmts have different lengths. Not using. Lengths %i, %i" \
                        % (len(Keys2WriteIn), len(Fmts2WriteIn))

        if HasInputs:
            Keys2Write = Keys2WriteIn
            Fmts2Write = Fmts2WriteIn
        else:
            Keys2Write = self.Keys2Write
            Fmts2Write = self.Fmts2Write

        if DBG:
            Keys2Write = ['sciframe','CommandedRoll','ROLL_AVG','Fleem']
            Fmts2Write = ['%18s','%9.4f', '%9.4f','%9.4f']

        outstring = ''
        for ikey in range(len(Keys2Write)):

            # if this particular frame does not have the keyword
            # populated, output a blank string
            if not Keys2Write[ikey] in Entry.keys():

                # lift out the length part of the format string and
                # produce a blank string of equal length
                lenthis = int(Fmts2Write[ikey][1:-1].split('.')[0])
                outstring = outstring + '-'*lenthis + ' '
            else:
                outstring = outstring + Fmts2Write[ikey]  \
                    % (Entry[Keys2Write[ikey]])

            outstring = outstring + ' '

        return outstring

class jitterballs(object):
    """Given a *jif.fits file, find its counterpart *jit.fits file to
    show the 3-second average pointing during each exposure.

    Uses the os module to call imagemagick to glue the png's of the jitter
    files into a single pdf report, one page per science frame."""

    def __init__(self,JifFits = None, \
                     DirJit = 'Jitters', \
                     DirBall = 'Histories'):
        self.Verbose=True
        self.JifFits = JifFits
        self.JitFits = None
        self.HDUlist = []
        self.DoAssemble = True
        self.DoCleanIntermeds = True
        
        # list of png files that have been produced by this class
        self.MadePNGList = []

        # output directories
        self.DirJit = DirJit
        self.DirBall = DirBall+'_'+self.JifFits.split('_jif.fits')[0]
        self.DirOut = self.DirJit+'/'+self.DirBall

        # filename for output pdf report
        self.OutPDF = None

        # make output directory if not present
        if not os.access(self.DirJit,os.R_OK):
            os.mkdir(self.DirJit)
        
        if not os.access(self.DirOut,os.R_OK) and self.JifFits:
            os.mkdir(self.DirOut)

        # pointing table
        self.ThisPointingTable = []
        self.ThisExpName = None
        self.ThisPointingPNG = None

        # which entries are we plotting?
        self.PlotX = ['Seconds','Seconds','SI_V2_AVG','Seconds']
        self.PlotY = ['SI_V2_AVG', 'SI_V3_AVG','SI_V3_AVG','Roll']
        self.UnitX = ['','',', arcsec','']
        self.UnitY = [', arcsec',', arcsec',', arcsec',', degrees']
        self.AScal = [False, False, True, False]

        self.ShowPix = True
        # plate scale, X and Y (arcsec per pix)
        self.PlateScaleX = 0.13
        self.PlateScaleY = 0.13 

    def JitfromJif(self):
        """Given input *jif.fits file, return the counterpart *jit.fits file"""
        if not self.JifFits:
            return

        self.JitFits = self.JifFits.split('jif.fits')[0]+'jit.fits'
        
    def ReadJitterBall(self):
        """Read in the HDUlist of pointing histories"""
        try:
            self.HDUlist = pyfits.open(self.JitFits)
        except:
            if self.Verbose:
                print "ReadJitterBall WARN - failed to load jitter file %s" \
                    % (self.JitFits)

    def CloseJitterBall(self):
        """Close pyfits HDUlist containing jitterballs if open"""

        if len(self.HDUlist) > 1:
            self.HDUlist.close()

    def ShowJitterBall(self,NewPlot = True):
        
        """Produce multiplot showing intra-exposure pointing information."""

        if len(self.ThisPointingTable.data) < 1:
            return

        if NewPlot:
            P.clf()

        # local variable to avoid typoes
        a = self.ThisPointingTable.data

        # go through the choices of stuff to plot
        for iplot in range(len(self.PlotX)):

            ThisSub = '22'+str(int(iplot+1))
            P.subplot(ThisSub)
            xplot = self.PlotX[iplot]
            yplot = self.PlotY[iplot]
            
            # scale - factors
            sX = 1.0
            sY = 1.0

            ThisUnitX = self.UnitX[iplot]
            ThisUnitY = self.UnitY[iplot]

            # meanval
            meanval = 0.0

            # convert arcsec to pixels?
            if self.ShowPix:
                if 'rcsec' in self.UnitX[iplot]:
                    sX = 1.0 / self.PlateScaleX
                    ThisUnitX = ', pix'

                if 'rcsec' in self.UnitY[iplot]:
                    sY = 1.0 / self.PlateScaleY
                    ThisUnitY = ', pix'

                if 'degr' in self.UnitY[iplot]:
                    meanval = N.mean(a.field(yplot))
                    sY = 3600.0
                    ThisUnitY = ', arcsec'

            P.plot(a.field(xplot)*sX,(a.field(yplot)-meanval)*sY,'bo',markersize=3)
            P.plot(a.field(xplot)*sX,(a.field(yplot)-meanval)*sY,color='k',alpha=0.5)
            
            P.xlabel(xplot+ThisUnitX)
            P.ylabel(yplot+ThisUnitY)

            # show the mean Y, std Y-value if a timeseries
            if 'Seconds' in xplot:
                P.title("%9.5f +/- %.2e" % (N.mean((a.field(yplot)-meanval)*sY) , \
                                                N.std(a.field(yplot)*sY) ))

            if N.abs(meanval) > 0.0:
                P.annotate('mean %9.5f degr' % (meanval), [0.03,0.05], \
                               xycoords='axes fraction', \
                               horizontalalignment='left', color='b')

            if self.AScal[iplot]:
                P.axes(ThisSub).set_aspect('equal','datalim')

        # which frame is this?
        P.annotate(self.ThisExpName+' - jitter information', \
                       [0.5,0.95], \
                       xycoords='figure fraction' ,\
                       horizontalalignment='center', \
                       verticalalignment='bottom')

        # save the figure
        P.savefig(self.ThisPointingPNG)

    def ProduceJitterFigures(self,clobber=False):
        """Mid-level wrapper. Loops through planes in jitter
        HDUlist and produces a .png file from each."""

        if len(self.HDUlist) < 1:
            return

        if self.Verbose:
            print "ProduceJitterFigures INFO - intermediate png images go to %s" % (self.DirOut)

        for iFrame in range(1,len(self.HDUlist)):
            
            self.ThisPointingTable = []  # re-initialize
            self.ThisPointingTable = self.HDUlist[iFrame]
            self.ThisExpName = None
            self.ThisPointingPNG = None

            try:
                self.ThisExpName = self.HDUlist[iFrame].header['EXPNAME'] 
                self.ThisPointingPNG = self.ThisExpName +'_jitterball.png'
                self.ThisPointingPNG = self.DirOut+'/'+self.ThisPointingPNG
                    
            except:
                dummy = 1

                
            if not self.ThisPointingPNG:
                continue

            self.MadePNGList.append(self.ThisPointingPNG)


            if os.access(self.ThisPointingPNG,os.R_OK) and not clobber:
                continue

            self.ShowJitterBall()

    def AssemblePDFJitters(self, clobber=True):
        
        """
        Given a set of jitter-ball images, wraps them into a single
        pdf for convenient viewing.

        Looks for the alias "convert" in the user's environment and
        determines whether this refers to ImageMagick convert or
        something else.

        If "convert" refers to ImageMagick convert, uses the syntax
        "convert *png collection.pdf" to assemble the png images into
        a single pdf file, in the same directory where jitter
        information is placed."""

        if not clobber:
            return

        if not self.DoAssemble:
            return

        # Count the number of times the string "ImageMagick" appears
        # in the manpage for the shell command "convert". If no
        # command is present, or if the command aliases to something
        # else, this will always return zero because the output of
        # "which" is piped into "grep -c"
        NumMagick = int(os.popen("man convert | grep -c ImageMagick").read().strip() )

        if NumMagick < 1:
            return

        # construct output command
        self.OutPDF = self.JifFits.split('_jif.fits')[0]+'_jitters.pdf'

        ShellConvCommand = "convert %s/*png %s/%s" % (self.DirOut, self.DirJit, self.OutPDF)

        if self.Verbose:
            print "AssemblePDFJitters INFO - executing: %s" % (ShellConvCommand)
        os.system(ShellConvCommand)

    def CleanIntermedPNGs(self):
        """If self.DoCleanIntermeds is set, remove the intermediate
        PNG images produced from the jitter balls for this
        association. If this leaves the intermediate directory empty,
        the intermediate directory is also removed.

        Does not remove intermediate products if no final product was
        produced."""

        if not self.DoCleanIntermeds:
            return

        if len(self.MadePNGList) < 1:
            return

        if not self.OutPDF:
            return

        if not os.access(self.DirJit+'/'+self.OutPDF, os.R_OK):
            if self.Verbose:
                print "CleanIntermedPNGs INFO - not cleaning intermediate PNGs because output pdf file %s/%s not found." % (self.DirJit,self.OutPDF)
            return

        if self.Verbose:
            print "CleanIntermedPNG INFO - removing intermediate PNG products from %s" % (self.DirOut)
        for ThisPNG in self.MadePNGList:
            if os.access(ThisPNG, os.W_OK):
                os.remove(ThisPNG)

        # if the intermediate directory is now empty, remove that.  
        if self.DirOut in os.getcwd():
            return

        if len(os.listdir(self.DirOut)) < 1:
            if self.Verbose:
                print "CleanIntermedPNG INFO - removing empty intermediate-product directory %s" % (self.DirOut)
            os.rmdir(self.DirOut)

class showrolls(object):
        
    """Given a pointings.dat file, read in pointing info and do some plotting.

    Known issue: handling of input text files not very robust."""

    def __init__(self,RollFile = None):
            
        """Initialise"""

        self.Verbose=True

        if not RollFile:
            self.RollFile = 'ib4102010_pointings.dat'
        else:
            self.RollFile = RollFile
        self.ARoll = N.array([])


        # columns
        self.cTime = 6
        self.cAlt = 5
        self.cCommanded = 1
        self.cAPT = 2
        self.cCalculated = 0

        # special entries
        self.VTime = N.array([])
        self.VHours = N.array([])

    # read in the file containing roll information
    def ReadRollFile(self):

        """Read in *pointings.dat file generated by class "hstroll",
        convert exptime to hours since first frame."""

        try:
            self.ARoll = N.transpose(N.genfromtxt(self.RollFile,dtype=None))
            self.VTime = self.ARoll[self.cTime]
            self.VHours = (self.VTime - N.min(self.VTime)) * 24.0
        except:
            dummy = 1
    
    def ShowRollInfo(self):

        """Throw up rollangle plots and save to *png files."""

        if N.size(self.ARoll) < 1:
            print "ShowRollInfo WARN - pointing information not present in self.ARoll. Cannot continue."
            return

        P.clf()

        # save typoes
        a = self.ARoll
            
        # HST Altitude vs time
        P.subplot(325)
        P.plot(self.VHours,a[self.cAlt],'bo')
        P.plot(self.VHours,a[self.cAlt],'b')
        P.xlabel('Time (hours)')
        P.ylabel('HST Altitude (km)')            
        P.title('HST Altitude vs time')

        # Calc - Command vs Command
        P.subplot(324)
        MinCommanded = N.min(a[self.cCommanded])
        # deltas - x and y
        deltaX = (a[self.cCommanded] - MinCommanded)*60.0
        deltaY =  (a[self.cCalculated] - MinCommanded)*60.0
        MeanDelta = N.mean(deltaY - deltaX)

        P.plot(deltaX, deltaY,'go')
        P.xlabel("(Commanded - ThetaRef), arcmin")
        P.ylabel("(Calculated - ThetaRef), arcmin")
        sref = "ThetaRef = %6.2f deg" % (MinCommanded)
        P.annotate(sref,[0.05,0.9], \
                       xycoords='axes fraction', \
                       horizontalalignment='left', \
                       verticalalignment='top')

        # show 1:1 line to guide the eye
        P.plot([N.min(deltaX),N.max(deltaX)], \
                   [0.0, N.max(deltaX)-N.min(deltaX)] + MeanDelta,'r--')
        P.annotate("y = x + %7.3f" % (MeanDelta), [0.05,0.80], \
                       xycoords='axes fraction', \
                       horizontalalignment='left', \
                       verticalalignment='top', color='r')
        P.title('Calculated vs Commanded V3 angle')

        # Calc - Command vs time
        P.subplot(323)
        P.plot(self.VHours,deltaY-deltaX,'go')
        P.title('Calculated - Commanded V3')
        P.xlabel('Time(hours)')
        P.ylabel('(Calculated - Commanded), degrees')

        # Calc vs Alt
        P.subplot(326)
        P.plot(a[self.cAlt],a[self.cCalculated],'ks')
        P.title('Calculated vs Altitude')
        P.xlabel('HST altitude (km)')
        P.ylabel('Calculated V3, degrees')

        # Calc vs Time
        P.subplot(321)
        P.plot(self.VHours,a[self.cCalculated],'gs')
        P.xlabel('Time(hours)')
        P.ylabel('Calculated V3, degrees')
        P.title('Calculated V3 from FGS')
        P.annotate("Mean %11.5f deg" % (N.mean(a[self.cCalculated])), \
                       [0.02,0.90], \
                       xycoords='axes fraction', \
                       horizontalalignment='left', \
                       verticalalignment='top', color='g')

        # APT ORIENT vs time
        P.subplot(322)
        P.plot(self.VHours,a[self.cAPT],'ks')

        P.title('Commanded ORIENT as in APT')
        P.annotate("Mean %11.5f deg" % (N.mean(a[self.cAPT])),  [0.02,0.90], \
                       xycoords='axes fraction', \
                       horizontalalignment='left', \
                       verticalalignment='top', color='g')
        P.xlabel('Time (hours)')
        P.ylabel('ORIENT (degrees)')

        P.subplots_adjust(hspace=0.40,wspace=0.35)
        
        FileStem = self.RollFile.split('.')[0]
        P.annotate(FileStem,[0.5,0.95], \
                       xycoords='figure fraction' ,\
                       horizontalalignment='center',verticalalignment='bottom')

        # output image filename
        outpng = FileStem+'.png'

        try:
            P.savefig(outpng)
            if self.Verbose:
                print "ShowRollInfo INFO - saved plots to %s" % (outpng)
        except:
            if self.Verbose:
                print "ShowRollInfo WARN - saving to %s failed." % (outpng)

def go(JitterFile='ib4101010_jif.fits', DirSci = None, DirOut = 'Jitters', \
           OrientTyp='ima', Verbose=True, ShowJitters = True):

    """
    .
    .
    Read in the roll-angle information from an obset, write to
    output files and then produce plots with information about the
    roll-angle history.

    Optionally produces PDF showing the pointing history within each
    science exposure of the association.

    Arguments:

    JitterFile -- ./*jif.fits file. Assumed to be present in current
    directory.

    DirSci -- relative path to the directory containing the science frames

    DirOut -- relative path for output files and plots. Directory
    generated if not found by os.access.

    OrientTyp -- type of frame searched for the relevant keywords when
    estimating the commanded roll (e.g. *ima.fits). If not any of
    'ima', 'flt' or 'cal', defaults to 'ima'.

    Verbose -- (T/F) Provide some information on the tasks as they run
    (e.g. type of science frames polled, where the outputs go). if
    False, routine is silent.
    
    ShowJitters -- (T/F) If True, uses the *jit.fits file to produce a
    pdf report with some pointing information within each exposure at
    3-second averages, with one page per science exposure. 
    .
    .

    """

    if not os.access(JitterFile,os.R_OK):
        print "hstjitter.go WARN - JitterFile %s not present in local directory." % (JitterFile)
        return

    # gather the pointing data...
    if Verbose:
        print "hstjitter.go INFO - gathering pointing information"
    A = hstroll(JitterFile, DirSci,DirOut = DirOut, OrientTyp=OrientTyp)
    A.Verbose = Verbose
    A.LoadJitterFits()
    A.MakePointingDict()
    A.EstCommandedRoll()
    A.WritePointingData()
    A.WritePointingInfo()
    
#    ... and prepare plots. NB A.OutLisData also includes the relative
#    path to the output list, which means the .png plots will fetch up
#    in A.DirOut.
    if Verbose:
        print "hstjitter.go INFO - throwing up roll plots"
    B = showrolls(A.OutLisData)
    B.Verbose=Verbose
    B.ReadRollFile()
    B.ShowRollInfo()

#   Produce PDF file for the jitter balls for each observation pointed
#   to by the association.
    if Verbose:
        print "hstjitter.go INFO - creating PDF jitter-ball report"
    C = jitterballs(JitterFile,DirJit = A.DirOut)
    C.Verbose = Verbose
    C.JitfromJif()
    C.ReadJitterBall()
    C.ProduceJitterFigures(clobber=False)  # clobber: overwrite
                                           # intermed pngs if present
    C.CloseJitterBall()
    C.AssemblePDFJitters() # assemble jitter balls into one PDF document
    C.CleanIntermedPNGs()  # get rid of the individual jitter-ball PNGs

