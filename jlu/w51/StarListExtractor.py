class StarListExtractor(object):
    def __init__(self, epochName, 
                 rootDir='/u/jlu/data/w51/', dirSuffix='',
                 filt='kp', coo='psf0', 
                 mcorr=0.8, scorr=0.6, ccorr=0.7, 
                 psfTableDir='/u/jlu/code/idl/w51/psftables/',
                 photTableDir='/u/jlu/data/w51/source_lists/'):
        
        self.epochName = epochName
        self.rootDir = rootDir
        self.dirSuffix = dirSuffix
        self.filt = filt
        self.coo = coo
        self.mcorr = mcorr
        self.scorr = scorr
        self.ccorr = ccorr
        self.psfTableDir = psfTableDir
        self.photTableDir = photTableDir


