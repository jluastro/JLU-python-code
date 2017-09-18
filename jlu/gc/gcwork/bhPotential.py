from numpy import *
from gcwork import analyticOrbits as aorb
import pickle
import pdb

class BHprops(object):
    """
    Contains the probability distribution of mass/Ro/focus positions
    properties of the black hole.
    """
    def __init__(self, ntrials=100000):
        self.ntrials = int(ntrials)
        #self.massCenter = 3.61e6
        #self.massSigma = 0.32e6
        #self.distCenter = 7620.0
        #self.distSigma = 320.0
        #self.x0Center = -0.001
        #self.x0Sigma = 0.001
        #self.y0Center = -0.005
        #self.y0Sigma = 0.002

        # Ghez et al. (2008)
        #self.massCenter = 4.1e6
        #self.massSigma = 0.6e6
        #self.distCenter = 7960.0
        #self.distSigma = 600.0
        #self.x0Center = -0.001
        #self.x0Sigma = 0.001
        #self.y0Center = -0.005
        #self.y0Sigma = 0.002
        # End Ghez et al. (2008)

        # Latest S0-2 orbit in aligndir/11_10_26/
        self.massCenter = 4.6e6
        self.massSigma = 0.7e6
        self.distCenter = 8232.9
        self.distSigma = 670.0
        self.x0Center = -0.003
        self.x0Sigma = 0.001
        self.y0Center = -0.008
        self.y0Sigma = 0.002
        

    def saveToFile(self, savefile):
        """
        SAVE all the results to a binary file that python can
        later read in for further plotting/analysis. This is
        called 'pickle' in python. To reload, simply call
        
             self = pickle.load(open(<file>, 'r'))
        """
        _pic = open(savefile, 'w')
        pickle.dump(self, _pic)

    def generate(self, efitFile=None):
        """
        Generate the list of masses, distances, and focus positions
        for the number of trials specified at initialization. The 
        resulting variables are arrays of mass, distance, x0, and y0:
        
        m  -- mass in solar masses
        r0 -- distance in pc
        x0 -- X origin in pixels
        y0 -- Y origin in pixels
        
        There are two ways to generate this list:

        1) Use the output from Andrea's efit mass/Ro monte carlos. The
        number of trials spit out by efit must be the same or larger
        than the number of trials specified here. To use this option,
        set the parameter 'efitFile' to the name of the file containing
        the efit monte carlo results.
        
        2) Use a gaussian distribution of mass/Ro/focus position. To
        specify this option (this is the default) just use efitFile=None
        The distributions are determined by the variables:

        self.massCenter    # solar masses
        self.massSigma     # solar masses
        self.distCenter    # parsec
        self.distSigma     # parsec
        self.x0Center      # pixel
        self.x0Sigma       # pixel
        self.y0Center      # pixel
        self.y0Sigma       # pixel

        which should be set after initialization but before calling
        generate(). If no values are specified, the default values from
        the Eisenhauer (2006) paper will be used.
        """


        m  = zeros(self.ntrials, dtype=float64)
        r0 = zeros(self.ntrials, dtype=float64)
        x0 = zeros(self.ntrials, dtype=float64)
        y0 = zeros(self.ntrials, dtype=float64)

        if (efitFile == None):
            gens = aorb.create_generators(4, self.ntrials)
            mgen = gens[0]
            r0gen = gens[1]
            x0gen = gens[2]
            y0gen = gens[3]

            for ii in range(self.ntrials):
                m[ii]  = mgen.gauss(self.massCenter, self.massSigma)
                r0[ii] = r0gen.gauss(self.distCenter, self.distSigma)
                x0[ii] = x0gen.gauss(self.x0Center, self.x0Sigma)
                y0[ii] = y0gen.gauss(self.y0Center, self.y0Sigma)

        else:
            # Read in M/Ro simluation results
            # Columns are:
            # Dist  x0   y0   a    P  e  t0  w  i  Ome
            # pc    pix  pix  mas  yr    yr   
            lines = open(efitFile, 'r').readlines()

            if (self.ntrials > len(lines)):
                print('EFIT file too short for number of trials = ', len(lines))

            for ii in range(self.ntrials):
                fields = lines[ii].split()

                r0[ii] = float(fields[0])   # in pc
                x0[ii] = float(fields[1])   # in pixels
                y0[ii] = float(fields[2])   # in pixels

                # Get the semi-major axis and period in order to get the mass
                a = float(fields[3])        # in mas
                p = float(fields[4])        # in years

                a *= r0[ii] / 1000.0

                # Mass in Msun
                m[ii] = a**3 / p**2

        self.m = m
        self.r0 = r0
        self.x0 = x0
        self.y0 = y0
                


    def generate_rho0(self):
        """
        Generate list of density normalizations for the number
        of trials given at initialization. Based on Schoedel
        et al. (2009), assuming M_BH = 4.1x10^6 Msun, gamma=1.0
        """

        rho0ave = 3.2e4  # solar masses/pc^3
        rho0sig = 1.3e4  # solar masses/pc^3
    
        rho0 = zeros(self.ntrials, dtype=float64)
        gens = aorb.create_generators(1, self.ntrials)
        rho0gen = gens[0]

        for ii in range(self.ntrials):
            rho0[ii] = rho0gen.gauss(rho0ave, rho0sig)

        self.rho0 = rho0
