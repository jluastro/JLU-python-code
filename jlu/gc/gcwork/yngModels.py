import asciidata
import pylab

def plotBaraffe():
    """
    Plot Baraffe Models at the GC distance. This will produce a
    K vs. K-L color magnitude diagram.
    """
    root = '/u/jlu/work/gc/stellar_models/B98_a1_'

    ages = [4.0, 6.0, 8.0]

    # Distance modulus
    dist = 8000.0    # pc
    distMod = 5.0 * pylab.log10(dist / 10.0)

    # Extinction
    AV = 25.0
    AH = 0.175 * AV  # Rieke & Lebofsky 1985
    AK = 0.112 * AV  # Rieke & Lebofsky 1985
    AL = 0.058 * AV  # Rieke & Lebofsky 1985
    AM = 0.058 * AV  # Viehmann et al. 2005

    masses = []
    hmags = []
    kmags = []
    lmags = []
    mmags = []
    for age in ages:
        filename = '%s%dmyr.models' % (root, age)
        table = asciidata.open(filename)

        # Masses
        mass = table[0].tonumarray()
        masses.append(mass)

        # Intrinsic Magnitudes
        hmag = table[9].tonumarray()
        kmag = table[10].tonumarray()
        lmag = table[11].tonumarray()
        mmag = table[12].tonumarray()

        # Switch to apparent magnitudes
        hmag += distMod + AH
        kmag += distMod + AK
        lmag += distMod + AL
        mmag += distMod + AM

        hmags.append(hmag)
        kmags.append(kmag)
        lmags.append(lmag)
        mmags.append(mmag)


    #----------
    #
    #  Plotting
    #
    #----------
    pylab.clf()
    pylab.plot(kmags[1]-mmags[1], kmags[1])
    pylab.plot(kmags[0]-mmags[0], kmags[0], 'k--')
    pylab.plot(kmags[2]-mmags[2], kmags[2], 'r--')
    pylab.axis([-1, 4, 28, 8])
    
    
    
