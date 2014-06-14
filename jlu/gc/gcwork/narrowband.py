# Take frames of regular standards.
# Look at their fluxes in JHK.
# Fit parabola.
# Pull out flux at wavelength of narrow band.
#
# Should be telluric corrected spectra. Or JHK magnitudes at airmass 0.


import asciidata, pyfits
import os, sys, math
from pylab import *
from numpy import *
from gcwork import starset
from gcwork import objects
from gcwork import util
from gcwork import young
from gcwork import starTables
from matplotlib import mlab

widthCO = 267.0

def plotBrg(root):
    # This should be align output with the following order:
    #    brg
    #    CO
    #    H2 1-0
    #    H2 2-1
    s = starset.StarSet(root)

    # Setup some filter properties
    filt = []
    filt.append({"name": 'h210', 
		    "on": 2.1112e4, "off": 2.1452e4})
    filt.append({"name": 'brg', 
		    "on": 2.1523e4, "off": 2.1849e4})
    filt.append({"name": 'h221', 
		    "on": 2.2428e4, "off": 2.2816e4})
    filt.append({"name": 'co', 
		    "on": 2.2757e4, "off": 2.3024e4})

    for fs in filt:
	fs["dw"] = fs["off"] - fs["on"]
	fs["w"] = fs["on"] + (fs["dw"] / 2.0)
	
    waveH210 = filt[0]["w"]
    waveBrg  = filt[1]["w"]
    waveH221 = filt[2]["w"]
    waveCo   = filt[3]["w"]

    # Keep the integration times
    tBrg = 30.0 * 2.0
    tCo = 20.0 * 2.0
    tH210 = 29.0 * 2.0
    tH221 = 25.0 * 2.0

    # We need to convert magnitudes to fluxes first
    fBrg  = s.getArrayFromEpoch(0, 'snr')
    fCo   = s.getArrayFromEpoch(1, 'snr')
    fH210 = s.getArrayFromEpoch(2, 'snr')
    fH221 = s.getArrayFromEpoch(3, 'snr')

    # Convert to cts per unit wavelength per unit time.
    fBrg  /= tBrg
    fCo   /= tCo
    fH210 /= tH210
    fH221 /= tH221

    fH210 /= filt[0]["dw"]
    fBrg  /= filt[1]["dw"]
    fH221 /= filt[2]["dw"]
    fCo   /= filt[3]["dw"]

    names = s.getArray('name')

    # Load up spectra taken from OSIRIS
    specRoot = '/u/tdo/osiris/06may/reduced_spectra/'
    s02specFile = specRoot + 's02.fits'
    s06specFile = specRoot + 's06_uncorrected.fits'
    s15specFile = specRoot + 's15_corrected.fits'

    s02specFile_2 = specRoot + 's02_uncorrected.fits'
    telluric = specRoot + 'telluric.fits'

    specHdr = pyfits.getheader(s02specFile)
    s02spec = pyfits.getdata(s02specFile)
    s06spec = pyfits.getdata(s06specFile)
    s15spec = pyfits.getdata(s15specFile)
    s02spec2 = pyfits.getdata(s02specFile_2)
    tell = pyfits.getdata(telluric)

    s02spec2 = s02spec2 / tell

    w0 = specHdr.get('CRVAL1')
    ws = specHdr.get('CDELT1')
    specX = w0 + (ws * arange(s02spec.size()))

    ##########
    # Now we need to integrate these spectra across the wavelengths
    # of interest.
    ##########

    # We need filter transmission curves. For now just assume a boxcar.
    transH210 = where((specX >= filt[0]["on"]) & 
		      (specX <= filt[0]["off"]), 1, 0)
    transBrg = where((specX >= filt[1]["on"]) & 
		     (specX <= filt[1]["off"]), 1, 0)
    transH221 = where((specX >= filt[2]["on"]) & 
		      (specX <= filt[2]["off"]), 1, 0)
    transCo = where((specX >= filt[3]["on"]) & 
		    (specX <= filt[3]["off"]), 1, 0)

    # S0-2
    s_s02H210 = (s02spec * transH210).sum() / filt[0]["dw"]
    s_s02Brg = (s02spec * transBrg).sum() / filt[1]["dw"]
    s_s02H221 = (s02spec * transH221).sum() / filt[2]["dw"]
    s_s02Co = (s02spec * transCo).sum() / filt[3]["dw"]

    # S0-6
    s_s06H210 = (s06spec * transH210).sum() / filt[0]["dw"]
    s_s06Brg = (s06spec * transBrg).sum() / filt[1]["dw"]
    s_s06H221 = (s06spec * transH221).sum() / filt[2]["dw"]
    s_s06Co = (s06spec * transCo).sum() / filt[3]["dw"]

    # S1-5
    s_s15H210 = (s15spec * transH210).sum() / filt[0]["dw"]
    s_s15Brg = (s15spec * transBrg).sum() / filt[1]["dw"]
    s_s15H221 = (s15spec * transH221).sum() / filt[2]["dw"]
    s_s15Co = (s15spec * transCo).sum() / filt[3]["dw"]

    ##########
    # Now find these 3 stars.
    ##########
    id_s02 = names.index('S0-2')
    id_s06 = names.index('S0-6')
    id_s15 = names.index('S1-5')

    nb_s02H210 = fH210[id_s02]
    nb_s02Brg = fBrg[id_s02]
    nb_s02H221 = fH221[id_s02]
    nb_s02Co = fCo[id_s02]

    nb_s06H210 = fH210[id_s06]
    nb_s06Brg = fBrg[id_s06]
    nb_s06H221 = fH221[id_s06]
    nb_s06Co = fCo[id_s06]

    nb_s15H210 = fH210[id_s15]
    nb_s15Brg = fBrg[id_s15]
    nb_s15H221 = fH221[id_s15]
    nb_s15Co = fCo[id_s15]

    ##########
    # Now determine scale factors for all 3 stars.
    ##########
    sc_s02H210 = s_s02H210 / nb_s02H210
    sc_s02Brg = s_s02Brg / nb_s02Brg
    sc_s02H221 = s_s02H221 / nb_s02H221
    sc_s02Co = s_s02Co / nb_s02Co

    sc_s06H210 = s_s06H210 / nb_s06H210
    sc_s06Brg = s_s06Brg / nb_s06Brg
    sc_s06H221 = s_s06H221 / nb_s06H221
    sc_s06Co = s_s06Co / nb_s06Co

    sc_s15H210 = s_s15H210 / nb_s15H210
    sc_s15Brg = s_s15Brg / nb_s15Brg
    sc_s15H221 = s_s15H221 / nb_s15H221
    sc_s15Co = s_s15Co / nb_s15Co

    print 'Scale factors: '
    print '%4s  %7s  %7s  %7s  %7s' % ('', 'H210', 'Brg', 'H221', 'Co')
    print '%4s  %7.4f  %7.4f  %7.4f  %7.4f' % \
          ('S0-2', sc_s02H210, sc_s02Brg, sc_s02H221, sc_s02Co)
    print '%4s  %7.4f  %7.4f  %7.4f  %7.4f' % \
          ('S0-6', sc_s06H210, sc_s06Brg, sc_s06H221, sc_s06Co)
    print '%4s  %7.4f  %7.4f  %7.4f  %7.4f' % \
          ('S1-5', sc_s15H210, sc_s15Brg, sc_s15H221, sc_s15Co)

    ##########
    # Take the average of the scale parameters
    ##########
    sc_H210 = average([sc_s02H210])
    sc_Brg = average([sc_s02Brg])
    sc_H221 = average([sc_s02H221])
    sc_Co = average([sc_s02Co])
    
    # Flux values taken from our NIRC2 spectra
    # 16C
    #name = 'irs16C'
    #f0H210 = 15.9264
    #f0Brg = 18.0492
    #f0H221 = 18.5307
    #f0Co = 18.7777
 
    # 16NE
    #name = 'irs16NE'
    #f0H210 = 18.3118
    #f0Brg = 19.1825
    #f0H221 = 19.0623
    #f0Co = 18.9131

    #slope = (f0H210 - f0H221) / (waveH210 - waveH221)
    #inter = f0H221 - (slope * waveH221)

    #f0Co = inter + (slope * waveCo)

    #print f0H210, f0Brg, f0H221, f0Co

    #Taken from starfinder .txt
    #f0Brg = 7.89456e06
    #f0Co = 4.15705e06
    #f0H210 = 7.24866e06
    #f0H221 = 1.63147e06
    #idx0 = names.index('irs16NE')

    #fBrg  = pow(10, -1.0*(mBrg - mBrg[idx0])/2.5) * f0Brg
    #fCo   = pow(10, -1.0*(mCo - mCo[idx0])/2.5) * f0Co
    #fH210 = pow(10, -1.0*(mH210 - mH210[idx0])/2.5) * f0H210
    #fH221 = pow(10, -1.0*(mH221 - mH221[idx0])/2.5) * f0H221


    ##########
    # We will attempt to flux calibrate our SED using S0-2.
    ##########
    #id = names.index(name)
    clf()
    plot(specX, s02spec * transH210, 'r-')
    plot(specX, s02spec * transBrg, 'b-')
    plot(specX, s02spec * transH221, 'g-')
    plot(specX, s02spec * transCo, 'm-')
    plot([filt[0]["w"], filt[1]["w"], filt[2]["w"], filt[3]["w"]],
	 [s_s02H210, s_s02Brg, s_s02H221, s_s02Co], 'co')

    # Flatten everything
    #flatH210 = fH210[id] / f0H210
    #flatBrg = fBrg[id] / f0Brg
    #flatH221 = fH221[id] / f0H221
    #flatCo = fCo[id] / f0Co

    #print 'Ratios found by calibrating with %s' % name
    #print '%6.4f  %7.2e %7.2e' % (waveH210, fH210[id], flatH210)
    #print '%6.4f  %7.2e %7.2e' % (waveBrg, fBrg[id], flatBrg)
    #print '%6.4f  %7.2e %7.2e' % (waveH221, fH221[id], flatH221)
    #print '%6.4f  %7.2e %7.2e' % (waveCo, fCo[id], flatCo)
    print '%6.4f - %6.4f' % (filt[0]["on"], filt[0]["off"])
    print '%6.4f - %6.4f' % (filt[1]["on"], filt[1]["off"])
    print '%6.4f - %6.4f' % (filt[2]["on"], filt[2]["off"])
    print '%6.4f - %6.4f' % (filt[3]["on"], filt[3]["off"])
    
    #fBrg /= flatBrg
    #fCo /= flatCo
    #fH210 /= flatH210
    #fH221 /= flatH221

    # Flux calibrate everything
    fH210 *= sc_H210
    fBrg *= sc_Brg
    fH221 *= sc_H221
    fCo *= sc_Co

    plot([filt[0]["w"], filt[1]["w"], filt[2]["w"], filt[3]["w"]],
	 [fH210[id_s02], fBrg[id_s02], 
	  fH221[id_s02], fCo[id_s02]], 
	 'rx')
    savefig('test.ps')

    w = [waveH210, waveH221]

    slopes = (fH210 - fH221) / (waveH210 - waveH221)
    interc = fH221 - (slopes * waveH221)

    contBrg = interc + (slopes * waveBrg)
    contCo = interc + (slopes * waveCo)
    
    i = 0
    print 'flux = %e + wave * %e' % (interc[i], slopes[i])
    print '%6.4f  %e  %e' % (waveH210, fH210[i], 
			     interc[i]+(slopes[i] * waveH210)) 
    print '%6.4f  %e  %e' % (waveBrg, fBrg[i],
			     interc[i]+(slopes[i] * waveBrg)) 
    print '%6.4f  %e  %e' % (waveH221, fH221[i],
			     interc[i]+(slopes[i] * waveH221)) 
    print '%6.4f  %e  %e' % (waveCo, fCo[i],
			     interc[i]+(slopes[i] * waveCo)) 

    
    # Plot in fluxes
    eqwBrg = 1.0 - (fBrg  / contBrg)
    #eqwBrg = 1.0 - (fBrg  / fH221)
    eqwCo = 1.0 - (fCo  / contCo)
    eqwCoSingle = 1.0 - (fCo  / fH221)

    # Might also plot in magnitudes
    fTot = fH210 + fBrg + fH221 + fCo
    zeropoint = 8.33
    magH210 = -2.5 * log10(fH210/fTot[0]) + zeropoint
    magBrg = -2.5 * log10(fBrg/fTot[0]) + zeropoint
    magH221 = -2.5 * log10(fH221/fTot[0]) + zeropoint
    magCo = -2.5 * log10(fCo/fTot[0]) + zeropoint
    magBrgCont = -2.5 * log10(contBrg/fTot[0]) + zeropoint
    magCoCont = -2.5 * log10(contCo/fTot[0]) + zeropoint


    # Can plot SED
    w = array([waveH210, waveBrg, waveH221, waveCo])

    clf()
    id = names.index('S0-2')
    plot(w, array([fH210[id], fBrg[id], fH221[id], fCo[id]])/fTot[id], 'k.')
    id = names.index('S0-4')
    plot(w, array([fH210[id], fBrg[id], fH221[id], fCo[id]])/fTot[id], 'kx')
    id = names.index('S0-5')
    plot(w, array([fH210[id], fBrg[id], fH221[id], fCo[id]])/fTot[id], 'rx')
    id = names.index('S0-6')
    plot(w, array([fH210[id], fBrg[id], fH221[id], fCo[id]])/fTot[id], 'bx')
    id = names.index('irs16SW-E')
    plot(w, array([fH210[id], fBrg[id], fH221[id], fCo[id]])/fTot[id], 'gx')
    title(names[id])
    plot(specX, s02spec/6.0)
    xlabel('Wavelength (Angstroms)')
    ylabel('Flux')
    legend(('S0-2', 'S0-4', 'S0-5', 'S0-6', 'irs16SW-E'), loc='lower right')
    #axis([2.1, 2.3, 0, 0.4])
    savefig('plotBrg_sed.eps')

    yngnames = young.youngStarNames()

    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = ['S0-6', 'S1-5', 'irs29S', 'irs13W', 'S3-22', 'S1-23', 'S4-4',
		'S1-17', 'S2-32', 'star_14', 'star_11', 'star_5', 'star_23', 
		'star_45', 'star_40', 'star_45', 'star_38']
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    newyng = ['S1-4', 'S1-1', 'S0-32', 'S0-9', 'S1-27', 'S1-33']
    ydx = []
    for newname in newyng:
	try:
	    ndx = names.index(newname)
	    ydx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0
    
    clf()
    plot(eqwBrg, magH221, 'k.')
    plot(eqwBrg[idx], magH221[idx], 'b.')
    plot(eqwBrg[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('BrGamma Index')
    ylabel('K (mag)')
    axis([rng[0], rng[1], rng[3], rng[2]])
    savefig('plotBrg_brg.eps')
    
    clf()
    plot(eqwCo, magH221, 'k.')
    plot(eqwCo[idx], magH221[idx], 'b.')
    plot(eqwCo[cdx], magH221[cdx], 'r.')
    plot(eqwCo[ydx], magH221[ydx], 'c.')
    rng = axis()
    xlabel('CO Index')
    ylabel('K (mag)')
    plot([-0.06, -0.06], [0.0, 13.4], 'k--')
    plot([-1, 1], [13.4, 13.4], 'k--')
    #axis([rng[0]+0.8, rng[1], rng[3], rng[2]])
    axis([-0.2, 0.2, rng[3], 9.0])
    savefig('plotBrg_co.eps')


    clf()
    plot(eqwCoSingle, magH221, 'k.')
    plot(eqwCoSingle[idx], magH221[idx], 'b.')
    plot(eqwCoSingle[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('CO Index (one Continuum)')
    ylabel('K (mag)')
    #axis([-0.4, 0.2, rng[3], rng[2]])
    axis([-0.35, 0.05, rng[3], 12.0])
    savefig('plotBrg_coSingle.eps')

    clf()
    plot(eqwCo, eqwBrg, 'k.')
    plot(eqwCo[idx], eqwBrg[idx], 'b.')
    plot(eqwCo[cdx], eqwBrg[cdx], 'r.')
    rng = axis()
    xlabel('CO Index')
    ylabel('BrGamma (mag)')
    axis([rng[0]+0.8, rng[1], rng[2], rng[3]])
    savefig('plotBrg_cobrg.eps')

    # Create a starlist for candidate young stars:
    #   mag limit = 13.4
    #   EQW upper limit = -0.06
    xpix = s.getArrayFromEpoch(1, 'xorig')
    ypix = s.getArrayFromEpoch(1, 'yorig')
    xarc = s.getArrayFromEpoch(1, 'x')
    yarc = s.getArrayFromEpoch(1, 'y')
    corr = s.getArrayFromEpoch(1, 'corr')
    nframes = s.getArrayFromEpoch(1, 'nframes')
    fwhm = s.getArrayFromEpoch(1, 'fwhm')

    new = where((magH221 <= 13.4) & 
		      (eqwCo <= -0.06))
    new = new[0]

    status = zeros(len(fwhm)) + -1.0

    for ii in range(len(status)):
	# Determine if this is a valid candidate young star
	if ((magH221[ii] <= 13.4) and (eqwCo[ii] <= -0.06)):
	    status[ii] = 0

	# Add in OSIRIS confirmed sources
	if ((names[ii]=='S0-32') or (names[ii]=='S0-9') or
            (names[ii]=='S1-33') or (names[ii]=='S1-27')):
	    status[ii] = 1

        # Check to see if this is an existing young star.
        # If so, then label as class 2
	try:
	    idx.index(ii)
            status[ii] = 2
	except ValueError:
            foo = 0

        # Check to see if this is a new young star.
        # If so, then label as class 1
	try:
	    ydx.index(ii)
            status[ii] = 1
	except ValueError:
            foo = 0

    _f = open('yngCandidates.lis', 'w')
    _f2 = open('yngCandidates_arcsec.lis', 'w')
    for nn in new:
        fmt = '%15s  %6.3f  %9.3f  %8.3f %8.3f   %7.3f  %4.2f  %5d  %8.3f\n'
        _f.write(fmt % (names[nn], magH221[nn], 2006.337,
                        xpix[nn], ypix[nn], eqwCo[nn],
                        corr[nn], status[nn], fwhm[nn]))
        _f2.write(fmt % (names[nn], magH221[nn], 2006.337,
                        xarc[nn], yarc[nn], eqwCo[nn],
                        corr[nn], status[nn], fwhm[nn]))
    _f.close()
    _f2.close()

    clf()
    plot(eqwCo, magH221, 'k.')
    old = plot(eqwCo[idx], magH221[idx], 'b.')
    late = plot(eqwCo[cdx], magH221[cdx], 'r.')
    #id = (where(status == 2))[0]
    #plot(eqwCo[id], magH221[id], 'co', mec='c')
    id = (where(status == 0))[0]
    new = plot(eqwCo[id], magH221[id], 'b^', mec='c')
    id = (where(status == 1))[0]
    osir = plot(eqwCo[id], magH221[id], 'co', mec='g')
    rng = axis()
    xlabel('CO Index')
    ylabel('K (mag)')
    legend((late, old, new),
           ('Late-Type Stars', 'Young Stars', 'Young Candidates'),
           numpoints=1, prop=matplotlib.font_manager.FontProperties('smaller'))

    plot([-0.06, -0.06], [0.0, 13.4], 'k--')
    plot([-1, 1], [13.4, 13.4], 'k--')
    plot([-1, 1], [15.0, 15.0], 'k--')
    #axis([rng[0]+0.8, rng[1], rng[3], rng[2]])
    axis([-0.2, 0.2, rng[3], 9.0])
    savefig('plotBrg_co_cand.eps')

    

def plotCo(root):
    # This should be align output with the following order:
    #    CO
    #    H2 1-0
    #    H2 2-1
    s = starset.StarSet(root)

    # Setup some filter properties
    filt = []
    filt.append({"name": 'h210', 
		    "on": 2.1112e4, "off": 2.1452e4})
    filt.append({"name": 'h221', 
		    "on": 2.2428e4, "off": 2.2816e4})
    filt.append({"name": 'co', 
		    "on": 2.2757e4, "off": 2.3024e4})

    for fs in filt:
	fs["dw"] = fs["off"] - fs["on"]
	fs["w"] = fs["on"] + (fs["dw"] / 2.0)
	
    waveH210 = filt[0]["w"]
    waveH221 = filt[1]["w"]
    waveCo   = filt[2]["w"]

    # Keep the integration times
    tCo = 29.0 * 3.0
    tH210 = 24.0 * 3.0
    tH221 = 22.0 * 3.0
    #tCo = 20.0 * 2.0
    #tH210 = 29.0 * 2.0
    #tH221 = 25.0 * 2.0

    # We need to convert magnitudes to fluxes first
    fCo   = s.getArrayFromEpoch(0, 'snr')
    fH210 = s.getArrayFromEpoch(1, 'snr')
    fH221 = s.getArrayFromEpoch(2, 'snr')
    #fCo   = s.getArrayFromEpoch(1, 'snr')
    #fH210 = s.getArrayFromEpoch(2, 'snr')
    #fH221 = s.getArrayFromEpoch(3, 'snr')

    # Convert to cts per unit wavelength per unit time.
    fCo   /= tCo
    fH210 /= tH210
    fH221 /= tH221

    fH210 /= filt[0]["dw"]
    fH221 /= filt[1]["dw"]
    fCo   /= filt[2]["dw"]

    names = s.getArray('name')

    # Load up spectra taken from OSIRIS
    specRoot = '/u/tdo/osiris/06may/reduced_spectra/'
    s02specFile = specRoot + 's02.fits'
    s06specFile = specRoot + 's06_uncorrected.fits'
    s15specFile = specRoot + 's15_corrected.fits'

    s02specFile_2 = specRoot + 's02_uncorrected.fits'
    telluric = specRoot + 'telluric.fits'

    specHdr = pyfits.getheader(s02specFile)
    s02spec = pyfits.getdata(s02specFile)
    s06spec = pyfits.getdata(s06specFile)
    s15spec = pyfits.getdata(s15specFile)
    s02spec2 = pyfits.getdata(s02specFile_2)
    tell = pyfits.getdata(telluric)

    s02spec2 = s02spec2 / tell

    w0 = specHdr.get('CRVAL1')
    ws = specHdr.get('CDELT1')
    foo = arange(len(s02spec), dtype=float)
    specX = w0 + (ws * foo)

    ##########
    # Now we need to integrate these spectra across the wavelengths
    # of interest.
    ##########

    # We need filter transmission curves. For now just assume a boxcar.
    transH210 = where((specX >= filt[0]["on"]) & 
		      (specX <= filt[0]["off"]), 1, 0)
    transH221 = where((specX >= filt[1]["on"]) & 
		      (specX <= filt[1]["off"]), 1, 0)
    transCo = where((specX >= filt[2]["on"]) & 
		    (specX <= filt[2]["off"]), 1, 0)

    # S0-2
    s_s02H210 = (s02spec * transH210).sum() / filt[0]["dw"]
    s_s02H221 = (s02spec * transH221).sum() / filt[1]["dw"]
    s_s02Co = (s02spec * transCo).sum() / filt[2]["dw"]

    # S0-6
    s_s06H210 = (s06spec * transH210).sum() / filt[0]["dw"]
    s_s06H221 = (s06spec * transH221).sum() / filt[1]["dw"]
    s_s06Co = (s06spec * transCo).sum() / filt[2]["dw"]

    # S1-5
    s_s15H210 = (s15spec * transH210).sum() / filt[0]["dw"]
    s_s15H221 = (s15spec * transH221).sum() / filt[1]["dw"]
    s_s15Co = (s15spec * transCo).sum() / filt[2]["dw"]

    ##########
    # Now find these 3 stars.
    ##########
    id_s02 = names.index('S0-2')
    id_s06 = names.index('S0-6')
    id_s15 = names.index('S1-5')

    nb_s02H210 = fH210[id_s02]
    nb_s02H221 = fH221[id_s02]
    nb_s02Co = fCo[id_s02]

    nb_s06H210 = fH210[id_s06]
    nb_s06H221 = fH221[id_s06]
    nb_s06Co = fCo[id_s06]

    nb_s15H210 = fH210[id_s15]
    nb_s15H221 = fH221[id_s15]
    nb_s15Co = fCo[id_s15]

    ##########
    # Now determine scale factors for all 3 stars.
    ##########
    sc_s02H210 = s_s02H210 / nb_s02H210
    sc_s02H221 = s_s02H221 / nb_s02H221
    sc_s02Co = s_s02Co / nb_s02Co

    sc_s06H210 = s_s06H210 / nb_s06H210
    sc_s06H221 = s_s06H221 / nb_s06H221
    sc_s06Co = s_s06Co / nb_s06Co

    sc_s15H210 = s_s15H210 / nb_s15H210
    sc_s15H221 = s_s15H221 / nb_s15H221
    sc_s15Co = s_s15Co / nb_s15Co

    print 'Scale factors: '
    print '%4s  %7s  %7s  %7s ' % ('', 'H210', 'H221', 'Co')
    print '%4s  %7.4f  %7.4f  %7.4f' % \
          ('S0-2', sc_s02H210, sc_s02H221, sc_s02Co)
    print '%4s  %7.4f  %7.4f  %7.4f' % \
          ('S0-6', sc_s06H210, sc_s06H221, sc_s06Co)
    print '%4s  %7.4f  %7.4f  %7.4f' % \
          ('S1-5', sc_s15H210, sc_s15H221, sc_s15Co)

    ##########
    # Take the average of the scale parameters
    ##########
    sc_H210 = average([sc_s02H210])
    sc_H221 = average([sc_s02H221])
    sc_Co = average([sc_s02Co])
    
    # Flux values taken from our NIRC2 spectra
    # 16C
    #name = 'irs16C'
    #f0H210 = 15.9264
    #f0H221 = 18.5307
    #f0Co = 18.7777
 
    # 16NE
    #name = 'irs16NE'
    #f0H210 = 18.3118
    #f0H221 = 19.0623
    #f0Co = 18.9131

    #slope = (f0H210 - f0H221) / (waveH210 - waveH221)
    #inter = f0H221 - (slope * waveH221)

    #f0Co = inter + (slope * waveCo)

    #print f0H210, f0Brg, f0H221, f0Co

    #Taken from starfinder .txt
    #f0Co = 4.15705e06
    #f0H210 = 7.24866e06
    #f0H221 = 1.63147e06
    #idx0 = names.index('irs16NE')

    #fCo   = pow(10, -1.0*(mCo - mCo[idx0])/2.5) * f0Co
    #fH210 = pow(10, -1.0*(mH210 - mH210[idx0])/2.5) * f0H210
    #fH221 = pow(10, -1.0*(mH221 - mH221[idx0])/2.5) * f0H221


    ##########
    # We will attempt to flux calibrate our SED using S0-2.
    ##########
    #id = names.index(name)
    clf()
    plot(specX, s02spec * transH210, 'r-')
    plot(specX, s02spec * transH221, 'g-')
    plot(specX, s02spec * transCo, 'm-')
    plot([filt[0]["w"], filt[1]["w"], filt[2]["w"]],
	 [s_s02H210, s_s02H221, s_s02Co], 'co')

    # Flatten everything
    #flatH210 = fH210[id] / f0H210
    #flatH221 = fH221[id] / f0H221
    #flatCo = fCo[id] / f0Co

    #print 'Ratios found by calibrating with %s' % name
    #print '%6.4f  %7.2e %7.2e' % (waveH210, fH210[id], flatH210)
    #print '%6.4f  %7.2e %7.2e' % (waveH221, fH221[id], flatH221)
    #print '%6.4f  %7.2e %7.2e' % (waveCo, fCo[id], flatCo)
    print '%6.4f - %6.4f' % (filt[0]["on"], filt[0]["off"])
    print '%6.4f - %6.4f' % (filt[1]["on"], filt[1]["off"])
    print '%6.4f - %6.4f' % (filt[2]["on"], filt[2]["off"])
    
    #fCo /= flatCo
    #fH210 /= flatH210
    #fH221 /= flatH221

    # Flux calibrate everything
    #fH210 *= sc_H210
    #fH221 *= sc_H221
    #fCo *= sc_Co
    fH210 *= fCo[0] / fH210[0]
    fH221 *= fCo[0] / fH221[0]

    plot([filt[0]["w"], filt[1]["w"], filt[2]["w"]],
	 [fH210[id_s02], fH221[id_s02], fCo[id_s02]], 'rx')
    savefig('test.ps')

    w = [waveH210, waveH221]

    slopes = (fH210 - fH221) / (waveH210 - waveH221)
    interc = fH221 - (slopes * waveH221)

    contCo = interc + (slopes * waveCo)
    
    i = 0
    print 'flux = %e + wave * %e' % (interc[i], slopes[i])
    print '%6.4f  %e  %e' % (waveH210, fH210[i], 
			     interc[i]+(slopes[i] * waveH210)) 
    print '%6.4f  %e  %e' % (waveH221, fH221[i],
			     interc[i]+(slopes[i] * waveH221)) 
    print '%6.4f  %e  %e' % (waveCo, fCo[i],
			     interc[i]+(slopes[i] * waveCo)) 

    
    # Plot in fluxes
    eqwCo = 1.0 - (fCo  / contCo)
    eqwCoSingle = 1.0 - (fCo  / fH221)

    # Might also plot in magnitudes
    fTot = fH210 + fH221 + fCo
    zeropoint = 8.33
    magH210 = -2.5 * log10(fH210/fTot[0]) + zeropoint
    magH221 = -2.5 * log10(fH221/fTot[0]) + zeropoint
    magCo = -2.5 * log10(fCo/fTot[0]) + zeropoint
    magCoCont = -2.5 * log10(contCo/fTot[0]) + zeropoint


    # Can plot SED
    w = array([waveH210, waveH221, waveCo])

    clf()
    id = names.index('S0-2')
    plot(w, array([fH210[id], fH221[id], fCo[id]])/fTot[id], 'k.')
    id = names.index('S0-4')
    plot(w, array([fH210[id], fH221[id], fCo[id]])/fTot[id], 'kx')
    id = names.index('S0-5')
    plot(w, array([fH210[id], fH221[id], fCo[id]])/fTot[id], 'rx')
    id = names.index('S0-6')
    plot(w, array([fH210[id], fH221[id], fCo[id]])/fTot[id], 'bx')
    id = names.index('irs16SW-E')
    plot(w, array([fH210[id], fH221[id], fCo[id]])/fTot[id], 'gx')
    title(names[id])
    plot(specX, s02spec/6.0)
    xlabel('Wavelength (Angstroms)')
    ylabel('Flux')
    legend(('S0-2', 'S0-4', 'S0-5', 'S0-6', 'irs16SW-E'), loc='lower right')
    #axis([2.1, 2.3, 0, 0.4])
    savefig('plotCo_sed.eps')

    yngnames = young.youngStarNames()
    
    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = ['S0-6', 'S1-5', 'irs29S', 'irs13W', 'S3-22', 'S1-23', 'S4-4',
		'S1-17', 'S2-32', 'star_14', 'star_11', 'star_5', 'star_23', 
		'star_45', 'star_40', 'star_45', 'star_38']
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    clf()
    plot(eqwCo, magH221, 'k.')
    plot(eqwCo[idx], magH221[idx], 'r.')
    plot(eqwCo[cdx], magH221[cdx], 'b.')
    rng = axis()
    xlabel('CO Index')
    ylabel('K (mag)')
    #axis([rng[0]+0.8, rng[1], rng[3], rng[2]])
    #axis([-0.2, 0.2, rng[3], 12.0])
    savefig('plotCo.eps')

    clf()
    plot(eqwCoSingle, magH221, 'k.')
    plot(eqwCoSingle[idx], magH221[idx], 'r.')
    plot(eqwCoSingle[cdx], magH221[cdx], 'b.')
    rng = axis()
    xlabel('CO Index (one Continuum)')
    ylabel('K (mag)')
    #axis([-0.4, 0.2, rng[3], rng[2]])
    #axis([-0.35, 0.05, rng[3], 12.0])
    savefig('plotCo_coSingle.eps')


def plotCo2(root):
    # This should be align output with the following order:
    #    CO
    #    H2 1-0
    #    H2 2-1
    s = starset.StarSet(root)

    names = s.getArray('name')
    yngnames = young.youngStarNames()
    
    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = ['S0-6', 'S1-5', 'irs29S', 'irs13W', 'S3-22', 'S1-23', 'S4-4',
		'S1-17', 'S2-32']#, 'star_14', 'star_11', 'star_5', 'star_23', 
		#'star_45', 'star_40', 'star_45', 'star_38']
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
            print oldname, names[ndx]
	except ValueError:
	    # Do nothing
	    foo = 0.0

    # Setup some filter properties in a dictionary.
    waveOn  = {'h210': 2.1112e4, 'h221': 2.2428e4, 'co': 2.2757e4}
    waveOff = {'h210': 2.1452e4, 'h221': 2.2816e4, 'co': 2.3024e4}
    wave = waveOn
    waveRng = waveOn
    for key in wave:
        waveRng[key] = waveOff[key] - waveOn[key]
        wave[key] = waveOn[key] + (waveRng[key] / 2.0)
	
    # Keep the integration times
    tCo = 29.0 * 3.0
    tH210 = 24.0 * 3.0
    tH221 = 22.0 * 3.0

    # We need to convert magnitudes to fluxes first
    mCo   = s.getArrayFromEpoch(0, 'mag')
    mH210 = s.getArrayFromEpoch(1, 'mag')
    mH221 = s.getArrayFromEpoch(2, 'mag')

    fCo = 10.0**(-(mCo - mCo[0]) / 2.5)
    fH210 = 10.0**(-(mH210 - mH210[0]) / 2.5)
    fH221 = 10.0**(-(mH221 - mH221[0]) / 2.5)

    slopes = (fH210 - fH221) / (wave['h210'] - wave['h221'])
    interc = fH221 - (slopes * wave['h221'])

    contCo = interc + (slopes * wave['co'])

    eqwCo = 1.0 - (fCo  / contCo)

    figure(1)
    clf()
    plot(eqwCo, mH221, 'k.')
    plot(eqwCo[idx], mH221[idx], 'r.')
    plot(eqwCo[cdx], mH221[cdx], 'c.')
    axis([-0.5, 0.5, 17, 8])
    xlabel('EQW of CO (mag)')
    ylabel('H221 (mag)')

    figure(2)
    clf()
    plot(mH210 - mH221, mH210, 'k.')
    plot(mH210[idx] - mH221[idx], mH210[idx], 'r.')
    plot(mH210[cdx] - mH221[cdx], mH210[cdx], 'c.')
    axis([-0.5, 0.5, 17, 8])
    xlabel('H210 - H221 (mag)')
    ylabel('H210 (mag)')

    figure(3)
    clf()
    plot(mH221 - mCo, mH221, 'k.')
    plot(mH221[idx] - mCo[idx], mH221[idx], 'r.')
    plot(mH221[cdx] - mCo[cdx], mH221[cdx], 'c.')
    axis([-0.5, 0.5, 17, 8])
    xlabel('H221 - CO (mag)')
    ylabel('H221 (mag)')

    figure(4)
    clf()
    plot(mH210 - mCo, mH210, 'k.')
    plot(mH210[idx] - mCo[idx], mH210[idx], 'r.')
    plot(mH210[cdx] - mCo[cdx], mH210[cdx], 'c.')
    axis([-0.5, 0.5, 17, 8])
    xlabel('H210 - CO (mag)')
    ylabel('H210 (mag)')



def plotSensitivity():
    """For a given photometric sensitivity and intrinsic equivalent
    width, how well can we detect the line?"""

    # Define array of equivalent widths
    ew = arange(20) * 5.0

    # Define array of photometric senstivities
    photErr = arange(20) * 0.005

    # For each photometric error, find what equivalent width
    # we can detect with 1 sigma, 2 sigma, and 3 sigma
    ewSigma1 = EWmatch(photErr, 1.0)
    ewSigma2 = EWmatch(photErr, 2.0)
    ewSigma3 = EWmatch(photErr, 3.0)

    clf()
    plot(photErr * 1.086, -ewSigma1, 'k-')
    plot(photErr * 1.086, -ewSigma2, 'k--')
    plot(photErr * 1.086, -ewSigma3, 'k-.')
    xlabel('Photometric Sensitivity (mag)')
    ylabel('Equivalent Width (Angstroms)')
    legend(('1 sigma', '2 sigma', '3 sigma'))
    axis([0, 0.10, -100.0, 0.0])
    show()
    savefig('nb_sensitivity.eps')
    savefig('nb_sensitivity.png')

    # Print out some info in a table for 3 sigma limits
    print 'For sigma = %5.3f mags we get EW = -%5.3f Angs' % \
	(0.028*1.086, EWmatch(0.028, 3.0))
    print '%12s\t %12s' % ('Sigma (mag)', 'EQ (Ang)')
    print '%12s\t %12s' % ('-----------', '--------')
    for i in range(len(photErr)):
	if (photErr[i] > 0.06):
	    break

	print '%12.3f\t %12.1f' % (photErr[i] * 1.086, -ewSigma3[i])
    

# These functions are defined based on the following assumptions
# f_c1 = f_c2
# f_c1_err = f_c2_err = f_CO_err
# f_CO_err / f_CO = f_c1_err / f_c1
def fluxRatio(ew):
    return (1.0 - (ew / widthCO))

def EWerr(ew, fluxErr):
    foo = pow(widthCO * fluxRatio(ew), 2)
    foo *= fluxErr * (1.0 + 1.48)
    foo = sqrt(foo)

def EWmatch(fluxErr, sigma):
    quad2err = sqrt(1.0 + 1.48)
    top = widthCO * fluxErr * quad2err
    bottom = (1.0 / sigma) + (fluxErr * quad2err)
    ew = (widthCO / 300.0) *  top / bottom
    return ew


def labelImage(starlist):
    imageFile = '/u/ghezgroup/data/gc/06maylgs1/combo/mag06maylgs1_kp.fits'
    sgra = [600.0, 687.0]
    scale = 0.00995

    img = pyfits.getdata(imageFile)

    imgsize = (img.shape)[0]
    pix = arange(0, imgsize)
    x = [(xpos - sgra[0])*-scale for xpos in pix]
    y = [(ypos - sgra[1])*scale for ypos in pix]

    # Read in starlist
    stars = asciidata.open(starlist)
    name = stars[0]
    x_arc = stars[3].tonumarray()
    y_arc = stars[4].tonumarray()
    # This tells us if previously known, or new candidate
    status = stars[7].tonumarray()

    # Plot
    clf()
    figure(2, figsize=(10,10))
    gray()
    imshow(log10(img+1), aspect='equal', interpolation='bicubic',
           extent=[max(x), min(x), min(y), max(y)], vmin=3, vmax=5,
           origin='lowerleft', cmap=cm.gray_r)
    ax = axis()

    # Status:
    #    2 = previously known
    #    1 = new... confirmed with osiris
    #    0 = new canddiates (were outside the OSIRIS FOV)
    old = (where(status == 2))[0]
    new = (where(status == 1))[0]
    cand = (where(status == 0))[0]
    
    oldPts = plot(x_arc[old], y_arc[old], 'bo', mfc=None, mec='b', mew=1.0, label='Previously Known')
    newPts = plot(x_arc[new], y_arc[new], 'bd', mfc=None, mec='b', mew=1.0, label='New OSIRIS')
    candPts = plot(x_arc[cand], y_arc[cand], 'b^', mfc=None, mec='b', mew=1.0, label='New Candidates')

    osir = [0.0, 0.0]
    osir_fov = [3.29, 2.38]
    osir_x = [osir[0] - (osir_fov[0]/2.0), osir[0] - (osir_fov[0]/2.0),
              osir[0] + (osir_fov[0]/2.0), osir[0] + (osir_fov[0]/2.0), osir[0] - (osir_fov[0]/2.0)]
    osir_y = [osir[1] - (osir_fov[1]/2.0), osir[1] + (osir_fov[1]/2.0),
              osir[1] + (osir_fov[1]/2.0), osir[1] - (osir_fov[1]/2.0), osir[1] - (osir_fov[1]/2.0)]
    plot(osir_x, osir_y, 'b--', label='OSIRIS FOV')

    legend((oldPts, newPts, candPts),
           ('Previously Known', 'OSIRIS Confirmed', 'Narrowband Candidates'), 
           numpoints=1, prop=matplotlib.font_manager.FontProperties('smaller'))
    axis([5, -5, -5, 5])
    savefig('yngCandidates.eps')




def plotCoNew(root, suffix=''):
    # This should be align output with the following order:
    #    CO
    #    H2 1-0
    #    H2 2-1
    s = starset.StarSet(root)

    # Setup some filter properties
    filt = []
    filt.append({"name": 'h210', 
		    "on": 2.1112e4, "off": 2.1452e4})
    filt.append({"name": 'h221', 
		    "on": 2.2428e4, "off": 2.2816e4})
    filt.append({"name": 'co', 
		    "on": 2.2757e4, "off": 2.3024e4})

    for fs in filt:
	fs["dw"] = fs["off"] - fs["on"]
	fs["w"] = fs["on"] + (fs["dw"] / 2.0)
	
    waveH210 = filt[0]["w"]
    waveH221 = filt[1]["w"]
    waveCo   = filt[2]["w"]

    # Keep the integration times
    tCo = 29.0 * 3.0
    tH210 = 24.0 * 3.0
    tH221 = 22.0 * 3.0

    # Get x, y, and calc r from center
    names = s.getArray('name')
    magH210 = s.getArray('mag')
    magH221 = s.getArray('mag')
    x = s.getArrayFromEpoch(0, 'xpix')
    y = s.getArrayFromEpoch(0, 'ypix')
    xarc = s.getArrayFromEpoch(0, 'x')
    yarc = s.getArrayFromEpoch(0, 'y')

    # We need to convert magnitudes to fluxes first
    fCo   = s.getArrayFromEpoch(0, 'fwhm')
    fH210 = s.getArrayFromEpoch(1, 'fwhm')
    fH221 = s.getArrayFromEpoch(2, 'fwhm')

    # We need to get errors out
    eCo   = s.getArrayFromEpoch(0, 'snr')
    eH210 = s.getArrayFromEpoch(1, 'snr')
    eH221 = s.getArrayFromEpoch(2, 'snr')

    # Cut for anisoplanatism
    xmid = (x.max() - x.min()) / 2.0
    ymid = (y.max() - y.min()) / 2.0
    rnirc2 = hypot(x - xmid, y - ymid)

    # Trim to central 3"
    idx = (where(rnirc2 < 350))[0]
    names = [names[i] for i in idx]
    magH210 = magH210[idx]
    magH221 = magH221[idx]
    x = x[idx]
    y = y[idx]
    xarc = xarc[idx]
    yarc = yarc[idx]
    rnirc2 = rnirc2[idx]
    fCo   = fCo[idx]
    fH210 = fH210[idx]
    fH221 = fH221[idx]
    eCo   = eCo[idx]
    eH210 = eH210[idx]
    eH221 = eH221[idx]

    

    #eCo = 1.0 / eCo
    #eH210 = 1.0 / eH210
    #eH221 = 1.0 / eH221

    eCo = sqrt(fCo)
    eH210 = sqrt(fH210)
    eH221 = sqrt(fH221)

    # Convert to cts per unit wavelength per unit time.
    fCo   /= tCo
    fH210 /= tH210
    fH221 /= tH221

    fH210 /= filt[0]["dw"]
    fH221 /= filt[1]["dw"]
    fCo   /= filt[2]["dw"]

    eCo   /= tCo
    eH210 /= tH210
    eH221 /= tH221

    eH210 /= filt[0]["dw"]
    eH221 /= filt[1]["dw"]
    eCo   /= filt[2]["dw"]

    # Flux calibrate everything
    scaleCo = 1.0
    scaleH210 = fCo[0] / fH210[0]
    scaleH221 = fCo[0] / fH221[0]
    #scaleCo = 1.0
    #scaleH210 = 0.981865
    #scaleH221 = 1.18

    fCo *= scaleCo
    fH210 *= scaleH210
    fH221 *= scaleH221

    eCo *= scaleCo
    eH210 *= scaleH210
    eH221 *= scaleH221

    print 'Scale Parameters Used for Flux Calibration:'
    print '   CO scale = ', scaleCo
    print ' H210 scale = ', scaleH210
    print ' H221 scale = ', scaleH221

    w = [waveH210, waveH221]

    slopes = (fH210 - fH221) / (waveH210 - waveH221)
    interc = fH221 - (slopes * waveH221)

    slopesErr = sqrt(eH210**2 + eH221**2) / abs(waveH210 - waveH221)
    intercErr = sqrt((eH221*waveH210)**2 + (eH210*waveH221)**2)
    intercErr /= abs(waveH210 - waveH221)

    #contCo = interc + (slopes * waveCo)
    #contCoErr = sqrt(intercErr**2 + (slopesErr * waveCo)**2)
    
    contCo =  fH210 * (waveCo - waveH221) / (waveH210 - waveH221)
    contCo -= fH221 * (waveCo - waveH210) / (waveH210 - waveH221)
    contCoErr = sqrt( (eH210 * (waveCo - waveH221) / (waveH210 - waveH221))**2 +
                      (eH221 * (waveCo - waveH210) / (waveH210 - waveH221))**2)
    
    i = 0
    print 'flux = %e + wave * %e' % (interc[i], slopes[i])
    print '%6.4f  %e  %e' % (waveH210, fH210[i], 
			     interc[i]+(slopes[i] * waveH210)) 
    print '%6.4f  %e  %e' % (waveH221, fH221[i],
			     interc[i]+(slopes[i] * waveH221)) 
    print '%6.4f  %e  %e' % (waveCo, fCo[i],
			     interc[i]+(slopes[i] * waveCo)) 

    
    # Plot in fluxes
    eqwCo = 1.0 - (fCo  / contCo)
    eqwCoErr = (fCo / contCo) * sqrt((eCo / fCo)**2 + (contCoErr / contCo)**2)

    eqwCoSingle = 1.0 - (fCo  / fH221)
    eqwCoSingleErr = (fCo / fH221) * sqrt((eCo / fCo)**2 + (eH221 / fH221)**2)

    # Might also plot in magnitudes
    #fTot = fH210 + fH221 + fCo
    #zeropoint = 8.33
    #magH210 = -2.5 * log10(fH210/fTot[0]) + zeropoint
    #magH221 = -2.5 * log10(fH221/fTot[0]) + zeropoint
    #magCo = -2.5 * log10(fCo/fTot[0]) + zeropoint
    #magCoCont = -2.5 * log10(contCo/fTot[0]) + zeropoint


    # Can plot SED
    w = array([waveH210, waveH221, waveCo])

    srcListRoot = '/u/ghezgroup/data/gc/source_list/'
    yngnames = young.youngStarNames(srcListRoot + 'young_new.dat')
    
    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = starTables.lateStarNames(srcListRoot + 'late.dat')
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    clf()
    plot(eqwCo, magH221, 'k.')
    errorbar(eqwCo, magH221, fmt='k.', xerr=eqwCoErr)
    plot(eqwCo[idx], magH221[idx], 'c.')
    plot(eqwCo[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('CO Index')
    ylabel('K (mag)')
    axis([-0.2, 0.2, 16.0, 8])
    #axis([rng[0], rng[1], rng[3], rng[2]])
    #axis([-0.2, 0.2, rng[3], 9.0])
    savefig('plots/plotCo'+suffix+'.eps')
    savefig('plots/plotCo'+suffix+'.png')

    clf()
    plot(eqwCoSingle, magH221, 'k.')
    errorbar(eqwCoSingle, magH221, fmt='k.', xerr=eqwCoSingleErr)
    plot(eqwCoSingle[idx], magH221[idx], 'c.')
    plot(eqwCoSingle[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('CO Index (one Continuum)')
    ylabel('K (mag)')
    axis([-0.2, 0.2, 16, 8])
    #axis([-0.35, 0.05, rng[3], 9.0])
    savefig('plots/plotCo_coSingle'+suffix+'.eps')
    savefig('plots/plotCo_coSingle'+suffix+'.png')

    # Print out a ds9 *.reg file with color coded circles
    # Find faint-end locus
    tdx = where(magH221 > 13)
    coLocus = eqwCo[tdx].mean()
    coLimit = coLocus
    magLimit = 13.5
    print 'CO Locus:  %5.2f' % coLocus
    print 'CO Limit:  %5.2f' % coLimit
    print 'Mag Limit: %5.2f' % magLimit

    rdx = (where((magH221 < magLimit) & (eqwCo < coLimit)))[0]
    bdx = (where((magH221 < magLimit) & (eqwCo > coLimit)))[0]

    _reg = open('tables/plotCo'+suffix+'.reg', 'w')
    _reg.write('# Region file format: DS9 version 3.0\n')
    _reg.write('global color=green font="helvetica 10 normal"')
    _reg.write('edit=1 move=1 delete=1 include=1 fixed=0\n')

    print ''
    print '%-13s  %4s  %5s  %7s  %7s (%4s, %4s)  CandidateType' % \
          ('#Name', 'Mag', 'eqwCO', 'x(")', 'y(")', 'xpix', 'ypix')

    cntEarlyCand = 0
    cntLateCand = 0
    cntEarlyConf = 0
    cntLateConf = 0
    cntEarlyWrong = 0
    cntLateWrong = 0
    
    for ii in range(len(magH221)):
        thick=1
        color = 'yellow'
        msg = ''
        label = ''
        
        if (ii in rdx):
            # Candidate Early
            color = 'blue'
            thick = 2
            msg += 'Early-Type?'
            label = 'text={%s}' % names[ii]
            cntEarlyCand += 1

        if (ii in bdx):
            # Candidate Late
            color = 'red'
            thick = 2
            msg += 'Late-Type? '
            label = 'text={%s}' % names[ii]
            cntLateCand += 1

        _reg.write('image;circle(%.3f, %.3f,5) # color=%s width=%d %s\n' \
                   % (x[ii]+1, y[ii]+1, color, thick, label))

        if ((ii in idx) or (ii in cdx)):
            # Spectral-Type is Known
            thick=2

            if (ii in idx):
                # Early
                color = 'blue'
                msg += ' (confirmed early)'

                if (ii in rdx):
                    cntEarlyConf += 1
                if (ii in bdx):
                    cntLateWrong += 1

            if (ii in cdx):
                # Late
                color = 'red'
                msg += ' (confirmed late)'

                if (ii in rdx):
                    cntEarlyWrong +=1
                if (ii in bdx):
                    cntLateConf += 1

            _reg.write('image;box(%.3f, %.3f,14,14,0) # color=%s width=%d\n' \
                       % (x[ii]+1, y[ii]+1, color, thick))


        if ((ii in rdx) or (ii in bdx) or (ii in idx) or (ii in cdx)):
            print '%-13s  %4.1f  %5.2f  %7.3f  %7.3f (%4d, %4d)   ## %s' % \
                  (names[ii], magH221[ii], eqwCo[ii], 
                   xarc[ii], yarc[ii], x[ii], y[ii], msg)

    print ''
    print 'Early:  %2d canddiates; %2d confirmed; %2d wrong' % \
          (cntEarlyCand, cntEarlyConf, cntEarlyWrong)
    print 'Late;   %2d candidates; %2d confirmed; %2d wrong' % \
          (cntLateCand, cntLateConf, cntLateWrong)
    _reg.close()
    


def plot05junlgs(root, suffix='_05junlgs'):
    # This should be align output with the following order:
    #    CO
    #    Kcont
    s = starset.StarSet(root)

    # Setup some filter properties
    filt = []
    filt.append({"name": 'kcont', 
		    "on": 2.2558e4, "off": 2.2854e4})
    filt.append({"name": 'co', 
		    "on": 2.2757e4, "off": 2.3024e4})

    for fs in filt:
	fs["dw"] = fs["off"] - fs["on"]
	fs["w"] = fs["on"] + (fs["dw"] / 2.0)
	
    waveKcont = filt[0]["w"]
    waveCo   = filt[1]["w"]

    # Keep the integration times
    tCo = 7.2 * 5.0
    tKcont = 11.9 * 5.0

    # We need to convert magnitudes to fluxes first
    fCo   = s.getArrayFromEpoch(0, 'fwhm')
    fKcont = s.getArrayFromEpoch(1, 'fwhm')
    print fCo[0:10]
    print fKcont[0:10]

    # We need to get errors out
    eCo   = s.getArrayFromEpoch(0, 'snr')
    eKcont = s.getArrayFromEpoch(1, 'snr')

    eCo = sqrt(fCo)
    eKcont = sqrt(fKcont)

    # Convert to cts per unit wavelength per unit time.
    fCo   /= tCo
    fKcont /= tKcont

    fKcont /= filt[0]["dw"]
    fCo   /= filt[1]["dw"]

    eCo   /= tCo
    eKcont /= tKcont

    eKcont /= filt[0]["dw"]
    eCo   /= filt[1]["dw"]


    names = s.getArray('name')

    # Flux calibrate everything
    scaleCo = 1.0
    scaleKcont = fCo[2] / fKcont[2]
    #scaleCo = 1.0
    #scaleKcont = 0.981865

    fCo *= scaleCo
    fKcont *= scaleKcont

    eCo *= scaleCo
    eKcont *= scaleKcont

    print 'Scale Parameters Used for Flux Calibration:'
    print '   CO scale = ', scaleCo
    print ' Kcont scale = ', scaleKcont

    # Plot in fluxes
    eqwCoSingle = 1.0 - (fCo  / fKcont)
    eqwCoSingleErr = (fCo / fKcont) * sqrt((eCo / fCo)**2 + \
                                           (eKcont / fKcont)**2)

    # Might also plot in magnitudes
    magKcont = s.getArray('mag')
    x = s.getArrayFromEpoch(0, 'xpix')
    y = s.getArrayFromEpoch(0, 'ypix')
    xarc = s.getArrayFromEpoch(0, 'x')
    yarc = s.getArrayFromEpoch(0, 'y')

    srcListRoot = '/u/ghezgroup/data/gc/source_list/'
    yngnames = young.youngStarNames(srcListRoot + 'young_new.dat')
    
    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = starTables.lateStarNames(srcListRoot + 'late.dat')
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    clf()
    plot(eqwCoSingle, magKcont, 'k.')
    errorbar(eqwCoSingle, magKcont, fmt='k.', xerr=eqwCoSingleErr)
    plot(eqwCoSingle[idx], magKcont[idx], 'c.')
    plot(eqwCoSingle[cdx], magKcont[cdx], 'r.')
    rng = axis()
    xlabel('CO Index (one Continuum)')
    ylabel('K (mag)')
    axis([-0.2, 0.2, 16, 8])
    #axis([-0.35, 0.05, rng[3], 9.0])
    savefig('plots/plotCo_coSingle'+suffix+'.eps')
    savefig('plots/plotCo_coSingle'+suffix+'.png')

    # Print out a ds9 *.reg file with color coded circles
    # Find faint-end locus
    tdx = where(magKcont > 13)
    coLocus = eqwCoSingle[tdx].mean()
    coLimit = coLocus
    magLimit = 13.5
    print 'CO Locus:  %5.2f' % coLocus
    print 'CO Limit:  %5.2f' % coLimit
    print 'Mag Limit: %5.2f' % magLimit

    rdx = (where((magKcont < magLimit) & (eqwCoSingle < coLimit)))[0]
    bdx = (where((magKcont < magLimit) & (eqwCoSingle > coLimit)))[0]

    _reg = open('tables/plotCo'+suffix+'.reg', 'w')
    _reg.write('# Region file format: DS9 version 3.0\n')
    _reg.write('global color=green font="helvetica 10 normal"')
    _reg.write('edit=1 move=1 delete=1 include=1 fixed=0\n')

    print ''
    print '%-13s  %4s  %5s  %7s  %7s (%4s, %4s)  CandidateType' % \
          ('#Name', 'Mag', 'eqwCO', 'x(")', 'y(")', 'xpix', 'ypix')

    cntEarlyCand = 0
    cntLateCand = 0
    cntEarlyConf = 0
    cntLateConf = 0
    cntEarlyWrong = 0
    cntLateWrong = 0
    
    for ii in range(len(magKcont)):
        thick=1
        color = 'yellow'
        msg = ''
        label = ''
        
        if (ii in rdx):
            # Candidate Early
            color = 'blue'
            thick = 2
            msg += 'Early-Type?'
            label = 'text={%s}' % names[ii]
            cntEarlyCand += 1

        if (ii in bdx):
            # Candidate Late
            color = 'red'
            thick = 2
            msg += 'Late-Type? '
            label = 'text={%s}' % names[ii]
            cntLateCand += 1

        _reg.write('image;circle(%.3f, %.3f,5) # color=%s width=%d %s\n' \
                   % (x[ii]+1, y[ii]+1, color, thick, label))

        if ((ii in idx) or (ii in cdx)):
            # Spectral-Type is Known
            thick=2

            if (ii in idx):
                # Early
                color = 'blue'
                msg += ' (confirmed early)'

                if (ii in rdx):
                    cntEarlyConf += 1
                if (ii in bdx):
                    cntLateWrong += 1

            if (ii in cdx):
                # Late
                color = 'red'
                msg += ' (confirmed late)'

                if (ii in rdx):
                    cntEarlyWrong +=1
                if (ii in bdx):
                    cntLateConf += 1

            _reg.write('image;box(%.3f, %.3f,14,14,0) # color=%s width=%d\n' \
                       % (x[ii]+1, y[ii]+1, color, thick))


        if ((ii in rdx) or (ii in bdx) or (ii in idx) or (ii in cdx)):
            print '%-13s  %4.1f  %5.2f  %7.3f  %7.3f (%4d, %4d)   ## %s' % \
                  (names[ii], magKcont[ii], eqwCoSingle[ii], 
                   xarc[ii], yarc[ii], x[ii], y[ii], msg)

    print ''
    print 'Early:  %2d canddiates; %2d confirmed; %2d wrong' % \
          (cntEarlyCand, cntEarlyConf, cntEarlyWrong)
    print 'Late;   %2d candidates; %2d confirmed; %2d wrong' % \
          (cntLateCand, cntLateConf, cntLateWrong)
    _reg.close()
    

def plot06maylgs1(root, suffix='_06maylgs1'):
    # This should be align output with the following order:
    #    CO
    #    H2 1-0
    #    H2 2-1
    s = starset.StarSet(root)

    # Setup some filter properties
    filt = []
    filt.append({"name": 'h210', 
		    "on": 2.1112e4, "off": 2.1452e4})
    filt.append({"name": 'h221', 
		    "on": 2.2428e4, "off": 2.2816e4})
    filt.append({"name": 'co', 
		    "on": 2.2757e4, "off": 2.3024e4})

    for fs in filt:
	fs["dw"] = fs["off"] - fs["on"]
	fs["w"] = fs["on"] + (fs["dw"] / 2.0)
	
    waveH210 = filt[0]["w"]
    waveH221 = filt[1]["w"]
    waveCo   = filt[2]["w"]

    # Keep the integration times
    tCo = 20.0 * 2.0
    tH210 = 29.0 * 2.0
    tH221 = 25.0 * 2.0

    # We need to convert magnitudes to fluxes first
    fCo   = s.getArrayFromEpoch(0, 'fwhm')
    fH210 = s.getArrayFromEpoch(1, 'fwhm')
    fH221 = s.getArrayFromEpoch(2, 'fwhm')

    # We need to get errors out
    eCo   = s.getArrayFromEpoch(0, 'snr')
    eH210 = s.getArrayFromEpoch(1, 'snr')
    eH221 = s.getArrayFromEpoch(2, 'snr')

    #eCo = 1.0 / eCo
    #eH210 = 1.0 / eH210
    #eH221 = 1.0 / eH221

    eCo = sqrt(fCo)
    eH210 = sqrt(fH210)
    eH221 = sqrt(fH221)

    # Convert to cts per unit wavelength per unit time.
    fCo   /= tCo
    fH210 /= tH210
    fH221 /= tH221

    fH210 /= filt[0]["dw"]
    fH221 /= filt[1]["dw"]
    fCo   /= filt[2]["dw"]

    eCo   /= tCo
    eH210 /= tH210
    eH221 /= tH221

    eH210 /= filt[0]["dw"]
    eH221 /= filt[1]["dw"]
    eCo   /= filt[2]["dw"]

    names = s.getArray('name')

    # Flux calibrate everything
    scaleCo = 1.0
    scaleH210 = fCo[0] / fH210[0]
    scaleH221 = fCo[0] / fH221[0]
    #scaleCo = 1.0
    #scaleH210 = 0.981865
    #scaleH221 = 1.18

    fCo *= scaleCo
    fH210 *= scaleH210
    fH221 *= scaleH221

    eCo *= scaleCo
    eH210 *= scaleH210
    eH221 *= scaleH221

    print 'Scale Parameters Used for Flux Calibration:'
    print '   CO scale = ', scaleCo
    print ' H210 scale = ', scaleH210
    print ' H221 scale = ', scaleH221

    w = [waveH210, waveH221]

    slopes = (fH210 - fH221) / (waveH210 - waveH221)
    interc = fH221 - (slopes * waveH221)

    slopesErr = sqrt(eH210**2 + eH221**2) / abs(waveH210 - waveH221)
    intercErr = sqrt((eH221*waveH210)**2 + (eH210*waveH221)**2)
    intercErr /= abs(waveH210 - waveH221)

    #contCo = interc + (slopes * waveCo)
    #contCoErr = sqrt(intercErr**2 + (slopesErr * waveCo)**2)
    
    contCo =  fH210 * (waveCo - waveH221) / (waveH210 - waveH221)
    contCo -= fH221 * (waveCo - waveH210) / (waveH210 - waveH221)
    contCoErr = sqrt( (eH210 * (waveCo - waveH221) / (waveH210 - waveH221))**2 +
                      (eH221 * (waveCo - waveH210) / (waveH210 - waveH221))**2)
    
    i = 0
    print 'flux = %e + wave * %e' % (interc[i], slopes[i])
    print '%6.4f  %e  %e' % (waveH210, fH210[i], 
			     interc[i]+(slopes[i] * waveH210)) 
    print '%6.4f  %e  %e' % (waveH221, fH221[i],
			     interc[i]+(slopes[i] * waveH221)) 
    print '%6.4f  %e  %e' % (waveCo, fCo[i],
			     interc[i]+(slopes[i] * waveCo)) 

    
    # Plot in fluxes
    eqwCo = 1.0 - (fCo  / contCo)
    eqwCoErr = (fCo / contCo) * sqrt((eCo / fCo)**2 + (contCoErr / contCo)**2)

    eqwCoSingle = 1.0 - (fCo  / fH221)
    eqwCoSingleErr = (fCo / fH221) * sqrt((eCo / fCo)**2 + (eH221 / fH221)**2)

    magH210 = s.getArray('mag')
    magH221 = s.getArray('mag')
    x = s.getArrayFromEpoch(0, 'xpix')
    y = s.getArrayFromEpoch(0, 'ypix')
    xarc = s.getArrayFromEpoch(0, 'x')
    yarc = s.getArrayFromEpoch(0, 'y')

    corr = s.getArrayFromEpoch(0, 'corr')
    r = hypot(xarc, yarc)

    idx = (where(r < 4.0))[0]
    eqwCo = eqwCo[idx]
    eqwCoErr = eqwCoErr[idx]
    eqwCoSingle = eqwCoSingle[idx]
    eqwCoSingleErr = eqwCoSingleErr[idx]
    magH221 = magH221[idx]
    x = x[idx]
    y = y[idx]
    xarc = xarc[idx]
    yarc = yarc[idx]
    names = [names[i] for i in idx]


    # Can plot SED
    w = array([waveH210, waveH221, waveCo])

    srcListRoot = '/u/ghezgroup/data/gc/source_list/'
    yngnames = young.youngStarNames(srcListRoot + 'young_new.dat')
    
    idx = []
    for yngname in yngnames:
	try:
	    ndx = names.index(yngname)
	    idx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    oldnames = starTables.lateStarNames(srcListRoot + 'late.dat')
    cdx = []
    for oldname in oldnames:
	try:
	    ndx = names.index(oldname)
	    cdx.append(ndx)
	except ValueError:
	    # Do nothing
	    foo = 0.0

    clf()
    plot(eqwCo, magH221, 'k.')
    errorbar(eqwCo, magH221, fmt='k.', xerr=eqwCoErr)
    plot(eqwCo[idx], magH221[idx], 'c.')
    plot(eqwCo[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('CO Index')
    ylabel('K (mag)')
    axis([-0.2, 0.2, 16, 8])
    #axis([rng[0], rng[1], rng[3], rng[2]])
    #axis([-0.2, 0.2, rng[3], 9.0])
    savefig('plots/plotCo'+suffix+'.eps')
    savefig('plots/plotCo'+suffix+'.png')

    clf()
    plot(eqwCoSingle, magH221, 'k.')
    errorbar(eqwCoSingle, magH221, fmt='k.', xerr=eqwCoSingleErr)
    plot(eqwCoSingle[idx], magH221[idx], 'c.')
    plot(eqwCoSingle[cdx], magH221[cdx], 'r.')
    rng = axis()
    xlabel('CO Index (one Continuum)')
    ylabel('K (mag)')
    axis([-0.2, 0.2, 16, 8])
    #axis([-0.35, 0.05, rng[3], 9.0])
    savefig('plots/plotCo_coSingle'+suffix+'.eps')
    savefig('plots/plotCo_coSingle'+suffix+'.png')

    # Print out a ds9 *.reg file with color coded circles
    # Find faint-end locus
    tdx = where(magH221 > 13)
    coLocus = eqwCo[tdx].mean()
    coLimit = coLocus
    magLimit = 13.5
    print 'CO Locus:  %5.2f' % coLocus
    print 'CO Limit:  %5.2f' % coLimit
    print 'Mag Limit: %5.2f' % magLimit

    rdx = (where((magH221 < magLimit) & (eqwCo < coLimit)))[0]
    bdx = (where((magH221 < magLimit) & (eqwCo > coLimit)))[0]

    _reg = open('tables/plotCo'+suffix+'.reg', 'w')
    _reg.write('# Region file format: DS9 version 3.0\n')
    _reg.write('global color=green font="helvetica 10 normal"')
    _reg.write('edit=1 move=1 delete=1 include=1 fixed=0\n')

    print ''
    print '%-13s  %4s  %5s  %7s  %7s (%4s, %4s)  CandidateType' % \
          ('#Name', 'Mag', 'eqwCO', 'x(")', 'y(")', 'xpix', 'ypix')

    cntEarlyCand = 0
    cntLateCand = 0
    cntEarlyConf = 0
    cntLateConf = 0
    cntEarlyWrong = 0
    cntLateWrong = 0
    
    for ii in range(len(magH221)):
        thick=1
        color = 'yellow'
        msg = ''
        label = ''
        
        if (ii in rdx):
            # Candidate Early
            color = 'blue'
            thick = 2
            msg += 'Early-Type?'
            label = 'text={%s}' % names[ii]
            cntEarlyCand += 1

        if (ii in bdx):
            # Candidate Late
            color = 'red'
            thick = 2
            msg += 'Late-Type? '
            label = 'text={%s}' % names[ii]
            cntLateCand += 1

        _reg.write('image;circle(%.3f, %.3f,5) # color=%s width=%d %s\n' \
                   % (x[ii]+1, y[ii]+1, color, thick, label))

        if ((ii in idx) or (ii in cdx)):
            # Spectral-Type is Known
            thick=2

            if (ii in idx):
                # Early
                color = 'blue'
                msg += ' (confirmed early)'

                if (ii in rdx):
                    cntEarlyConf += 1
                if (ii in bdx):
                    cntLateWrong += 1

            if (ii in cdx):
                # Late
                color = 'red'
                msg += ' (confirmed late)'

                if (ii in rdx):
                    cntEarlyWrong +=1
                if (ii in bdx):
                    cntLateConf += 1

            _reg.write('image;box(%.3f, %.3f,14,14,0) # color=%s width=%d\n' \
                       % (x[ii]+1, y[ii]+1, color, thick))


        if ((ii in rdx) or (ii in bdx) or (ii in idx) or (ii in cdx)):
            print '%-13s  %4.1f  %5.2f  %7.3f  %7.3f (%4d, %4d)   ## %s' % \
                  (names[ii], magH221[ii], eqwCo[ii], 
                   xarc[ii], yarc[ii], x[ii], y[ii], msg)

    print ''
    print 'Early:  %2d canddiates; %2d confirmed; %2d wrong' % \
          (cntEarlyCand, cntEarlyConf, cntEarlyWrong)
    print 'Late;   %2d candidates; %2d confirmed; %2d wrong' % \
          (cntLateCand, cntLateConf, cntLateWrong)
    _reg.close()
    
