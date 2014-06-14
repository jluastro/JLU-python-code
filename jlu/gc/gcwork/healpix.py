import math
import numpy


"""
; -----------------------------------------------------------------------------
;
;  Copyright (C) 1997-2005  Krzysztof M. Gorski, Eric Hivon, Anthony J. Banday
;
;
;
;
;
;  This file is part of HEALPix.
;
;  HEALPix is free software; you can redistribute it and/or modify
;  it under the terms of the GNU General Public License as published by
;  the Free Software Foundation; either version 2 of the License, or
;  (at your option) any later version.
;
;  HEALPix is distributed in the hope that it will be useful,
;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;  GNU General Public License for more details.
;
;  You should have received a copy of the GNU General Public License
;  along with HEALPix; if not, write to the Free Software
;  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
;
;  For more information about HEALPix see http://healpix.jpl.nasa.gov
;
; -----------------------------------------------------------------------------
"""

def ang2pix_ring(nside, theta, phi):
    """
;**************************************************************************
;+
; ANG2PIX_RING, Nside, Theta, Phi, Ipring
;
;        renders the RING scheme pixel number Ipring for a pixel which, given th
e
;        map resolution parameter Nside, contains the point on the sphere
;        at angular coordinates Theta and Phi
;
; INPUT
;    Nside     : determines the resolution (Npix = 12* Nside^2)
;       SCALAR
;    Theta : angle (along meridian), in [0,Pi], theta=0 : north pole,
;       can be an ARRAY
;    Phi   : angle (along parallel), in [0,2*Pi]
;       can be an ARRAY of same size as theta
;
; OUTPUT
;    Ipring  : pixel number in the RING scheme of HEALPIX pixelisation in [0,Npi
x-1]
;       can be an ARRAY of same size as Theta and Phi
;    pixels are numbered along parallels (ascending phi), 
;    and parallels are numbered from north pole to south pole (ascending theta)
;
;
; SUBROUTINE
;    nside2npix
;
; HISTORY
;    June-October 1997,  Eric Hivon & Kris Gorski, TAC, 
;            original ang_pix
;    Feb 1999,           Eric Hivon,               Caltech
;            name changed to ang2pix_ring
;    Sept 2000,          EH
;           free memory by discarding unused variables
;    June 2003,  EH, replaced STOPs by MESSAGEs
;    Aug  2004,  EH, use !PI as theta upper-bound instead of !DPI
;-
;*******************************************************************************
**
"""
    npix = nside2npix(nside)

    np0 = len(theta)
    np1 = len(phi) 

    if (np0 != np1):
        print 'inconsistent theta and phi'

    if ((min(theta) < 0.0) or (max(theta) > math.pi)):
        print 'theta out of range'

    ipring = numpy.zeros(np0, dtype=numpy.long)

    nside  = long(nside)
    pion2 = math.pi * 0.5
    twopi = math.pi * 2.0
    nl2   = 2*nside
    nl4   = 4*nside
    ncap  = nl2*(nside-1)
    
    cth0 = 2.0/3.0
    
    cth_in = numpy.cos(theta)
    phi_in = phi % twopi
    phi_in = phi + numpy.where(phi <= 0.0, 1, 0)*twopi

    # Equatorial Strip
    pix_eqt = (numpy.where((cth_in <= cth0) & (cth_in > -cth0)))[0]
    n_eqt = len(pix_eqt)

    if (n_eqt > 0):
        tt = phi_in[pix_eqt] / pion2

        # Increasing edge line index
        jp = numpy.array(nside * (0.5 + tt - cth_in[pix_eqt]*0.75), dtype=numpy.long)
        # Decreasing edge line index
        jm = numpy.array(nside*(0.5 + tt + cth_in[pix_eqt]*0.75), dtype=numpy.long)
        
        # in {1,2n+1} (ring number counted from z=2/3)
        ir = (nside + 1) + jp - jm
        # k=1 if ir even, and 0 otherwise
        k =  numpy.where( (ir % 2) == 0, 1, 0 )   
        
        # in {1,4n}
        ip = ( ( jp+jm+k + (1-nside) ) / 2 ) + 1 
        ip = ip - nl4*numpy.where(ip > nl4, 1, 0)

        ipring[pix_eqt] = ncap + nl4*(ir-1) + ip - 1
        
        cth0 = 2.0/3.0

        cth_in = numpy.cos(theta)
        phi_in = phi % twopi
        phi_in = phi + numpy.where(phi < 0.0, 1, 0)*twopi
        
    # equatorial strip
    pix_eqt = (numpy.where((cth_in <= cth0) & (cth_in > -cth0)) )[0]
    n_eqt = len(pix_eqt)
      
    if (n_eqt > 0):
        tt = phi_in[pix_eqt] / pion2
        # increasing edge line index
        jp = numpy.array(nside*(0.50 + tt - cth_in[pix_eqt]*0.75), dtype=numpy.long) 
        # decreasing edge line index
        jm = numpy.array(nside*(0.50 + tt + cth_in[pix_eqt]*0.75), dtype=numpy.long) 
        
        # in {1,2n+1} (ring number counted from z=2/3)
        ir = (nside + 1) + jp - jm
        # k=1 if ir even, and 0 otherwise
        k =  numpy.where( (ir % 2) == 0, 1, 0)
        
        # in {1,4n}
        ip = ( ( jp+jm+k + (1-nside) ) / 2 ) + 1 
        ip = ip - nl4*numpy.where(ip > nl4, 1, 0)
        
        ipring[pix_eqt] = ncap + nl4*(ir-1) + ip - 1
        tt = 0
        jp = 0
        jm = 0
        ir = 0
        k = 0
        ip = 0
        pix_eqt = 0

    # north caps
    pix_np  = (numpy.where(cth_in >  cth0))[0]
    n_np = len(pix_np)
    
    if (n_np > 0):
        tt = phi_in[pix_np] / pion2
        tp = tt % 1.0
        tmp = numpy.sqrt( 3.0*(1.0 - abs(cth_in[pix_np])) )

        # increasing edge line index
        jp = numpy.array( nside * tp         * tmp, dtype=numpy.long )
        # decreasing edge line index
        jm = numpy.array( nside * (1.0 - tp) * tmp, dtype=numpy.long ) 
        
        ir = jp + jm + 1         # ring number counted from the closest pole
        ip = numpy.array( tt * ir, dtype=numpy.long ) + 1 # in {1,4*ir}
        ir4 = 4*ir
        ip = ip - ir4*numpy.where(ip > ir4, 1, 0)
        
        ipring[pix_np] =  2*ir*(ir-1) + ip - 1
        tt = 0
        tp = 0
        tmp =0
        jp = 0
        jm = 0
        ir = 0
        ip = 0
        ir4 = 0
        pix_np = 0
        

    # south pole
    pix_sp  = (numpy.where(cth_in <= -cth0))[0]
    n_sp = len(pix_sp)
    
    if (n_sp > 0):
        tt = phi_in[pix_sp] / pion2
        tp = tt % 1.0
        tmp = numpy.sqrt( 3.0*(1.0 - abs(cth_in[pix_sp])) )

        # increasing edge line index
        jp = numpy.array( nside * tp         * tmp, dtype=numpy.long )
        # decreasing edge line index
        jm = numpy.array( nside * (1.0 - tp) * tmp, dtype=numpy.long )  
        
        ir = jp + jm + 1         # ring number counted from the closest pole
        ip = numpy.array( tt * ir, dtype=numpy.long ) + 1 # in {1,4*ir}
        ir4 = 4*ir
        ip = ip - ir4*numpy.where(ip > ir4, 1, 0)

        ipring[pix_sp] = npix - 2*ir*(ir+1) + ip - 1
        tt = 0
        tp = 0
        tmp = 0
        jp = 0
        jm = 0
        ir = 0
        ip = 0
        ir4 = 0
        pix_sp = 0

    return ipring

def pix2ang_ring(nside, ipix):
    """
;****************************************************************************************
;+
; PIX2ANG_RING, nside, ipix, theta, phi
; 
;       renders Theta and Phi coordinates of the nominal pixel center
;       given the RING scheme pixel number Ipix and map resolution parameter Nside
;
; INPUT
;    Nside     : determines the resolution (Npix = 12* Nside^2)
;	SCALAR
;    Ipix  : pixel number in the RING scheme of Healpix pixelisation in [0,Npix-1]
;	can be an ARRAY 
;       pixels are numbered along parallels (ascending phi), 
;       and parallels are numbered from north pole to south pole (ascending theta)
;
; OUTPUT
;    Theta : angle (along meridian = co-latitude), in [0,Pi], theta=0 : north pole,
;	is an ARRAY of same size as Ipix
;    Phi   : angle (along parallel = azimut), in [0,2*Pi]
;	is an ARRAY of same size as Ipix
;
; SUBROUTINE
;    nside2npix
;
; HISTORY
;    June-October 1997,  Eric Hivon & Kris Gorski, TAC
;    Aug  1997 : treats correctly the case nside = 1
;    Feb 1999,           Eric Hivon,               Caltech
;         renamed pix2ang_ring
;    Sept 2000,          EH
;           free memory by discarding unused variables
;    June 2003,  EH, replaced STOPs by MESSAGEs
;
;-
;****************************************************************************************
    """

    npix = nside2npix(nside)

    nside = long(nside)
    nl2 = 2*nside
    nl3 = 3*nside
    nl4 = 4*nside
    ncap = nl2*(nside-1)
    nsup = nl2*(5*nside+1)
    fact1 = 1.5*nside
    fact2 = (3.0*nside)*nside
    np = len(ipix)
    theta = numpy.zeros(np, dtype=numpy.float)
    phi   = numpy.zeros(np, dtype=numpy.float)

    min_pix = ipix.min()
    max_pix = ipix.max()
    if (min_pix < 0):
        print 'pixel index : %10d' % min_pix
        print 'is out of range : %2d %8d' % (0, npix-1)

    if (max_pix > npix-1):
        print 'pixel index : %10d' % max_pix
        print 'is out of range : %2d %8d' % (0,npix-1)


    # north polar cap
    pix_np = (numpy.where(ipix < ncap))[0]
    n_np = len(pix_np)   
    if (n_np > 0): 
       ip = numpy.array(ipix[pix_np] + 1.0, dtype=numpy.long)

       # counted from NORTH pole, starting at 1
       iring = numpy.array(numpy.sqrt( (ip/2.0) - numpy.sqrt(ip/2) ) + 1, dtype=numpy.long)
       iphi  = numpy.array(ip - ( 2*iring*(iring-1) ), dtype=numpy.long)

       theta[pix_np] = numpy.arccos( 1.0 - iring**2 / fact2 )
       phi[pix_np]   = (iphi - 0.5) * math.pi/(2.0*iring)
       ip = 0
       iring = 0
       iphi = 0    # free memory
       pix_np = 0  # free memory


    # equatorial strip
    pix_eq = (numpy.where((ipix >= ncap) & (ipix < nsup)))[0]
    n_eq = len(pix_eq)

    if (n_eq > 0):
        # counted from NORTH pole
        ip    = numpy.array(ipix[pix_eq] - ncap, dtype=numpy.long)
        iring = numpy.array((ip / nl4) + nside, dtype=numpy.long)
        iphi  = ( ip % nl4 ) + 1
        
        # 1 if iring is odd, 1/2 otherwise
        fodd  = 0.5 * (1 + ((iring+nside) % 2)) 

        theta[pix_eq] = numpy.arccos( (nl2 - iring) / fact1 )
        phi[pix_eq]   = (iphi - fodd) * math.pi/(2.0*nside)
        ip = 0
        iring = 0
        iphi = 0    # free memory
        pix_eq = 0  # free memory


    # south polar cap
    pix_sp = (numpy.where(ipix >= nsup))[0]
    n_sp = len(pix_sp)   

    if ((n_np + n_sp + n_eq) != np):
        print 'Error!'

    if (n_sp > 0):
       ip =  numpy.array(npix - ipix[pix_sp], dtype=numpy.long)
       # counted from SOUTH pole, starting at 1
       iring = numpy.array(numpy.sqrt( (ip/2.0) - numpy.sqrt(ip/2) ) + 1, dtype=numpy.long)
       iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1))

       theta[pix_sp] = numpy.arccos( -1.0 + iring**2 / fact2 )
       phi[pix_sp]   = (iphi - 0.5) * math.pi/(2.0*iring)

       # free memory
       ip = 0
       iring = 0
       iphi = 0
       pix_sp = 0

    return (theta, phi)


def nside2npix(nside):
    """
    ;+
    ; npix = nside2npix(nside)
    ;
    ; returns npix = 12*nside*nside
    ; number of pixels on a Healpix map of resolution nside
    ;
    ; if nside is not a power of 2 <= 8192,
    ; -1 is returned and the error flag is set to 1
    ;
    ; MODIFICATION HISTORY:
    ;
    ;     v1.0, EH, Caltech, 2000-02-11
    ;     v1.1, EH, Caltech, 2002-08-16 : uses !Healpix structure
    ;-
    """
    npix = 12 * long(nside)**2

    return npix
