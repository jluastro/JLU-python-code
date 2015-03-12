##############################
#
# ------  PYRAF ONLY  -------
#
# GSAOI Reduction Script for
#
# 2014 Feb 13-15
#
##############################

###############################################################################
# STEP 1: Initialize the required packages                                    #
###############################################################################
gemini
gsaoi
from pyraf.iraf import gemini
from pyraf.iraf import gsaoi

print "Unlearning tasks"
unlearn gemini
unlearn gsaoi

###############################################################################
# STEP 2: Define any variables, the database and the logfile.                  #
###############################################################################

# Define the logfile
gsaoi.logfile = "gsaoi_2014_02.log"

# Don't change these... bugs happen.
red_dir = "/u/jlu/data/gsaoi/GS-2014A-Q-19/reduce_2014_02/"
raw_dir = red_dir
prep_dir = red_dir + "g"

###############################################################################
# STEP 3: Prepare the raw data
###############################################################################
gaprepare('*.fits', rawpath=raw_dir, outpref=prep_dir,
          fl_vardq="yes", fl_sat="yes",
          logfile=prep_dir+"gaprepare.log")

###############################################################################
# STEP 4: Create a master flat image                                          #
###############################################################################

##########
# Kprime Flats 2014-02
##########
flat_root = "flat_kprime_2014_02"
first_frame = "S20140218S0393"
gemlist("S20140218S", "393-401,432-441", Stdout=flat_root+".lis")
imdelete(flat_root + ".fits", verify="no")
delete(flat_root + ".log", verify="no")
gaflat("@" + flat_root + ".lis", outsufx=flat_root,
       fl_vardq="yes", fl_dqprop="yes",
       rawpath=raw_dir, gaprep_pref=prep_dir, logfile=flat_root + ".log")
rename("g" + first_frame + "_" + flat_root + ".fits", flat_root + ".fits")

##########    
# Flats Jcont 2014-02
##########    
flat_root = "flat_jcont_2014_02"
first_frame = "S20140218S0462"
gemlist("S20140218S", "462-471", Stdout=flat_root + ".lis")
imdelete(flat_root + ".fits", verify="no")
delete(flat_root + ".log", verify="no")
gaflat("@" + flat_root + ".lis", outsufx=flat_root,
       fl_vardq="yes", fl_dqprop="yes",
       rawpath=raw_dir, gaprep_pref=prep_dir, logfile=flat_root + ".log")
rename("g" + first_frame + "_" + flat_root + ".fits", flat_root + ".fits")

##########    
# Flats Hcont 2014-02
##########    
flat_root = "flat_hcont_2014_02"
first_frame = "S20140218S0472"
gemlist("S20140218S", "472-501", Stdout=flat_root + ".lis")
imdelete(flat_root + ".fits", verify="no")
delete(flat_root + ".log", verify="no")
gaflat("@" + flat_root + ".lis", outsufx=flat_root,
       fl_vardq="yes", fl_dqprop="yes",
       rawpath=raw_dir, gaprep_pref=prep_dir, logfile=flat_root + ".log")
rename("g" + first_frame + "_" + flat_root + ".fits", flat_root + ".fits")

##########    
# Flats Hcont 2014-02
##########
flat_root = "flat_kcntlong_2014_02"
first_frame = "S20140219S0472"
gemlist("S20140219S", "472-473,475-482", Stdout=flat_root + ".lis")
imdelete(flat_root + ".fits", verify="no")
delete(flat_root + ".log", verify="no")
gaflat("@" + flat_root + ".lis", outsufx=flat_root,
       fl_vardq="yes", fl_dqprop="yes",
       rawpath=raw_dir, gaprep_pref=prep_dir, logfile=flat_root + ".log")
rename("g" + first_frame + "_" + flat_root + ".fits", flat_root + ".fits")



###############################################################################
# STEP 5: Create a master sky frame                                           #
###############################################################################
gsaoi.gaimchk.rawpath = ""
gsaoi.gaimchk.gaprep_pref = "g"

##########
# Sky Kprime 2014-02
##########
flat_root = "flat_kprime_2014_02"
sky_root = "sky_kprime_2014_02"
gemlist("S20140213S", "176-178", Stdout=sky_root + ".lis")
imdelete(sky_root + ".fits", verify="no")
delete(sky_root + ".log", verify="no")

gasky("g@" + sky_root + ".lis", outimage=sky_root + ".fits",
      fl_vardq=yes, fl_dqprop=yes,
      flatimg=red_dir + flat_root + ".fits",
      logfile=sky_root + ".log")

##########
# Sky Jcont 2014-02
##########
flat_root = "flat_jcont_2014_02"
sky_root = "sky_jcont_2014_02"
gemlist("S20140213S", "170-172", Stdout=sky_root + ".lis")
imdelete(sky_root + ".fits", verify="no")
delete(sky_root + ".log", verify="no")

gasky("g@" + sky_root + ".lis", outimage=sky_root + ".fits",
      fl_vardq=yes, fl_dqprop=yes,
      flatimg=red_dir + flat_root + ".fits",
      logfile=sky_root + ".log")


##########
# Sky Hcont 2014-02
##########
flat_root = "flat_hcont_2014_02"
sky_root = "sky_hcont_2014_02"
gemlist("S20140213S", "173-175", Stdout=sky_root + ".lis")
imdelete(sky_root + ".fits", verify="no")
delete(sky_root + ".log", verify="no")

gasky("g@" + sky_root + ".lis", outimage=sky_root + ".fits",
      fl_vardq=yes, fl_dqprop=yes,
      flatimg=red_dir + flat_root + ".fits",
      logfile=sky_root + ".log")

##########
# Sky Kcntlong 2014-02
##########
flat_root = "flat_kcntlong_2014_02"
sky_root = "sky_kcntlong_2014_02"
gemlist("S20140213S", "164-169", Stdout=sky_root + ".lis")
imdelete(sky_root + ".fits", verify="no")
delete(sky_root + ".log", verify="no")

gasky("g@" + sky_root + ".lis", outimage=sky_root + ".fits",
      fl_vardq=yes, fl_dqprop=yes,
      flatimg=red_dir + flat_root + ".fits",
      logfile=sky_root + ".log")


###############################################################################
# STEP 8: Reduce science images                                               #
###############################################################################

# By default the output will be converted to electrons.
##########
# Wd 1 pos 2 Kprime 2014-02
##########
wd1_root = 'wd1_p1_kprime_2014_02'
gemlist("S20140213S", "130,139,148,157", Stdout=wd1_root + ".lis")
imdelete("rg@" + wd1_root + ".lis", verify="no")
delete(wd1_root + ".log")

gareduce ("g@" + wd1_root + ".lis",
          fl_vardq="yes", fl_dqprop="yes", fl_dark="no",
          fl_flat="yes", flatimg="flat_kprime_2014_02.fits",
          fl_sky="yes", skyimg="sky_kprime_2014_02.fits", 
          calpath=red_dir, logfile=wd1_root + ".log")

##########
# Wd 1 pos 2 Jcont 2014-02
##########
wd1_root = 'wd1_p1_jcont_2014_02'
gemlist("S20140213S", "128,137,146,155", Stdout=wd1_root + ".lis")
imdelete("rg@" + wd1_root + ".lis", verify="no")
delete(wd1_root + ".log")

gareduce ("g@" + wd1_root + ".lis",
          fl_vardq="yes", fl_dqprop="yes", fl_dark="no",
          fl_flat="yes", flatimg="flat_jcont_2014_02.fits",
          fl_sky="yes", skyimg="sky_jcont_2014_02.fits", 
          calpath=red_dir, logfile=wd1_root + ".log")

##########
# Wd 1 pos 2 Hcont 2014-02
##########
wd1_root = 'wd1_p1_hcont_2014_02'
gemlist("S20140213S", "129,138,147,156", Stdout=wd1_root + ".lis")
imdelete("rg@" + wd1_root + ".lis", verify="no")
delete(wd1_root + ".log")

gareduce ("g@" + wd1_root + ".lis",
          fl_vardq="yes", fl_dqprop="yes", fl_dark="no",
          fl_flat="yes", flatimg="flat_hcont_2014_02.fits",
          fl_sky="yes", skyimg="sky_hcont_2014_02.fits", 
          calpath=red_dir, logfile=wd1_root + ".log")

##########
# Wd 1 pos 2 Kcntlong 2014-02
##########
wd1_root = 'wd1_p1_kcntlong_2014_02'
gemlist("S20140213S", "131-136,140-145,149-154,158-163", Stdout=wd1_root + ".lis")
imdelete("rg@" + wd1_root + ".lis", verify="no")
delete(wd1_root + ".log")

gareduce ("g@" + wd1_root + ".lis",
          fl_vardq="yes", fl_dqprop="yes", fl_dark="no",
          fl_flat="yes", flatimg="flat_kcntlong_2014_02.fits",
          fl_sky="yes", skyimg="sky_kcntlong_2014_02.fits", 
          calpath=red_dir, logfile=wd1_root + ".log")

# The outputs will, by default, have names the same as the inputs but prefixed
# with "rg".

###############################################################################
# STEP 9: Mosaic reduced science images                                       #
###############################################################################

##########
# Wd 1 pos 2 Kprime 2014-02
##########
wd1_root = 'wd1_p1_kprime_2014_02'
imdelete ("mrg//@" + wd1_root + ".lis", verify=no)
gamosaic ("rg@" + wd1_root + ".lis", fl_vardq=yes)

##########
# Wd 1 pos 2 Jcont 2014-02
##########
wd1_root = 'wd1_p1_jcont_2014_02'
imdelete ("mrg//@" + wd1_root + ".lis", verify=no)
gamosaic ("rg@" + wd1_root + ".lis", fl_vardq=yes)

##########
# Wd 1 pos 2 Hcont 2014-02
##########
wd1_root = 'wd1_p1_hcont_2014_02'
imdelete ("mrg//@" + wd1_root + ".lis", verify=no)
gamosaic ("rg@" + wd1_root + ".lis", fl_vardq=yes)

##########
# Wd 1 pos 2 Kcntlong 2014-02
##########
wd1_root = 'wd1_p1_kcntlong_2014_02'
imdelete ("mrg//@" + wd1_root + ".lis", verify=no)
gamosaic ("rg@" + wd1_root + ".lis", fl_vardq=yes)

# This, by default, will create files with names the same as the input with "m"
# prefixed to them.
# Mosaic the four individual data extensions into one extension. This is done
# for each of the extensions within the image, e.g., "SCI", "VAR" and "DQ".


###############################################################################
# Split into 4 images (for Starfinder) and copy into the clean directory.
###############################################################################
from astropy.table import Table
from jlu.gsaoi import reduce_quick

##########
# Wd 1 pos 2 Kprime 2014-02
##########
wd1_root = 'wd1_p1_kprime_2014_02'
clean_dir = '../clean/wd1_p1/kprime/'
lis = Table.read(wd1_root + '.lis', format='ascii',
                 data_start=0, names=['Frame'])
for ii in range(len(lis)):
    fitsfile = "rg" + lis['Frame'][ii] + ".fits"
    reduce_quick.convertFile(fitsfile, outputDir=clean_dir, clobber=True)

##########
# Wd 1 pos 2 Jcont 2014-02
##########
wd1_root = 'wd1_p1_jcont_2014_02'
clean_dir = '../clean/wd1_p1/jcont/'
lis = Table.read(wd1_root + '.lis', format='ascii',
                 data_start=0, names=['Frame'])
for ii in range(len(lis)):
    fitsfile = "rg" + lis['Frame'][ii] + ".fits"
    reduce_quick.convertFile(fitsfile, outputDir=clean_dir, clobber=True)

##########
# Wd 1 pos 2 Hcont 2014-02
##########
wd1_root = 'wd1_p1_hcont_2014_02'
clean_dir = '../clean/wd1_p1/hcont/'
lis = Table.read(wd1_root + '.lis', format='ascii',
                 data_start=0, names=['Frame'])
for ii in range(len(lis)):
    fitsfile = "rg" + lis['Frame'][ii] + ".fits"
    reduce_quick.convertFile(fitsfile, outputDir=clean_dir, clobber=True)

##########
# Wd 1 pos 2 Kcntlong 2014-02
##########
wd1_root = 'wd1_p1_kcntlong_2014_02'
clean_dir = '../clean/wd1_p1/kcntlong/'
lis = Table.read(wd1_root + '.lis', format='ascii',
                 data_start=0, names=['Frame'])
for ii in range(len(lis)):
    fitsfile = "rg" + lis['Frame'][ii] + ".fits"
    reduce_quick.convertFile(fitsfile, outputDir=clean_dir, clobber=True)
            
        


