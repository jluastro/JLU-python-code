# bfixpix - to fix pixels using medians of surrounding pixels
# by Ben Sugerman, ben@astro.columbia.edu
#
procedure bfixpix (input, mask, clean)
#
string  input       {prompt = "Input file to fixpix"}
string  mask	    {prompt = "mask file (0 == goodpix, >0 == badpix)"}
string  clean       {"", prompt = "cleaned version of input file, if necc"}
string  outsuffix   {"default", prompt = "fixpix'ed output suffix"}
string  msksuffix   {"default", prompt = "badpix significance mask suffix"}
bool    box	        {yes, prompt = "3x3 box (yes) or 3x3 cross (no)"}
#
begin
#

string inp,msk,cle,sout,mout
string outf,outm
string medimg,medscl
real ct,it,st

inp = input
msk = mask
cle = clean
sout = outsuffix
mout = msksuffix

stsdas
toolbox
imgtools

if (sout == "default" || sout == "def") outf = inp//"f"
if (mout == "default" || mout == "def") outm = inp//"_s"
if (cle == "") cle = inp
if (access(outf)) imdelete(outf,verify=no)
#imcopy(inp,outf,verbose = yes)
printf("%s -> %s\n",inp,outf)

medimg = "medimg.fits"
medscl = "medscl.fits"

if (access(medimg)) imdelete(medimg, verify=no)
if (access(medscl)) imdelete(medscl, verify=no)

# median the image

if (box) {
   median(cle,medimg,3,3,
          zloreject = INDEF, zhireject = INDEF, boundary = "nearest",
          constant = 0.,verbose = no)
} else {
   rmedian(cle,medimg,0,1.4,
           ratio = 1., theta = 0., zloreject = INDEF, zhireject = INDEF,
           boundary = "nearest",constant = 0., verbose = no)
}

# scale the median image to input image

if (cle != inp) {
 imgets (cle,param="exptime",value="1.")
 ct= real(imgets.value)

 imgets (inp,param="exptime",value="1.")
 it= real(imgets.value)

 st=it/ct

 imarith(medimg,"*",st,medscl,
        title = "",divzero = 0.,hparams = "",pixtype = "real",
        calctype = "real",verbose = no,noact = no)
} else {

 imcopy(medimg,medscl,verbose=no)
}

# generate the pixel files

# !im3 is the mask normalized and inverted such that 0->1,1->0
# abs((!im3)-1.) is the normalized mask 

imcalc(inp//","//medscl//","//msk,outf,"(im1*(!im3))+(im2*abs((!im3)-1.))",
     pixtype = "old",nullval = 0.,verbose = no)

# generate the badpix significance file

imcalc(inp//","//medscl//","//msk,outm,"(im1-im2)*abs((!im3)-1.)",
     pixtype = "old",nullval = 0.,verbose = no)

# clean up files

#delete(nmask ,verify=no)
#delete(imask,verify=no)
imdelete(medimg,verify=no)
imdelete(medscl,verify=no)

end
