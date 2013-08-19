# bfixpix - to fix pixels using medians of surrounding pixels
# by Ben Sugerman, ben@astro.columbia.edu
#
procedure bfixpix (input, mask, clean)
#
string  input       {prompt = "Input file (or list of files) to fixpix"}
string  mask	    {prompt = "mask file (0 == goodpix, >0 == badpix)"}
string  clean       {"", prompt = "cleaned version of input file, if necc"}
string  outsuffix   {"default", prompt = "fixpix'ed output suffix"}
string  msksuffix   {"default", prompt = "badpix significance mask suffix"}
bool    box	    {yes, prompt = "3x3 box (yes) or 3x3 cross (no)"}
#
begin
#

string inp,msk,cle,sout,mout
bool bx
string fileName

inp = input
msk = mask
cle = clean
sout = outsuffix
mout = msksuffix
bx = box

# Check whether 'input' refers to a list of files
if ( substr(inp,1,1) == "@" ) {
    list = ""
    list = substr(inp,2,strlen(inp))
    while ( fscan(list, fileName) != EOF) {
    	bfixpix_one( fileName, msk, cle, outsuffix=sout, msksuffix=mout, box=bx )
    }
} else {
    bfixpix_one( inp, msk, cle, outsuffix=sout, msksuffix=mout, box=bx )
}

end
