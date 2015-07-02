#!/usr/bin/env python
# $Id$
#
# Copyright (c) 2007, Michael P. Fitzgerald (mpfitz@berkeley.edu)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Michael P. Fitzgerald may not be used to endorse
#       or promote products derived from this software without specific
#       prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY MICHAEL P. FITZGERALD ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MICHAEL P. FITZGERALD BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Some of the conversion code ported from J. Baker's nat2jour.pl
# part of the astronat package.
#
# Commands for compressing figures ported from M. Perrin's
# submission-prep Perl script.

import sys, os, re, shutil, string, pdb

default_maxauths = 8
default_res = 75      # [dpi]
default_thresh = 200  # [kilobytes]


def check_file(fn, exit=True):
    """
    Check if a file is readable.  If 'exit' is set, then the program will
    exit on failure.
    """
    if os.access(fn, os.R_OK): return True
    print "Cannot read %s !" % fn
    if exit: sys.exit(1)
    return False

def check_ext(s, ext):
    "get filename with default extension"
    if not os.access(s, os.R_OK):
        s += ext
    if not os.access(s, os.R_OK):
        print "can't find %s ..." % s
    return s

def which(program):
    "check if a program name is executable"
    # http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.exists(fpath) and os.access(fpath, os.X_OK)
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def compress_figure(eps_fn, res = default_res, thresh = default_thresh):
    """
    Compress an eps figure for arXiv submission.  'res' is resolution
    in dpi; 'thresh' is threshold for compression in kb
    """
    # sanity check: only compress eps figures
    ext = eps_fn.split('.')[-1]
    if ext != 'eps':
        print "\tskipping %s -- not an eps file!" % eps_fn
        return

    # compress if less than threshold
    thresh *= 1024 # [kilobytes] -> [bytes]
    if os.path.getsize(eps_fn) > thresh:
        # check that gs and convert commands are available
        if not which('gs'):
            print "`gs` not found on path!  skipping compression for %s ..." % eps_fn
            return
        if not which('convert'):
            print "`convert` not found on path!  skipping compression for %s ..." % eps_fn
            return

        # determine filenames
        print "\tprocessing %s ..." % eps_fn
        jpg_fn = eps_fn.replace('.eps', '.jpg')
        new_eps_fn = eps_fn.replace('.eps', '_p.eps')
        # create processed figure
        os.system("gs -r%d -dEPSCrop -dTextAlphaBits=4 -sDEVICE=jpeg -sOutputFile=%s -dBATCH -dNOPAUSE %s" % (res, jpg_fn, eps_fn))
        os.system("convert %s %s" % (jpg_fn, new_eps_fn))
        os.remove(jpg_fn)
        # use new figure if smaller
        if os.path.getsize(new_eps_fn) < os.path.getsize(eps_fn):
            os.rename(new_eps_fn, eps_fn)
        else:
            os.remove(new_eps_fn)
    else:
        print "\tskipping %s ..." % eps_fn

def massage_bbl(bbl_infn, bbl_outfn, maxauths):

    bbl_inf = open(bbl_infn)

    procbib = False # currently processing bibitems
    bibinfo = {} # contains bibliography information for each key
    bibkeys = [] # ordered list of keys
    n_auths = {} # number of authors

    for line in bbl_inf:
        # new bibitem
        if re.match(r"\\bibitem", line):
            item = line
            procbib = True
            continue

        # append line to current item string
        if procbib:
            item += line

        # end of item -- process
        if re.match(r"\n", line) and procbib:
            procbib = False

            # strip newlines
            item = re.sub(r"\n", '', item)

            # strip '\natexlab's
            item = re.sub(r"\{\\natexlab\{(.*?)\}\}", r"\1", item)

            # strip \noopsort{}'s
            item = re.sub(r"\{\\noopsort\{(.*?)\}\}", '', item)

            # parse entry
            m = re.match(r"\\bibitem\[\{(.*?)\((\d{4}[a-z]*)\)(.*?)\}\]\{(.*?)\}(.*)", item)
            if m:
                shortlist, year, longlist, key, ref = m.groups()
                authlist = longlist and longlist or shortlist

                # save number of authors
                n_auths[key] = authlist.count(',')+1

                # shorten author list in reference if too long
                if n_auths[key] > maxauths:
                    # extract author list, last comma before year (if any) and year
                    m = re.match(r"(.*?)(\,)?\s*(\d{4})", ref)
                    if not m:
                        print "Can't find year in reference:\n%s" % ref
                    auths, lastcomma = m.group(1, 2)
                    lastcomma = lastcomma or '' # set blank string if no match
                    end = m.start(3) # position of year

                    if not re.match(r"\w", auths): # avoid refs like "---, 1998, ..."
                        #n_commas = maxauths*2
                        n_commas = 2
                        if n_commas > auths.count(','):
                            print "Error trying to truncate author list:\n%s" % ref
                        else:
                            # find the index of the Nth comma
                            pos = 0
                            for i in range(n_commas):
                                pos = auths.index(',', pos)
                                pos += 1
                            # replace with etal
                            ref = ref[0:pos]+" {et~al.}"+lastcomma+ref[end-1:]

                # output entry
                bibkeys.append(key)
                bibinfo[key] = shortlist, year, longlist, ref

            else: # didn't match
                print "weird bibitem: %s" % item

    bbl_inf.close()

    # output info to bbl file
    bbl_outf = open(bbl_outfn, 'w')
    bbl_outf.write("\\begin{thebibliography}{%d}\n\n" % len(bibkeys))
    for key in bibkeys:
        shortlist, year, longlist, ref = bibinfo[key]
        bbl_outf.write("\\bibitem[{%s(%s)%s}]{%s}\n%s\n\n" % (shortlist, year, longlist, key, ref))
    bbl_outf.write("\\end{thebibliography}\n")
    bbl_outf.close()

    return bibkeys


class SubmissionPreparer(object):
    "Class for preparing a manuscript for submission"

    # the possible extensions for graphics files, in order of
    # preference (in principle, should match LaTeX's preferencing)
    _graphics_exts
     = ('ps', 'eps', 'pdf')

    def get_submission_dir(self, fileroot):
        "convention for submission directory"
        return fileroot+'_submission/'

    def figname(self, num, i, n, ext='eps'):
        "filename convention for processed figures"
        if n == 1:
            letter = ''
        else:
            letter = string.ascii_lowercase[i]
        return "f%d%s.%s" % (num, letter, ext)

    def get_orig_figname(self, gfile):
        "Returns the filename and extension of a graphics file."
        ext = gfile.split('.')[-1]
        if ext not in self._graphics_exts:
            # figure out extension
            for x in self._graphics_exts:
                if os.access(gfile+'.'+x, os.R_OK):
                    # extension found, append to graphics filename
                    ext = x
                    gfile += '.'+ext
                    break
            if ext not in self._graphics_exts:
                print "Couldn't find graphics file %s !" % gfile
                sys.exit(1)
        return gfile, ext

    def parse_figures(self, line, firstpass=True):
        "parse a graphics object line (uncommented)"
        # FIXME  handle case when more than one command per line

        # strip leading/trailing whitespace
        line = line.strip()

        if re.match(r"\\begin\{figure\*?\}", line):
            self.cursubfig = 0
            self.in_figure = True
            if firstpass:
                self.curfig_graphics = [] # a list of graphics info for the current figure (the list contains info for f#a, f#b, etc.)

        m = re.match(r"[^%]*\\plotone\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
            self.cursubfig += 1

        m = re.match(r"[^%]*\\includegraphics(?:\[.*?\])?\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
            self.cursubfig += 1

        m = re.match(r"[^%]*\\plottwo\s*\{(.*?)\}\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
                self.curfig_graphics.append((m.group(2), None)) 
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
                line = line.replace(self.figinfo[self.curfig][self.cursubfig+1][0],
                                    self.figinfo[self.curfig][self.cursubfig+1][2])
            self.cursubfig += 2

        m = re.match(r"[^%]*\\epsfig\s*\{(?:.*,)?file=(.*?)(?:,.*)?\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None))
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
                self.cursubfig += 1

        m = re.match(r"[^%]*\\printonlygray\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics[-1] = (self.curfig_graphics[-1][0], m.group(1)) # graphic name, pog name
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig-1][3],
                                    self.figinfo[self.curfig][self.cursubfig-1][5])



        if re.match(r"\\end\{figure\*?\}", line):
            # got all graphics objects in this figure environment
            self.curfig += 1
            self.in_figure = False

            if firstpass:
                # store filename information for graphics in current figure
                curfig_info = []
                self.figinfo.append(curfig_info)

                for i, (orig_gfile, orig_pog_gfile) in enumerate(self.curfig_graphics):
                    # get full filename and extension
                    gfile, ext = self.get_orig_figname(orig_gfile)

                    # get the filename for the new version
                    newfile = self.figname(self.curfig, i, len(self.curfig_graphics), ext=ext)
                    # put figure in list
                    self.figlist.append((gfile, newfile))

                    # same for print-only gray version
                    if orig_pog_gfile is not None:
                        pog_gfile, pog_ext = self.get_orig_figname(orig_pog_gfile)
                        pog_newfile = self.figname(self.curfig, i, len(self.curfig_graphics), ext=pog_ext).replace('.'+pog_ext, '_gray.'+pog_ext)
                        self.figlist.append((pog_gfile, pog_newfile))

                        curfig_info.append((orig_gfile, gfile, newfile,
                                            orig_pog_gfile, pog_gfile, pog_newfile))

                    else:
                        curfig_info.append((orig_gfile, gfile, newfile,
                                            None, None, None))

        return line

    def process_bbl(self, bbl_infn, bbl_outfn, maxauths):
        "Process the bibliography file.  Returns a list of bibliography keys in order of bib. entry."
        
        self.filelist.append(bbl_outfn)
        return massage_bbl(bbl_infn, self.subdir+bbl_outfn, maxauths)


    def parse_extra(self, line):
        "any extra processing"
        return line
    
    def process_manuscript(self, ms_infn, ms_outfn, strip, bbl_outfn, bibkeys=None):
        "Processes the manuscript"

        if bibkeys is None: bibkeys = []

        ms_inf = open(ms_infn)
        self.filelist.append(ms_outfn)
        ms_outf = open(self.subdir+ms_outfn, 'w')

        filebegin = True # True if we haven't encountered any non-whitespace/comments yet
        in_document = False # True in document environment
        first_citation = dict(zip(bibkeys, (True,)*len(bibkeys)))

        # process line-by-line
        # NOTE can't handle things when commands are broken across lines
        firstpass_outlines = []
        for line in ms_inf:

            if strip:
                # skip processing if blank
                if line == '\n':
                    if not filebegin: firstpass_outlines.append(line)
                    continue

                # strip comments
                line = re.sub(r"^%.*\n?", '\n', line)
                line = re.sub(r"^(.*[^\\])%.*\n?", r"\1\n", line)
                if line == '\n':
                    continue

            filebegin = False # now we're processing real commands
            # check if we're in the meat of the text
            if re.match(r"[^%]*\\begin\{document\}", line):
                in_document = True
            if re.match(r"[^%]*\\end\{document\}", line):
                in_document = False

            # handle any extra processing
            line = self.parse_extra(line)

            # handle graphics objects
            try:
                line = self.parse_figures(line)
            except AssertionError:
                print "Error parsing figure line:\n%s" % line
                sys.exit(1)

            # include files
            m = re.match(r"[^%]*\\input\{(.*)\}", line)
            if m:
                infile = m.groups()[0]
                infile = check_ext(infile, '.tex')
                if do_input:
                    # input file
                    # NOTE  assumes one input command per line
                    firstpass_outlines.extend(open(infile).readlines())
                    line = re.sub(r"\\input\{.*\}", '', line)
                else:
                    # copy file
                    self.filelist.append(infile)
                    print "%s -> %s" % (infile, self.subdir+infile)
                    shutil.copyfile(infile, self.subdir+infile)


            # if this is the first citation for a ref w/ 3 authors, use the long format
            # NOTE  no longer required by ApJ (?)
##             if in_document:
##                 # iter over all citations in this line
##                 for m in re.finditer(r"\\([Cc]ite.*?)(\[.*\])?\{(.*?)\}", line):
##                     citecomm, opt, keystr = m.groups()
##                     opt = opt or '' # set to blank string if no match
##                     comm = "\\%s%s{%s}" % (citecomm, opt, keystr)
##                     # iter over all keys in this citation
##                     keys = re.split(r"\,\s*", keystr)
##                     for key in keys:
##                         if first_citation[key]:
##                             first_citation[key] = False

##                             if n_auths[key] == 3:
##                                 if len(keys) > 1:
##                                     print "warning: won't use long format for 1st occ. in mult. cit.: %s" % comm
##                                 else:
##                                     # make sure long-format
##                                     if not citecomm.endswith('*'):
##                                         # replace first occurrence with long-format version
##                                         newcomm = "\\%s*%s{%s}" % (citecomm, opt, keystr)
##                                         line = line.replace(comm, newcomm, 1)

            # remove bibliography style commands
            if re.match(r"\\bibliographystyle", line):
                continue

            # replace bibliography entry
            line = re.sub(r"\\bibliography\{.*\}", r"\\input{"+bbl_outfn+"}", line)

            # comment out natbib package
            # NOTE  must be alone on a line
            line = re.sub(r"\\usepackage\{natbib\}", r"%\\usepackage{natbib}", line)

            # write line to output
            firstpass_outlines.append(line)

        ms_inf.close()

        # do second pass
        secondpass_outlines = []
        self.curfig, self.cursubfig = 0, 0
        for line in firstpass_outlines:
            # handle graphics objects
            try:
                line = self.parse_figures(line, firstpass=False)
            except AssertionError:
                print "Error parsing figure line:\n%s" % line
                sys.exit(1)
            secondpass_outlines.append(line)

        # write to file
        if '\n' not in secondpass_outlines[0]:
            ms_outf.write('\n'.join(secondpass_outlines))
        else:
            ms_outf.writelines(secondpass_outlines)
        ms_outf.close()


    def setup(self, fileroot):
        "set up internal variables"

        self.filelist = [] # holds list of files
        self.curfig = 0    # index for current figure number
        self.cursubfig = 0 # index for current subfigure number
        self.figlist = []  # holds list of figure graphics filenames
        self.figinfo = []  # list of lists containing graphics info for each figure
        self.curfig_graphics = [] # helper list for graphics info of current figure in figure parser
        self.in_figure = False # currently in a figure environment

        # make submission directory
        self.subdir = self.get_submission_dir(fileroot)
        if not os.access(self.subdir, os.W_OK):
            os.mkdir(self.subdir)


    def do_prep(self, fileroot, strip=True, maxauths=8, do_input=True, do_figcheck=False, **kwargs):
        """Prepares given tex file for submission.

        fileroot    (string) root of manuscript filename (fileroot.tex)
        strip       (bool)   option to strip blank lines/comments
        maxauths    (int)    option for maximum number of authors to list
        do_input    (bool)   option to place \input'ted files into master document, instead of copy
                             NOTE:  does not handle recursion
        do_figcheck (bool)   create fileroot_figcheck.tex, useful for examining figures

        """

        # file/path names
        bbl_infn = fileroot+'.bbl'
        bbl_outfn = 'ms.bbl'
        ms_infn = fileroot+'.tex'
        ms_outfn = 'ms.tex'
        if do_figcheck:
            figcheck_fn = fileroot+'_figcheck.tex'

        # sanity checks
        check_file(ms_infn)
        have_bbl = check_file(bbl_infn, exit=False)

        # convert bibliography file (.bbl)
        if have_bbl:
            bibkeys = self.process_bbl(bbl_infn, bbl_outfn, maxauths)
        else: bibkeys = []

        # convert manuscript
        self.process_manuscript(ms_infn, ms_outfn, strip, bbl_outfn, bibkeys=bibkeys)

        # copy figures
        for gfile, newfile in self.figlist:
            print "%s -> %s" % (gfile, newfile)
            self.filelist.append(newfile)
            shutil.copyfile(gfile, self.subdir+newfile)

        # make figure-check file
        if do_figcheck:
            f = open(figcheck_fn, 'w')
            f.write("""\\documentclass[preprint2]{aastex}
        \\begin{document}
        """)
            for gfile, newfile in self.figlist:
                f.write("\\includegraphics[width=.4\\textwidth]{%s}\n\n" % gfile)
            f.write("\\end{document}\n")
            f.close()

    def prep(self, fileroot, **kwargs):
        "wrapper for preparation"

        self.setup(fileroot)
        self.do_prep(fileroot, **kwargs)
        

_ApJSubmissionPreparerBase = SubmissionPreparer
class ApJSubmissionPreparer(_ApJSubmissionPreparerBase):
    "Class for preparing a manuscript for submission to ApJ"

    def get_submission_dir(self, fileroot):
        "convention for submission directory"
        return fileroot+'_apj/'

    def do_prep(self, fileroot, **kwargs):
        """Prepares given tex file for ApJ submission.

        fileroot    (string) root of manuscript filename (fileroot.tex)
        strip       (bool)   option to strip blank lines/comments
        maxauths    (int)    option for maximum number of authors to list
        do_input    (bool)   option to place \input'ted files into master document, instead of copy
                             NOTE:  does not handle recursion
        do_figcheck (bool)   create fileroot_figcheck.tex, useful for examining figures

        """

        # file/path names
        readme_infn = fileroot+'.README'
        readme_outfn = 'README'
        response_infn = 'ref_rept_response.txt'
        response_outfn = 'response'

        # copy README, if it exists, otherwise make it
        if check_file(readme_infn, exit=False):
            print "%s -> %s" % (readme_infn, readme_outfn)
            shutil.copyfile(readme_infn, self.subdir+readme_outfn)
        else:
            print "Creating %s ..." % readme_outfn
            open(self.subdir+readme_outfn, 'w').close()
        self.filelist.append(readme_outfn)

        # copy referee rept. response (if any)
        if os.access(response_infn, os.R_OK):
            print "%s -> %s" % (response_infn, response_outfn)
            self.filelist.append(response_outfn)
            shutil.copyfile(response_infn, self.subdir+response_outfn)

        # process
        _ApJSubmissionPreparerBase.do_prep(self, fileroot, **kwargs)

        # append file list to README
        f = open(self.subdir+readme_outfn, 'a')
        f.write('Included files:\n')
        for fn in self.filelist:
            if fn.count('gray') == 1:
                fn += '\t  (print-only gray version)'
            f.write("\t%s\n" % fn)
        f.write('\n')
        f.close()


_ArXivSubmissionPreparerBase = SubmissionPreparer
class ArXivSubmissionPreparer(_ArXivSubmissionPreparerBase):
    "Class for preparing a manuscript for submission to arXiv"

    def get_submission_dir(self, fileroot):
        "convention for submission directory"
        return fileroot+'_arxiv/'

    def parse_figures(self, line, firstpass=True):
        "parse a graphics object line (uncommented)"
        # FIXME  handle case when more than one command per line

        # strip leading/trailing whitespace
        line = line.strip()

        if re.match(r"\\begin\{figure\*?\}", line):
            self.cursubfig = 0
            self.in_figure = True
            if firstpass:
                self.curfig_graphics = [] # a list of graphics info for the current figure (the list contains info for f#a, f#b, etc.)

        m = re.match(r"[^%]*\\plotone\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])

            self.cursubfig += 1

        m = re.match(r"[^%]*\\includegraphics(?:\[.*?\])?\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])

            self.cursubfig += 1

        m = re.match(r"[^%]*\\plottwo\s*\{(.*?)\}\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None)) # graphic name, pog name
                self.curfig_graphics.append((m.group(2), None)) 
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
                line = line.replace(self.figinfo[self.curfig][self.cursubfig+1][0],
                                    self.figinfo[self.curfig][self.cursubfig+1][2])

            self.cursubfig += 2

        m = re.match(r"[^%]*\\epsfig\s*\{(?:.*,)?file=(.*?)(?:,.*)?\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics.append((m.group(1), None))
            else: # replace filename
                line = line.replace(self.figinfo[self.curfig][self.cursubfig][0],
                                    self.figinfo[self.curfig][self.cursubfig][2])
                self.cursubfig += 1

        m = re.match(r"[^%]*\\printonlygray\s*\{(.*?)\}", line)
        if m:
            assert self.in_figure
            if firstpass:
                self.curfig_graphics[-1] = (self.curfig_graphics[-1][0], m.group(1)) # graphic name, pog name
            else:
                # strip command
                line = re.sub(r"\\printonlygray\s*\{(.*?)\}", '', line)


        if re.match(r"\\end\{figure\*?\}", line):
            # got all graphics objects in this figure environment
            self.curfig += 1
            self.in_figure = False

            if firstpass:
                # store filename information for graphics in current figure
                curfig_info = []
                self.figinfo.append(curfig_info)

                for i, (orig_gfile, orig_pog_gfile) in enumerate(self.curfig_graphics):
                    # get full filename and extension
                    gfile, ext = self.get_orig_figname(orig_gfile)

                    # get the filename for the new version
                    newfile = self.figname(self.curfig, i, len(self.curfig_graphics), ext=ext)

                    # put figure in list
                    self.figlist.append((gfile, newfile))

                    # same for print-only gray version
                    if orig_pog_gfile is not None:
                        pog_gfile, pog_ext = self.get_orig_figname(orig_pog_gfile)
                        pog_newfile = self.figname(self.curfig, i, len(self.curfig_graphics), ext=pog_ext).replace('.'+pog_ext, '_gray.'+pog_ext)

                        curfig_info.append((orig_gfile, gfile, newfile,
                                            orig_pog_gfile, pog_gfile, pog_newfile))


                    else:
                        curfig_info.append((orig_gfile, gfile, newfile,
                                            None, None, None))

        return line

    def parse_extra(self, line):
        "extra processing to remove printonlygray command"
        line = re.sub(r"\\newcommand\{\\printonlygray\}\[1\]\{.*\}", '', line)
        return line
    
    def do_prep(self, fileroot, res=default_res, thresh=default_thresh, **kwargs):
        """Prepares given tex file for arXiv submission.

        fileroot    (string) root of manuscript filename (fileroot.tex)
        strip       (bool)   option to strip blank lines/comments
        maxauths    (int)    option for maximum number of authors to list
        do_input    (bool)   option to place \input'ted files into master document, instead of copy
                             NOTE:  does not handle recursion
        do_figcheck (bool)   create fileroot_figcheck.tex, useful for examining figures
        res         (int)    resolution for compressing figures [dpi]
        thresh      (int)    threshold for compressing figures [kb]

        """

        # process
        _ArXivSubmissionPreparerBase.do_prep(self, fileroot, **kwargs)

        # process figures for size requirements
        for dum, eps_fn in self.figlist:
            compress_figure(self.subdir+eps_fn, res=res, thresh=thresh)



if __name__ == '__main__':
    usage = """prep_jour.py    $Rev$
Copyright (c) 2007, Michael P. Fitzgerald (fitz@astro.berkeley.edu)
All rights reserved.

    Prepares a given tex file for submission to a journal service.

Usage:  prep_jour.py [-s] [-a <n>] [-i] [-m <mode>] <fileroot>

Required arguments:
    fileroot  root of manuscript filename (fileroot.tex)

Options:
    -s        disable stripping of blank lines/comments
    -a <n>    maximum number of authors to list in bibliography (default %d)
    -i        disable placement of \input'ted files into master document
    -f        create fileroot_figcheck.tex, useful for examining figures
    -m <mode> preparation mode, i.e. apj (default), arxiv
    -r <dpi>  resolution for compressed figures (arxiv mode, default %d dpi)
    -t <kb>   threshold for compressing figures (arxiv mode, default %d kb)
    
Description
-----------
    This program creates a directory called fileroot_submission which
    contains the processed manuscript and supporting files.  It
    expects your manuscript to follow a certain layout:

       fileroot.tex            Manuscript
       fileroot.bbl            Bibliography
    for ApJ:
       fileroot.README         README file (e.g. contact information)
       ref_rept_response.txt   (optional) response to referee


Print-only Gray Figures
-------------------------
    To specify print-only gray figures for ApJ (online-only color),
    this program expects the following:

    In your manuscript, create an \printonlygray command in the
    preamble, e.g.
    
       \newcommand{\printonlygray}[1]{\notetoeditor{Print-only grayscale figure: #1}}
    
    Then, for every grayscale figure that has an online-only color
    counterpart, use the color version in your manuscript and tag
    the grayscale version with the \printonlygray command, e.g.

       \plotone{example_color.eps}\printonlygray{example.eps}
    
""" % (default_maxauths, default_res, default_thresh)

    # get command line args/options
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'sa:ifm:r:t:')
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    # default options
    strip = True
    maxauths = default_maxauths
    do_input = True
    do_figcheck = False
    mode = 'apj'
    res = default_res
    thresh = default_thresh
    # process options
    for o, a in opts:
        if   o == '-s': strip = False
        elif o == '-a': maxauths = int(a)
        elif o == '-i': do_input = False
        elif o == '-f': do_figcheck = True
        elif o == '-m': mode = a
        elif o == '-r': res = int(a)
        elif o == '-t': thresh = int(a)
    # process arguments
    if not args:
        print usage
        sys.exit(1)
    fileroot = args.pop(0)

    # do preparation
    prepdict = {'apj':ApJSubmissionPreparer,
                'arxiv':ArXivSubmissionPreparer,
                }
    if mode not in prepdict:
        print "invalid mode!  must be one of %s" % repr(prepdict.keys())
        sys.exit(3)
    preparer = prepdict[mode]()
    preparer.prep(fileroot, strip=strip, maxauths=maxauths,
                  do_input=do_input, do_figcheck=do_figcheck,
                  res=res, thresh=thresh)
