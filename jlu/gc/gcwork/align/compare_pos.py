#!/usr/bin/env python
import optparse
import textwrap
import numpy as np
import pylab as py
import math
import sys
from gcwork import starset
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.lines as lines

##################################################
# 
# Help formatter for command line arguments. 
# This is very generic... skip over for main code.
#
##################################################
class IndentedHelpFormatterWithNL(optparse.IndentedHelpFormatter):
    def format_description(self, description):
        if not description: return ""
        desc_width = self.width - self.current_indent
        indent = " "*self.current_indent
        # the above is still the same
        bits = description.split('\n')
        formatted_bits = [
            textwrap.fill(bit,
                          desc_width,
                          initial_indent=indent,
                          subsequent_indent=indent)
            for bit in bits]
        result = "\n".join(formatted_bits) + "\n"
        return result

    def format_option(self, option):
        # The help for each option consists of two parts:
        #   * the opt strings and metavars
        #   eg. ("-x", or "-fFILENAME, --file=FILENAME")
        #   * the user-supplied help string
        #   eg. ("turn on expert mode", "read data from FILENAME")
        #
        # If possible, we write both of these on the same line:
        #   -x    turn on expert mode
        #
        # But if the opt string list is too long, we put the help
        # string on a second line, indented to the same column it would
        # start in if it fit on the first line.
        #   -fFILENAME, --file=FILENAME
        #       read data from FILENAME
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            opts = "%*s%s\n" % (self.current_indent, "", opts)
            indent_first = self.help_position
        else: # start help on same line as opts
            opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            # Everything is the same up through here
            help_lines = []
            for para in help_text.split("\n"):
                help_lines.extend(textwrap.wrap(para, self.help_width))
            # Everything is the same after here
            result.append("%*s%s\n" % (
                    indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line)
                           for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)



##################################################
#
# Main body of compare_pos
#
##################################################
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Read options and check for errors.
    options = read_command_line(argv)
    if (options == None):
        return

    s = starset.StarSet(options.align_root)

    nEpochs = len(s.stars[0].years)
    nStars = len(s.stars)

    names = np.array(s.getArray('name'))

    if (options.center_star != None):
        idx = np.where(names == options.center_star)[0]

        if (len(idx) > 0):
            options.xcenter = s.stars[idx].x
            options.ycenter = s.stars[idx].y
        else:
            print 'Could not find star to center, %s. Reverting to Sgr A*.' % \
                  (options.center_star)

    # Create a combined error term (quad sum positional and alignment)
    combineErrors(s)
    
    yearsInt = np.floor(s.years)

    # Set up a color scheme
    cnorm = colors.normalize(s.years.min(), s.years.max()+1)
    cmap = cm.gist_ncar

    colorList = []
    for ee in range(nEpochs):
        colorList.append( cmap(cnorm(yearsInt[ee])) )

    py.close(2)
    py.figure(2, figsize=(10,10))

    previousYear = 0.0
    for ee in range(nEpochs):
        x = s.getArrayFromEpoch(ee, 'x')
        y = s.getArrayFromEpoch(ee, 'y')

        xe = s.getArrayFromEpoch(ee, 'xerr')
        ye = s.getArrayFromEpoch(ee, 'yerr')

        mag = s.getArrayFromEpoch(ee, 'mag')

        idx = np.where((x > -1000) & (y > -1000))[0]
        x = x[idx]
        y = y[idx]
        xe = xe[idx]
        ye = ye[idx]
        mag = mag[idx]

        tmpNames = names[idx]
        
        if yearsInt[ee] != previousYear:
            previousYear = yearsInt[ee]
            label = '%d' % yearsInt[ee]
        else:
            label = '_nolegend_'

        (line, foo1, foo2) = py.errorbar(x, y, xerr=xe, yerr=ye,
                                         color=colorList[ee], fmt='k^',
                                         markeredgecolor=colorList[ee],
                                         label=label, picker=4)

        #selector = HighlightSelected(line, )

    class HighlightSelected(lines.VertexSelector):
        def __init__(self, line):
            lines.VertexSelector.__init__(self, line)
            self.markers, = self.axes.plot([], [], 'k^', markerfacecolor='none')

        def process_selected(self, ind, xs, ys):
            self.markers.set_data(xs, ys)
            self.canvas.draw()

    def onpick(event):
        polyCollection = event.artist
        polyCollection.get_xdata()
        polyCollection.get_xdata()
        ind = event.ind
        


    xlo = options.xcenter + (options.range)
    xhi = options.xcenter - (options.range)
    ylo = options.ycenter - (options.range)
    yhi = options.ycenter + (options.range)

    py.axis('equal')
    py.axis([xlo, xhi, ylo, yhi])
    py.legend(numpoints=1, loc='lower left')
    py.show()

    return
            



def read_command_line(argv):
    p = optparse.OptionParser(usage='usage: %prog [options] [starlist]',
                              formatter=IndentedHelpFormatterWithNL())

    p.add_option('-e', '--errors', dest='plot_errors', default=False,
                 action='store_true', 
                 help='Plot error bars on all the points (quad sum of '+
                 'positional and alignment errors.')
    p.add_option('-s', '--star', dest='center_star', default=None,
                 metavar='[star]',
                 help='Named star to center initial plot on.')
    p.add_option('-r', '--range', dest='range', default=0.4, type=float,
                 metavar='[arcsec]',
                 help='Sets the half width of the X and Y axis in arcseconds'+
                 'from -xcen and -ycen  or 0,0 (default: %default)')
    p.add_option('-x', '--xcen', dest='xcenter', default=0, type=float, 
                 metavar='[arcsec]',
                 help='The X center point of the plot in arcseconds offset' +
                 'from Sgr A*.')
    p.add_option('-y', '--ycen', dest='ycenter', default=0, type=float, 
                 metavar='[arcsec]',
                 help='The Y center point of the plot in arcseconds offset' +
                 'from Sgr A*.')
    p.add_option('-f', '--find', dest='isInteractive', default=True,
                 action='store_true',
                 help='Turns on interactive mode.')
    p.add_option('-l', '--label', dest='labelOn', default=False,
                 action='store_false',
                 help='Label the name of the star at the last valid epoch.')

    options, args = p.parse_args(argv)
    
    # Keep a copy of the original calling parameters
    options.originalCall = ' '.join(argv)

    # Read the input filename
    options.align_root = None
    if len(args) == 1:
        options.align_root = args[0]
    else:
        print ''
        p.print_help()
        return None

    return options
    

def combineErrors(s):
    for ss in range(len(s.stars)):
        star = s.stars[ss]
        
        for ee in range(len(s.stars[0].e)):
            epoch = star.e[ee]
            
            epoch.xerr = math.sqrt(epoch.xerr_p**2 + epoch.xerr_a**2)
            epoch.yerr = math.sqrt(epoch.yerr_p**2 + epoch.yerr_a**2)

if __name__ == '__main__':
    main()
