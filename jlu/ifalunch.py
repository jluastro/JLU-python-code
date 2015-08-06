import time
import numpy as np

options = {"Red Ginger": 50,
           "Bangkok Chef": 12.5,
           "Morning Glass": 20,
           "Serg's or \n Wings 'n Things": 12.5,
           "Pho Viet": 6.25,
           "Andy's": 6.25}

def ifalunch(verbose=True):
    if verbose:
        print("Today is " + time.ctime())

    # Get the weights
    weights = options.values()

    # Cumulative sum the weights
    cs = np.cumsum(weights)

    # Renormalize so that the total sum = 1.0
    cs /= cs[-1]

    # Randomly select a value
    hit = np.sum(cs < np.random.rand())

    # Results.
    if verbose:
        print("Go eat at " + options.keys()[hit])

    return hit

# Main Function (command line call)
if __name__ == "__main__":
    ifalunch()


def test():
    import pylab as py

    foo = np.zeros(1000)

    for ii in range(len(foo)):
        foo[ii] = ifalunch()

    py.clf()
    py.subplots_adjust(bottom=0.3, right=0.95, top=0.95)
    py.hist(foo, bins=range(7), align='right', normed=True,
            rwidth=0.8)
    py.xlim(0.5, 6.5)

    ax = py.gca()
    py.xticks(np.arange(7)+1, rotation=45, horizontalalignment='right')
    ax.set_xticklabels(options.keys())
