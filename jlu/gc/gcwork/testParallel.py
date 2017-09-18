import ipython1.kernel.api as kernel

def test():
    ipc = kernel.RemoteController(('127.0.0.1', 10105))
    print(ipc.getIDs())
    print(ipc.addr)

    ipc.executeAll('import time')
    foobar = ipc.executeAll('print time.ctime(time.time())')
    print(foobar)
    ipc.executeAll( 'from papers import lu06yng as lu' )
    print(ipc.executeAll('print lu.root'))
    ipc.executeAll( 'import healpix' )

    ipc.executeAll( 'from numarrray import *' )
    nodeIDs = ipc.getIDs()
    
    ipc.executeAll('pixIdx = arange(12288, type=Int)')
    ipc.executeAll('(ipix, opix) = healpix.pix2ang_ring(32, pixIdx)')
    ipix = ipc.gatherAll('ipix')

    print(ipix.shape)
    
    
