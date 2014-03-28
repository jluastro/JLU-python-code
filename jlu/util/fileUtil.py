import os, errno, shutil

def rmall(files):
    """Remove list of files without confirmation."""
    for file in files:
        if os.access(file, os.F_OK): os.remove(file)


def mkdir(dir):
    """Make directory if it doesn't already exist."""
    try: 
        os.makedirs(dir)
    except OSError, exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise
