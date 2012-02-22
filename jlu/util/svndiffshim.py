#!/usr/local/bin/python

# inspired by http://www.mikeash.com/blog/pivot/entry.php?id=7#body
#
# This version's verbosity is devoted to generating useful filenames 
# that show up in the title bar of the FileMerge window.
# I have no idea if this is remotely universal, but it works for me.
#
# Patches welcome. bbum@mac.com

import sys
import os
import tempfile
from optparse import OptionParser
import re
import shutil

parser = OptionParser("usage: %prog <options>")
parser.add_option('-u', None,
                  dest="unifiedFlag",
                  action="store_true",
                  help="Probably asks diff to be unified.  Ignored.")
parser.add_option('-L', None,
                  dest="fileNamesAndVersions",
                  action="append",
                  help="The file naming informaiton.")

(globalOptions, args) = parser.parse_args()

if len(args) < 2:
    parser.print_usage()
    sys.exit(1)

file1RawPath = args[0]
file2RawPath = args[1]

tempDir = tempfile.gettempdir()
tempDir = os.path.join(tempDir, "svndiffshim-%d" % os.getuid())
if not os.path.isdir(tempDir):
   os.mkdir(tempDir)

infoRE = re.compile('(?P<filename>[^\t]*)(\t|  )(\(revision (?P<version>[^)]*)|\(working copy\)|\(local\))')

def randomStringGenerator():
    randomFD = open('/dev/random', 'r')
    while 1:
       yield "".join(["%02x" % ord(x[0]) for x in randomFD.read(4)])
randomString = randomStringGenerator()

def fileNameFromSubversionInfo(info):
    info = "-".join(info.split("/"))
    match = infoRE.match(info)
    if not match:
       print "Failed to match: %s" % info
       return None
    version = match.group('version')
    if not version:
       version = "LocalVersion"
    return "%s-%s-%s" % (match.group('filename'), version, randomString.next())

fileInfoArray = globalOptions.fileNamesAndVersions
file1Path = os.path.join(tempDir, fileNameFromSubversionInfo(fileInfoArray[0]))
file2Path = os.path.join(tempDir, fileNameFromSubversionInfo(fileInfoArray[1]))
shutil.copy(file1RawPath, file1Path)
shutil.copy(file2RawPath, file2Path)

os.execlp("opendiff", "opendiff", file1Path, file2Path)
