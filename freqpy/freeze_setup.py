import sys
from cx_Freeze import setup, Executable
import matplotlib
import scipy
import os
import stat
import subprocess
import glob

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"packages": ["os", "wx", "matplotlib", "scipy", 
                                  "sklearn.utils.lgamma", 
                                  "sklearn.neighbors.typedefs",
                                  "sklearn.utils.sparsetools._graph_validation",
                                  "sklearn.utils.weight_vector"],
                     "include_files": ["SETTINGS", "Data", "Readme"],
                     "excludes": ["tkinter", 
                                  "collections.sys",
                                  "collections._weakref",
                                  "tcl"]}

bdist_msi_options = {
                    'upgrade_code': '{111111-1010101-666A}',
                    'add_to_path': True
                  }

build_dmg_options = {
                    'applications_shortcut': True,
                    'volume_label': 'FreqPy'
                  }

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(  name = "FreqPy",
        version = "0.00001",
        description = "Analyze Data for Population Coding",
        options = {
                    "build_exe": build_exe_options,
                    "bdist_msi": bdist_msi_options,
                    "build_dmg": build_dmg_options
                    },
        executables = [Executable("freqgui.py", base=base)])


### Fix for mac
# this is a straight copy paste from macdist.py in the cx_Freeze code
def setRelativeReferencePaths(binDir):
    """ For all files in Contents/MacOS, check if they are binaries
        with references to other files in that dir. If so, make those
        references relative. The appropriate commands are applied to all
        files; they will just fail for files on which they do not apply."""
    files = []
    for root, dirs, dir_files in os.walk(binDir):
        files.extend([os.path.join(root, f).replace(binDir + "/", "")
                      for f in dir_files])
    for fileName in files:

        # install_name_tool can't handle zip files or directories
        filePath = os.path.join(binDir, fileName)
        if fileName.endswith('.zip'):
            continue

        # ensure write permissions
        mode = os.stat(filePath).st_mode
        if not (mode & stat.S_IWUSR):
            os.chmod(filePath, mode | stat.S_IWUSR)

        # let the file itself know its place
        subprocess.call(('install_name_tool', '-id',
                         '@executable_path/' + fileName, filePath))

        # find the references: call otool -L on the file
        otool = subprocess.Popen(('otool', '-L', filePath),
                                 stdout=subprocess.PIPE)
        references = otool.stdout.readlines()[1:]

        for reference in references:

            # find the actual referenced file name
            referencedFile = reference.decode().strip().split()[0]

            if (referencedFile.startswith('@loader_path/.dylibs/')
                or referencedFile.startswith('@loader_path/.')
                or referencedFile.startswith('@rpath')):
                # this file is likely an hdf5 file
                print "Found hdf5 file {} referencing {}".format(filePath, referencedFile)

            elif referencedFile.startswith('@'):
                # the referencedFile is already a relative path
                continue

            path, name = os.path.split(referencedFile)

            # some referenced files have not previously been copied to the
            # executable directory - the assumption is that you don't need
            # to copy anything fro /usr or /System, just from folders like
            # /opt this fix should probably be elsewhere though
            # if (name not in files and not path.startswith('/usr') and not
            #        path.startswith('/System')):
            #    print(referencedFile)
            #    self.copy_file(referencedFile,
            #                   os.path.join(self.binDir, name))
            #    files.append(name)

            # see if we provide the referenced file;
            # if so, change the reference
            if name in files:
                newReference = '@executable_path/' + name
                print "Fixing", filePath, "from", referencedFile, "to", newReference
                subprocess.call(('install_name_tool', '-change',
                                 referencedFile, newReference, filePath))


def fix_library_references(built_bin_dir=None):
    setRelativeReferencePaths(binDir=built_bin_dir)
    # dumb hack just to fix PIL._imaging.so
    try:
        PIL_Imaging_file = glob.glob(os.path.join(built_bin_dir, 'PIL._imaging.*'))[0]
        # find the references: call otool -L on the file
        otool = subprocess.Popen(('otool', '-L', PIL_Imaging_file),
                                 stdout=subprocess.PIPE)
        references = otool.stdout.readlines()[1:]
        for reference in references:
            # find the actual referenced file name
            referencedFile = reference.decode().strip().split()[0]
            if referencedFile.startswith('@'):
                # the referencedFile is already a relative path
                # continue
                # for this file we need to correct these
                pass
            path, name = os.path.split(referencedFile)
            if (not path.startswith('/usr')):
                newReference = '@executable_path/' + name
                subprocess.call(('install_name_tool', '-change',
                                 referencedFile, newReference, PIL_Imaging_file))
    except Exception as e:
        print "Could not file PIL._imaging file in", built_bin_dir


builddir = os.getcwd()

if os.path.isdir(os.path.join(builddir, "build/FreqPy-0.00001.app/Contents/MacOS")):
    # fix files in the app directory as well
    fix_library_references(built_bin_dir=os.path.join(builddir, "build/FreqPy-0.00001.app/Contents/MacOS"))

built_bin_dir = glob.glob(os.path.join(builddir, 'build/exe*'))[0]
if os.path.isdir(built_bin_dir):
    # fix files in the app directory as well
    fix_library_references(built_bin_dir=built_bin_dir)

