# Import default Python libraries
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import gc

from astropy.io import fits, ascii
import numpy as np
import dask.array as da

import time as testtime

from modules.functions import get_taskids


# Doesn't seem to catch the errors from dask. :/
np.seterr(divide='ignore', invalid='ignore')

###################################################################

parser = ArgumentParser(description="Stack psf's from cubes that have been combined.",
                        formatter_class=RawTextHelpFormatter)

# default=['191231012', '210718041']
parser.add_argument('-f', '--field', default='S1345+5324', required=True,
                    help='Required: Specify the field to stack. If taskid given, assemble taskids for \n'
                         ' that field. No default.')

parser.add_argument('-b', '--beams', default='34',
                    help='Specify a range (0-39) or list (3,5,7,11) of beams on which to do source finding\n'
                         ' (default: %(default)s).')

parser.add_argument('-c', '--cubes', default='2',
                    help='Specify the cubes on which to do source finding (default: %(default)s).')

parser.add_argument('-force', '--force',
                    help='Force creation of barycent corrected cubes even if not all observations have been processed.',
                    action='store_true')

###################################################################

# Parse the arguments above
args = parser.parse_args()

field = args.field
cubes = [int(c) for c in args.cubes.split(',')]
if '-' in args.beams:
    b_range = args.beams.split('-')
    beams = np.array(range(int(b_range[1]) - int(b_range[0]) + 1)) + int(b_range[0])
else:
    beams = [int(b) for b in args.beams.split(',')]
c = cubes[0]

package_dir = os.path.dirname(__file__)

# Find out what taskids contribute to cube
taskids, processed_ids = get_taskids(field)

if (len(taskids) == len(processed_ids)) or args.force:
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))

    # Iterate over beams:
    for b in beams:
        data_all = []
        rms = []
        if len(processed_ids) > 1:
            for t in processed_ids:
                # Check if noise.txt file is in place for the processed_ids:
                if os.path.isfile(str(t) + "/B0" + str(b).zfill(2) + "/noise.txt"):
                    print("\tFound noise.txt file for taskid {}, beam {}, cube {}.".format(t, b, c))
                    noise_values = ascii.read(str(t) + "/B0" + str(b).zfill(2) + "/noise.txt")
                    skip_chan = noise_values['chan'][0]
                    n_chans = len(noise_values)
                    filename = str(t) + '/B0' + str(b).zfill(2) + '/HI_beam_cube' + str(c) + '.fits'
                    try:
                        data_all.append(fits.getdata(filename)[int(skip_chan):int(skip_chan + n_chans), :, :])
                        rms.append(noise_values['noise'])
                    except FileNotFoundError:
                        print("\tERROR: psf file for taskid {}, beam {}, cube {} hasn't been downloaded.".format(t, b, c))
                        print("\t\tExiting. Download appropriate psf cube from ALTA!")
                        exit()
                else:
                    # Fill with nan values if the psf cube doesn't exist.
                    print("\tWARNING: noise values for taskid {}, beam {}, cube {} doesn't exist.".format(t, b, c))
                    if os.path.isfile(str(t) + "/B0" + str(b).zfill(2) + "/HI_image_cube" + str(c) + ".fits"):
                        print("\tWARNING: image cube DOES exist, not included in mosaic??")

            # Prepare data arrays
            data_all = np.array(data_all)
            rms = np.array(rms)
            weights = 1 / rms ** 2

            # Weight cube, combine, and normalize data cubes
            tic = testtime.perf_counter()
            # dataweight = data_all.transpose() * weights.transpose()
            num = da.nansum(data_all.transpose() * weights.transpose(), axis=-1)
            del data_all
            gc.collect()
            denom = da.nansum(weights, axis=0)
            combo_cube = (num / denom).compute()
            del denom
            del num
            gc.collect()
            # End counter
            toc = testtime.perf_counter()
            print(f"Do median: {toc - tic:0.4f} seconds")

            # Get a template header
            print("\tGet template header from {}".format(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits'))
            header = fits.getheader(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits')
            header['CRPIX1'] = 661.0
            header['CRPIX2'] = 661.0

            hdu_new = fits.PrimaryHDU(data=combo_cube.transpose(), header=header)
            hdu_new.writeto(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_psf.fits', overwrite=True)
            print("\tFinished field {} beam {}.".format(field, b))

        elif len(processed_ids) == 1:
            print("\tOnly one taskid.  Writing to a folder with the field naming scheme & barycent image header.")
            filename = str(processed_ids[0]) + '/B0' + str(b).zfill(2) + '/HI_beam_cube' + str(c) + '.fits'
            combo_cube = fits.getdata(filename)

            # Get a template header
            print("\tGet template header from {}".format(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits'))
            header = fits.getheader(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits')
            header['CRPIX1'] = 661.0
            header['CRPIX2'] = 661.0

            hdu_new = fits.PrimaryHDU(data=combo_cube, header=header)
            hdu_new.writeto(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_psf.fits', overwrite=True)
            print("\tFinished field {} beam {}.".format(field, b))

        else:
            print("\tNo data processed yet for this field: {}.".format(field))

else:
    print("\tNot all the data sets have been processed for this field: {}.".format(field))
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))

