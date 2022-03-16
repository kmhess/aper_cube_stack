# Import default Python libraries
from argparse import ArgumentParser, RawTextHelpFormatter
import os

from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits, ascii
from astropy.time import Time
import astropy.units as u
import numpy as np
import dask.array as da

import time as testtime

from modules.functions import get_taskids, topo2bary_corr
from modules.telescope_params import westerbork


###################################################################

parser = ArgumentParser(description="Stack cubes for source finding.",
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

# taskids = ['190913045','191207034','200302074']
if len(taskids) == len(processed_ids):
    print("\tTASKIDS: {}".format(taskids))
    if len(taskids) > 1:
        # Find field center for a given pointing (work even with potentially legacy files):
        master_list = ascii.read(package_dir + '/data/apertif_v12.21apr06.txt', format='fixed_width')
        pointing, entry = next(([s, m] for s, m in zip(master_list['name'], master_list) if field[1:] in s), None)
        barycent_pos = SkyCoord(ra=entry['ra'], dec=entry['dec'], unit='deg')
        time = []
        for b in beams:
            # Calculate barycentric shifts for each cube
            delta_chan = []
            new_crval3 = []
            for ii, t in enumerate(taskids):
                # Get info from the header
                filename = str(t) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
                hdu = fits.open(filename)
                header = hdu[0].header
                if b == beams[0]:
                    time.append(Time(header['DATE-OBS']))
                if t == taskids[0]:
                    beam_pos = SkyCoord(ra=header['CRVAL1'], dec=header['CRVAL2'], unit='deg', frame='fk5')
                    cdelt3 = header['CDELT3']

                # Build in a check if the header is already in BARYCENT!
                if hdu[0].header['SPECSYS'] == 'BARYCENT':
                    print("\tCube already in barycentric reference frame, continuing.")
                    delta_crval3 = 0 * u.Hz
                else:
                    print("\tAssuming cube is in topecentric reference frame; transforming to barycentric")
                    # Calculate barycentric correction
                    spec_coord = SpectralCoord(header['CRVAL3'], unit='Hz',
                                               observer=westerbork().get_itrs(obstime=time[ii]),
                                               target=barycent_pos)
                    bary_spec_coord = spec_coord.with_observer_stationary_relative_to('icrs')
                    delta_crval3 = spec_coord - bary_spec_coord
                    # hdu[0].header['CRVAL3'] = bary_spec_coord.value
                    # hdu[0].header['SPECSYS'] = 'BARYCENT'
                    # hdu[0].header['CTYPE3'] = 'FREQ'
                    # hdu.flush()

                hdu.close()
                # Calculate int(channel shift)
                new_crval3.append(bary_spec_coord.value)
                delta_chan.append(np.round(np.array(delta_crval3.value) / cdelt3))

            # Create subcubes
            n_chans = int(header['NAXIS3'] - np.abs(np.min(delta_chan)-np.max(delta_chan)))
            skip_chans = delta_chan - np.min(delta_chan)
            combo_cube = np.empty((int(n_chans), header['NAXIS2'], header['NAXIS2']))

            tic = testtime.perf_counter()
            data_all = []
            rms = []
            # Calculate noise, weight for each cube
            for j, t in zip(skip_chans, taskids):
                data_all.append(fits.getdata(str(t) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) +
                                             '.fits')[int(j):int(j+n_chans), :, :])
                rms.append(da.nanstd(da.array(data_all)[-1, :, :, :], axis=(1, 2)).compute())

                # Write noise, weight as a function of channel to a text file for each cube.
                ascii.write([list(range(int(j), int(j)+n_chans)), np.array(rms)[-1, :]],
                            str(t) + '/B0' + str(b).zfill(2) + '/noise.txt',
                            names=['chan', 'noise'], overwrite=True)

            data_all = np.array(data_all)
            rms = np.array(rms)
            weights = 1/rms**2

            # Weight cube, combine, and normalize
            dataweight = data_all.transpose()*weights.transpose()
            num = da.nansum(dataweight.transpose(), axis=0)
            denom = da.nansum(weights, axis=0)
            combo_cube2 = (num.transpose()/denom).compute().transpose()
            toc = testtime.perf_counter()
            print(f"Do median: {toc - tic:0.4f} seconds")

            # Make a new directory if it doesn't already exist:
            if not os.path.isdir(field):
                os.system('mkdir {}'.format(field))
            # if not os.path.isdir(field + '/B0' + str(b).zfill(2)):
            #     os.system('mkdir ' + field + '/B0' + str(b).zfill(2))

            # Write new cube & save in a logical place
            header['CRVAL3'] = np.array(new_crval3)[skip_chans == 0.][0]
            header['NAXIS3'] = n_chans
            hdu = fits.PrimaryHDU(data=combo_cube2, header=header)
            tic1 = testtime.perf_counter()
            hdu.writeto(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits', overwrite=True)
            toc1 = testtime.perf_counter()
            print(f"Do write: {toc1 - tic1:0.4f} seconds")
    else:
        print("\tOnly one taskid.  Copying to a folder with the field naming scheme.")
        # Make a new directory if it doesn't already exist:
        if not os.path.isdir(field):
            os.system('mkdir {}'.format(field))
        # if not os.path.isdir(field + '/B0' + str(b).zfill(2)):
        #     os.system('mkdir ' + field + '/B0' + str(b).zfill(2))
        os.system('cp ' + filename + ' ' + field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits')

else:
    print("\tNot all the data sets have been processed for this field: {}.".format(field))
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))
