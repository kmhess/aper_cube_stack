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
cdelt3 = 0
all_nan = True

if (len(taskids) == len(processed_ids)) or args.force:
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))
    if len(processed_ids) > 1:
        # Find field center for a given pointing (work even with potentially legacy files):
        master_list = ascii.read(package_dir + '/data/apertif_v12.21apr06.txt', format='fixed_width')
        pointing, entry = next(([s, m] for s, m in zip(master_list['name'], master_list) if field[1:] in s), None)
        # Assign the bary center based on the center of the pointing itself (always beam 0 saved in files)
        barycent_pos = SkyCoord(ra=entry['ra'], dec=entry['dec'], unit='deg')
        time = [None] * len(processed_ids)
        data_all = []
        for b in beams:
            # Calculate barycentric shifts for each cube
            delta_chan = []
            new_crval3 = []
            for ii, t in enumerate(taskids):
                # Get info from the header
                filename = str(t) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
                try:
                    os.system('cp {} {}_test.fits'.format(filename, filename[:-5]))
                    hdu = fits.open(filename)
                    # hdu = fits.open(filename[:-5] + '_test.fits', mode=update)
                    print("\tFound cube {} for {} beam {:02}".format(c, t, b))
                    all_nan = False
                except FileNotFoundError:
                    print('\t{} has no cube {} for beam {:02}'.format(t, c, b))
                    new_crval3.append(np.nan)
                    delta_chan.append(np.nan)
                    continue
                header = hdu[0].header
                if time[ii] == None:
                    time[ii] = Time(header['DATE-OBS'])
                if cdelt3 == 0.0:
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

            # Skip beam if there is no contributing data from any taskids
            if all_nan == True:
                print("\tNo data for beam {:02} in any taskid covering field {}.".format(b, field))
                continue

            # Create subcubes
            n_chans = int(header['NAXIS3'] - np.abs(np.nanmin(delta_chan)-np.nanmax(delta_chan)))
            skip_chans = delta_chan - np.nanmin(delta_chan)
            nan_cube = np.empty((int(n_chans), header['NAXIS2'], header['NAXIS2'])) * np.nan

            tic = testtime.perf_counter()
            data_all = []
            rms = []
            # Calculate noise, weight for each cube
            for j, t in zip(skip_chans, taskids):
                try:
                    data_all.append(fits.getdata(str(t) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) +
                                                 '.fits')[int(j):int(j+n_chans), :, :])
                    rms.append(da.nanstd(da.array(data_all)[-1, :, :, :], axis=(1, 2)).compute())
                    # Write noise, weight as a function of channel to a text file for each cube.
                    ascii.write([list(range(int(j), int(j) + n_chans)), np.array(rms)[-1, :]],
                                str(t) + '/B0' + str(b).zfill(2) + '/noise.txt',
                                names=['chan', 'noise'], overwrite=True)
                except FileNotFoundError:
                    # Fill with nan values if the beam cube doesn't exist.
                    data_all.append(nan_cube)
                    rms.append(da.std(da.array(data_all)[-1, :, :, :], axis=(1, 2)).compute())

            data_all = np.array(data_all)
            rms = np.array(rms)
            weights = 1/rms**2

            # Weight cube, combine, and normalize
            dataweight = data_all.transpose()*weights.transpose()
            num = da.nansum(dataweight.transpose(), axis=0)
            denom = da.nansum(weights, axis=0)
            combo_cube = (num.transpose()/denom).compute().transpose()
            toc = testtime.perf_counter()
            print(f"Do median: {toc - tic:0.4f} seconds")

            # Make a new directory if it doesn't already exist:
            if not os.path.isdir(field):
                os.system('mkdir {}'.format(field))

            # Write new cube & save in a logical place
            header['CRVAL3'] = np.array(new_crval3)[skip_chans == 0.][0]
            header['NAXIS3'] = n_chans
            header['SPECSYS'] = 'BARYCENT'
            header['CTYPE3'] = 'FREQ'
            hdu_new = fits.PrimaryHDU(data=combo_cube, header=header)
            tic1 = testtime.perf_counter()
            hdu_new.writeto(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits', overwrite=True)
            toc1 = testtime.perf_counter()
            print(f"Do write: {toc1 - tic1:0.4f} seconds")

    elif len(processed_ids) == 1:
        print("\tOnly one taskid.  Doing barycent correction & writing to a folder with the field naming scheme.")
        # Find field center for a given pointing (work even with potentially legacy files):
        master_list = ascii.read(package_dir + '/data/apertif_v12.21apr06.txt', format='fixed_width')
        pointing, entry = next(([s, m] for s, m in zip(master_list['name'], master_list) if field[1:] in s), None)
        barycent_pos = SkyCoord(ra=entry['ra'], dec=entry['dec'], unit='deg')
        for b in beams:
            # Get info from the header
            filename = str(processed_ids[0]) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
            try:
                hdu = fits.open(filename)
                print("\tFound cube {} for {} beam {:02}".format(c, t, b))
            except FileNotFoundError:
                print('\t{} has no cube {} for beam {:02}'.format(t, c, b))
                continue
            header = hdu[0].header
            if b == beams[0]:
                time = Time(header['DATE-OBS'])
            # Build in a check if the header is already in BARYCENT!
            if hdu[0].header['SPECSYS'] == 'BARYCENT':
                print("\tCube already in barycentric reference frame, continuing.")
                # Make a new directory if it doesn't already exist:
                if not os.path.isdir(field):
                    os.system('mkdir {}'.format(field))
                    os.system('cp ' + filename + ' ' + field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) +
                              '_image.fits')
            else:
                print("\tAssuming cube is in topecentric reference frame; transforming to barycentric")
                # Calculate barycentric correction
                spec_coord = SpectralCoord(header['CRVAL3'], unit='Hz', observer=westerbork().get_itrs(obstime=time),
                                           target=barycent_pos)
                bary_spec_coord = spec_coord.with_observer_stationary_relative_to('icrs')
                header['CRVAL3'] = bary_spec_coord.value
                header['SPECSYS'] = 'BARYCENT'
                header['CTYPE3'] = 'FREQ'

                if not os.path.isdir(field):
                    os.system('mkdir {}'.format(field))
                hdu_new = fits.PrimaryHDU(data=hdu[0].data, header=header)
                tic1 = testtime.perf_counter()
                hdu_new.writeto(field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits', overwrite=True)
                toc1 = testtime.perf_counter()
                print(f"Do write: {toc1 - tic1:0.4f} seconds")
            hdu.close()

    else:
        print("\tNo data processed yet for this field: {}.".format(field))

else:
    print("\tNot all the data sets have been processed for this field: {}.".format(field))
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))
