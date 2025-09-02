# Import default Python libraries
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import gc

from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits, ascii
from astropy.time import Time
import numpy as np
import dask.array as da

import time as testtime

from modules.functions import get_taskids, get_common_spectrum
from modules.telescope_params import westerbork


#Barbara: talk to Kelley how she wants to implement this keyword, maybe only for cube = 3? 
# KH: for now let's 'hard-code' to cube 3's only.  I don't think we will have the resources to come back to the other cubes...
# use_Barbara_func = True

# Doesn't seem to catch the errors from dask. :/
np.seterr(divide='ignore', invalid='ignore')

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

parser.add_argument('-d', '--directory', default='.', required=False,
                    help='Specify the directory where taskid/field folders live containing the data (default: %(default)s).')

parser.add_argument('-force', '--force',
                    help='Force creation of barycent corrected cubes even if not all observations have been processed.',
                    action='store_true')

###################################################################

# Parse the arguments above
args = parser.parse_args()

d = args.directory
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
    print("\tWill combine data for beams: {}".format(beams))
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))

    if len(processed_ids) > 1:
        # Find field center for a given pointing (including files where survey prefix may have changed):
        master_list = ascii.read(package_dir + '/data/apertif_v12.21apr06.txt', format='fixed_width')
        pointing, entry = next(([s, m] for s, m in zip(master_list['name'], master_list) if field[1:] in s), None)

        # Assign the bary center based on the center of the pointing itself (always beam 0 saved in files)
        barycent_pos = SkyCoord(ra=entry['ra'], dec=entry['dec'], unit='deg')

        # Find common spectrum across all beams for a field.
        
        # if use_Barbara_func:
        if c == 3:
            delta_chan, ref_index, new_crval3_ref_obs, spec_res_mid_axis, bary_fin_cdelt3, naxis2, naxis3, a_lin_param, b_lin_param = get_common_spectrum_barbara(barycent_pos, processed_ids, beams, c, d)
        else:
            delta_chan, new_crval3, naxis2, naxis3 = get_common_spectrum(barycent_pos, processed_ids, beams, c, d)
            

        print("\tCreate barycentric corrected cubes for the specified beams.")
        for b in beams:
            # Create subcubes
            n_chans = int(naxis3 - np.abs(np.nanmin(delta_chan)-np.nanmax(delta_chan)))
            skip_chans = delta_chan - np.nanmin(delta_chan)
            nan_cube = np.empty((int(n_chans), naxis2, naxis2)) * np.nan

            # Start counter
            tic = testtime.perf_counter()
            data_all = []
            rms = []
            all_nan = True
            # Calculate noise, weight for each cube
            for j, t in zip(skip_chans, taskids):
                try:
                    filename = d + '/' + str(t) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
                    data_all.append(fits.getdata(filename)[int(j):int(j+n_chans), :, :])
                    rms.append(da.nanstd(da.array(data_all)[-1, :, :, :], axis=(1, 2)).compute())
                    # Write noise, weight as a function of channel to a text file for each cube.
                    ascii.write([list(range(int(j), int(j) + n_chans)), np.array(rms)[-1, :]],
                                filename[:19] + 'noise_cube' + str(c) + '.txt',
                                names=['chan', 'noise'], overwrite=True)
                    all_nan = False
                    last_file = filename
                except FileNotFoundError:
                    # Fill with nan values if the beam cube doesn't exist.
                    data_all.append(nan_cube)
                    rms.append(da.std(da.array(data_all)[-1, :, :, :], axis=(1, 2)).compute())
                    
            # Skip beam if there is no contributing data from any taskids
            if all_nan:
                print("\tNo data for beam {:02} in any taskid covering field {}.".format(b, field))
                continue

            # Get a template header
            header = fits.getheader(last_file)
            # Prepare data arrays
            data_all = np.array(data_all)
            rms = np.array(rms)
            weights = 1/rms**2

            # Weight cube, combine, and normalize data cubes
            num = da.nansum(data_all.transpose() * weights.transpose(), axis=-1)
            del data_all
            gc.collect()
            denom = da.nansum(weights, axis=0)
            combo_cube = (num/denom).compute().transpose()
            # End counter
            toc = testtime.perf_counter()
            print(f"Do median: {toc - tic:0.4f} seconds")

            # Make a new directory if it doesn't already exist:
            if not os.path.isdir(d + '/' + field):
                os.system('mkdir {}/{}'.format(d,field))

            # Write new cube & save in a logical place
            # if use_Barbara_func:
            if c == 3:
                header['CRVAL3'] = new_crval3_ref_obs + skip_chans[ref_index]*bary_fin_cdelt3
                header['CDELT3'] = bary_fin_cdelt3
                #parameters to calculate spectral resolution (in Hz) as a function of channel number as: SPRESA*nr_chan + SPRESB
                header['SPRESA'] = a_lin_param 
                header['SPRESB'] = b_lin_param
            else:
                header['CRVAL3'] = np.array(new_crval3)[skip_chans == 0.][0]
                #Barbara: add CDELT3 change due to the change to the barycent frame?
            
            header['NAXIS3'] = n_chans
            header['SPECSYS'] = 'BARYCENT'
            header['CTYPE3'] = 'FREQ'
            hdu_new = fits.PrimaryHDU(data=combo_cube, header=header)
            tic1 = testtime.perf_counter()
            hdu_new.writeto(d + '/' + field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits', overwrite=True)
            toc1 = testtime.perf_counter()
            print("\tFinished field {} beam {}.".format(field, b))
            print(f"Do write: {toc1 - tic1:0.4f} seconds")

    elif len(processed_ids) == 1:

        print("\tOnly one taskid.  Doing barycent correction & writing to a folder with the field naming scheme.")
        # Find field center for a given pointing (work even with potentially legacy files):
        master_list = ascii.read(package_dir + '/data/apertif_v12.21apr06.txt', format='fixed_width')
        pointing, entry = next(([s, m] for s, m in zip(master_list['name'], master_list) if field[1:] in s), None)

        # Assign the bary center based on the center of the pointing itself (always beam 0 saved in files)
        barycent_pos = SkyCoord(ra=entry['ra'], dec=entry['dec'], unit='deg')

        time = None
        for b in beams:
            # Get info from the header
            filename = str(d + '/' + processed_ids[0]) + '/B0' + str(b).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
            try:
                hdu = fits.open(filename)
                print("\tFound cube {} for {} beam {:02}".format(c, processed_ids, b))
            except FileNotFoundError:
                print("\tNo data for beam {:02} in taskid {} covering field {}.".format(b, processed_ids, field))
                continue
            header = hdu[0].header
            if b == beams[0] or (not time):
                time = Time(header['DATE-OBS'])
            # Build in a check if the header is already in BARYCENT!
            if hdu[0].header['SPECSYS'] == 'BARYCENT':
                print("\tCube already in barycentric reference frame, continuing.")
                # Make a new directory if it doesn't already exist:
                if not os.path.isdir(d + '/' + field):
                    os.system('mkdir {}/{}'.format(d, field))
                    os.system('cp ' + filename + ' ' + d + '/' + field + '/HI_B0' + str(b).zfill(2) + 
                              '_cube' + str(c) + '_image.fits')
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
                    os.system('mkdir {}/{}'.format(d, field))
                hdu_new = fits.PrimaryHDU(data=hdu[0].data, header=header)
                tic1 = testtime.perf_counter()
                hdu_new.writeto(d + '/' + field + '/HI_B0' + str(b).zfill(2) + '_cube' + str(c) + '_image.fits', overwrite=True)
                toc1 = testtime.perf_counter()
                print(f"Do write: {toc1 - tic1:0.4f} seconds")
            hdu.close()

    else:
        print("\tNo data processed yet for this field: {}.".format(field))

else:
    print("\tNot all the data sets have been processed for this field: {}.".format(field))
    print("\tTASKIDS: {}".format(taskids))
    print("\tPROCESSED TASKIDS: {}".format(processed_ids))
