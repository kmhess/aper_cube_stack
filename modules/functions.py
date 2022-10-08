import os

from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.table import Table, unique
import astropy.units as u
import numpy as np

from modules.telescope_params import westerbork


dir = os.path.dirname(__file__)

HI_restfreq = 1420405751.77 * u.Hz
optical_HI = u.doppler_optical(HI_restfreq)


###################################################################


def get_taskids(field):
    """
    :param field:
    :return:
    """

    observations = Table.read(dir + '/../data/obscensus.csv', comment='#')
    processed = Table.read(dir + '/../data/pointings.dat', format='ascii')

    taskids = observations[observations['name'] == field]['taskID']
    processed_ids = unique(processed[processed['field'] == field], keys='0taskID')['0taskID']

    return taskids, processed_ids


def topo2bary_corr(beam_pos, time):
    """
    :param loc:
    :param time:
    :return:
    """
    barycorr = beam_pos.radial_velocity_correction(obstime=time, location=westerbork())
    print("Velocity: ", barycorr)
    barycorr_Hz = barycorr.to(u.Hz, equivalencies=optical_HI)
    print("Converted to Hz: ", barycorr_Hz)
    delta_crval3 = HI_restfreq - barycorr_Hz
    print("Change in Hz: ", delta_crval3)

    return delta_crval3


def get_common_spectrum(barycent_pos, taskids, beams, c):
    """
    :param barycent_pos:
    :param taskids:
    :param c:
    :param beams:
    :return:
    """
    print("\tAttempting to figure out common spectrum settings.")
    # beams = range(0, 40)
    # Calculate barycentric shifts for each taskid (must be same for all beams)
    delta_chan = []
    new_crval3 = []
    time = [None] * len(taskids)

    for ii, t in enumerate(taskids):
        b = 0
        # For each taskid, try beams until appropriate content is populated. (Assume all beams have same bary corr.)
        while (time[ii] == None) and (b < len(beams)):

            # Get info from the header
            filename = str(t) + '/B0' + str(beams[b]).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
            try:
                header = fits.getheader(filename)
                print("\tFound beam {:02} cube {} for taskid {}".format(beams[b], c, t))
            except FileNotFoundError:
                b += 1
                continue

            # Build in a check if the header is already in BARYCENT!
            # THIS ASSUMES ALL BEAMS ARE IN BARYCENT FOR A TASKID!
            if header['SPECSYS'] == 'BARYCENT':
                print("\tCube already in barycentric reference frame, continuing.")
                delta_crval3 = 0 * u.Hz
            else:
                print("\tAssuming cube is in topecentric reference frame; transforming to barycentric")
                # Calculate barycentric correction
                time[ii] = Time(header['DATE-OBS'])
                cdelt3 = header['CDELT3']
                spec_coord = SpectralCoord(header['CRVAL3'], unit='Hz',
                                           observer=westerbork().get_itrs(obstime=time[ii]),
                                           target=barycent_pos)
                bary_spec_coord = spec_coord.with_observer_stationary_relative_to('icrs')
                delta_crval3 = spec_coord - bary_spec_coord

            # Calculate int(channel shift)
            new_crval3.append(bary_spec_coord.value)
            delta_chan.append(np.round(np.array(delta_crval3.value) / cdelt3))

        if b == len(beams):
            print("\tNo valid data for taskid {} !".format(t))
            new_crval3.append(np.nan)
            delta_chan.append(np.nan)

    # Assumes these are the same for all original beams and at least one taskid, beam, cube combo will work.
    naxis2 = header['NAXIS2']
    naxis3 = header['NAXIS3']

    return delta_chan, new_crval3, naxis2, naxis3


# test = get_taskids('M1403+5324')
# print(test)

from astropy.time import Time
sc = SkyCoord(ra=2.04884572000e2, dec=5.44064750000e1, unit='deg', frame='fk5')
t = Time('2019-12-31T06:52:55.6')
test = topo2bary_corr(sc, t)
t2 = Time('2021-07-18T17:20:30.600')
test2 = topo2bary_corr(sc, t2)
#
# beam_pos = SkyCoord(ra=2.04884572000e2, dec=5.44064750000e1, unit='deg', frame='fk5')
# obs = westerbork().get_itrs(obstime=Time('2019-12-31T06:52:55.6'))
# print(obs)
# print(beam_pos)
# spec_coord = SpectralCoord(1373645484.116311, unit='Hz', observer=westerbork().get_itrs(obstime=Time('2019-12-31T06:52:55.6')), target=beam_pos)
# print(spec_coord)
# #
# print(spec_coord.with_observer_stationary_relative_to('icrs'))
# print((spec_coord.with_observer_stationary_relative_to('icrs')-spec_coord)/1420e6*3e5)