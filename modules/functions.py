from astropy.table import Table, unique
import astropy.units as u

from modules.telescope_params import westerbork


HI_restfreq = 1420405751.77 * u.Hz
optical_HI = u.doppler_optical(HI_restfreq)


###################################################################


def get_taskids(field):
    """
    :param field:
    :return:
    """

    observations = Table.read('../data/obscensus.csv', comment='#')
    processed = Table.read('../data/pointings.dat', format='ascii')

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

# test = get_taskids('S1102+5815')
# print(test)

from astropy.coordinates import SkyCoord, SpectralCoord
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