import os

from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.table import Table, unique
from astropy.time import Time
import astropy.units as u
from itertools import product
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

    observations['name'][observations['name']=='S2214+3130'] = 'M2214+3130'
    processed['field'][processed['field']=='S2214+3130'] = 'M2214+3130'

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


def get_common_spectrum(barycent_pos, taskids, beams, c, d):
    """
    :param barycent_pos:
    :param taskids:
    :param beams:
    :param c:
    :param d:
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
            filename = d + '/' + str(t) + '/B0' + str(beams[b]).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
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

    # Calculate optimal shift (to be added by Barbara)
    
            
    # Assumes these are the same for all original beams and at least one taskid, beam, cube combo will work.
    naxis2 = header['NAXIS2']
    naxis3 = header['NAXIS3']

    return delta_chan, new_crval3, naxis2, naxis3


#Barbara: function to estimate the true spectral resolution of the co-added cube as a function of the channel number of the reference spectral axis
def spec_res_n(n, n_chan, del_f_mid, cdelt3s, cdelt3_fin):
    """
    :param n:
    :param n_chan:
    :param del_f_mid:
    :param cdelt3s:
    :param cdelt3_fin:
    """
    n = n - int(n_chan/2) 
    new_sig = np.std(del_f_mid + n * (cdelt3s - cdelt3_fin))

    return np.sqrt(cdelt3_fin**2 + (2 * np.sqrt(2 * np.log(2)) * new_sig)**2) 


def get_common_spectrum_barbara(barycent_pos, taskids, beams, c, d):
    """
    :param barycent_pos:
    :param taskids:
    :param beams:
    :param c:
    :param d:
    :return:
    """
    print("\tAttempting to figure out common spectrum settings.")
    # beams = range(0, 40)
    # Calculate barycentric shifts for each taskid (must be same for all beams)
#     delta_chan = []
#     new_crval3 = []
    time = [None] * len(taskids)
    
    bary_mid_freqs = []
    bary_cdelt3 = []
    for ii, t in enumerate(taskids):
        b = 0
        # For each taskid, try beams until appropriate content is populated. (Assume all beams have same bary corr.)
        while (time[ii] == None) and (b < len(beams)):

            # Get info from the header
            filename = d + '/' + str(t) + '/B0' + str(beams[b]).zfill(2) + '/HI_image_cube' + str(c) + '.fits'
            
            try:
                header = fits.getheader(filename)
                print("\tFound beam {:02} cube {} for taskid {}".format(beams[b], c, t))
            except FileNotFoundError:
                b += 1
                continue
            
            cdelt3 = header['CDELT3']
            N_chan = header['NAXIS3']
            mid_freq = header['CRVAL3'] + int(N_chan/2) * cdelt3
            # Build in a check if the header is already in BARYCENT!
            # THIS ASSUMES ALL BEAMS ARE IN BARYCENT FOR A TASKID!
            if header['SPECSYS'] == 'BARYCENT':
                print("\tCube already in barycentric reference frame, continuing.")#                 delta_crval3 = 0 * u.Hz
                bary_mid_freqs.append(mid_freq)
                bary_cdelt3.append(cdelt3)
            else:
                print("\tAssuming cube is in topocentric reference frame; transforming to barycentric")
                # Calculate barycentric correction
                time[ii] = Time(header['DATE-OBS'])
                spec_coord = SpectralCoord(mid_freq, unit='Hz',
                                           observer=westerbork().get_itrs(obstime=time[ii]),
                                           target=barycent_pos)
                bary_mid_freqs.append(spec_coord.with_observer_stationary_relative_to('icrs').value)
                bary_cdelt3.append(cdelt3*bary_mid_freqs[-1]/mid_freq)

        if b == len(beams):
            print("\tNo valid data for taskid {} !".format(t))
            bary_mid_freqs.append(np.nan)
            bary_cdelt3.append(np.nan)

    # Calculate optimal shift 
    bary_mid_freqs = np.array(bary_mid_freqs)
    bary_cdelt3 = np.array(bary_cdelt3)
    
    for i,freq in enumerate(bary_mid_freqs):
        if not np.isnan(freq):
            bary_ref_freq = freq
            ref_index = i
            break

    N_obs = len(taskids)
    delta_mid_freq = bary_ref_freq - bary_mid_freqs
#   # We divide each frequency shift by its own cdelt3, important for correct alignment
    delta_chan_true = delta_mid_freq/bary_cdelt3  

    indexes = np.array(list(product([0,1], repeat=N_obs)))
    delta_chan_round_j = np.zeros(N_obs, dtype=float)

    possibilities = np.zeros((N_obs,2), dtype=float)
    for j in range(N_obs):
        possibilities[j,:] = np.array([np.floor(delta_chan_true[j]), np.ceil(delta_chan_true[j])])

    # Check if the reference frequency corresponds to one of the observations', i.e. rounding up and down gives the same number.
    # Then remove all possibilities that assume a different rounding for this observation, i.e. all possibilities that have value 1 for this observation.
    same_round = np.where([possibilities[i,0]==possibilities[i,1] for i in range(N_obs)])[0]
    if same_round.size > 0:
        where_ind = []
        for i in same_round:
            where_ind.append(np.where(indexes[:,i] == 1))

        where_ind = np.ndarray.flatten(np.array(where_ind))
        ind_need = np.array([i for i in range(2**N_obs) if i not in where_ind])
    else:
        ind_need = np.array([i for i in range(2**N_obs)])

    sigma_dif_all = np.zeros(len(ind_need), dtype=float)
    mean_dif_all = np.zeros(len(ind_need), dtype=float)
    dif_freq_mid = np.zeros(len(ind_need), dtype=object)
    delta_chan_round = np.zeros((len(ind_need),N_obs), dtype=float)

    for j in range(len(ind_need)):
        for k in range(N_obs):
            delta_chan_round_j[k] = possibilities[k, int(indexes[ind_need[j]][k])]

        delta_chan_round[j,:] = delta_chan_round_j
        difference = delta_chan_true - delta_chan_round_j
        difference_freq = difference * bary_cdelt3
        dif_freq_mid[j] = difference_freq
        sigma_dif_all[j] = np.nanstd(difference_freq)      #approx for std coming from the shifts in frequency for case j
        mean_dif_all[j] = np.nanmean(difference_freq)

    index_optimal_case = np.where(sigma_dif_all == np.min(sigma_dif_all))[0][0]
    delta_chan = delta_chan_round[index_optimal_case]

    bary_fin_cdelt3 = np.nanmean(bary_cdelt3)
    spec_res_mid_axis = np.sqrt(bary_fin_cdelt3**2 + (2 * np.sqrt(2*np.log(2)) * sigma_dif_all[index_optimal_case])**2)
    new_crval3_ref_obs = bary_ref_freq + mean_dif_all[index_optimal_case] - int(N_chan/2)*bary_fin_cdelt3

    # Assumes these are the same for all original beams and at least one taskid, beam, cube combo will work.
    naxis2 = header['NAXIS2']
    naxis3 = header['NAXIS3']
    
    #find how spectral resolution changes along the spectral axes, approximate with a linear function and parameters of slope (a) and y-intercept (b)
    
    #first find the first common channel that will be channel 0 in the co-added cube (skip_chans_ref_obs), and the final number of channels of the co-added cube (n_chans)
    skip_chans_ref_obs = delta_chan[ref_index] - np.nanmin(delta_chan)
    n_chans = int(naxis3 - np.abs(np.nanmin(delta_chan) - np.nanmax(delta_chan)))

    #find the true spectral resolution in the first and last channels of the co-added cube
    y1 = spec_res_n(skip_chans_ref_obs, N_chan, dif_freq_mid[index_optimal_case], bary_cdelt3, bary_fin_cdelt3)
    y2 = spec_res_n(skip_chans_ref_obs + n_chans,N_chan, dif_freq_mid[index_optimal_case], bary_cdelt3, bary_fin_cdelt3)
    
    #calculate the parameters of the linear approximation to put in the header of the co-added cube
    a_lin_par = (y2 - y1) / (n_chans - 1)
    b_lin_par = y1

    return delta_chan, ref_index, new_crval3_ref_obs, spec_res_mid_axis, bary_fin_cdelt3, naxis2, naxis3, a_lin_par, b_lin_par
