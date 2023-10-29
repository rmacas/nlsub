# Example script used to download data
import numpy as np
from gwpy.timeseries import TimeSeries

# output name suffix
suffix = 'event_4096s'

gps = 1264316116  # event int(gps) time

# start/end times for channels
start_gps = gps - 2048
end_gps = gps + 2048

# sampling rate
frate = 512

# ASD time
# found by visually inspecting timeseries, see "asd_inspecting.ipynb"
gps_asd = gps - 450
asd_win = 256
start_gps_asd = gps_asd - asd_win
end_gps_asd = gps_asd + asd_win

# frame and channels
frame = 'L1_HOFT_C01'
channels = ['L1:DCS-CALIB_STRAIN_CLEAN_C01', 'L1:LSC-POP_A_RF45_I_ERR_DQ',
            'L1:LSC-POP_A_RF45_Q_ERR_DQ', 'L1:LSC-POP_A_RF9_I_ERR_DQ']

for channel in channels:

    # get tseries and resample it
    tseries = TimeSeries.get(channel=channel, start=start_gps, end=end_gps, frametype=frame).resample(frate)

    # estimate the ASD
    tseries_quiet = TimeSeries.get(channel=channel, start=start_gps_asd,
                                   end=end_gps_asd,
                                   frametype=frame).resample(frate)
    tseries_asd = tseries_quiet.asd(4, 2, method='median')

    # inflate the ASD below 10Hz
    cutoff_idx = np.argmin(np.abs(tseries_asd.frequencies.value - 10))
    max_val = np.max(tseries_asd)
    tseries_asd[0:cutoff_idx] = max_val

    # whiten the tseries using the estimated ASD
    tseries_whitened = tseries.whiten(asd=tseries_asd)

    # convert tseries to float 32 np array
    tseries_array = np.array([tseries_whitened.times, tseries_whitened]).T

    # write final tseries to a file
    np.save(f'{channel[3:]}_{frate}Hz_{suffix}', tseries_array)
