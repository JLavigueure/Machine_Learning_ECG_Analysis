""""
Load the pickled dataframe containing the metadata and the raw ECG data. Calculate the ECG features
and append those to the dataframe. Save the new dataframe as a pickle file. 
Note: Errors are expected during ecg processing due to the nature of the data. These errors are
caught and NaN values are returned for the features in these cases.

Usage: 
    python signal_processing.py <sampling_rate>
""" 

import pandas as pd
import numpy as np
import sys              # for command line arguments
import pickle           # for saving data
import neurokit2 as nk  # for ECG processing
import progressbar        # for progress bar
import warnings         # for ignoring warnings

def calculate_ecg_features(data: pd.DataFrame, sampling_rate: int = 100, log: bool = True) -> pd.DataFrame:
    """
    Master function which calls all other ecg processing functions. Given the dataframe 
    containing the raw ECG data, return a new dataframe with the new features appended.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing metadata and ECG data.
    sampling_rate : int
        Sampling rate of the ECG signal.
    log : bool
        Whether to log errors or not.
        

    Returns
    -------
    df : DataFrame
        Original dataframe with new features appended.
    """

    # Define the columns for the new features
    columns = [         
        # Heart rate
        "heart_rate_avg", "heart_rate_sd",
        
        # Durations 
        "qrs_duration_avg", "qrs_duration_sd",
        "p_duration_avg", "p_duration_sd",
        "t_duration_avg", "t_duration_sd",
        "pr_duration_avg", "pr_duration_sd",
        "rt_duration_avg", "rt_duration_sd",
        
        # Voltages (12 leads)
        "qrs_voltage_avg_1", "qrs_voltage_avg_2", "qrs_voltage_avg_3", "qrs_voltage_avg_4", "qrs_voltage_avg_5", "qrs_voltage_avg_6", "qrs_voltage_avg_7", "qrs_voltage_avg_8", "qrs_voltage_avg_9", "qrs_voltage_avg_10", "qrs_voltage_avg_11", "qrs_voltage_avg_12",
        "qrs_voltage_sd_1", "qrs_voltage_sd_2", "qrs_voltage_sd_3", "qrs_voltage_sd_4", "qrs_voltage_sd_5", "qrs_voltage_sd_6", "qrs_voltage_sd_7", "qrs_voltage_sd_8", "qrs_voltage_sd_9", "qrs_voltage_sd_10", "qrs_voltage_sd_11", "qrs_voltage_sd_12",
        "p_voltage_avg_1", "p_voltage_avg_2", "p_voltage_avg_3", "p_voltage_avg_4", "p_voltage_avg_5", "p_voltage_avg_6", "p_voltage_avg_7", "p_voltage_avg_8", "p_voltage_avg_9", "p_voltage_avg_10", "p_voltage_avg_11", "p_voltage_avg_12",
        "p_voltage_sd_1", "p_voltage_sd_2", "p_voltage_sd_3", "p_voltage_sd_4", "p_voltage_sd_5", "p_voltage_sd_6", "p_voltage_sd_7", "p_voltage_sd_8", "p_voltage_sd_9", "p_voltage_sd_10", "p_voltage_sd_11", "p_voltage_sd_12",
        "t_voltage_avg_1", "t_voltage_avg_2", "t_voltage_avg_3", "t_voltage_avg_4", "t_voltage_avg_5", "t_voltage_avg_6", "t_voltage_avg_7", "t_voltage_avg_8", "t_voltage_avg_9", "t_voltage_avg_10", "t_voltage_avg_11", "t_voltage_avg_12",
        "t_voltage_sd_1", "t_voltage_sd_2", "t_voltage_sd_3", "t_voltage_sd_4", "t_voltage_sd_5", "t_voltage_sd_6", "t_voltage_sd_7", "t_voltage_sd_8", "t_voltage_sd_9", "t_voltage_sd_10", "t_voltage_sd_11", "t_voltage_sd_12",
        "total_voltage_1", "total_voltage_2", "total_voltage_3", "total_voltage_4", "total_voltage_5", "total_voltage_6", "total_voltage_7", "total_voltage_8", "total_voltage_9", "total_voltage_10", "total_voltage_11", "total_voltage_12",

        # ST segment (12 leads)
        "st_height_1", "st_height_2", "st_height_3", "st_height_4", "st_height_5", "st_height_6", "st_height_7", "st_height_8", "st_height_9", "st_height_10", "st_height_11", "st_height_12",
        "st_slope_1", "st_slope_2", "st_slope_3", "st_slope_4", "st_slope_5", "st_slope_6", "st_slope_7", "st_slope_8", "st_slope_9", "st_slope_10", "st_slope_11", "st_slope_12",
        "st_curve_1", "st_curve_2", "st_curve_3", "st_curve_4", "st_curve_5", "st_curve_6", "st_curve_7", "st_curve_8", "st_curve_9", "st_curve_10", "st_curve_11", "st_curve_12"
    ]

    # Create a dataframe (filled with NaN) with new features
    df = pd.DataFrame(np.nan, index=data.index, columns=columns)

    # Number of leads
    lead_count = 12
    
    # Start progress bar
    bar = progressbar.ProgressBar(widgets=[
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.ETA(),
        ], maxval=len(data)+1).start()

    # For each row in the dataframe
    for index, row in data.iterrows():
        
        # Initialize feature arrays
        feature_count = 22
        heart_rate_avg, heart_rate_sd, \
        qrs_duration_avg, qrs_duration_sd, \
        qrs_voltage_avg, qrs_voltage_sd, \
        p_duration_avg, p_duration_sd, \
        p_voltage_avg, p_voltage_sd, \
        t_duration_avg, t_duration_sd, \
        t_voltage_avg, t_voltage_sd, \
        pr_duration_avg, pr_duration_sd, \
        rt_duration_avg, rt_duration_sd, \
        st_height, st_slope, st_curve, \
        total_voltage = [np.full((lead_count,), np.nan) for _ in range(feature_count)]
        
        # Process each lead
        for lead in range(lead_count):
            signal = row['ecg'][lead]
            try: 
                # see https://neuropsychology.github.io/NeuroKit/functions/ecg.html
                ecg_signals, ecg_info = nk.ecg_process(signal, sampling_rate=sampling_rate, method='hamilton2002')

                # calculate the sum of the entire ecg
                total_voltage[lead] = calculate_total_voltage(ecg_signals, sampling_rate=sampling_rate)
                
                # calculate average heart rate and the heart rate's standard deviation
                heart_rate_avg[lead], heart_rate_sd[lead] =\
                calculate_heart_rate(ecg_signals, ecg_info, sampling_rate=sampling_rate)
                
                # calcualte average QRS duration and voltage
                qrs_duration_avg[lead], qrs_duration_sd[lead], qrs_voltage_avg[lead], qrs_voltage_sd[lead] =\
                calculate_qrs_duration_and_voltage(ecg_signals, sampling_rate=sampling_rate)
    
                # calculate average P wave duration and peak voltage
                p_duration_avg[lead], p_duration_sd[lead], p_voltage_avg[lead], p_voltage_sd[lead] =\
                calculate_p_duration_and_voltage(ecg_signals, sampling_rate=sampling_rate)
                
                # calculate average T wave duration and peak voltage
                t_duration_avg[lead], t_duration_sd[lead], t_voltage_avg[lead], t_voltage_sd[lead] =\
                calculate_t_duration_and_voltage(ecg_signals, sampling_rate=sampling_rate)
    
                # calculate PR duration and 
                pr_duration_avg[lead], pr_duration_sd[lead] =\
                calculate_pr_duration(ecg_signals, sampling_rate=sampling_rate)
    
                # calculate PR duration and 
                rt_duration_avg[lead], rt_duration_sd[lead] =\
                calculate_rt_duration(ecg_signals, sampling_rate=sampling_rate)
                
                # calculate the ST segment height relative to the ecg baseline (set by PR segment)
                st_height[lead], st_slope[lead], st_curve[lead] =\
                calculate_st_height_and_poly(ecg_signals, sampling_rate=sampling_rate)
                
            except Exception as e:
                if log:
                    # hide progress bar to log errors
                    sys.stdout.write('\r\033[K') # Clear line

                    print(f'Error @ index {index}, lead {lead}: {e}', file=sys.stderr)

        # Calculate the median of the heart rate
        df.loc[index,'heart_rate_avg'] = np.nanmedian(heart_rate_avg)
        df.loc[index,'heart_rate_sd'] = np.nanmedian(heart_rate_sd)

        # Calculate the median of the durations
        df.loc[index, 'qrs_duration_avg'] = np.nanmedian(qrs_duration_avg)
        df.loc[index, 'qrs_duration_sd'] = np.nanmedian(qrs_duration_sd)
        df.loc[index, 'p_duration_avg'] = np.nanmedian(p_duration_avg)
        df.loc[index, 'p_duration_sd'] = np.nanmedian(p_duration_sd)
        df.loc[index, 't_duration_avg'] = np.nanmedian(t_duration_avg)
        df.loc[index, 't_duration_sd'] = np.nanmedian(t_duration_sd)
        df.loc[index, 'pr_duration_avg'] = np.nanmedian(pr_duration_avg)
        df.loc[index, 'pr_duration_sd'] = np.nanmedian(pr_duration_sd)
        df.loc[index, 'rt_duration_avg'] = np.nanmedian(rt_duration_avg)
        df.loc[index, 'rt_duration_sd'] = np.nanmedian(rt_duration_sd)

        # Store voltage and ST segment features for each lead
        for lead in range(lead_count):
            
            # Voltages
            df.loc[index, f"qrs_voltage_avg_{lead+1}"] = qrs_voltage_avg[lead] 
            df.loc[index, f"qrs_voltage_sd_{lead+1}"] = qrs_voltage_sd[lead]
            df.loc[index, f"p_voltage_avg_{lead+1}"] = p_voltage_avg[lead]
            df.loc[index, f"p_voltage_sd_{lead+1}"] = p_voltage_sd[lead]
            df.loc[index, f"t_voltage_avg_{lead+1}"] = t_voltage_avg[lead]
            df.loc[index, f"t_voltage_sd_{lead+1}"] = t_voltage_sd[lead]
            df.loc[index, f"total_voltage_{lead+1}"] = total_voltage[lead]
            
            # ST segment features
            df.loc[index, f"st_height_{lead+1}"] = st_height[lead]
            df.loc[index, f"st_slope_{lead+1}"] = st_slope[lead]
            df.loc[index, f"st_curve_{lead+1}"] = st_curve[lead]
        
        # Update progress bar
        bar.update(index+1)

    # Finish progress bar
    bar.finish()

    # Append features to original dataframe
    return pd.concat([data, df], axis=1)
            
# Heart Rate, Heart Rate Standard Deviation
def calculate_heart_rate(ecg_signals: pd.DataFrame, ecg_info: dict, sampling_rate: int = 100) -> tuple[float, float]:
    """
    Calculate the heart rate from the ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    ecg_info : dict
        Dictionary containing the ECG information from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal
        
    Returns
    -------
    heart_rate : float
        Average heart rate from the ECG signal.
    heart_rate_sd : float
        Standard deviation of the heart rate.
    """
    # get the R-peaks from the processed ECG signal
    r_peaks = ecg_info['ECG_R_Peaks']

    if len(r_peaks) == 0:
        return 0, 0
    
    # calculate the RR intervals (time between consecutive R-peaks)
    rr_intervals = np.diff(r_peaks) / sampling_rate  # Convert sample indices to seconds
    
    # calculate the heart rate from the RR intervals (HR = 60 / RR_interval in seconds)
    heart_rate = 60 / rr_intervals
    
    # calculate average and standard deviation of the heart rate
    return (np.nanmean(heart_rate), np.nanstd(heart_rate))
    
# QRS Duration, QRS Voltage
def calculate_qrs_duration_and_voltage(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float, float, float]:
    """
    Calculate the QRS duration and voltage from the ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal
        
    Returns
    -------
    avg_qrs_duration : float
        Average QRS duration from the ECG signal.
    std_qrs_duration : float
        Standard deviation of the QRS duration.
    avg_qrs_voltage : float
        Average QRS voltage from the ECG signal.
    std_qrs_voltage : float
        Standard deviation of the QRS voltage.
    """
    #extract Q and S peak locations
    q_peaks = np.where(ecg_signals["ECG_Q_Peaks"] == 1)[0]
    s_peaks = np.where(ecg_signals["ECG_S_Peaks"] == 1)[0]

    #ensure we have corresponding Q and S peaks
    if len(q_peaks) == 0 or len(s_peaks) == 0:
        return np.nan  # Return NaN if no peaks detected

    #compute QRS durations for each detected Q-S pair
    qrs_durations = []
    qrs_voltages = []
    for q in q_peaks:
        #find the closest S peak after the Q peak
        s_after_q = s_peaks[s_peaks > q]
        if len(s_after_q) > 0:
            #Calculate duration
            qrs_duration = (s_after_q[0] - q) / sampling_rate * 1000  # Convert to ms
            qrs_durations.append(qrs_duration)
            
            #Calculate Voltage
            qrs_signal = ecg_signals["ECG_Clean"][q:s_after_q[0]]
            qrs_voltage = np.mean(np.abs(qrs_signal))  # Absolute value to get magnitude
            qrs_voltages.append(qrs_voltage)

    #calculate average QRS duration and voltage
    avg_qrs_duration = np.nanmean(qrs_durations) if qrs_durations else np.nan
    avg_qrs_voltage = np.nanmean(qrs_voltages) if qrs_voltages else np.nan

    # Calculate standard deviation of QRS duration and voltage
    std_qrs_duration = np.nanstd(qrs_durations) if len(qrs_durations) > 1 else np.nan
    std_qrs_voltage = np.nanstd(qrs_voltages) if len(qrs_voltages) > 1 else np.nan

    return (avg_qrs_duration, std_qrs_duration, avg_qrs_voltage, std_qrs_voltage)

# P-wave duration, P-wave peak voltage
def calculate_p_duration_and_voltage(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float, float, float]:
    """
    Calculate the P-wave duration and peak voltage from the ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal
    
    Returns
    -------
    p_wave_duration_avg : float
        Average P-wave duration from the ECG signal.
    p_wave_duration_sd : float
        Standard deviation of the P-wave duration.
    p_peak_voltage_avg : float
        Average P-wave peak voltage from the ECG signal.
    p_peak_voltage_sd : float
        Standard deviation of the P-wave peak voltage.
    """
    # get pertinent peaks
    p_peaks = np.where(ecg_signals["ECG_P_Peaks"] == 1)[0]
    p_onsets = np.where(ecg_signals["ECG_P_Onsets"] == 1)[0]
    p_offsets = np.where(ecg_signals["ECG_P_Offsets"] == 1)[0]

    # calculate p-wave duration (from onset to offset)
    p_wave_durations = []
    for p_onset in p_onsets:
        # find the next p offset after the current p onset
        p_offset_after_onset = p_offsets[p_offsets > p_onset]
        if len(p_offset_after_onset) > 0:
            p_offset = p_offset_after_onset[0]
            p_wave_duration = (p_offset - p_onset) / sampling_rate * 1000  # convert to ms
            p_wave_durations.append(p_wave_duration)

    # get peak voltage values at the P-peaks
    p_peak_voltages = ecg_signals["ECG_Clean"][p_peaks]
    
    # calculate average p wave duration and voltage
    p_wave_duration_avg = np.nanmean(p_wave_durations) if p_wave_durations else np.nan
    p_peak_voltage_avg = np.nanmean(np.abs(p_peak_voltages)) if len(p_peak_voltages) > 0 else np.nan
    
    # calculate p wave duration standard deviation and p wave voltage standward deviation
    p_wave_duration_sd = np.nanstd(p_wave_durations) if p_wave_durations else np.nan
    p_peak_voltage_sd = np.nanstd(np.abs(p_peak_voltages)) if len(p_peak_voltages) > 0 else np.nan

    return (p_wave_duration_avg, p_wave_duration_sd, p_peak_voltage_avg, p_peak_voltage_sd)

# T-wave duration, T-wave peak voltage
def calculate_t_duration_and_voltage(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float, float, float]:
    """'
    Calculate the T-wave duration and peak voltage from the ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal
    
    Returns
    -------
    t_wave_duration_avg : float
        Average T-wave duration from the ECG signal.
    t_wave_duration_sd : float
        Standard deviation of the T-wave duration.
    t_peak_voltage_avg : float
        Average T-wave peak voltage from the ECG signal.
    t_peak_voltage_sd : float
        Standard deviation of the T-wave peak voltage.
    """
    # Get important peaks for T-wave
    t_peaks = np.where(ecg_signals["ECG_T_Peaks"] == 1)[0]
    t_onsets = np.where(ecg_signals["ECG_T_Onsets"] == 1)[0]
    t_offsets = np.where(ecg_signals["ECG_T_Offsets"] == 1)[0]
    
    # Calculate T-wave duration (from onset to offset)
    t_wave_durations = []
    for t_onset in t_onsets:
        # Find the next T offset after the current T onset
        t_offset_after_onset = t_offsets[t_offsets > t_onset]
        if len(t_offset_after_onset) > 0:
            t_offset = t_offset_after_onset[0]
            t_wave_duration = (t_offset - t_onset) / sampling_rate * 1000  # Convert to ms
            t_wave_durations.append(t_wave_duration)

    
    # get peak voltage values at the T-peaks
    t_peak_voltages = ecg_signals["ECG_Clean"][t_peaks]
    
    # calculate average p wave duration and voltage
    t_wave_duration_avg = np.nanmean(t_wave_durations) if t_wave_durations else np.nan
    t_peak_voltage_avg = np.nanmean(np.abs(t_peak_voltages)) if len(t_peak_voltages) > 0 else np.nan
    
    # calculate p wave duration standard deviation and p wave voltage standward deviation
    t_wave_duration_sd = np.nanstd(t_wave_durations) if t_wave_durations else np.nan
    t_peak_voltage_sd = np.nanstd(np.abs(t_peak_voltages)) if len(t_peak_voltages) > 0 else np.nan

    return (t_wave_duration_avg, t_wave_duration_sd, t_peak_voltage_avg, t_peak_voltage_sd)
    
# PR Duration
def calculate_pr_duration(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float]:
    """    
    Calculate the PR duration from the ECG signal.

    Parameters:
    -----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal.

    Returns:
    --------
    pr_duration_avg : float
        Average PR duration from the ECG signal.
    pr_duration_sd : float
        Standard deviation of the PR duration.
    """
    # Get P-wave onsets and R-peaks
    p_onsets = np.where(ecg_signals["ECG_P_Onsets"] == 1)[0]
    r_peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Compute PR durations (time from P onset to the next R peak)
    pr_durations = []
    for p_onset in p_onsets:
        # Find the closest R peak after the P onset
        r_after_p = r_peaks[r_peaks > p_onset]
        if len(r_after_p) > 0:
            pr_duration = (r_after_p[0] - p_onset) / sampling_rate * 1000  # Convert to ms
            pr_durations.append(pr_duration)

    # Calculate average PR duration and standard deviation
    pr_duration_avg = np.nanmean(pr_durations) if pr_durations else np.nan
    pr_duration_sd = np.nanstd(pr_durations) if pr_durations else np.nan

    return pr_duration_avg, pr_duration_sd

# RT duration
def calculate_rt_duration(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float]:
    """
    Calculate the RT duration from the ECG signal.

    Parameters:
    -----------
    ecg_signals : 
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal.
    
    Returns:
    --------
    rt_duration_avg : float
        Average RT duration from the ECG signal.
    rt_duration_sd : float
        Standard deviation of the RT duration.
    """
    #get r-peaks and t-offsets
    r_peaks = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]
    t_offsets = np.where(ecg_signals["ECG_T_Offsets"] == 1)[0]

    #compute rt durations (time from r peak to the next t offset)
    rt_durations = []
    for r_peak in r_peaks:
        #find the closest t offset after the r peak
        t_after_r = t_offsets[t_offsets > r_peak]
        if len(t_after_r) > 0:
            rt_duration = (t_after_r[0] - r_peak) / sampling_rate * 1000  #convert to ms
            rt_durations.append(rt_duration)

    #calculate average rt duration and standard deviation
    rt_duration_avg = np.nanmean(rt_durations) if rt_durations else np.nan
    rt_duration_sd = np.nanstd(rt_durations) if rt_durations else np.nan

    return rt_duration_avg, rt_duration_sd

# ST Height, ST polynomial Coefficient 1, ST polynomial Coefficient 2
def calculate_st_height_and_poly(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> tuple[float, float, float]:
    """
    Calculate the ST height and polynomial coefficients of the ST segment from the ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal

    Returns
    -------
    avg_st_height : float
        Average ST height from the ECG signal.
    avg_st_coefficient_1 : float
        Average first degree coefficient of the ST segment.
    avg_st_coefficient_2 : float
        Average second degree coefficient of the ST segment.
    """
    # Get S peaks and T onsets
    s_peaks = np.where(ecg_signals["ECG_S_Peaks"] == 1)[0]
    t_onsets = np.where(ecg_signals["ECG_T_Onsets"] == 1)[0]

    st_segment_heights = []
    first_degree_coefficient = []
    second_degree_coefficient = []

    for s in s_peaks:
        # Find the closest T onset after the S peak
        t_after_s = t_onsets[t_onsets > s]
        if len(t_after_s) > 0:
            # Extract the ECG signal between S peak and T onset
            st_segment = ecg_signals["ECG_Clean"][s:t_after_s[0]]
            
            # Calculate the average height of the ST segment
            avg_st_height = np.nanmean(st_segment)
            st_segment_heights.append(avg_st_height)
            
            # Calculate slope and curve of ST segment            
            mask = ~np.isnan(st_segment)
            st_segment = st_segment[mask]
            polynomial_degrees = 2
            if len(st_segment) > polynomial_degrees:
                poly = np.polyfit(np.arange(len(st_segment)), st_segment, polynomial_degrees)
                first_degree_coefficient.append(poly[1])
                second_degree_coefficient.append(poly[0])

    # Calculate average ST height and average ST angle
    avg_st_height = np.nanmean(st_segment_heights) if st_segment_heights else np.nan
    avg_st_coefficient_1 = np.nanmean(first_degree_coefficient) if first_degree_coefficient else np.nan
    avg_st_coefficient_2 = np.nanmean(second_degree_coefficient) if second_degree_coefficient else np.nan

    return avg_st_height, avg_st_coefficient_1, avg_st_coefficient_2

# Total Sum
def calculate_total_voltage(ecg_signals: pd.DataFrame, sampling_rate: int = 100) -> float:
    """
    Calculate the sum of the entire ECG signal.

    Parameters
    ----------
    ecg_signals : pd.DataFrame
        Dictionary containing the processed ECG signals from neurokit2.
    sampling_rate : int
        Sampling rate of the ECG signal

    Returns
    -------
    total_voltage : float
        Sum of the entire ECG signal.
    """
    return np.nansum(ecg_signals['ECG_Clean'])


def main():
    # Check if correct number of arguments are passed
    if len(sys.argv) != 2:
        print('Usage: python signal_processing.py <sampling_rate>')
        return

    # Path to sampling rate to use
    sampling_rate = int(sys.argv[1])

    # Check if sampling rate is valid
    if sampling_rate not in [100, 500]:
        print('Invalid sampling rate.')
        return

    # Load the pickled dataframe
    with open('data/raw_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Silence warnings
    warnings.filterwarnings("ignore")
    
    # Calculate the ECG features
    data = calculate_ecg_features(data)

    # Restore default behavior for all warnings
    warnings.filterwarnings("default")
    
    # Save the dataframe as a pickle file
    with open('data/processed_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()