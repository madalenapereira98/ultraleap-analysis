"""
Calculates features per tap and features per block.
Saves features in .json files.
"""

import numpy as np
import movement_calc.helpfunctions as hp
import matplotlib.pyplot as plt
import os
import json
import importlib
from scipy.ndimage import uniform_filter1d

importlib.reload(hp)

def get_repo_path_in_notebook():
    """
    Finds path of repo from Notebook.
    Start running this once to correctly find
    other modules/functions
    """
    path = os.getcwd()
    repo_name = 'ultraleap_analysis'

    while path[-len(repo_name):] != 'ultraleap_analysis':

        path = os.path.dirname(path)

    return path

repo_path = get_repo_path_in_notebook()


def calculate_feat(folder, dist, block: str, sub: str, cond: str, cam: str, task: str, side: str, to_plot = False, to_save_plot = False, to_save = False):

    """
    Function that uses other functions defined 
    in this .py file to calculate features
    per tap and features per block.

    Input:
        - folder (str), dist (DataFrame), 
    Output:
        -
    """

    if task == 'ft':
        dist_col = 'dist'
    elif task == 'oc':
        dist_col = 'mean'
    elif task == 'ps':
        dist_col = 'ang'


    # get minima and maxima
    idx_min, idx_max = hp.find_min_max(np.array(dist[dist_col]), task=task)

    # get time & dist lists per tap and tap duration
    ls_time, ls_dist, tap_duration = tap_times(dist, dist_col, idx_min)

    # get time & dist list per opening and closing movement for speed calculation
    ls_opening_time, ls_opening_dist, ls_closing_time, ls_closing_dist = tap_times_for_open_close(dist, dist_col, idx_min, idx_max)

    # spe_over_taps = speed_over_time_tap(ls_time, ls_dist)
    # # spe_over_openings = speed_over_time_opening(ls_opening_time, ls_opening_dist, idx_min, idx_max)
    # # spe_over_closings = speed_over_time_closing(ls_closing_time, ls_closing_dist, idx_min, idx_max)
    spe_over_openings = speed_over_time_opening(ls_opening_time, ls_opening_dist)
    spe_over_closings = speed_over_time_closing(ls_closing_time, ls_closing_dist)

    # extract features per tap
    ft_dict_tap = get_feat_tap(ls_dist, spe_over_openings, spe_over_closings, tap_duration, idx_max)
    ft_dict_block = get_feat_block(ft_dict_tap, dist,
                                   block, idx_max, sub, cond, cam, task, side
                                   )

    if to_save:
        # save features per tap in json files
        ft_tap_path = os.path.join(repo_path, 'features', folder, sub, task, cond, 'features_tap')
        if not os.path.exists(ft_tap_path):
            os.makedirs(ft_tap_path)

        ft_dict_open = open(os.path.join(ft_tap_path, f'{block}_{sub}_{cond}_{cam}_{task}_{side}.json'), 'w')
        json.dump(ft_dict_tap, ft_dict_open)
        ft_dict_open.close()

        # save features per block in json files
        if all(key in ft_dict_block for key in ['jerkiness', 'entropy']):
            ft_block_path = os.path.join(repo_path, 'features',folder, sub, task, cond,'features_block')
        else:
            ft_block_path = os.path.join(repo_path, 'other_features', folder, sub, task, cond,'features_block')
        if not os.path.exists(ft_block_path):
            os.makedirs(ft_block_path)

        ft_dict_block_open = open(os.path.join(ft_block_path, f'{block}_{sub}_{cond}_{cam}_{task}_{side}.json'), 'w')
        json.dump(ft_dict_block, ft_dict_block_open)
        ft_dict_open.close()

    # plot max_min
    if to_plot:
            plot_max_min(folder, dist, dist_col, block, sub, cond, cam, task, side, idx_max, idx_min, to_save = to_save_plot)

    return

def plot_max_min(folder, dist, dist_col, block, sub, cond, cam, task, side, idx_max, idx_min, to_save = False):

    """
    Function that plots the block with the
    minima & maxima calculated with the
    find_min_max function.

    Input:
        - dist (Dataframe) with the euclidean distances,
        maxima indexes, minima indexes, camera position,
        distance column (e.g. for  oc: 'mean', for ft: 'dist')

    Output:
        - figures with minima and maxima.
    """
    if (len(idx_min) != 0) and (len(idx_max) != 0):

        # x = np.linspace(0,1,len(dist))
        print('plot')
        x = np.array(dist['time'])

        fig = plt.figure(figsize=(8,6))
        plt.plot(x, dist[dist_col], color='grey')
        plt.plot(x[idx_max],
                    np.array(dist[dist_col])[idx_max],
                    "o", label="max", color='blue')
        plt.plot(x[idx_min],
                    np.array(dist[dist_col])[idx_min],
                    "o", label="max", color='red')
        plt.xlabel('time (s)')
        print('before labelling')
        if np.logical_or(task == 'ft', task == 'oc'):
            plt.ylabel('distance (m)')
        elif task == 'ps':
            plt.ylabel('angle (degree)')

        if to_save:
            fig_path = os.path.join(repo_path,
                                    'figures',
                                    'distances_min_max',
                                    folder,
                                    task,
                                    f'{cam}_max_min'
                                    )
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            fig.savefig(os.path.join(fig_path,
                                        f'{block}_{sub}_{cond}_{cam}_{task}_{side}'),
                                        dpi = 300, facecolor = 'w',
                                        )
            print('after saving')
        plt.close()
    return


def tap_times(block_df, dist_col, min_idx):

    """ Function that gives all time and
        dist values in between two minima
        -> values per tap

        Input:
            - block dataframe, column name of the dist_time df,
            min indices - numpy.ndarray

        Output:
            - list of lists for each tap containing tap_start_time,
            tap_end_time, tap_duration
    """

    print('in tap_times')

    ls_time = []
    ls_dist = []
    ls_tap_duration = []

    for idx, i_min in enumerate(min_idx[:-1]):

        t_start = block_df.iloc[i_min]['time']
        t_end = block_df.iloc[min_idx[idx+1]]['time']
        tap_duration = t_end-t_start

        df = block_df[np.logical_and(block_df['time']>=t_start, block_df['time']<=t_end)]

        ls_time.append(df['time'].tolist())
        ls_dist.append(df[dist_col].tolist())

        ls_tap_duration.append(tap_duration)
    
    return ls_time, ls_dist, ls_tap_duration


def tap_times_for_open_close(block_df, dist_col, min_idx, max_idx):
    """
    Function that gives all time and dist values 
    between one minimum and the next maximum (opening movement)
    and between one maximum and the next minimum (closing movement).

    Input:
        - block dataframe, column name of the dist_time df,
          min indices - numpy.ndarray, max indices - numpy.ndarray

    Output:
        - list of lists for each opening movement containing 
          tap_start_time, tap_end_time, tap_duration, dist_values, time_values,
        - list of lists for each closing movement containing 
          tap_start_time, tap_end_time, tap_duration, dist_values, time_values,
    """

    ls_opening_time = []
    ls_opening_dist = []
    # ls_opening_duration = []

    ls_closing_time = []
    ls_closing_dist = []
    # ls_closing_duration = []

    for i in range(len(min_idx) - 1):
        t_start = block_df.iloc[min_idx[i]]['time']
        t_end = block_df.iloc[max_idx[i]]['time']
        # tap_duration = t_end - t_start

        opening_df = block_df[np.logical_and(block_df['time'] >= t_start, block_df['time'] <= t_end)]
        closing_df = block_df[np.logical_and(block_df['time'] >= t_end, block_df['time'] <= block_df.iloc[min_idx[i+1]]['time'])]

        ls_opening_time.append(opening_df['time'].tolist())
        ls_opening_dist.append(opening_df[dist_col].tolist())
        # ls_opening_duration.append(tap_duration)

        ls_closing_time.append(closing_df['time'].tolist())
        ls_closing_dist.append(closing_df[dist_col].tolist())
        # ls_closing_duration.append(closing_df.iloc[-1]['time'] - t_end)

    # return ls_opening_time, ls_opening_dist, ls_opening_duration, ls_closing_time, ls_closing_dist, ls_closing_duration
    return ls_opening_time, ls_opening_dist, ls_closing_time, ls_closing_dist


# Calculate speed between each min-max pair for opening movements
def speed_over_time_opening(ls_opening_time, ls_opening_dist):
    speed_opening = []

    for time_list, dist_list in zip(ls_opening_time, ls_opening_dist):
        if len(dist_list) >= 2:
            dif_dist = abs(dist_list[-1] - dist_list[0])
            dif_time = abs(time_list[-1] - time_list[0])
            speed_single = dif_dist / dif_time
            speed_opening.append(speed_single)

    return speed_opening


# Calculate speed between each max-min pair for closing movements
def speed_over_time_closing(ls_closing_time, ls_closing_dist):
    speed_closing = []

    for time_list, dist_list in zip(ls_closing_time, ls_closing_dist):
        if len(dist_list) >= 2:
            dif_dist = abs(dist_list[-1] - dist_list[0])
            dif_time = abs(time_list[-1] - time_list[0])
            speed_single = dif_dist / dif_time
            speed_closing.append(speed_single)

    return speed_closing



def speed_over_time_tap(ls_time, ls_dist):
    """
    Function that calculates the speed over time.

    Input:
        - ls_time (list) containing time lists
        for each tap, ls_dist (list) containing
        distance lists for each tap.
    Output:
        - list of lists containing the speed over
        time per tap.
    """

    speed = []

    for j in range(0, len(ls_dist)-1):
        dif_dist = abs(np.diff(ls_dist[j]))
        dif_time = np.diff(ls_time[j])
        speed_single = dif_dist/dif_time

        speed.append(speed_single)

    return speed


def get_feat_tap(ls_dist, spe_over_openings, spe_over_closings, ls_tap_duration, idx_max):
    print('in get_feat_tap')
    total_ft_dict = {}
    ft_names = [
        'num_events',
        'max_dist',
        'open_vel',
        'close_vel',
        'tap_dur',
        'rms',
        ]

    for ft in ft_names:
        total_ft_dict[ft] = []
    
    total_ft_dict['num_events'].append(len(ls_dist))

    for d, s_o, s_c, t in zip(ls_dist, spe_over_openings, spe_over_closings, ls_tap_duration):
      
        total_ft_dict['max_dist'].append(np.nanmax(d))
        total_ft_dict['open_vel'].append(s_o)
        total_ft_dict['close_vel'].append(s_c)
        total_ft_dict['tap_dur'].append(t)
        total_ft_dict['rms'].append(np.sqrt(np.mean(d)**2))

    return total_ft_dict


def get_feat_block(feat_dict, dist_df, block, idx_max, sub, cond, cam, task, side):
    print('in get_feat_block')
    feat_names = ['num_events',
    'mean_max_dist', 'sd_max_dist', 'coef_var_max_dist', 'slope_max_dist', 'decr_max_dist',

    'max_open_vel', 'mean_open_vel', 'sd_open_vel', 'coef_var_open_vel',
    'slope_open_vel',

    'max_close_vel', 'mean_close_vel', 'sd_close_vel', 'coef_var_close_vel',
    'slope_close_vel',

    'mean_tap_dur', 'sd_tap_dur', 'coef_var_tap_dur', 'slope_tap_dur',
    'mean_rms', 'sd_rms', 'slope_rms', 'sum_rms',
    
    'jerkiness',
    'entropy'
    ]

    dict_feat_block = {}

    for feat in feat_names:
        dict_feat_block[feat] = []

    if feat_dict['num_events'][0] > 2:
        try:
            dict_feat_block['num_events'] = feat_dict['num_events']

            # distance
            dict_feat_block['mean_max_dist'].append(np.nanmean(feat_dict['max_dist']))
            dict_feat_block['sd_max_dist'].append(np.nanstd(feat_dict['max_dist']))
            dict_feat_block['coef_var_max_dist'].append(np.nanstd(feat_dict['max_dist'])/np.nanmean(feat_dict['max_dist']))
            # print(np.polyfit(np.arange(len(feat_dict['max_dist'])), feat_dict['max_dist'], 1)[0])
            dict_feat_block['slope_max_dist'].append(np.polyfit(np.arange(len(feat_dict['max_dist'])), feat_dict['max_dist'], 1)[0])
            # get decrement feature
            decr = decrement(dist_df,task,idx_max)
            dict_feat_block['decr_max_dist'].append(decr)

            # opening speed
            dict_feat_block['max_open_vel'].append(np.nanmax(feat_dict['open_vel']))
            dict_feat_block['mean_open_vel'].append(np.nanmean(feat_dict['open_vel']))
            dict_feat_block['sd_open_vel'].append(np.nanstd(feat_dict['open_vel']))
            dict_feat_block['coef_var_open_vel'].append(np.nanstd(feat_dict['open_vel'])/np.nanmean(feat_dict['open_vel']))
            dict_feat_block['slope_open_vel'].append(np.polyfit(np.arange(len(feat_dict['open_vel'])), feat_dict['open_vel'], 1)[0])
         
            # closing speed
            dict_feat_block['max_close_vel'].append(np.nanmax(feat_dict['close_vel']))
            dict_feat_block['mean_close_vel'].append(np.nanmean(feat_dict['close_vel']))
            dict_feat_block['sd_close_vel'].append(np.nanstd(feat_dict['close_vel']))
            dict_feat_block['coef_var_close_vel'].append(np.nanstd(feat_dict['close_vel'])/np.nanmean(feat_dict['close_vel']))
            dict_feat_block['slope_close_vel'].append(np.polyfit(np.arange(len(feat_dict['close_vel'])), feat_dict['close_vel'], 1)[0])

            # tap_duration
            dict_feat_block['mean_tap_dur'].append(np.nanmean(feat_dict['tap_dur']))
            dict_feat_block['sd_tap_dur'].append(np.nanstd(feat_dict['tap_dur']))
            dict_feat_block['coef_var_tap_dur'].append(np.nanstd(feat_dict['tap_dur'])/np.nanmean(feat_dict['tap_dur']))
            dict_feat_block['slope_tap_dur'].append(np.polyfit(np.arange(len(feat_dict['tap_dur'])), feat_dict['tap_dur'], 1)[0])

            # root mean square
            dict_feat_block['mean_rms'].append(np.nanmean(feat_dict['rms']))
            dict_feat_block['sd_rms'].append(np.nanstd(feat_dict['rms']))
            dict_feat_block['slope_rms'].append(np.polyfit(np.arange(len(feat_dict['rms'])), feat_dict['rms'], 1)[0])
            dict_feat_block['sum_rms'].append(np.sum(feat_dict['rms']))
        
            # jerkiness
            dict_feat_block['jerkiness'].append(jerkiness(task, dist_df, unit_to_assess = 'trace', n_hop = 1))

            # entropy
            dict_feat_block['entropy'].append(calc_entropy(task, dist_df))

        except TypeError:
            print(f'Impossible to calculate features for combination: {block, sub, cond, cam, task, side}')

    return dict_feat_block


def decrement(dist, task, max_idx):

    if task == 'ft':
        dist_col = 'dist'
    elif task == 'oc':
        dist_col = 'mean'
    elif task == 'ps':
        dist_col = 'ang'

    decrement = (dist[dist_col][max_idx[-1]]-dist[dist_col][max_idx[0]])/dist[dist_col][max_idx[0]]

    return decrement


# def sampling_frequency(signal):
    
#     sfreq = signal.shape[0]/(signal['time'].iloc[-1]-signal['time'].iloc[0])
    
#     return sfreq

# Hesitations Function
def jerkiness(
    task: str,
    accsig,
    unit_to_assess: str,
    n_hop: int = 1,
):
    """
    Detects the number of small changes in
    direction of acceleration.
    Hypothesized is that best tappers, have
    the smoothest acceleration-trace and
    therefore lower numbers of small
    slope changes
    Inputs:
        - accsig (array): tri-axial acceleration
            signal from e.g. 10-s tapping
        - fs: sample freq
        - tap_indices: list with arrays of tap-timing-indices
        - unit_to_assess: calculated per tap or per
            whole trace
        - n_hop (int): the number of samples used
            to determine the difference between
            two points
        - smooth_samples: number of samples to
            smooth signal over

    Returns:
        - trace_count: total sum of differential
            changes in all thee axes, returned per
            tap or per whoel trace. Both are norma-
            lised against the tap/trace duration
    """

    if task == 'ft':
        dist_col = 'dist'
    elif task == 'oc':
        dist_col = 'mean'
    elif task == 'ps':
        dist_col = 'ang'

    assert unit_to_assess in ['trace', 'taps'], print(
        f'given unit_to_asses ({unit_to_assess}) is incorrect'
    )
    fs = accsig.shape[0]/(accsig['time'].iloc[-1]-accsig['time'].iloc[0])
    if unit_to_assess == 'trace':

        trace_count = 0
        sig_diff = np.diff(accsig[dist_col])
        # plt.plot(accsig['dist'])
        # plt.plot(sig_diff)
        for i in np.arange(sig_diff.shape[0] - n_hop):
            if (sig_diff[i + n_hop] * sig_diff[i]) < 0:  # removed if -1 < sigdiff...
                trace_count += 1
    
        # normalise for duration of trace
        duration_trace = accsig.shape[0] / fs
        trace_count = trace_count / duration_trace

        return trace_count


    return np.array(trace_count)  # return as array for later calculations


from math import log, e

def calc_entropy(task, signal, base=None):
    """
    Computes entropy of label distribution.

    adjusted from: https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    """
    if task == 'ft':
        dist_col = 'dist'
    elif task == 'oc':
        dist_col = 'mean'
    elif task == 'ps':
        dist_col = 'ang'

    len_signal = len(signal[dist_col])

    if len_signal <= 1:
        return 0

    _, counts = np.unique(signal[dist_col], return_counts=True)
    probs = counts / len_signal
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent