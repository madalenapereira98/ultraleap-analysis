"""
Importing Handtrack-Files from Ultraleap
version data output 01.09.2022

Most recent version prior to repo-deletion
"""

# Import public packages and fucntions
import numpy as np
import pandas as pd
import datetime
from itertools import product
import os 
import matplotlib.pyplot as plt

import traces
from datetime import datetime
from datetime import timedelta

# Import own functions
import import_data.preprocessing_meta_info as meta_info
import import_data.import_and_convert_data as import_dat

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
code_path = os.path.join(repo_path, 'code')


def block_extraction(
        folder, df, sub, task, side, cond, cam, to_save = False, to_plot = False
):
    """
    Divides the cleaned data between two time points (can 
    be used in cases where all task are in the same file).
    If to_save=True the block(s) is/are saved in .csv files.
    If to_plot=True the raw blocks are plotted with the 
    excel extracted timestamps.
        
    Input:
        - cleaned df (DataFrame),
        time1: global_time of the start of a task, 
        time2: global_time of the end of a task.
        
    Output:
        - new dataframe for specific task/block.
    """
    # get block timestamps corresponding to data
    blocktimes = meta_info.load_block_timestamps(folder,
        sub=sub, task=task, side=side)
    
    if cam == 'desktop': cam = 'dt'
    
    # reading excel file
    ### CHANGED !!!!
    if cond == 'm0': run_row = blocktimes.loc[cam.lower()]
    else: run_row = blocktimes.loc[f'{cond.lower()}_{cam.lower()}']

    block_times = {}
    blocks_dict = {}
    new_blocks = []

    # creating a dictionary with ul_start and ul_end timepoints from excel table
    for block, time in product(['b1', 'b2', 'b3'],['start','end']):
        try:
            # extracting ul_start and ul_end timepoints
            t = run_row[f'{block}_{time}_ul']     
       
        except KeyError:
            continue

        if type(t) != float:
      
            t = t.strftime("%H:%M:%S")
            block_times[f'{block}_{time}'] = t
            new_blocks.append(block)

    # use set, otherwise blocknames will appear duplicated
    new_blocks = list(set(new_blocks))

    for b_idx, b in enumerate(new_blocks):
        b_idx += 1
      
        blocks_dict[f'b{b_idx}'] = df[
            np.logical_and(
                df['global_time'] >= block_times[f'{b}_start'],
                df['global_time'] <= block_times[f'{b}_end']
            )
        ].reset_index(drop=True)

        reg_block = import_dat.remove_double_and_onlyNan_rows(
            regularize_block(blocks_dict[f'b{b_idx}'], 1000/80)
        )
      
        # saving block dataframes as csv files
        if to_save:
            block_path = os.path.join(
                        repo_path, 'data', folder, sub, task, cond, 'cleaned_blocks')

            if reg_block.empty:
                print(f'b{b_idx}_{sub}_{cond}_{cam}_{task}_{side} is empty')
                continue
            
            if not os.path.exists(block_path) and not reg_block.empty:
                os.makedirs(block_path)

            reg_block.to_csv(os.path.join(
                block_path, f'b{b_idx}_{sub}_{cond}_{cam}_{task}_{side}.csv'))
            
        if to_plot:
            plot_timestamps(folder, df, sub, cond, cam, task, side, block_times.values())
            
    return 


def regularize_block(or_block, new_timedelta_ms):

    """
    Function to regularize UltraLeap-data to have 
    the same sampling frequency for the whole dataset.

    Input:
        - cleaned block without NaNs & double rows 
        - timedelta: sampling period
    
    Returns:
        - regularized dataframe
    
    """

    glob_time_to_date_time = [datetime.strptime(or_block['global_time'].iloc[row], '%H:%M:%S:%f') for row in np.arange(0, or_block.shape[0])]
    
    if 'date_time' not in or_block.columns:
        or_block.insert(loc = 0, column = 'date_time', value = glob_time_to_date_time)

    new_df = pd.DataFrame()
    
    for col in or_block.columns:
        if col.startswith('pinch'): continue
        if not col[-1] in ['x', 'y', 'z']: continue

        ls_tuple = []
        for row in np.arange(0, or_block.shape[0]):
            ls_tuple.append((
                or_block['date_time'][row], 
                or_block[col][row]
            ))

        ts = traces.TimeSeries(ls_tuple)

        new_col = ts.sample(
            sampling_period = timedelta(milliseconds = new_timedelta_ms),
            start = or_block['date_time'][0].round('1s'),
            end = or_block['date_time'][or_block.shape[0]-1],
            interpolate = 'linear',
        )

        new_times = []
        new_values = []

        for t in new_col:
            new_times.append(t[0])
            new_values.append(t[1])
        
        if 'date_time' not in new_df.columns:
            new_df.insert(loc = 0, column = 'date_time', value = new_times)
        
        new_df.insert(loc = new_df.shape[1], column = col, value = new_values)
    
    new_df.insert(loc = 1, column = 'program_time', value = np.linspace(0,new_timedelta_ms,new_df.shape[0]))
        
    return new_df



def plot_timestamps(folder, df, sub, cond, cam, task, side, times):

    """
    Function that plots the raw data with the 
    timestamps that are used to cut the blocks.

    Input:
    - times -> block_times.values() with the ul_start and ul_end time points

    """

    glob_time_to_date_time = [datetime.strptime(df['global_time'].iloc[row], '%H:%M:%S:%f') for row in np.arange(0, df.shape[0])]
    ls_times = list(times)

    fig = plt.figure(figsize=(8,6))

    x_values = [datetime.strptime(ls_times[time], '%H:%M:%S') for time in np.arange(0, len(ls_times))]
    y_values = len(list(times))*[np.mean(df['index_tip_x'])]

    plt.scatter(x_values, y_values, s=75, c='red')
    plt.plot(glob_time_to_date_time, df['index_tip_x'], alpha = .7)

    fig_dir = os.path.join(repo_path, 'figures', 'raw_data_with_timepoints', folder, sub, task, cond)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig.savefig(
        os.path.join(fig_dir, f'{sub}_{cond}_{cam}_{task}_{side}'),
        dpi = 300, facecolor = 'w',
        )
    plt.close()

    return 