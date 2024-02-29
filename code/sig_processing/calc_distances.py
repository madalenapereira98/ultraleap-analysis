"""
Calculates the distances for each task and 
creates .csv files with distances and times.
"""

# Import public packages and functions
import numpy as np
import pandas as pd
import os
import importlib

# Import own functions
import movement_calc.helpfunctions as hp
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

def calculate_distances(folder, sub, task, cond, block, file):
    """
    Calculates distances differently for each task. 
    For opening-closing it calculates the distances 
    based on every finger and saves a .csv file with
    the euclidean distance between each finger and 
    the palm. 

    Input:
        - sub, task, cond, block, file (str)
    """
   
    if task == 'oc':
        ls_fing = ['index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        dist = {}
        for fing in ls_fing:
            dist[fing] = []
        for fing in ls_fing:
            dist[fing] = hp.calc_distances(block, fing, 'palm')
        
        dist_time = pd.DataFrame(dist, columns=ls_fing)
        
        # insert mean column
        mean = np.mean(dist_time[ls_fing], axis=1)
        dist_time['mean'] = mean
        dist_time['time'] = block.program_time.to_list()

        ref_time = [0]
        cum_time = 0
        for time in np.diff(block.program_time):
            cum_time += time
            ref_time.append(cum_time)
        dist_time['ref_time'] = ref_time
        dist_time['date_time'] = block.date_time.to_list()
    
    elif task == 'ft':
        dist = hp.calc_distances(block, 'index_tip', 'thumb_tip')
        dist_time = pd.DataFrame(dist, columns=['dist'])
        dist_time['time'] = block.program_time.to_list()
        
        ref_time = [0]
        cum_time = 0
        for time in np.diff(block.program_time):
            cum_time += time
            ref_time.append(cum_time)
        dist_time['ref_time'] = ref_time
        dist_time['date_time'] = block.date_time.to_list()
        
    elif task == 'ps':
        dist = hp.calc_ps_angle(block, 'thumb_tip', 'middle_tip', 'palm')
        dist_time = pd.DataFrame(dist, columns=['ang'])
        dist_time['time'] = block.program_time.to_list()

        ref_time = [0]
        cum_time = 0
        for time in np.diff(block.program_time):
            cum_time += time
            ref_time.append(cum_time)
        dist_time['ref_time'] = ref_time
        dist_time['date_time'] = block.date_time.to_list()

    else:
        print('postural tremor was not analysed yet')

    dist_time_path = os.path.join(
        repo_path, 'data', folder, sub, task, cond, 'distances')
    if not os.path.exists(dist_time_path):
        os.makedirs(dist_time_path)
    dist_time.to_csv(os.path.join(
        dist_time_path, f'{file}'))

    return 