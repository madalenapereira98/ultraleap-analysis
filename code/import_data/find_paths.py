"""
Find Ultraleap Subject files' directories
"""

import os
import numpy as np
import import_data.import_and_convert_data as import_dat


def find_onedrive_path(
    subfolder
):
    """
    Function that finds the one drive 
    path where the ul_data is stored.

    Input:
        - folder (str) specifying control or patientdata.

    Output:
        - onedrive path (str).

    """
    path = os.getcwd()

    while os.path.dirname(path)[-5:] != 'Users':
        path = os.path.dirname(path)
    # path is now Users/username
    onedrive_f = [
        f for f in os.listdir(path) if np.logical_and(
            'onedrive' in f.lower(),
            'charit' in f.lower()
        )
    ]  # gives list
    onedrivepath = os.path.join(path, onedrive_f[0])

    if subfolder.lower() == 'data':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',
            'movement_analysis',
            'data',
        )

    elif subfolder.lower() == 'patientdata':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',
            'movement_analysis',
            'data',
            'patientdata'
        )

    elif subfolder.lower() == 'control':
        onedrivepath = os.path.join(
            onedrivepath,
            'Ultraleap-hand-tracking',
            'movement_analysis',
            'data',
            'control'
        )

    return onedrivepath


def find_available_subs(folder, exc=False):
    """
    Function that finds gives a list of subjects.

    Input:
        - folder (str) specifying control or patientdata, 
        exc (bool) excluding subject 'ul007' if exc = True (default
        includes all patients present in folder = 'patientdata).
    Output:
        - list of subjects in the specified folder.

    """

    subs = os.listdir(find_onedrive_path(folder))

    if folder == 'patientdata':
        if exc:
            subs = [s for s in subs if s[:2].lower() == 'ul' and s!='ul007']
            
        else:
            subs = [s for s in subs if s[:2].lower() == 'ul']
    elif folder == 'control':
        subs = [s for s in subs if s[:7].lower() == 'control']

    return subs


def find_raw_data_filepath(
    folder: str,
    sub: str, cam_pos: str, task: str,
    condition: str, side: str
):
    """
    Function to find specific path with defined
    files.

    Input:
        - folder (str), sub (str), cam_pos (str), 
        task (str), condition (str), side (str). 

    Output:
        - pathfile (str) selected by file specifications.
    """
    assert side in ['left', 'right'], (
        f'given side ({side}) should be "left" or "right"'
    )
    # assert cam_pos in []

    # find folder with defined data
    # if len(sub) == 3: sub = f'ul{sub}'
    subpath = os.path.join(find_onedrive_path(folder), sub)

    cam_folder = os.path.join(subpath, cam_pos.lower())

    # only take folder with defined task
    files = os.listdir(cam_folder)

    ### CHANGED !!!!!
    if condition == 'm0': sel_files = [f for f in files if task.lower() in f]
    else: sel_files = [f for f in files if (task.lower() in f and condition.lower() in f)]

    if len(sel_files) == 0:
        print(
            'No files available for combination: '
            f'{sub, cam_pos, task, condition, side}')
        return ''

    sel_folder = os.path.join(cam_folder, sel_files[0])
    data_files = os.listdir(sel_folder)

    # select on side
    if side.lower() == 'left':
        data_files = [f for f in data_files if 'lh' in f]

    elif side.lower() == 'right':
        data_files = [f for f in data_files if 'rh' in f]

    pathfile = os.path.join(sel_folder, data_files[0])

    return pathfile
