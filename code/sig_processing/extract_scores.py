"""
Extracts scores from excel table.
Removes non-scored/non existing rows from X_df.
"""

# Import public packages and functions
import numpy as np
import pandas as pd
import os

# Import own function
import import_data.find_paths as find_paths

def get_labels_for_feat_df(X_df):

    y = []  # list to store labels

    ids = X_df['file']
    if ids[0].startswith('feat'): ids = [x[5:-5] for x in ids]
    else: ids = [x[:-5] for x in ids]

    ids = [x.split('_') for x in ids]

    for id_list in ids:
        block, sub, cond, cam, task, side = id_list
        value = get_scores(block, sub, cond, cam, task, side)
        y.append(value)

    return y


def get_scores(block, sub, cond, cam, task, side):

    """
    Function that extracts the scores from 
    the scores excel table.

    Input:
        - sub, cond, cam, task, side, block (str).
    Output:
        - score (int or nan)
    """

    read_scores = pd.read_excel(os.path.join(
        find_paths.find_onedrive_path('patientdata'),
        f'scores_JB_JH_JR.xlsx'),
        usecols='A:I'
        )

    read_scores.set_index('sub_cond_cam', inplace = True)

    if side == 'left': side='lh'
    elif side == 'right': side='rh'
    
    # read scores for all blocks of a subject in the same cond, cam per side
    ext_scores = read_scores.loc[f'{sub}_{cond}_{cam}'][f'{task}_{side}']

    if type(ext_scores) != float:

        if isinstance(ext_scores, int):
            ls_int_sc = [ext_scores,]
        else:
            ls_int_sc = [int(s) for s in ext_scores if s in ['0', '1', '2', '3', '4']]
        
        if block == 'b1':
            score = ls_int_sc[0]
        elif block == 'b2':
            try:
                score = ls_int_sc[1]
            except IndexError:
                score = ls_int_sc[0]     
        elif block == 'b3':
            score = ls_int_sc[2]
    else:
        score = np.nan
        print(f'No scores for block {block, sub, cond, cam, task, side} or this combination does not exist')
            
    return score


def remove_non_score_rows(X_df, y):

    new_X_df = X_df.drop(np.where(np.isnan(y))[0])

    return new_X_df