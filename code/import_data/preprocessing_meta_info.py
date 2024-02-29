"""
Importing metadata (timestamps) 
for preprocessing Ultraleap-data.

"""

# Import public packages and fucntions
import os
import numpy as np
import pandas as pd

# impoprt own functions
from import_data.find_paths import find_onedrive_path


def load_block_timestamps(
        folder: str,
        sub: str, 
        task: str, 
        side: str
):
    """"
    Function that reads timestamps excel table.

    Input:
        - file specifications: sub (str), 
        task (str), side (str).

    Output:
        - blocktimes (DataFrame) for specified 
        task and side.

    """
    # prevent incorrent side variable
    if side == 'lh': side = 'left'
    elif side == 'rh': side = 'right'

    # prevent incorrect task variable
    # if sub[:2].lower() == 'ul': sub = sub[2:]

    blocktimes = pd.read_excel(
        os.path.join(
            find_onedrive_path(folder),
            # f'ul{sub}',
            # f'ul{sub}_block_timestamps.xlsx'),
            sub,
            f'{sub}_block_timestamps.xlsx'),
        sheet_name=f'{task}_{side}',
    )
    if folder == 'patientdata': blocktimes.set_index('cond_cam', inplace = True)
    else: blocktimes.set_index('cam', inplace = True)

    return blocktimes
