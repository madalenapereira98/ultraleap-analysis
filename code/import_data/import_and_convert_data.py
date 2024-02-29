"""
Importing Handtrack-Files from Ultraleap.
Removing duplicate and only NaN rows.
Getting clean Ultraleap-data as DataFrames.

"""

# Import public packages and fucntions
import os
import numpy as np
import pandas as pd

# import own functions
from import_data import find_paths


def import_string_data(
    file_path: str,
    removeNaNs: bool=True,
):
    """
    Function that loads Ul-data separated by 
    commas and converts it into DataFrame.

    Input:
        - file_path (str): directory of data file
        - removeNaNs (bool): defines if double- and
        nan-rows have to be deleted (default=True); 
        uses remove_double_and_onlyNan_rows(). 
    
    Output:
        - df: pd DataFrame with correct data
    """

    try:

        # read in original data
        dat = np.loadtxt(file_path, dtype=str)

        # split keys to list
        keys = dat[0]
        keys = keys.split(',')

        # remove key row from data
        dat = dat[1:]

        list_of_values = []

        for row in np.arange(len(dat)):

            # create value list per row

            # split big string in pieces
            datsplit = dat[row].split(',')

            # take out single values global time and is pinching
            glob_time = datsplit[0]
        
            # take pinching boolean value
            try:
                is_pinch = int(datsplit[-5])
        
            # if is_pinching is missing (nan) because hand was not recorded
            except ValueError:
                
                if datsplit[-5] == 'nan':

                    is_pinch = np.nan

            # remove boolean values from rest
            datsplit.pop(0)
            datsplit.pop(-5)

            # fill new list with floats
            values = []

            for i in np.arange(0, len(datsplit) - 1, 2):

                # create float from two string parts
                try:
                    values.append(
                        float(f'{datsplit[i]}.{datsplit[i + 1]}')
                    )
                
                # add nan if no values are recorded
                except ValueError:
                    
                    if np.logical_or(
                        datsplit[i] == 'nan',
                        datsplit[i + 1] == 'nan'
                    ):
                        values.append(np.nan)

            # insert single values in correct order to keys
            values.insert(0, glob_time)
            values.insert(-4, is_pinch)

            list_of_values.append(values)

        # convert list of lists to DataFrame
        df = pd.DataFrame(data=list_of_values, columns=keys)    

    except (ValueError, AssertionError, UnboundLocalError):

        df = pd.read_csv(file_path)

        if 'Unnamed: 0' in df.columns:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    if removeNaNs:
            df = remove_double_and_onlyNan_rows(df)

    return df


def remove_double_and_onlyNan_rows(
    df
):
    """
    Removes every row containing only 
    NaN values and/or duplicate rows.
    
    Input:
        - raw df (DataFrame) that has to be cleaned.
    
    Output:
        - cleaned_df (DataFrame) without only-nan 
        and/or duplicate rows.
    """
    values = df.values  # use np-array for computational-speed
    # create list to store selection
    to_keep = [False]  # start with 1 bool because of range - 1 in for-loop
    for i in np.arange(1, df.shape[0]): # loop over every row (i)
        # if all values are nan
        if np.isnan(list(values[i, 3:])).all():
            to_keep.append(False)
            continue
        # if all values are identical
        if (values[i, 3:] == values[i - 1, 3:]).all():
            to_keep.append(False)

        # if xx% of values are identical
        elif sum(values[i, 3:] == values[i - 1, 3:]) / len(values[i, 3:]) > .8:
            to_keep.append(False)

        
        else:
            # keep row if not all-nan, or all-identical
            to_keep.append(True)
    # create new dataframe with rows-to-keep
    clean_df = df[to_keep].reset_index(drop=True)

    return clean_df


def get_data(folder: str,
    sub: str, task: str, condition: str, side: str,
    cam_pos: str
):
    """
    Function that gets the data as DataFrame 
    without only-nan and duplicate rows.

    Input:
        - folder (str), sub (str), task (str), 
        condition (str), side (str), cam_pos (str).
    
    Output:
        - cleaned data (DataFrame).
    
    Raises:
        - ValueErrors if cam_pos or side are incorrect

    """
    if side.lower() not in ['left', 'right']:
        raise ValueError('incorrect side variable')
    
    if cam_pos.lower() not in ['vr', 'dt', 'st']:
        raise ValueError('incorrect camera variable')
    
    # find path of defined data
    pathfile = find_paths.find_raw_data_filepath(folder=folder,
        sub=sub, cam_pos=cam_pos, task=task,
        condition=condition, side=side,
    )
    
    if len(pathfile) == 0: return
    
    assert os.path.exists(pathfile), (
        f'selected path does not exist {pathfile}')
  
    # load selected file
    data = import_string_data(pathfile)
    
    # clean data
    data = remove_double_and_onlyNan_rows(data)    

    return data



