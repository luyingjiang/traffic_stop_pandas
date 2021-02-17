'''
Analyzing traffic stop data.

Luying Jiang
'''

import numpy as np
import pandas as pd
import os

# Defined constants for column names
ARREST_CITATION = 'arrest_or_citation'
IS_ARRESTED = 'is_arrested'
YEAR_COL = 'stop_year'
MONTH_COL = 'stop_month'
DATE_COL = 'stop_date'
STOP_SEASON = 'stop_season'
STOP_OUTCOME = 'stop_outcome'
SEARCH_TYPE = 'search_type'
SEARCH_CONDUCTED = 'search_conducted'
AGE_CAT = 'age_category'
OFFICER_ID = 'officer_id'
STOP_ID = 'stop_id'
DRIVER_AGE = 'driver_age'
DRIVER_RACE = 'driver_race'
DRIVER_GENDER = 'driver_gender'
VIOLATION = "violation"

SEASONS_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11]}

NA_DICT = {
    'drugs_related_stop': False,
    'search_basis': "UNKNOWN"
    }

AGE_BINS = [0, 21, 36, 50, 65, 100]
AGE_LABELS = ['juvenile', 'young_adult', 'adult', 'middle_aged', 'senior']

SUCCESS_STOPS = ['Arrest', 'Citation']

CATEGORICAL_COLS = [AGE_CAT, DRIVER_GENDER, DRIVER_RACE,
                    STOP_SEASON, STOP_OUTCOME, VIOLATION]

SEARCHING_TYPES = ['Probable Cause', 'Incident to Arrest']

# Task 1a
def read_and_process_allstops(csv_file):
    '''
    Purpose: read in csv and process it according to the assignment
      requirements.

    Inputs:
      csv_file (string): path to the csv file to open

    Returns: (dataframe): a processed dataframe,
      or None if the file does not exist
    '''
    if not os.path.exists(csv_file):
        return None
    
    col_type = {STOP_ID: int, DATE_COL: str, OFFICER_ID: str, 
        DRIVER_GENDER: str, DRIVER_AGE: float, DRIVER_RACE: str, 
        VIOLATION: str, IS_ARRESTED: bool, STOP_OUTCOME: str}
    df = pd.read_csv(csv_file, dtype = col_type)
    
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[YEAR_COL] = df[DATE_COL].dt.year
    df[MONTH_COL] = df[DATE_COL].dt.month

    df[STOP_SEASON] = "default"
    for key, value in SEASONS_MONTHS.items():
        df[STOP_SEASON] = np.where(df[MONTH_COL].isin(value), 
                                    key, df[STOP_SEASON])

    df[AGE_CAT] = pd.cut(x = df[DRIVER_AGE], bins = AGE_BINS, 
                        labels = AGE_LABELS)

    df['arrest_or_citation'] = np.where(df[STOP_OUTCOME].isin(SUCCESS_STOPS),
                                        True, False) 

    df[OFFICER_ID] = df[OFFICER_ID].fillna('UNKNOWN')

    for i in CATEGORICAL_COLS:
        df[i] = df[i].astype('category')

    return df


# Task 1b
def read_and_process_searches(csv_file, fill_na_dict=None):
    '''
    Purpose: read in csv and process it according to the assignment
        requirements.

    Inputs:
        csv_file (string): path to the csv file to open
        fill_na_dict (dict): of the form {colname: fill value}

    Returns: (dataframe): a processed dataframe,
      or None if the file does not exist
    '''
    if fill_na_dict is None:
        # Handle fill_na_dict parameter not supplied
        fill_na_dict = NA_DICT

    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)

    for key in fill_na_dict:   
        df[key] = df[key].fillna(fill_na_dict[key])

    return df


# Task 2a
def apply_val_filters(df, filter_info):
    '''
    Purpose: apply a value filter to a dataframe

    Inputs:
        df (dataframe)
        filter_info (dict): of the form {'column_name':
            ['value1', 'value2', ...]}

    Returns: (dataframe) filtered dataframe,
      or None if a specified column does not exist
    '''
    for key, value in filter_info.items():
        if key not in df.columns:
            return None
        df = df[df[key].isin(value)]
            
    return df


# Task 2b
def apply_range_filters(df, filter_info):
    '''
    Purpose: apply a range filter to a dataframe

    Inputs:
        df (dataframe)
        filter_info (dict): of the form {'column_name': ['value1', 'value2']}

    Returns: (dataframe) filtered dataframe,
      or None if a specified column does not exist
    '''
    for key, value in filter_info.items():
        if key not in df.columns:
            return None
        df = df[df[key].between(value[0], value[1])]
            
    return df


# Task 3
def get_summary_statistics(df, group_col_list, summary_col=DRIVER_AGE):
    '''
    Purpose: produce a dataframe of aggregations

    Inputs:
        df (dataframe): the dataframe to get aggregations from
        group_col_list (list of str colnames): a list of columns to group by
        summary_col (str colname): a numeric column to aggregate

    Returns: (dataframe) a dataframe constructed from aggregations
    '''
    if len(group_col_list) == 0:
        return None

    for i in group_col_list:
        if i not in df.columns:
            return None

    global_mean = df[summary_col].mean()
    grouped = df.groupby(group_col_list)
    df_agg = grouped[summary_col].agg([np.median, np.mean])
    df_agg['mean_diff'] = df_agg['mean'] - global_mean
    
    return df_agg


# Task 4
def get_rates(df, cat_col, outcome_col):
    '''
    Purpose: returns dataframe of rates given in outcome column

    Inputs:
        df (dataframe)
        cat_col (list) of the column names to group by
        outcome_col (str) column name of outcome column

    Returns: (dataframe) dataframe with the rates for each outcome.
    '''
    for i in cat_col:
        if i not in df.columns or outcome_col not in df.columns:
            return None

    counts_per_group = df.groupby(cat_col).size()
    groups = cat_col + [outcome_col]
    outcome_per_group = df.groupby(groups).size()
    pct_per_group = outcome_per_group/counts_per_group
    pct_per_group_df = pct_per_group.unstack()
    pct_per_group_df = pct_per_group_df.fillna(0)

    return pct_per_group_df


# Task 5
def compute_search_share(
        stops_df, searches_df, cat_col, M_stops=25):
    '''
    Purpose: return a sorted dataframe of cat_cols by share of search
        conducted
    Inputs:
        stops_df (dataframe)
        searches_df (dataframe)
        cat_cols (list) of the column names to group by
        M_stops (int) minimum number of stops to retain

    Returns (dataframe): dataframe of search rates given by cat_col,
      or None if no officers meet M_stops criterion
    '''

    merged_df = pd.merge(stops_df, searches_df, on = STOP_ID, how = 'left')

    if SEARCH_CONDUCTED in merged_df.columns:
        merged_df[SEARCH_CONDUCTED] = merged_df[SEARCH_CONDUCTED].fillna(False)
    else:
        merged_df[SEARCH_CONDUCTED] = np.where(merged_df[SEARCH_TYPE].\
            isin(SEARCHING_TYPES), True, False)

    officer_count = merged_df.groupby([OFFICER_ID]).size()
    officer_count = officer_count[officer_count >= M_stops]

    if officer_count.shape[0] == 0:
        return None

    index = officer_count.index
    merged_df = merged_df[merged_df[OFFICER_ID].isin(index)]
    new_df = get_rates(merged_df, cat_col, SEARCH_CONDUCTED)
    
    if new_df.shape[1] == 1:
        if False in new_df.columns:
            new_df[True] = 0
        else:
            new_df[False] = 0

    sorted_df = new_df.sort_values(by = True)
    
    return sorted_df