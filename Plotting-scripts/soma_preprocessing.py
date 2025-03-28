# This file will contains functions to preprocess SOMA data

import zarr #open files
import sklearn.metrics as sn
from sklearn.decomposition import PCA 
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

import math
import os
import textwrap
from adjustText import adjust_text
from itertools import combinations
import datetime

channel_targets = {
'ch_0': 'E2-barcode-R1',
'ch_1': 'S1-barcode-R1',
'ch_2': 'ALFA-barcode-R1',
'ch_3': 'Ty1-barcode-R2',
'ch_4': 'GFAP-marker-R2',
'ch_5': 'ALFA-barcode-R2',
'ch_6': 'VGAT-marker-R3',
'ch_7': 'GABRA1-marker-R3',
'ch_8': 'HA-barcode-R3',
'ch_9': 'ALFA-barcode-R3',
'ch_10': 'PSD-95-marker-R4',
'ch_11': 'Bassoon-marker-R4',
'ch_12': 'ALFA-barcode-R4',
'ch_13': 'Beta-Amyloid-marker-R5',
'ch_14': 'c-Myc-barcode-R5',
'ch_15': 'ALFA-barcode-R5',
'ch_16': 'T7-barcode-R6',
'ch_17': 'ALFA-barcode-R6',
'ch_18': 'Synapsin-marker-R6',
'ch_19': 'VSVG-barcode-R6',
'ch_20': 'Shank2-marker-R7',
'ch_21': 'Bassoon-marker-R7',
'ch_22': 'ALFA-barcode-R7',
'ch_23': 'ALFA-barcode-R8',
'ch_24': 'AU5-barcode-R8',
'ch_25': 'ALFA-barcode-R9',
'ch_26': 'NWS-barcode-R9',
'ch_27': 'SunTag-barcode-R9',
'ch_28': 'ETAG-barcode-R9',
'ch_29': 'SPOT-barcode-R10',
'ch_30': 'MoonTag-barcode-R10',
'ch_31': 'ALFA-barcode-R10',
'ch_32': 'HSV Tag-barcode-R10',
'ch_33': 'Protein C-barcode-R11',
'ch_34': 'Tag100-barcode-R11',
'ch_35': 'ALFA-barcode-R11',
'ch_36': 'CMyc-barcode-R11',
'ch_37': 'OLLAS-barcode-R12',
'ch_38': 'GFP-marker-R12',
'ch_39': 'AU5-barcode-R12'
}

target_channels = [
'E2-barcode-R1',
'S1-barcode-R1',
'ALFA-barcode-R1',
'Ty1-barcode-R2',
'HA-barcode-R3',
'T7-barcode-R6',
'VSVG-barcode-R6',
'AU5-barcode-R8',
'NWS-barcode-R9',
'SunTag-barcode-R9',
'ETAG-barcode-R9',
'SPOT-barcode-R10',
'MoonTag-barcode-R10',
'HSV Tag-barcode-R10',
'Protein C-barcode-R11',
'Tag100-barcode-R11',
'CMyc-barcode-R11',
'OLLAS-barcode-R12'
]


# input: csv file containing (index, segment_id)

def import_zarr(file):
    vals = zarr.open(file, mode="r")
    df = pd.DataFrame({'coords': vals['coords'][:].tolist(),
                   'score': vals['score'][:].tolist(),
                   'is_cell': vals['is_cell']})
    return df


def process_zarr(df):
    max_score_length = max(len(score) for score in df['score'])
    score_columns = [f'ch_{i}' for i in range(max_score_length)]
    
    # Split the 'score' column into individual columns
    df[score_columns] = pd.DataFrame(df['score'].tolist(), index=df.index)

    #get coordinates for each cell
    df[['z', 'y', 'x']] = pd.DataFrame(df['coords'].tolist(), index=df.index)

    return df, score_columns


def filter_somas(df, csv_path):

   
    # in the case of these plots, there is no filtering anymore because we are lookign at all somas. 
    # 1. Load the CSV file with soma information
    somas_csv = pd.read_csv(csv_path)

    # 2. Filter out somas with segment ID of x: here we are not filtering anything out. 
    id_column = 'segment_id' 
    valid_somas = somas_csv[somas_csv[id_column] != 10000000] # change filter to 0 for filtering out non skeleton somas.

    # 3. Get the valid indices to keep in the processed_df
    # These indices correspond to the positions in processed_df
    valid_indices = valid_somas.index.tolist()

    # 4. Filter processed_df to keep only the rows with valid indices
    processed_df_filtered = df.iloc[valid_indices]

    # 5. Continue with the renaming columns logic
    rename_mapping = {}
    for column_name, target_name in channel_targets.items():
    # Only include columns that exist in the DataFrame
        if column_name in processed_df_filtered.columns:
            rename_mapping[column_name] = target_name

    # Rename the columns in the DataFrame
    processed_df_filtered = processed_df_filtered.rename(columns=rename_mapping)

    # Define the non-channel columns you want to keep
    non_channel_columns = ['coords', 'score', 'is_cell', 'z', 'y', 'x']

    # Combine non-channel columns with target channels to create the list of columns to keep
    columns_to_keep = non_channel_columns + target_channels

    # Filter the DataFrame to include only the specified columns
    filtered_df = processed_df_filtered[columns_to_keep]


    return processed_df_filtered


def generate_barcode_array():

    zarr_path = "/home/kathleen/LS-somas_AM.zarr"
    df = import_zarr(zarr_path)
    processed_df, score_columns = process_zarr(df)

    # soma filtering dataset, in this case we aren't actually filtering anything out. 
    csv_path = "/home/aashir/repos/barcode_analysis/Preprint-Barcode-Analysis/Data/LS-somas.csv"  # Using absolute path
    filtered_df = filter_somas(processed_df, csv_path)

    soma_barcodes = filtered_df[target_channels].values
    # Convert soma_barcodes to a numpy array
    soma_barcodes = np.array(soma_barcodes)

    return soma_barcodes


# Running this script

# soma_barcodes array, this is what is used for downstream plot analysis
soma_barcodes = generate_barcode_array()
print(soma_barcodes.shape)
    
