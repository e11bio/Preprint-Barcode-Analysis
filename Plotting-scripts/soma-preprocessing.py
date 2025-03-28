# This file will contains functions to preprocess SOMA data

# input: csv file containing (index, segment_id)

def import_zarr(file):
    vals = zarr.open(file, mode="r")
    df = pd.DataFrame({'coords': vals['coords'][:].tolist(),
                   'score': vals['score'][:].tolist(),
                   'is_cell': vals['is_cell']})
    return df


def process_zarr(zarr_path):
    max_score_length = max(len(score) for score in df['score'])
    score_columns = [f'ch_{i}' for i in range(max_score_length)]
    
    # Split the 'score' column into individual columns
    df[score_columns] = pd.DataFrame(df['score'].tolist(), index=df.index)

    #get coordinates for each cell
    df[['z', 'y', 'x']] = pd.DataFrame(df['coords'].tolist(), index=df.index)

    return df, score_columns


def filter_df(df, csv_path):

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
    

    # in the case of these plots, there is no filtering anymore because we are lookign at all somas. 
    # 1. Load the CSV file with soma information
    somas_csv = pd.read_csv(csv_path)

    # 2. Filter out somas with segment ID of 0
    id_column = 'segment_id' 
    valid_somas = somas_csv[somas_csv[id_column] != 100000] # change filter to 0 for filtering out non skeleton somas.

    # 3. Get the valid indices to keep in the processed_df
    # These indices correspond to the positions in processed_df
    valid_indices = valid_somas.index.tolist()

    # 4. Filter processed_df to keep only the rows with valid indices
    processed_df_filtered = df.iloc[valid_indices]

    return df


