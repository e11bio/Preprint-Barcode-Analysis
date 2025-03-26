import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from barcode_heatmap import create_barcode_heatmap

# Import your data - use whichever method is appropriate for your current data state
# Option 1: If you have the data already processed as soma_barcodes
try:
    # Try to load from a saved numpy array if available
    soma_barcodes = np.load('Data/soma_barcodes.npy')
    print(f"Loaded soma barcodes with shape: {soma_barcodes.shape}")
except:
    print("Could not find pre-saved soma barcodes. Please specify the correct path or method to load your data.")
    # Option 2: If you need to process from the DataFrame
    # Uncomment and modify these lines as needed
    # import zarr
    # file_path = "path/to/your/data.zarr"
    # 
    # def import_zarr(file):
    #     vals = zarr.open(file, mode="r")
    #     df = pd.DataFrame({'coords': vals['coords'][:].tolist(),
    #                     'score': vals['score'][:].tolist(),
    #                     'is_cell': vals['is_cell']})
    #     return df
    # 
    # df = import_zarr(file_path)
    # 
    # # Process zarr data
    # def process_zarr(df):
    #     max_score_length = max(len(score) for score in df['score'])
    #     score_columns = [f'ch_{i}' for i in range(max_score_length)]
    #     
    #     # Split the 'score' column into individual columns
    #     df[score_columns] = pd.DataFrame(df['score'].tolist(), index=df.index)
    # 
    #     #get coordinates for each cell
    #     df[['z', 'y', 'x']] = pd.DataFrame(df['coords'].tolist(), index=df.index)
    # 
    #     return df, score_columns
    # 
    # processed_df, score_columns = process_zarr(df)
    # 
    # # Filter if needed
    # filtered_df = processed_df[processed_df['is_cell'] == True]  # adjust filtering as needed
    # 
    # # Choose target channels
    # target_channels = [col for col in filtered_df.columns if 'barcode' in col and 'marker' not in col]
    # 
    # # Create the soma_barcodes array
    # soma_barcodes = filtered_df[target_channels].values
    # soma_barcodes = np.array(soma_barcodes)
    
    # If you have some other method to load or create your data, use that here
    # ...

# Once you have the soma_barcodes array, run the analysis
if 'soma_barcodes' in locals():
    print("Creating barcode heatmap...")
    distance_matrix = create_barcode_heatmap(soma_barcodes)
    
    # Optionally save results
    # np.save('Data/hamming_distance_matrix.npy', distance_matrix)
    print("Analysis complete!") 