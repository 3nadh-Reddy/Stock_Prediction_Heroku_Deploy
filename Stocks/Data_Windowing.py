import os
import pandas as pd
from datetime import datetime
from start_end_date import start_date, end_date

# Define the input and output directories
input_folder = r'D:\Program Files\STOCK_PREDICTION\final_data'
output_folder = r'D:\Program Files\STOCK_PREDICTION\Data_windowing_yahoo'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Define the date range
# start_date = '17-11-2017'
# end_date = '15-06-2023'
# start_date_dt = datetime.strptime(start_date, '%d-%m-%Y')
# end_date_dt = datetime.strptime(end_date, '%d-%m-%Y')

# Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Check if the 'Date' column exists
        if 'Date' in df.columns:
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

            # Filter the data within the date range
            df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

            # sort wrt date
            df_filtered.sort_values(by='Date', inplace=True)

            # Convert the date format to dd-mm-yyyy
            df_filtered['Date'] = df_filtered['Date'].dt.strftime('%d-%m-%Y')

            # Save the filtered data to a new CSV file in the output folder
            output_file_path = os.path.join(output_folder, filename)
            df_filtered.to_csv(output_file_path, index=False, date_format='%d-%m-%Y')
#         else:
#             print(f"Skipping {filename}: 'Date' column not found.")

# print('Data filtering and conversion completed!')


























# import os
# import pandas as pd

# # Define the folder paths
# input_folder = r"D:\Program Files\project\NIFTY50_transformed"
# output_folder = r"D:\Program Files\project\NIFTY50_buffer"

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Define the date range
# start_date = pd.to_datetime('2017-11-17')
# end_date = pd.to_datetime('2023-06-15')

# # Iterate through each CSV file in the input folder
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(input_folder, filename)
        
#         # Read the CSV file
#         df = pd.read_csv(file_path)
        
#         # Convert the date column to datetime format if it's not already
#         df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        
#         # Filter the dataframe based on the date range
#         filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
#         # Save the filtered dataframe to a new CSV file in the output folder
#         output_file_path = os.path.join(output_folder, filename)
#         filtered_df.to_csv(output_file_path, index=False)
        
#         print(f"Processed {filename}")

# print("All files have been processed.")

# import os
# import pandas as pd

# # Define the folder paths
# input_folder = r"D:\Program Files\project\NIFTY50_transformed"
# output_folder = r"D:\Program Files\project\NIFTY50_buffer"

# # Ensure the output folder exists
# os.makedirs(output_folder, exist_ok=True)

# # Define the date range
# start_date = pd.to_datetime('2017-11-17')
# end_date = pd.to_datetime('2023-06-15')

# # Iterate through each CSV file in the input folder
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(input_folder, filename)
        
#         # Read the CSV file
#         df = pd.read_csv(file_path)
        
#         # Convert the date column to datetime format if it's not already
#         df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.strftime('%d-%m-%Y')
        
#         # Filter the dataframe based on the date range
#         filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
#         # Save the filtered dataframe to a new CSV file in the output folder
#         output_file_path = os.path.join(output_folder, filename)
#         filtered_df.to_csv(output_file_path, index=False)
        
#         print(f"Processed {filename}")

# print("All files have been processed.")