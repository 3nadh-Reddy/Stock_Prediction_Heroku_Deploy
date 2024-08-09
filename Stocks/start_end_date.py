import os
import pandas as pd
from datetime import datetime

folder_path = r"D:\Program Files\STOCK_PREDICTION\Stocks\NIFTY50_transformed"

start_dates = []
end_dates = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'Date' column exists
        if 'Date' in df.columns:
            try:
                # Convert the 'Date' column to datetime
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
                if df['Date'].isnull().values.any():
                    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
            except Exception as e:
                print(f"Error parsing dates in {file_path}: {e}")
                continue
            
            df = df.dropna(subset=['Date'])
            
            # Get the start date (minimum date) from the 'Date' column
            if not df.empty:
                start_date = df['Date'].min()
                end_date = df['Date'].max()
                start_dates.append(start_date)
                end_dates.append(end_date)
            # else:
            #     print(f"Warning: No valid dates found in {file_path}")
        # else:
        #     print(f"Warning: 'Date' column not found in {file_path}")

# print("List of starting dates from each CSV file:")
# for date in start_dates:
#     print(date.strftime('%d-%m-%Y'))
# print()

# print("List of end dates from each CSV file:")
# for date in end_dates:
#     print(date.strftime('%d-%m-%Y'))
# print()

# print("Max data:", max(start_dates).strftime('%d-%m-%Y'))
# print("Min data:", min(end_dates).strftime('%d-%m-%Y'))
# print("difference in dates:", (min(end_dates) - max(start_dates)).days)
# print(len(start_dates))

start_date = max(start_dates).strftime('%d-%m-%Y')
end_date = min(end_dates).strftime('%d-%m-%Y')
# print(start_date, end_date) 


























# import os
# import pandas as pd
# from datetime import datetime

# folder_path = r"D:\Program Files\project\NIFTY50_transformed"

# start_dates = []
# end_dates = []

# # Define the list of possible date formats
# date_formats = ['%d-%m-%Y', '%Y-%m-%d']

# for filename in os.listdir(folder_path):
#     if filename.endswith(".csv"):
#         file_path = os.path.join(folder_path, filename)
        
#         # Read the CSV file
#         df = pd.read_csv(file_path)
        
#         # Check if 'Date' column exists
#         if 'Date' in df.columns:
#             # Initialize a variable to store the correctly parsed dates
#             parsed_dates = pd.Series([None] * len(df))
            
#             for date_format in date_formats:
#                 # Try converting the 'Date' column to datetime using the current format
#                 temp_dates = pd.to_datetime(df['Date'], format=date_format, errors='coerce')
#                 # Fill the non-null values in parsed_dates
#                 parsed_dates = parsed_dates.fillna(temp_dates)
            
#             df['Date'] = parsed_dates.dropna()
            
#             # Drop rows where date conversion failed
#             df = df.dropna(subset=['Date'])
            
#             # Get the start date (minimum date) and end date (maximum date) from the 'Date' column
#             if not df.empty:
#                 start_date = df['Date'].min()
#                 end_date = df['Date'].max()
#                 start_dates.append(start_date)
#                 end_dates.append(end_date)
#             else:
#                 print(f"Warning: No valid dates found in {file_path}")
#         else:
#             print(f"Warning: 'Date' column not found in {file_path}")

# # print("List of starting dates from each CSV file:")
# # for date in start_dates:
# #     print(date.strftime('%d-%m-%Y'))

# # print("List of end dates from each CSV file:")
# # for date in end_dates:
# #     print(date.strftime('%d-%m-%Y'))

# # if start_dates and end_dates:
# #     print("Max start date:", max(start_dates).strftime('%d-%m-%Y'))
# #     print("Min end date:", min(end_dates).strftime('%d-%m-%Y'))
# #     print("Difference in dates:", (min(end_dates) - max(start_dates)).days)
# #     print("Number of valid CSV files processed:", len(start_dates))
# # else:
# #     print("No valid dates found in the files.")

# start_date = max(start_dates).strftime('%d-%m-%Y')
# end_date = min(end_dates).strftime('%d-%m-%Y')
# print(start_date, end_date)