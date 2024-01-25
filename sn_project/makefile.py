import pandas as pd

def select_last_n_rows(input_file, output_file, n):
    # Read the CSV file
    df = pd.read_csv(input_file, low_memory=False)

    # Select the last n rows
    df_subset = df.tail(n)

    # Save the selected rows to a new CSV file
    df_subset.to_csv(output_file, index=False)


input_csv = '../dataset/US_youtube_trending_data_sample.csv'
output_csv = '../dataset/usdataset.csv'

n_rows_to_select = 2100

select_last_n_rows(input_csv, output_csv, n_rows_to_select)
