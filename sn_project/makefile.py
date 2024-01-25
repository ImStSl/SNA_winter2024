import pandas as pd

def select_last_n_rows_and_filter(input_file, output_file, n, column_name, filter_value):
    # Read the CSV file
    df = pd.read_csv(input_file, low_memory=False)

    # Exclude rows where the specified column has the filter value
    df_filtered = df[df[column_name] != filter_value]

    # Select the last n rows from the filtered DataFrame
    df_subset = df_filtered.tail(n)

    # Save the selected and filtered rows to a new CSV file
    df_subset.to_csv(output_file, index=False)


input_csv = '../dataset/US_youtube_trending_data_sample.csv'
output_csv = '../dataset/dataset.csv'

n_rows_to_select = 2100
column_to_filter = 'video_id'
filter_value = '#NAME?'

select_last_n_rows_and_filter(input_csv, output_csv, n_rows_to_select, column_to_filter, filter_value)
