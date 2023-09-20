'''
Take the data in raw/ and write to data/derived/ with suitable numeric data
'''
import pandas as pd

def fill_missing_data(df):
    df

def winsorize_outliers(df):
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join("data", "raw", "my_data.csv"))

    df = fill_missing_data(df)
    df = winsorize_outliers(df)

    df.to_csv(os.path.join("data", "derived", "my_preprocesed_data.csv"))
