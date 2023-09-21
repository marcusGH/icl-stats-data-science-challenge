'''
Take the data in raw/ and write to data/derived/ with suitable numeric data
'''
import pandas as pd
import os


def get_preprocessed_data(data_dir):
    """
    :param data_dir: path to the input directory
    :return:
    """
    train_path = os.path.join(data_dir, "raw", "train.csv")
    preprocessed_path = os.path.join(data_dir, "derived", "train-preprocessed.csv")

    with open(train_path, "rt") as fin:
        with open(preprocessed_path, "wt") as fout:
            for line in fin:
                fout.write(line.replace('\\"', "'"))

    return pd.read_csv(preprocessed_path, sep=r'\s+', quotechar='"')

def fill_missing_data(df):
    return df

def winsorize_outliers(df):
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join("data", "raw", "my_data.csv"))

    df = fill_missing_data(df)
    df = winsorize_outliers(df)

    df.to_csv(os.path.join("data", "derived", "my_preprocesed_data.csv"))
