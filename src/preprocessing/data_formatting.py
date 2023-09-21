'''
Take the data in raw/ and write to data/derived/ with suitable numeric data
'''
import pandas as pd
import os


def get_preprocessed_data(data_dir, skip_preprocess=True, get_test=False):
    """
    :param data_dir: path to the input directory
    :return:
    """
    if get_test:
        name_prefix = "test"
    else:
        name_prefix = "train"
    train_path = os.path.join(data_dir, "raw", f"{name_prefix}.csv")
    preprocessed_path = os.path.join(data_dir, "derived", f"{name_prefix}-preprocessed.csv")

    if not skip_preprocess:
        with open(train_path, "rt") as fin:
            with open(preprocessed_path, "wt") as fout:
                for line in fin:
                    fout.write(line.replace('\\"', "'"))

    return pd.read_csv(preprocessed_path, sep=r'\s+', quotechar='"')

def fill_missing_data(df):
    return df

def winsorize_outliers(df):
    return df


def fill_remaining_cols(df):
    extended_cols = ['AnB', 'AnC', 'AnD', 'BnC', 'BnD', 'CnD',
                     'AnBnC', 'AnBnD', 'AnCnD', 'BnCnD', 'AnBnCnD']
    for i in range(len(df['A'])):
        for c in extended_cols:
            fill_val = True
            for l in ['A', 'B', 'C', 'D']:
                if l in c:
                    fill_val = fill_val and (df.at[i, l] == 1)
            # modify the cell
            df.at[i, c] = int(fill_val)

def submit_dataframe(df, submit_path):
    df = df.reset_index()
    assert len(df.columns) == 29
    df.to_csv(submit_path, index=False, quotechar='"', sep=' ')

if __name__ == '__main__':
    df = pd.read_csv(os.path.join("data", "raw", "my_data.csv"))

    df = fill_missing_data(df)
    df = winsorize_outliers(df)

    df.to_csv(os.path.join("data", "derived", "my_preprocesed_data.csv"))
