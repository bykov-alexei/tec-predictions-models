import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--file', dest='file')
args = parser.parse_args()
index_data_flie = os.path.join(args.file)

dsc_table = pd.read_csv(index_data_flie, delim_whitespace=True, header=None)
dsc_table.columns = ['YEAR', 'DOY', 'UT', 'Kp', 'R', 'ap', 'f10_7', 'AE', 'AL', 'AU']
dsc_table.loc[dsc_table['f10_7'] == 999.9, 'f10_7'] = np.nan
dsc_table['f107_ma05'] = dsc_table.f10_7.rolling(12).mean()
dsc_table['f107_ma11'] = dsc_table.f10_7.rolling(264).mean()
dsc_table['f107_ma81'] = dsc_table.f10_7.rolling(972).mean()
dsc_table['f107_sd05'] = dsc_table.f10_7.rolling(12).std()
dsc_table['f107_sd11'] = dsc_table.f10_7.rolling(264).std()
dsc_table['f107_sd81'] = dsc_table.f10_7.rolling(972).std()
dsc_table = dsc_table.fillna(0)

dsc_table['COY'] = np.cos(2*np.pi*dsc_table.DOY/365.5)
dsc_table['SOY'] = np.sin(2*np.pi*dsc_table.DOY/365.5)

dsc_table.to_csv(os.path.join('indexes_data', 'meta.csv'))
