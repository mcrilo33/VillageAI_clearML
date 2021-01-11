#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import peakutils
from tqdm import tqdm
from clearml import Task, StorageManager
from glob import glob


PROJECT_NAME = 'AltaroadCML'

# Connecting ClearML with the current process,
# from here on everything is logged automatically
task = Task.init(
    project_name=PROJECT_NAME,
    task_type=Task.TaskTypes.data_processing,
)
# program arguments
# Use either dataset_task_id to point to a tasks artifact or
# use a direct url with dataset_url
args = {
    'data_path': '/Users/mcrilo33/Repos/VillageAI_clearML/datasets/raw/CEREMA/m5152/*.npz',
    'mapping_path': '/Users/mcrilo33/Repos/VillageAI_clearML/datasets/raw/CEREMA/CEREMA_5152_ALL_091220_Good.csv',
    'passages_used': 5,
    'time_size': 150,
    'DEBUG': True
}

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

data_path = args['data_path']
mapping_path = args['mapping_path']
passages_used = args['passages_used']
if not isinstance(passages_used, list):
    passages_used = [passages_used]
time_size = args['time_size']
DEBUG = args['DEBUG']

if DEBUG:
    print('->> Warning in DEBUG MODE <<-\n')

def get_peaks(x, dropped_channels=[], min_dist=170, thres=0.25, abs=True):
    x = np.delete(x, dropped_channels, 1)
    return peakutils.indexes(np.mean(np.abs(x) if abs else np.maximum(x, 0), axis=1), 
                             thres=thres, min_dist=min_dist)

def _normalize_signal(x):
    median = np.median(x, axis=0)
    return x - median, median.mean()

files = sorted(glob(data_path))
QUADRANTS = ['bottom_left', 'top_left', 'bottom_right', 'top_right']
SENSORS_M51_BOTTOM_RIGHT = [48,49,50,51,52,53,54,55,63,62,61,60,59,58,57,56]
SENSORS_M51_TOP_RIGHT = [32,33,34,35,36,37,38,39,47,46,45,44,43,42,41,40]
SENSORS_M51_BOTTOM_LEFT = [16,17,18,19,20,21,22,23,31,30,29,28,27,26,25,24]
SENSORS_M51_TOP_LEFT = [0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8]
SENSORS_M51 = {'bottom_left': SENSORS_M51_BOTTOM_LEFT, 'top_left': SENSORS_M51_TOP_LEFT,
              'bottom_right': SENSORS_M51_BOTTOM_RIGHT, 'top_right': SENSORS_M51_TOP_RIGHT}

count = 0
data = {}

data5152 = {}

# get signal and normalized it.
for f in tqdm(files):
    array = np.load(f, allow_pickle=True).get('arr_0')
    for v in array:
        if 'X' in f:
            x = v[0]
            data5152[int(v[1])] = {'signal': x}
            count += 1
        if 'y' in f:
            pass
    if DEBUG and count>10:
        break
print('files loaded.')

# get peaks from signal.
count = 0
for p in tqdm(list(data5152.keys())):
    count += 1
    signal, median = _normalize_signal(data5152[p]['signal'])
    passage_peaks = get_peaks(signal, min_dist=200)

    for quadrant in QUADRANTS:
        s = signal[:, SENSORS_M51[quadrant]]
        row = {
            'signal': s,
            'median': median,
            'passage_peaks': passage_peaks,
            'n_passage_peaks': len(passage_peaks),
            'temperatures': signal[:, 64:]
        }
        data[('m5152', p, quadrant)] = row
    if DEBUG==True and len(data)>200:
        break

# keep [4, 5] passage peaks.
# from peaks get signal around peak.
peak_data = {}

for idx in data.keys():
    dataset, p, q = idx
    s = data[idx]
    if s['n_passage_peaks'] in passages_used:
        for axle, peak in enumerate(s['passage_peaks']):
            signal_around_peak = s['signal'][peak-time_size:peak+time_size]
            if signal_around_peak.shape[0] == 2*time_size:
                peak_data[(dataset, p, axle, q)] = {
                    'peak_time': peak,
                    'signal': signal_around_peak,
                    'n_passage_peaks': s['n_passage_peaks'],
                    'median': s['median'],
                    'temperatures': s['temperatures'][peak-time_size:peak+time_size]
                }
peaks = pd.DataFrame(peak_data).T
peaks.index.set_names(['dataset', 'timestamp', 'axle', 'quadrant'], inplace=True)

# get mapping_df infos.
mapping_df_ = pd.read_csv(mapping_path, sep=';', index_col=0)
mapping_df = mapping_df_.dropna(subset=['timestamp_m5152'])
mapping_df = mapping_df.drop_duplicates(subset=['timestamp_m5152'])
mapping_df = mapping_df[~mapping_df['poids_essieu_1'].astype(str).str.contains('ass')]
mapping_df['poids_essieu_1'] = pd.to_numeric(mapping_df['poids_essieu_1'])
mappind_df = mapping_df.dropna(subset=['poids_essieu_1'])
merged_df = pd.merge(
    peaks.reset_index(),
    mapping_df[['timestamp_m5152'] + [c for c in mapping_df.columns if 'poids' in c]],
    left_on='timestamp',
    right_on='timestamp_m5152'
)
for i, axle_n in enumerate(passages_used):
    if i==0:
        inconsistent_number_of_axles = (
            (merged_df['n_passage_peaks']==axle_n)
            & (merged_df['poids_essieu_{}'.format(axle_n)].isna())
        )
    else:
        inconsistent_number_of_axles = (
            (inconsistent_number_of_axles) |
            (
                (merged_df['n_passage_peaks']==axle_n)
                & (merged_df['poids_essieu_{}'.format(axle_n)].isna())
            )
        )
extracted_df = merged_df[~inconsistent_number_of_axles]

# delete 'passe trop vite'
for c in ['poids_essieu_{}'.format(i+1) for i in range(5)]:
    extracted_df[c] = pd.to_numeric(extracted_df[c], errors='coerce')
extracted_df['axle_mass'] = np.sum(np.eye(5)[extracted_df.axle.values] * extracted_df[['poids_essieu_{}'.format(i+1) for i in range(5)]].replace(np.nan, 0).values, axis=1)
extracted_df = extracted_df.drop([c for c in extracted_df.columns if 'poids' in c] + ['timestamp_m5152'], axis=1)
extracted_df = extracted_df.drop_duplicates(subset=['dataset', 'timestamp', 'axle', 'quadrant'])

# keep only passage with right number of axles
groupby_time = extracted_df.groupby('timestamp')
timestamps = groupby_time.first()[groupby_time.axle.nunique().isin(passages_used)].index.values
extracted_df = extracted_df[extracted_df.timestamp.isin(timestamps)]

# upload processed data
print('Uploading process dataset')
task.upload_artifact('extracted_df', [extracted_df])

print('Done')
