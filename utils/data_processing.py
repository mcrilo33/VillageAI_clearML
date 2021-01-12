#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import peakutils
import yaml
import re
import json
import pprint as pp
import tensorflow as tf
import pandas as pd
import numpy as np
# import utils.q_outliers as q_outliers
from glob import glob
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle
from clearml import Task, StorageManager
from sklearn.model_selection import train_test_split

def lateral_pos_detection(df, threshold=0.2, ma_size=5):
    
    def _lateral_pos_detection(row, threshold, ma_size):
        signal = np.mean(np.abs(row['signal']), axis=0)
        pos = {}
        amp = np.max(signal)
        threshold = threshold*amp
        where = np.argwhere(signal>threshold)
        left, right = np.min(where[:,0]), np.max(where[:,0])
        pos['left_pos'] = left
        pos['right_pos'] = right
        pos['width'] = right - left
        
        signal_padded = np.pad(signal, (ma_size//2, ma_size-1-ma_size//2), mode='linear_ramp', end_values=signal.min())
        signal_smooth = np.convolve(signal_padded, np.ones((ma_size,))/ma_size, mode='valid')
        pos['max_pos'] = np.argmax(signal_smooth)
        
        pos['left_width'] = pos['max_pos'] - pos['left_pos']
        pos['right_width'] = pos['right_pos'] - pos['max_pos']

        new_row = {**row, **pos}
        
        return new_row

    new_df = df.apply(lambda x: _lateral_pos_detection(x, threshold, ma_size=ma_size), axis=1, result_type='expand')
    return new_df

def remove_cropped(df):
    
    # AXLE 1 IS VERY BADLY DETECTED  -> TODO
    # df = df[((df['axle']==1) | ((df['right_pos']<15) & (df['left_pos']>0)))]
    # (df['axle']==1) | ((df['max_pos']>=4) & (df['max_pos']<=11))
    # is_full = df.groupby(['timestamp', 'axle']).axle.count().rename("count")
    # full_passages = is_full[is_full == 4].reset_index().drop('count', axis=1)
    # df = pd.merge(df, full_passages, on=['timestamp', 'axle'])

    cropped_axles = (df['axle'] != 1) & ((df['max_pos'] < 4) | (df['max_pos'] > 11))
    cropped_passages = df[cropped_axles].timestamp.unique()
    clean_passages = pd.DataFrame(list(set(df.timestamp) - set(cropped_passages)), columns=['timestamp'])
    df = pd.merge(df, clean_passages, on=['timestamp'])

    return df
    # return df[((df['right_pos']<15) & (df['left_pos']>0))]

def centered_only(df):

    not_centered_axles = (df['axle'] != 1) & ((df['max_pos'] >= 10) | (df['max_pos'] <= 5))
    not_centered_passages = df[not_centered_axles].timestamp.unique()
    clean_passages = pd.DataFrame(list(set(df.timestamp) - set(not_centered_passages)), columns=['timestamp'])
    df = pd.merge(df, clean_passages, on=['timestamp'])

    return df

def speed(df, coeffs, **kwargs):
    
    def _normalize_time_axis(signal, min_shape):

        to_delete = np.rint(
            np.linspace(0, len(signal)-1, np.shape(signal)[0]-min_shape)
        ).astype('int')
        return np.delete(signal, to_delete, axis=0)

        signal
    dfs = [df]
    signal = np.stack(df['signal'].values)
    signal = np.expand_dims(signal, axis=3)
    temperatures = np.stack(df['temperatures'].values)
    temperatures = np.expand_dims(temperatures, axis=3)
    if 'speed_augm' not in df:
        df['speed_augm'] = False
    
    for coeff in coeffs:
        new_signal = tf.image.resize(
            signal,
            (round(signal.shape[1]*coeff), signal.shape[2]),
            preserve_aspect_ratio=False,
            **kwargs
        )
        new_temperatures = tf.image.resize(
            signal,
            (round(temperatures.shape[1]*coeff), temperatures.shape[2]),
            preserve_aspect_ratio=False,
            **kwargs
        )
        new_df = df.copy(deep=True)
        new_df['signal'] = [x for x in np.squeeze(new_signal)]
        new_df['temperatures'] = [x for x in np.squeeze(new_temperatures)]
        new_df['speed_augm'] = coeff
        dfs.append(new_df)
        
    new_df = pd.concat(dfs).reset_index(drop=True)
    min_shape = new_df.signal.apply(np.shape).min()[0]
    new_df['signal'] = new_df.signal.apply(
        lambda x: _normalize_time_axis(x, min_shape)
    )
    new_df['temperatures'] = new_df.temperatures.apply(
        lambda x: _normalize_time_axis(x, min_shape)
    )

    return new_df

def smoothing(df, kernel_name, **kwargs):
    
    from scipy.signal import gaussian
    
    duration = kwargs['duration'] if 'duration' in kwargs else 100
    std = kwargs['std'] if 'std' in kwargs else 50
    
    if kernel_name=='uniform':
        kernel = [1/duration]*duration
    elif kernel_name=='gaussian':
        kernel = gaussian(duration, std=std)
        kernel /= np.sum(kernel)
    else:
        raise ValueError('bad kernel_name.')
    
    df['signal'] = df.signal.apply(
        lambda s: np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=s)
    )
    
    return df

def drop_threshold(df, threshold=0.1):

    return df[(df.signal.apply(lambda x: np.max(np.abs(x)))>threshold)]

def downsampling(df, downsample=5, augmented=False):
    
    if df.signal.iloc[0].shape[0]%downsample!=0:
        raise ValueError(
            'Time axis length ({}) must be a multiple of downsample ({}).'.format(
                df.signal.iloc[0].shape[0],
                downsample
            )
        )
        
    df = df.copy(deep=True)
    dfs = [df]
    
    if augmented:
        if 'downsample_augm' not in df:
            df['downsample_augm'] = 0
        for i in range(1,downsample):
            new_df = df.copy(deep=True)
            new_df['signal'] = new_df['signal'].apply(lambda x: x[i::downsample]) 
            new_df['temperatures'] = new_df['temperatures'].apply(lambda x: x[i::downsample]) 
            new_df['downsample_augm'] = i
            dfs.append(new_df)
    df['signal'] = df['signal'].apply(lambda x: x[::downsample]) 
    df['temperatures'] = df['temperatures'].apply(lambda x: x[::downsample]) 
    dfs[0] = df
        
    return pd.concat(dfs).reset_index(drop=True)

def symmetries(df, sym1=True, sym2=True, sym3=True):
    
    def _sym1(row): # symmetrized between left mat and right mat
        tmp = row.loc[row['quadrant']=='top_left', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='top_left', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='top_right', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='top_right', 'signal'].iloc[0] = tmp
        tmp = row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0] = tmp
        
        return row
        
    def _sym2(row): # symmetrized for each mat along vertical axis
        row.loc[row['quadrant']=='top_left', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='top_left', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='top_right', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='top_right', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0][:,::-1]
        
        return row
        
    def _sym3(row): # symmetrized along horizontal axis <=> inverse line1 with line2
        tmp = row.loc[row['quadrant']=='top_left', 'signal'].iloc[0]
        row.loc[row['quadrant']=='top_left', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_left', 'signal'].iloc[0] = tmp
        tmp = row.loc[row['quadrant']=='top_right', 'signal'].iloc[0]
        row.loc[row['quadrant']=='top_right', 'signal'].iloc[0] = \
            row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0][:,::-1]
        row.loc[row['quadrant']=='bottom_right', 'signal'].iloc[0] = tmp
        
        return row
                                                        
    augmented_columns = list(df.columns[df.columns.str.contains('_augm')])
    if sym1:
        if 'sym1_augm' not in df:
            df['sym1_augm'] = False
        df_sym1 = df.groupby(['timestamp','axle'] + augmented_columns).apply(_sym1)
        df_sym1['sym1_augm'] = True
    else:
        df_sym1 = None
    if sym2:
        if 'sym2_augm' not in df:
            df['sym2_augm'] = False
        df_sym2 = df.groupby(['timestamp','axle'] + augmented_columns).apply(_sym2)
        df_sym2['sym2_augm'] = True
    else:
        df_sym2 = None
    if sym3:
        if 'sym3_augm' not in df:
            df['sym3_augm'] = False
        df_sym3 = df.groupby(['timestamp','axle'] + augmented_columns).apply(_sym3)
        df_sym3['sym3_augm'] = True
    else:
        df_sym3 = None
                                                        
    return pd.concat([df,df_sym1,df_sym2,df_sym3]).reset_index(drop=True)
        
def translations(df, coeffs=[-3, 3], drop_cropped=True):
    
    def _translate(row, coeff, drop_cropped):
        s = row['signal']
        if coeff<0:
            row['signal'] = np.hstack([s[:,-coeff:], np.zeros((s.shape[0],-coeff))])
        else:
            row['signal'] = np.hstack([np.zeros((s.shape[0],coeff)),s[:,:-coeff]])
        row['left_pos'] += coeff
        row['right_pos'] += coeff
        return row
        
    dfs = [df]
    
    if 'translation_augm' not in df:
        df['translation_augm'] = 0
    for coeff in coeffs:
        new_df = df.copy(deep=True)
        new_df = new_df.apply(lambda x: _translate(x,coeff,drop_cropped), axis=1)
        if drop_cropped:
            # AXLE 1 IS VERY BADLY DETECTED  -> TODO
            # new_df[((new_df['axle']==1) | ((new_df['right_pos']<15) & (new_df['left_pos']>0)))]
            # new_df = new_df[((new_df['axle']==1) | ((new_df['max_pos']>=4) & (new_df['max_pos']<=11)))]
            # new_df[((new_df['right_pos']<15) & (new_df['left_pos']>0))]
            cropped_axles = (df['axle'] != 1) & ((df['max_pos'] < 4) | (df['max_pos'] > 11))
            cropped_passages = df[cropped_axles].timestamp.unique()
            clean_passages = pd.DataFrame(list(set(df.timestamp) - set(cropped_passages)), columns=['timestamp'])
            df = pd.merge(df, clean_passages, on=['timestamp'])

        new_df['translation_augm'] = coeff
        dfs.append(new_df)
    
    return pd.concat(dfs).reset_index(drop=True)

def last_axles_only(df):
    return df[df['axle'] > 1]

def extract_quadrants(df):

    augmented_col = df.columns[df.columns.str.contains('_augm')]
    grouped_by = df.groupby(['timestamp', 'axle'] + list(augmented_col))
    df = pd.concat([
            grouped_by.signal.apply(np.stack),
            grouped_by.temperatures.apply(lambda x: np.stack(x)[0]),
            grouped_by[[
                'peak_time',
                'axle_mass',
                'width',
                'left_width',
                'right_width',
                'max_pos',
                'median'
            ]].first(),
            grouped_by.axle.count().rename("count")
        ],
        axis=1
    ).reset_index()

    df['signal'] = df.signal.apply(lambda x : np.transpose(x, axes=[1, 2, 0]))

    return df

def add_fake_axles(df):

    def _add_fake_axles(gb):

        for i in range(max_axles - len(gb)):
            new_row = gb.iloc[-1]
            new_row.loc['axle'] = i + len(gb)
            new_row.loc['signal'] = np.zeros(new_row.signal.shape)
            new_row.loc['axle_mass'] = 0
            gb = gb.append(new_row)
        return gb

    groupby_n_axles = df.groupby('timestamp').axle.nunique()
    max_axles = groupby_n_axles.max()
    missing_axles_times = groupby_n_axles.loc[groupby_n_axles<max_axles].index.values
    new_df = df.loc[df.timestamp.isin(missing_axles_times)].groupby('timestamp').apply(_add_fake_axles).reset_index(drop=True)
    
    return pd.concat([df.loc[~df.timestamp.isin(missing_axles_times)], new_df])

def pipeline(dataset, config, save_path=None):

    print('Computing lateral position...')
    if 'lateral_pos_detection' in config and config['lateral_pos_detection']:
        dataset = lateral_pos_detection(
            dataset,
            **config['lateral_pos_detection']
        )
    else:
        dataset = lateral_pos_detection(dataset)
    if 'drop_threshold' in config and config['drop_threshold']:
        print('Removing under threshold...')
        dataset = drop_threshold(
            dataset,
            **config['drop_threshold']
        )
        print('New shape : {}'.format(dataset.shape))
    if 'remove_cropped' in config and config['remove_cropped']:
        print('Removing cropped...')
        dataset = remove_cropped(dataset)
        print('New shape : {}'.format(dataset.shape))
    if 'centered_only' in config and config['centered_only']:
        print('Keeping centered trajectories only...')
        dataset = centered_only(dataset)
        print('New shape : {}'.format(dataset.shape))
    if 'speed' in config and config['speed']:
        print('Speed augmentation...')
        dataset = speed(dataset, **config['speed'])
        print('New shape : {}'.format(dataset.shape))
    if 'smoothing' in config and config['smoothing']:
        print('Smoothing...')
        dataset = smoothing(dataset, **config['smoothing'])
        print('New shape : {}'.format(dataset.shape))
    if 'downsampling' in config and config['downsampling']:
        print('Downsampling...')
        dataset = downsampling(dataset, **config['downsampling'])
        print('New shape : {}'.format(dataset.shape))
    if 'symmetries' in config and config['symmetries']:
        print('Symmetry augmentation...')
        dataset = symmetries(dataset, **config['symmetries'])
        print('New shape : {}'.format(dataset.shape))
    if 'translations' in config and config['translations']:
        print('Translations...')
        dataset = translations(dataset, **config['translations'])
        print('New shape : {}'.format(dataset.shape))
    if 'last_axles_only' in config and config['last_axles_only']:
        print('Keeping last axles only')
        dataset = last_axles_only(dataset)
        print('New shape : {}'.format(dataset.shape))
    print('Extracting quadrants...')
    dataset = extract_quadrants(dataset)
    print('New shape : {}'.format(dataset.shape))
    if 'add_fake_axles' in config and config['add_fake_axles']:
        print('Adding fake axles to normalize the inputs...')
        dataset = add_fake_axles(dataset)
        print('New shape : {}'.format(dataset.shape))

    if save_path:
        print('Saving...')
        # q_outliers_path = os.path.basename(save_path)
        # q_outliers_path = re.sub(r'\..*$', '', q_outliers_path) + '_q.csv'

        # q_outliers_path = os.path.join(
            # './datasets/q_outliers/', q_outliers_path
        # )
        # os.makedirs(os.path.dirname(q_outliers_path), exist_ok=True)
        dataset.to_pickle(save_path)
        # save_q_outliers(dataset, q_outliers_path)
        print('Dataset saved at {}'.format(save_path))
        # print('Q_outliers saved at {}'.format(q_outliers_path))
        

    return dataset

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
    'extracted_df_url': 'https://files.community.clear.ml/AltaroadCML/read_npz.e4e8ec4d4de740c2895ddc562a79bc0e/artifacts/extracted_df/extracted_df.pkl',
    'config_path': '/Users/mcrilo33/Repos/VillageAI_clearML/configs/datasets/GSD.yaml',
}

config_path = \
    '/Users/mcrilo33/Repos/VillageAI_clearML/configs/datasets/GSD.yaml'

# store arguments, later we will be able to change them from outside the code
task.connect(args)
print('Arguments: {}'.format(args))

# only create the task, we will actually execute it later
task.execute_remotely()

# args['config'] = json.loads(args['config'])
extracted_pickle = StorageManager.get_local_copy(remote_url=args['extracted_df_url'])
extracted_df = pickle.load(open(extracted_pickle, 'rb'))
extracted_df = extracted_df[0]

print('Loading config at {}'.format(config_path))
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pp.pprint(config)
task.connect_configuration(
    config,
    name='Config',
    description='Specify each processing steps with their parameters.'
)

dataset = pipeline(extracted_df, config)

# upload processed data
print('Uploading processed dataset')
task.upload_artifact('dataset', [dataset])
