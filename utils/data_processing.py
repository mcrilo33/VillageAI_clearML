#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import peakutils
import yaml
import re
import pprint as pp
import tensorflow as tf
import utils.q_outliers as q_outliers
from glob import glob
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def read_m5152_npz(files, MAPPING_PATH, passages_used, time_size=150, DEBUG=False):

    def get_peaks(x, dropped_channels=[], min_dist=170, thres=0.25, abs=True):
        x = np.delete(x, dropped_channels, 1)
        return peakutils.indexes(np.mean(np.abs(x) if abs else np.maximum(x, 0), axis=1), 
                                 thres=thres, min_dist=min_dist)

    def _normalize_signal(x):
        median = np.median(x, axis=0)
        return x - median, median.mean()

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
    mapping_df_ = pd.read_csv(MAPPING_PATH, sep=';', index_col=0)
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

    return extracted_df

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

def save_q_outliers(df, path):

    q_outliers_df = df[['timestamp']].groupby('timestamp').first()
    for method in [m for m in q_outliers.__dict__.keys() if m.startswith('___')]:
        col = q_outliers.__dict__[method](df)
        method = method[3:]
        q_outliers_df[method] = False
        q_min=0.02
        q_max=0.98
        q_outliers_df[method].loc[
            (col>col.quantile(q_max)) | (col<col.quantile(q_min))
        ] = True
    q_outliers_df.to_csv(path)

def pipeline(config_path, data_path, label_path, save_path=None, save=True, DEBUG=False):

    if DEBUG:
        print('->> Warning in DEBUG MODE <<-\n')
    print('Loading config at {}'.format(config_path))
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pp.pprint(config)

    print('Loading data with glob({})'.format(data_path))
    files = sorted(glob(data_path))
    dataset = read_m5152_npz(
        files,
        label_path,
        config['passages_used'],
        time_size=config['time_size'],
        DEBUG=DEBUG
    )

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

    if save:
        print('Saving...')
        if save_path:
            q_outliers_path = os.path.basename(save_path)
            q_outliers_path = re.sub(r'\..*$', '', q_outliers_path) + '_q.csv'
        else:
            filename = re.sub(r'\.yaml', '', os.path.basename(config_path))
            save_path = './datasets/processed/' + filename + '.pkl'
            q_outliers_path = filename + '_q.csv'
        q_outliers_path = os.path.join(
            './datasets/q_outliers/', q_outliers_path
        )
        os.makedirs(os.path.dirname(q_outliers_path), exist_ok=True)
        dataset.to_pickle(save_path)
        save_q_outliers(dataset, q_outliers_path)
        print('Dataset saved at {}'.format(save_path))
        print('Q_outliers saved at {}'.format(q_outliers_path))
        

    return dataset
