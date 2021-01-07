import numpy as np
import pandas as pd

def get_original_dataset(df):

    return df[~df[df.columns[df.columns.str.contains('_augm')]].any(axis=1)]

def train_test_split(dataset, timestamp_column='timestamp', test_size=0.2):

    unique_timestamps = np.unique(dataset[timestamp_column])
    np.random.shuffle(unique_timestamps)
    train_timestamps = unique_timestamps[:-int(test_size*len(unique_timestamps))]
    test_timestamps = unique_timestamps[-int(test_size*len(unique_timestamps)):]

    return train_timestamps, test_timestamps
    
def get_signal_and_label_onehot_for_model(dataset, label_regex='mass', merged=False):

    def _axle_to_one_hot(x):
        new = np.zeros(5)
        new[x-1] = 1
        return new
    
    label_columns = [c for c in dataset.columns if label_regex in c]

    signals = np.stack(dataset.signal.values)
    labels = dataset[label_columns].astype(float).values
    if not merged:
        one_hots = np.stack(dataset.axle.apply(_axle_to_one_hot))
    else:
        one_hots = np.stack(dataset.n_axles.apply(_axle_to_one_hot))

    return signals, labels, one_hots, label_columns

def merge_axles_to_passages(dataset):
    augmented_columns = list(dataset.columns[dataset.columns.str.contains('_augm')])
    groupby = dataset.groupby(['timestamp'] + augmented_columns)

    length = len(dataset.signal.iloc[0])
    dataset.signal = dataset.signal.apply(lambda x: np.reshape(np.transpose(x, [0, 2, 1]), (length, 64, 1)))

    dataset = pd.concat([groupby.signal.apply(np.stack), groupby.axle_mass.sum(), groupby.axle_mass.apply(np.stack), groupby.peak_time.apply(np.stack), groupby.axle.max()], axis=1)
    dataset.columns = ['signal', 'total_mass', 'mass_list', 'peak_times', 'n_axles']
    dataset['n_axles'] = dataset.mass_list.apply(len)
    dataset = dataset.reset_index()

    axle_mass_df = pd.DataFrame(dataset.mass_list.tolist(), index=dataset.index)
    axle_mass_df.columns = ['mass_axle_{}'.format(i) for i in axle_mass_df.columns]
    dataset = pd.concat([dataset, axle_mass_df], axis=1)

    return dataset   

def generate_bias_and_weights(target):
    n = len(target)
    n_pos = sum(target)
    n_neg = n - n_pos

    initial_bias = np.log([n_pos/n_neg])
    output_bias = initial_bias
    print(n, n_pos, n_neg, output_bias)
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / n_neg)*(n)/2.0
    weight_for_1 = (1 / n_pos)*(n)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return output_bias, class_weight    