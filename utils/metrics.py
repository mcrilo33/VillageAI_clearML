#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import utils.q_outliers as q_outliers

def custom_mean_error_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred-y_true)/y_true, axis=-1)

def custom_under10_metric(y_true, y_pred):
    under_10 = (tf.abs(y_pred-y_true)/y_true) < 0.1
    return tf.reduce_mean(tf.cast(under_10, tf.float32), axis=-1)

def custom_under5_metric(y_true, y_pred):
    under_5 = (tf.abs(y_pred-y_true)/y_true) < 0.05
    return tf.reduce_mean(tf.cast(under_5, tf.float32), axis=-1)

def classif_metrics():
    class_METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc')
    ]
    return class_METRICS

def custom_BCE(y_true, y_pred):
    epsilon=0.001
    K=0
    bce = -1*((1+K)*y_true * tf.math.log(y_pred+epsilon) + (1-K)*(1-y_true) * tf.math.log(1-y_pred+epsilon))
    return tf.reduce_mean(bce, axis=-1)  # Note the `axis=-1`

def get_errors(y_test, y_pred):

    error_abs = np.abs(y_test-y_pred)
    error = error_abs/y_test
    
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(y_test.values[:, None], y_pred.values)
    slope = reg.coef_[0]

    results = {'Mean error (%)': 100*np.mean(error),
               'Max error (%)': 100*np.max(error),
               '% <5%': 100* np.mean(error < 0.05),
               '% <10%': 100* np.mean(error < 0.10),
               'Mean error (T)': np.mean(error_abs)/1000,
               'Max error (T)': np.max(error_abs)/1000,
               'U250 (%)': 100* np.mean(error_abs < 250),
               'U500 (%)': 100* np.mean(error_abs < 500),
               'U1000 (%)': 100* np.mean(error_abs < 1000),
               'Slope': slope,
               'Corrcoef': np.corrcoef(y_test, y_pred)[0, 1],
               'Sample size': len(y_test)}
    return results


def get_list_errors(y_tests, y_preds):

    errors = [get_errors(y_test, y_pred) for y_test, y_pred in zip(y_tests, y_preds)]
    if len(errors)==0:
        return [0]
    keys = errors[0].keys()
    values = [list(x.values()) for x in errors]
    means = np.mean(values, axis=0)
    mean_errors = {k: np.round(e, 2) for k, e in zip(keys, means)}    
    return mean_errors


def plot_predictions(y_test_df, truck_preds, save_path, label_column='axle_mass', merged_dataset=False):
    plt.style.use('seaborn')
    if merged_dataset:
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(truck_preds[label_column], truck_preds['pred'], '.')
        plt.title('Test set Trucks')
        plt.ylabel('prediction')
        plt.xlabel('mass')

        plt.subplot(1, 2, 2)
        plt.plot(truck_preds[label_column], truck_preds['pred_avg'], '.')
        plt.title('Test set Trucks - Dropout prediction')
        plt.ylabel('prediction')
        plt.xlabel('mass')

    else:
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.plot(y_test_df[label_column], y_test_df['pred'], '.')
        plt.title('Test set Axles')
        plt.ylabel('prediction')
        plt.xlabel('mass')

        plt.subplot(2, 2, 2)
        plt.plot(y_test_df[label_column], y_test_df['pred_avg'], '.')
        plt.title('Test set Axles - Dropout prediction')
        plt.ylabel('prediction')
        plt.xlabel('mass')

        plt.subplot(2, 2, 3)
        plt.plot(truck_preds[label_column], truck_preds['pred'], '.')
        plt.title('Test set Trucks')
        plt.ylabel('prediction')
        plt.xlabel('mass')

        plt.subplot(2, 2, 4)
        plt.plot(truck_preds[label_column], truck_preds['pred_avg'], '.')
        plt.title('Test set Trucks - Dropout prediction')
        plt.ylabel('prediction')
        plt.xlabel('mass')
    plt.savefig(save_path)
    plt.close()

def generate_dropout_predictions(val, model_dropout, n_dropout_preds):
    # Generating dropout predictions
    print('Generating dropout predictions...')
    # f = tf.keras.backend.function([l.input for l in model.layers if 'input' in l.name] + [keras.backend.learning_phase()], [l.output for l in model.layers if 'output' in l.name])
    all_dropout_preds = np.stack([np.sum(model_dropout.predict(val), axis=-1) for i in tqdm(range(n_dropout_preds))])

    return all_dropout_preds

def get_performance_by_mass_range(truck_preds, label_column='axle_mass', ranges=[10000, 20000, 30000, 40000, 50000]):
    perfs_per_mass_range = {}
    for ir in range(len(ranges)-1):
        range_preds = truck_preds[(truck_preds[label_column] < ranges[ir+1]) & (truck_preds[label_column] >= ranges[ir])]
        if len(range_preds):
            perfs_per_mass_range['{}->{}'.format(ranges[ir], ranges[ir+1])] = get_errors(range_preds[label_column], range_preds.pred_avg)
        
    return perfs_per_mass_range

def get_performance_by_axle(axle_preds, label_column='axle_mass'):
    perfs_per_axle = {}
    axles = sorted(axle_preds.axle.unique())
    for axl in axles:
        axl_preds = axle_preds[axle_preds.axle == axl]
        perfs_per_axle['Axle_{}'.format(axl)] = get_errors(axl_preds[label_column], axl_preds.pred)
    for axl in axles:
        axl_preds = axle_preds[axle_preds.axle == axl]
        perfs_per_axle['Axle_{}_dropout'.format(axl)] = get_errors(axl_preds[label_column], axl_preds.pred_avg)

    return perfs_per_axle

def get_errors_classif(y_test, y_pred, p=0.5):
    cm = confusion_matrix(y_test, y_pred > p)
    precision = sklearn.metrics.precision_score(y_test, y_pred > p)
    recall = sklearn.metrics.recall_score(y_test, y_pred> p)
    f1_score = sklearn.metrics.f1_score(y_test, y_pred >p)
    f2_score = sklearn.metrics.fbeta_score(y_test, y_pred >p, beta=2)
    true_negative, false_positive = cm[0]
    false_negative, true_positive = cm[1]
    
    results = {'precision': precision,
               'recall': recall,
               'f1_score': f1_score,
               'f2_score': f2_score,
               'True Neg': true_negative,
               'False Pos': false_positive,
               'False Neg': false_negative,
               'True Pos': true_positive}
    return results

def plot_predictions_class(y_test, y_pred, save_path, p=0.5):
    cm = confusion_matrix(y_test, y_pred > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def save_infos(logs_dir, model, model_dropout, dataset, val, y_test_df, scaling_factor=1e4, label_column='axle_mass', merged_dataset=False, n_dropout_preds=50):
    weights_path = os.path.join(logs_dir, 'model.weights')
    best_weights_path = os.path.join(logs_dir, 'best_model.weights')
    history_path = os.path.join(logs_dir, 'model_history.pkl')
    df_path = os.path.join(logs_dir, 'axle_preds.csv')
    truck_df_path = os.path.join(logs_dir, 'truck_preds.csv')
    figure_path = os.path.join(logs_dir, 'predictions_plots.png')
    perfs_path = os.path.join(logs_dir, 'performance.csv')
    perfs_per_mass_range_path = os.path.join(logs_dir, 'performance_mass_range.csv')
    perfs_per_axle_path = os.path.join(logs_dir, 'performance_axle.csv')

    # Save Model weights
    model.save_weights(weights_path)
    print('Model weights saved at {}'.format(weights_path))
    
    # Load the best model weigths
    model.load_weights(best_weights_path)
    model_dropout.load_weights(best_weights_path)
    print('Weights of the best model loaded')

    # Save Model history
    fhistory = open(history_path, 'wb')
    print('Model history saved at {}'.format(history_path))
    pkl.dump(model.history.history, fhistory)
    fhistory.close()
    
    all_dropout_preds = scaling_factor * generate_dropout_predictions(val, model_dropout, n_dropout_preds=n_dropout_preds)
    # Axle stats
    y_pred_avg = np.mean(all_dropout_preds, axis=0)
    y_pred = np.sum(model.predict(val) * scaling_factor, axis=1)
    y_true = y_test_df[label_column]
    y_test_df['pred'] = y_pred
    y_test_df['pred_avg'] = y_pred_avg
    y_test_df['error'] = (y_pred-y_true)/y_true
    y_test_df['std'] = np.sqrt(np.var(all_dropout_preds, axis=0))
    y_test_df.to_csv(df_path)
    print('Axle predictions saved at {}'.format(df_path))

    # Truck stats
    truck_preds = y_test_df.groupby('timestamp')[[label_column] + ['pred', 'pred_avg']].sum()
    truck_preds.to_csv(truck_df_path)
    print('Truck predictions saved at {}'.format(truck_df_path))

    # Write performance
    errors = get_errors(truck_preds[label_column], truck_preds.pred)
    errors = pd.Series(errors).rename('Preds', inplace=True)
    errors_avg = get_errors(truck_preds[label_column], truck_preds.pred_avg)
    errors_avg = pd.Series(errors_avg).rename('Preds_avg', inplace=True)
    results = pd.concat([errors, errors_avg], axis=1)
    results.to_csv(perfs_path, index=True)

    # Performance per mass range
    perfs_per_mass_range = get_performance_by_mass_range(truck_preds, label_column=label_column)
    pd.DataFrame(perfs_per_mass_range).to_csv(perfs_per_mass_range_path, float_format='%.2f')

    # Performance per q_outlier
    q_outliers_path = os.path.join(logs_dir, 'q_outliers.xlsx')
    writer = pd.ExcelWriter(q_outliers_path, engine='xlsxwriter')
    for method in [m for m in q_outliers.__dict__.keys() if m.startswith('___')]:
        df = q_outliers_analysis(
            q_outliers.__dict__[method](dataset),
            truck_preds,
            q_min=0.02,
            q_max=0.98,
            label_column=label_column
        )
        method = method[3:]
        df.to_excel(writer, sheet_name=method)
    writer.save()

    # Save p_outliers
    p_outliers = (truck_preds[label_column] - truck_preds.pred_avg)/truck_preds.pred_avg
    p_outliers = p_outliers[p_outliers>p_outliers.quantile(0.95)]
    p_outliers_path = os.path.join(logs_dir, 'p_outliers.csv'.format(method))
    p_outliers.to_csv(p_outliers_path)

    if not merged_dataset:
        # Performance per axle
        perfs_per_axle = get_performance_by_axle(y_test_df, label_column=label_column)
        pd.DataFrame(perfs_per_axle).to_csv(perfs_per_axle_path, float_format='%.2f')

    # Plot predictions
    plot_predictions(y_test_df, truck_preds, figure_path, label_column=label_column, merged_dataset=merged_dataset)


def save_infos_classif(logs_dir, model, model_dropout, dataset, val, y_test_df, label_column='axle_mass', merged_dataset=False, n_dropout_preds=50):
    weights_path = os.path.join(logs_dir, 'model.weights')
    best_weights_path = os.path.join(logs_dir, 'best_model.weights')
    history_path = os.path.join(logs_dir, 'model_history.pkl')
    truck_df_path = os.path.join(logs_dir, 'truck_preds.csv')
    perfs_path = os.path.join(logs_dir, 'performance.csv')
    figure_path = os.path.join(logs_dir, 'predictions_plots.png')

    # Save Model weights
    model.save_weights(weights_path)
    print('Model weights saved at {}'.format(weights_path))

    # Load the best model weigths
    model.load_weights(best_weights_path)
    print('Weights of the best model loaded')

    # Save Model history
    fhistory = open(history_path, 'wb')
    print('Model history saved at {}'.format(history_path))
    pkl.dump(model.history.history, fhistory)
    fhistory.close()

    all_dropout_preds = generate_dropout_predictions(val, model_dropout, n_dropout_preds=n_dropout_preds)
    # Axle stats
    y_pred_avg = np.mean(all_dropout_preds, axis=0)
    y_proba = np.sum(model.predict(val), axis=1)
    y_true = y_test_df[label_column]
    y_test_df['proba'] = y_proba
    y_test_df['proba_avg'] = y_pred_avg
    y_test_df['std'] = np.sqrt(np.var(all_dropout_preds, axis=0))

    perfs = get_errors_classif(y_true, y_proba)
    perfs_dropout = get_errors_classif(y_true, y_pred_avg)
    performance = pd.DataFrame([perfs, perfs_dropout], index=['y_proba', 'y_proba_avg']).T
    performance.to_csv(perfs_path)

    plot_predictions_class(y_true, y_proba, figure_path)

    y_test_df.drop('axle', axis=1).to_csv(truck_df_path)

    # TODO : ADAPT Q_OUTLIERS_ANALYSIS TO CLASSIFICATION
    # for method in [m for m in q_outliers.__dict__.keys() if m.startswith('___')]:
    #     df = q_outliers_analysis(
    #         q_outliers.__dict__[method](dataset),
    #         truck_preds,
    #         q_min=0.02,
    #         q_max=0.98,
    #         label_column=label_column
    #     )
    #     q_outliers_path = os.path.join(logs_dir, 'q_out{}.csv'.format(method))
    #     df.to_csv(q_outliers_path)    


def save_exp_performance(
    exp_directories,
    label_column='axle_mass',
    output_file=None,
    ranges=[10000,
            20000,
            30000,
            40000,
            50000],
    force=False
):

    truck_dfs = []
    for i, folder in enumerate(exp_directories):
        path = os.path.join(folder, 'truck_preds.csv')
        if os.path.isfile(path):
            df = pd.read_csv(path)
            if isinstance(label_column, list):
                df['target'] = df[label_column[i]].copy()
            truck_dfs.append(df)
        elif not force:
            raise ValueError('Missing truck_preds.csv at {}'.format(path))
    
    if isinstance(label_column, list):
        label_column = 'target'

    # Perf all splits
    perfs_dropout = get_list_errors([tdf[label_column] for tdf in truck_dfs], [tdf.pred_avg for tdf in truck_dfs])
    perfs = get_list_errors([tdf[label_column] for tdf in truck_dfs], [tdf.pred for tdf in truck_dfs])
    macro_perfs = {'pred': perfs, 'pred_dropout': perfs_dropout}

    # Perf per experience
    preds_error = {'{}_pred'.format(os.path.basename(exp_directories[i])): get_list_errors([tdf[label_column]], [tdf.pred]) for i, tdf in enumerate(truck_dfs)}
    preds_dropout_error = {'{}_pred_dropout'.format(os.path.basename(exp_directories[i])): get_list_errors([tdf[label_column]], [tdf.pred_avg]) for i, tdf in enumerate(truck_dfs)}

    # Perf per mass range
    perfs_per_mass_range = {}
    for range_idx in range(len(ranges)-1):
        m, M = ranges[range_idx], ranges[range_idx+1]
        range_dfs = [tdf[(tdf[label_column] >= m) & (tdf[label_column] < M)] for tdf in truck_dfs]
        perfs_per_mass_range['{}->{}'.format(m, M)] = get_list_errors([tdf[label_column] for tdf in range_dfs], [tdf.pred_avg for tdf in range_dfs])

    macro_perfs.update(perfs_per_mass_range)
    macro_perfs.update(preds_error)
    macro_perfs.update(preds_dropout_error)

    performance = pd.DataFrame(macro_perfs)
    if output_file:
        performance.to_csv(output_file)
        print('Performance saved at {}'.format(output_file))
    return performance


def q_outliers_analysis(values, preds, q_min=0.02, q_max=0.98,
                        label_column='axle_mass'):
    
    def _get_outliers(col, q_min=0.01, q_max=0.99):

        if isinstance(col, list):
            for i,c in enumerate(col):
                if i==0:
                    out_bol = ((c<c.quantile(q_min)) | (c>c.quantile(q_max)))
                else:
                    out_bol = (out_bol) | ((c<c.quantile(q_min)) | (c>c.quantile(q_max)))
                
            return col[0][out_bol], col[0][~out_bol]
                    
            min_outliers = col[col<col.quantile(q_min)]
            max_outliers = col[col>col.quantile(q_max)]
            not_outliers = col[(col>=col.quantile(q_min)) & (col<=col.quantile(q_max))]
            
        min_outliers = col[col<col.quantile(q_min)]
        max_outliers = col[col>col.quantile(q_max)]
        not_outliers = col[(col>=col.quantile(q_min)) & (col<=col.quantile(q_max))]

        return pd.concat([min_outliers,max_outliers]), not_outliers

    df = []
    outliers, not_outliers = _get_outliers(values, q_min=q_min, q_max=q_max)
    not_outliers_index = list(set(not_outliers.index).intersection(set(preds.index)))

    perfs = pd.Series(get_errors(
        preds['pred_avg'].loc[not_outliers_index].dropna(),
        preds[label_column].loc[not_outliers_index].dropna()
    ))
    perfs.rename('All - q_outliers', inplace=True)
    perfs_all = pd.Series(get_errors(
        preds['pred_avg'],
        preds[label_column]
    ))
    perfs_all.rename('All', inplace=True)
    perfs['Removed (%)'] = (1 - perfs['Sample size']/perfs_all['Sample size'])*100
    perfs_all['Removed (%)'] = 0
    df.append(perfs)
    df.append(perfs_all)
    
    ranges = [10000, 20000, 30000, 40000, 50000]
    perfs_list = get_performance_by_mass_range(
        preds.loc[not_outliers_index].dropna(),
        ranges=ranges,
        label_column=label_column
    )
    perfs_all_list = get_performance_by_mass_range(
        preds,
        ranges=ranges,
        label_column=label_column
    )
    for i,p in enumerate(perfs_list):
        range_name = '[{} -> {}] '.format(ranges[i], ranges[i+1])
        perfs = perfs_list[p]
        perfs_all = perfs_all_list[p]
        perfs['Removed (%)'] = (1 - perfs['Sample size']/perfs_all['Sample size'])*100
        perfs_all['Removed (%)'] = 0
        perfs = pd.Series(perfs)
        perfs.rename(range_name + ' - q_outliers', inplace=True)
        perfs_all = pd.Series(perfs_all)
        perfs_all.rename(range_name, inplace=True)
        df.append(perfs)
        df.append(perfs_all)

    return pd.concat(df, axis=1)

