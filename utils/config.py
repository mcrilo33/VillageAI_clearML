#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import yaml
import pprint as pp
import utils.metrics

def load_metrics(metric_names):

    metrics = []
    for name in metric_names:
        if re.search(r'^metrics\.', name):
            name = re.sub(r'^metrics\.', '', name)
            metrics.append(utils.metrics.__dict__[name])
        else:
            metrics.append(name)
    return metrics

def load_experiment_config(path):

    path = re.sub('.yaml', '', path) + '.yaml'
    print('Loading experiment {}'.format(path))
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pp.pprint(config)

    if 'metrics' in config:
        config['metrics'] = load_metrics(config['metrics'])
    else:
        config['metrics'] = []

    return config
