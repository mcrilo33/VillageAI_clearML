#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import pprint as pp
import utils.data_processing as data_processing
from clearml import Task
from clearml.automation.controller import PipelineController


if __name__ == "__main__":
    print(open('./utils/ascii-art.txt', 'r').read())
    parser = argparse.ArgumentParser(
        description='Altaroad data processing step.',
    )
    parser.add_argument(
        'config',
        metavar='Config',
        type=str,
        help='YAML config path.',
    )
    parser.add_argument(
        'data',
        metavar='Data',
        type=str,
        help='Data path to load *.npz files : (passed through glob so it can be a regex)',
    )
    parser.add_argument(
        'label',
        metavar='Label',
        type=str,
        help='Label data path to load *.csv file',
    )
    parser.add_argument(
        '--output',
        '-o',
        metavar='Output file',
        type=str,
        help='Output data path.'
    )
    parser.add_argument(
        '--debug',
        help='Debug mode',
        action='store_true'
    )
    args = vars(parser.parse_args())
    config_path = args['config']
    data_path = args['data']
    label_path = args['label']
    debug_mode = args['debug']

    if 'output' in args:
        save_path = args['output']
    else:
        save_path = None

    # data_processing.pipeline(
        # config_path,
        # data_path,
        # label_path,
        # save_path=save_path,
        # DEBUG=debug_mode
    # )

    PROJECT_NAME = 'AltaroadCML'
    task = Task.init(project_name=PROJECT_NAME, task_name='Data Pipeline',
                     task_type=Task.TaskTypes.data_processing, reuse_last_task_id=False)

    print('Loading config at {}'.format(config_path))
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pp.pprint(config)
    task.connect_configuration(
        config,
        name='Config',
        description='Specify each processing steps with their parameters.'
    )

    pipe = PipelineController(default_execution_queue='default', add_pipeline_tags=False)
    pipe.add_step(
        name='Read *.npz files', base_task_project=PROJECT_NAME,
        base_task_name='read_npz',
        parameter_override={
            'data_path': data_path,
            'mapping_path': label_path,
            'passages_used': config['passages_used'],
            'time_size': config['time_size'],
            'DEBUG': debug_mode
        }
    )
    pipe.add_step(
        name='Lateral position detection', base_task_project=PROJECT_NAME,
        parents=['read_npz'],
        base_task_name='lateral_pos_detection',
        parameter_override={
            'extracted_df_task_id': '68cec756d79147de9b2970e0ffb1564d',
            'config': config,
        }
    )
    # Starting the pipeline (in the background)
    print('Starting pipe...')
    pipe.start()
    # Wait until pipeline terminates
    pipe.wait()
    print('Pipe finished.')
    # cleanup everything
    pipe.stop()
    print('Pipe done')
