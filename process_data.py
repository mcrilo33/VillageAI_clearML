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
        '--config',
        '-c',
        metavar='Config',
        type=str,
        help='YAML config path.'
    )
    parser.add_argument(
        '--data',
        '-d',
        metavar='Data',
        type=str,
        help='Data path to load *.npz files : (passed through glob so it can be a regex)'
    )
    parser.add_argument(
        '--label',
        '-l',
        metavar='Label',
        type=str,
        help='Label data path to load *.csv file'
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
        '-b',
        metavar='Debug',
        type=bool,
        default=False,
        help='Debug mode'
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

    PROJECT_NAME = 'Altaroad'
    task = Task.init(project_name=PROJECT_NAME, task_name='Data Processing',
                     task_type=Task.TaskTypes.data_processing, reuse_last_task_id=False)

    print('Loading config at {}'.format(config_path))
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    pp.pprint(config)
    task.upload_artifact('config', artifact_object=config)

    pipe = PipelineController(default_execution_queue='default', add_pipeline_tags=False)
    pipe.add_step(name='stage_data', base_task_project=PROJECT_NAME, base_task_name='pipeline step 1 dataset artifact')
    # pipe.add_step(
        # name='read_data', base_task_project=PROJECT_NAME,
        # base_task_name='read_npz',
        # parameter_override={
            # 'data_path': data_path,
            # 'mapping_path': label_path,
            # 'passages_used': 5,
            # 'time_size': 150,
            # 'DEBUG': debug_mode
        # }
    # )
    # Starting the pipeline (in the background)
    print('Starting pipe...')
    pipe.start()
    # Wait until pipeline terminates
    pipe.wait()
    print('Pipe finished.')
    # cleanup everything
    pipe.stop()
    print('Pipe done')
