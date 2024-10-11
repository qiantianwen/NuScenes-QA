# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import os, copy
from src.datasets.nuscenes_qa import NuScenes_QA
from src.execution.train_engine import train_engine
from src.execution.train_engine import test_engine
import ipdb

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading dataset........')
        self.dataset = NuScenes_QA(__C)

        # If trigger the evaluation after every epoch
        # Will create a new cfgs with RUN_MODE = 'val'
        self.dataset_eval = None
        if __C.EVAL_FREQUENCY != 0:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation........')
            self.dataset_eval = NuScenes_QA(__C_eval)


    def run(self, run_mode):
        if run_mode == 'train':
            if self.__C.RESUME is False:
                self.empty_log(self.__C.VERSION)
            train_engine(self.__C, self.dataset, self.dataset_eval)

        elif run_mode == 'val':
            test_engine(self.__C, self.dataset, save_eval_result=True)

        elif run_mode == 'test':
            test_engine(self.__C, self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file........')
        if (os.path.exists(self.__C.LOG_PATH + '/log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + '/log_run_' + version + '.txt')
        print('Finished!')
        print('')