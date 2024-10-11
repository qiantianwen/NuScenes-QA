# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import os

class PATH:
    def __init__(self):
        self.init_path()


    def init_path(self):

        self.DATA_ROOT = './data'


        self.FEATS_PATH = {
            'BEVDet': {
                'train': self.DATA_ROOT + '/features/BEVDet',
                'val': self.DATA_ROOT + '/features/BEVDet'},
            'CenterPoint': {
                'train': self.DATA_ROOT + '/features/CenterPoint',
                'val': self.DATA_ROOT + '/features/CenterPoint'},
            'MSMDFusion': {
                'train': self.DATA_ROOT + '/features/MSMDFusion',
                'val': self.DATA_ROOT + '/features/MSMDFusion'}
        }


        self.RAW_PATH = {
            'train': self.DATA_ROOT + '/questions' + '/NuScenes_train_questions.json',
            'val': self.DATA_ROOT + '/questions' + '/NuScenes_val_questions.json'}


        self.SPLIT = {
            'train': 'train',
            'val': 'val',
        }


        self.LOG_PATH = './outputs/log'
        self.CKPTS_PATH = './outputs/ckpts'
        self.RESULT_PATH = './outputs/result'


        if 'log' not in os.listdir('./outputs'):
            os.mkdir('./outputs/log')

        if 'ckpts' not in os.listdir('./outputs'):
            os.mkdir('./outputs/ckpts')
        
        if 'result' not in os.listdir('./outputs'):
            os.mkdir('./outputs/result')


    def check_path(self, vis_feat):
        print('Checking Data Path ........')

        
        for item in self.FEATS_PATH[vis_feat]:
            if not os.path.exists(self.FEATS_PATH[vis_feat][item]):
                print(self.FEATS_PATH[vis_feat][item], 'NOT EXIST')
                exit(-1)

        for item in self.RAW_PATH:
            if not os.path.exists(self.RAW_PATH[item]):
                print(self.RAW_PATH[item], 'NOT EXIST')
                exit(-1)

        print('Data Path Checking Finished!')
        print('')
