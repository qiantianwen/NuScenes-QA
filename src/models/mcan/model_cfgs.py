# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

from src.configs.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.LAYER = 6
        self.HIDDEN_SIZE = 512
        self.BBOXFEAT_EMB_SIZE = 512
        self.FF_SIZE = 2048
        self.MULTI_HEAD = 8
        self.DROPOUT_R = 0.1
        self.FLAT_MLP_SIZE = 512
        self.FLAT_GLIMPSES = 1
        self.FLAT_OUT_SIZE = 1024
        self.USE_BBOX_FEAT = False
        self.BBOX_NORMALIZE = False
