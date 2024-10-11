# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

from src.configs.base_cfgs import BaseCfgs


class Cfgs(BaseCfgs):
    def __init__(self):
        super(Cfgs, self).__init__()

        self.HIDDEN_SIZE = 1024
        self.DROPOUT_R = 0.2
        self.CLASSIFER_DROPOUT_R = 0.5
        self.FLAT_OUT_SIZE = 2048