# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

from src.models.butd.tda import TDA

import torch.nn as nn
import torch.nn.functional as F
from src.ops.layer_norm import LayerNorm
from torch.nn.utils.weight_norm import weight_norm
import torch


# ------------------------------
# ---- Adapt object feature ----
# ------------------------------
class Adapter(nn.Module):
    def __init__(self, __C):
        super(Adapter, self).__init__()
        self.__C = __C

        obj_feat_linear_size = __C.FEAT_SIZE['OBJ_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(__C.FEAT_SIZE['BBOX_FEAT_SIZE'][1], __C.BBOXFEAT_EMB_SIZE)
            obj_feat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.obj_linear = nn.Linear(obj_feat_linear_size, __C.HIDDEN_SIZE)

    def forward(self, obj_feat, bbox_feat):
        obj_feat = obj_feat.to(torch.float32)
        bbox_feat = bbox_feat.to(torch.float32)
        obj_feat_mask = make_mask(obj_feat)
        
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            obj_feat = torch.cat((obj_feat, bbox_feat), dim=-1)
        
        obj_feat = self.obj_linear(obj_feat)

        return obj_feat, obj_feat_mask

# -------------------------
# ---- Main BUTD Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.rnn = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = TDA(__C)

        
        layers = [
            weight_norm(nn.Linear(__C.HIDDEN_SIZE,
                                  __C.FLAT_OUT_SIZE), dim=None),
            nn.ReLU(),
            nn.Dropout(__C.CLASSIFER_DROPOUT_R),
            weight_norm(nn.Linear(__C.FLAT_OUT_SIZE, answer_size), dim=None)
        ]
        self.classifer = nn.Sequential(*layers)
        # self.proj_norm = LayerNorm(__C.HIDDEN_SIZE)
        # self.proj = nn.Linear(__C.HIDDEN_SIZE, answer_size)

    def forward(self, obj_feat, bbox_feat, ques_ix):

        # Pre-process Language Feature
        # lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.rnn(lang_feat)

        obj_feat, _ = self.adapter(obj_feat, bbox_feat)

        # Backbone Framework
        joint_feat = self.backbone(
            lang_feat[:, -1],
            obj_feat
        )

        # Classification layers
        # proj_feat = self.proj_norm(joint_feat)
        # proj_feat = self.proj(proj_feat)
        proj_feat = self.classifer(joint_feat)

        return proj_feat
    
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)