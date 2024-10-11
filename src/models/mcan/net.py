# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

from src.ops.fc import MLP
from src.ops.layer_norm import LayerNorm
from src.models.mcan.mca import MCA_ED

import torch.nn as nn
import torch.nn.functional as F
import torch

import ipdb


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


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
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

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, obj_feat, bbox_feat, ques_ix):
        '''
        obj_feat
            shape: [bsz, num_obj, obj_dim]
            dtype: torch.float32
        bbox_feat
            shape: [bsz, num_obj, box_dim]
            dtype: torch.float32
        ques_ix
            shape: [bsz, max_word_len]
            dtype: torch.int32
        '''

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        obj_feat, obj_feat_mask = self.adapter(obj_feat, bbox_feat)

        # Backbone Framework
        lang_feat, obj_feat = self.backbone(
            lang_feat,
            obj_feat,
            lang_feat_mask,
            obj_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        obj_feat = self.attflat_img(
            obj_feat,
            obj_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + obj_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat


def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)