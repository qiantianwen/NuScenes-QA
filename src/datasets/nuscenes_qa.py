# ------------------------------------------------------------------
# NuScenes-QA
# Written by Tianwen Qian https://github.com/qiantianwen/NuScenes-QA
# ------------------------------------------------------------------

import numpy as np
import torch
import glob, json, re, en_vectors_web_lg
import torch.utils.data as Data
import ipdb

class NuScenes_QA(Data.Dataset):
    def __init__(self, __C):
        super(NuScenes_QA).__init__()
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        qa_dict_preread = {
            'train': json.load(open(__C.RAW_PATH['train'], 'r')),
            'val': json.load(open(__C.RAW_PATH['val'], 'r'))}
        
        self.qa_list = []
        split = __C.SPLIT[__C.RUN_MODE]
        self.qa_list += qa_dict_preread[split]['questions']
        self.data_size = self.qa_list.__len__()

        print(split, ' dataset size:', self.data_size)
        
        # {question id} -> {question}
        self.qid2ques = self.ques_load(self.qa_list)

        # Loading scene features path
        scene_feat_path_list = glob.glob(__C.FEATS_PATH[__C.VISUAL_FEATURE][split] + '/*.npz')
        # {scene token} -> {scene feature absolutely path}
        self.stk2featpath = self.scene_feat_path_load(scene_feat_path_list)

        # Tokenize and load glove embedding
        if __C.MODEL != 'clip-adpter':
            self.token2ix, self.pretrained_emb = self.tokenize(qa_dict_preread)
            self.token_size = self.token2ix.__len__()
        else:
            self.token_size = None
            self.pretrained_emb = None

        self.ans2ix, self.ix2ans = self.load_ans_table('./src/datasets/answer_dict.json')
        self.ans_size = self.ans2ix.__len__()
        print('Data Loading Finished!')
        print('')


    def ques_load(self, qa_list):
        qid2ques = {}

        qid = 0
        for item in qa_list:
            qid2ques[qid] = item
            qid += 1

        return qid2ques
    
    def scene_feat_path_load(self, path_list):
        stk2path = {}

        for path in path_list:
            stk = str(path.split('/')[-1].split('.')[0])
            stk2path[stk] = path

        return stk2path
    
    def tokenize(self, qa_dict):
        token2ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb = []
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)

        ques_list = []
        for split in qa_dict:
            qa_list = qa_dict[split]['questions']
            for item in qa_list:
                ques_list.append(item['question'])

        for ques in ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token2ix:
                    token2ix[word] = len(token2ix)
                    pretrained_emb.append(spacy_tool(word).vector)
        
        pretrained_emb = np.array(pretrained_emb)

        return token2ix, pretrained_emb
    
    def load_ans_table(self, answer_table):
        ans2ix, ix2ans = json.load(open(answer_table, 'r'))
        return ans2ix, ix2ans
    

    def __getitem__(self, idx):

        ques_ix_iter, ans_iter, scene_token = self.load_ques_ans(idx)
        obj_feat_iter, bbox_feat_iter = self.load_obj_feats(scene_token)

        return \
            torch.from_numpy(obj_feat_iter),\
            torch.from_numpy(bbox_feat_iter),\
            torch.from_numpy(ques_ix_iter),\
            torch.from_numpy(ans_iter)
    
    
    def __len__(self):
        return self.data_size
    

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        ques = self.qa_list[idx]['question']
        scene_token = self.qa_list[idx]['sample_token']

        ques_ix_iter = self.proc_ques(ques, max_token=30)
        ans_iter = np.zeros(1)

        if self.__C.RUN_MODE in ['train']:
            ans = self.qa_list[idx]['answer']
            ans_iter = self.proc_ans(ans, self.ans2ix)
        
        return ques_ix_iter, ans_iter, scene_token
    
    def load_obj_feats(self, scene_token):
        det_results = np.load(self.stk2featpath[scene_token], allow_pickle=True)['results']
        num_obj = det_results.shape[0]
        obj_feat = []
        bbox = []
        label = []
        for i in range(num_obj):
            obj = det_results[i]
            obj_feat.append(obj['feats'])
            bbox.append(obj['box'][:7])
            label.append(obj['label'])
        # empty detection
        if obj_feat == []:
            obj_feat = np.zeros((1, 512)).astype(np.float32)
            bbox = np.zeros((1, 7)).astype(np.float32)
        obj_feat = np.stack(obj_feat, axis=0) # [num_obj, feat_dim]
        bbox = np.stack(bbox, axis=0) # [num_obj, 7]
        obj_feat_iter = self.proc_scene_feat(obj_feat, feat_pad_size=self.__C.FEAT_SIZE['OBJ_FEAT_SIZE'][0])
        bbox_feat_iter = self.proc_scene_feat(self.proc_bbox_feat(bbox, None), feat_pad_size=self.__C.FEAT_SIZE['BBOX_FEAT_SIZE'][0])

        return obj_feat_iter.astype(np.float32), bbox_feat_iter.astype(np.float32)


    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_ques(self, ques, max_token):
        if self.__C.MODEL != 'clip-adpter':
            token2ix = self.token2ix
            ques_ix = np.zeros(max_token, np.int64)

            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for ix, word in enumerate(words):
                if word in token2ix:
                    ques_ix[ix] = token2ix[word]
                else:
                    ques_ix[ix] = token2ix['UNK']

                if ix + 1 == max_token:
                    break
        
        else:
            # use clip tokenizer
            import clip
            ques_ix = clip.tokenize([ques], context_length=max_token).numpy()[0]
        
        return ques_ix
    
    def proc_ans(self, ans, ans2ix):
        ans = str(ans)
        ans_ix = np.zeros(1, np.int64)
        ans_ix[0] = ans2ix[ans]

        return ans_ix
    
    def proc_scene_feat(self, feat, feat_pad_size):
        if feat.shape[0] > feat_pad_size:
            feat = feat[:feat_pad_size]
        
        feat = np.pad(
            feat,
            ((0, feat_pad_size-feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return feat
    
    def proc_bbox_feat(self, bbox, scene_shape):
        if self.__C.BBOX_NORMALIZE:
            raise NotImplementedError()
        
        return bbox
        