import random
import torch
from transformers import AutoTokenizer

from data.datasets.constant import PromptType

from ..data_utils import make_bce_label, pad_sequence_2d, pad_sequence
from .dataset_wrapper import DATASETWRAPPER_REGISTRY
from torch.utils.data import Dataset, default_collate
from .scanfamily_wrapper import ScanFamilyDatasetWrapper,ScanFamilyDatasetWrapper_Single


# 0 for the refer task, 1 for the QA task, 2 for the caption task
dataset2task_id = {'ScanRefer': 0, 'Sr3D': 0, 'Nr3D': 0, 'Multi3DRefer': 0,
                   'ScanQA': 1, 'SQA3D': 1,
                   'Scan2Cap': 2,
                   'ScanReferSceneVerse': 0, 'Sr3DSceneVerse': 0, 'Nr3DSceneVerse': 0, 'Multi3DReferSceneVerse': 0,
                   'ScanQASceneVerse': 1, 'SQA3DSceneVerse': 1,
                   'Scan2CapSceneVerse' : 2,
                   'SequentialGroundingSingleStepScanNet': 0, 'SequentialGroundingSingleStep3RScan': 0, 'SequentialGroundingSingleStepMultiScan': 0,
                   'SequentialGroundingSingleStepARKitScenes': 0, 'SequentialGroundingSingleStepHM3D': 0,'SequentialGroundingScanNet': 0, 'SequentialGrounding3RScan': 0, 'SequentialGroundingMultiScan': 0,
                   'SequentialGroundingARKitScenes': 0, 'SequentialGroundingHM3D': 0
                   }

@DATASETWRAPPER_REGISTRY.register()
class UnifiedTaskDatasetWrapper(ScanFamilyDatasetWrapper):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset)
        self.dataset_name = dataset.__class__.__name__
        self.task_id = dataset2task_id[self.dataset_name]
        self.train = self.dataset.split == 'train'
        self.train_keys = ['task_id', 'prompt_type', 'response','step_prompts','step_masks','step_num',
                           'tgt_object_id', 'tgt_object_label', 'prompt','query_pad_masks','query_locs','seg_center','pc_seg_fts','pc_seg_pad_masks',
                          'obj_fts', 'obj_locs', 'obj_labels', 'obj_boxes', 'obj_pad_masks','coord_min', 'coord_max']
        if 'qa' in cfg.model.heads:
            self.train_keys.append('answer_label')
        generation_tokenizer_name = getattr(cfg.data_wrapper, 'generation_tokenizer', 't5-small')
        self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_tokenizer_name)
        self.responses = self.init_response()
        if dataset.split == 'train' and dataset.train_duplicate > 1:
            print(f'Duplicating train data {dataset.train_duplicate} times for {self.dataset_name}.')
            self.duplicate_data(dataset.train_duplicate)

    def duplicate_data(self, duplicate):
        self.dataset.lang_data = self.dataset.lang_data * duplicate
        self.tokenized_step = self.tokenized_step * duplicate
        self.responses = self.responses * duplicate
        
    def init_response(self):
        if self.task_id == 0:
            encoded_input = self.generation_tokenizer([''] * len(self.dataset), add_special_tokens=True, truncation=True)
            tokenized_responses = encoded_input.input_ids
        elif self.task_id == 1:
            tokenized_responses = []
            for i in range(len(self.dataset)):
                sample = self.dataset.get_lang(i)[-1]
                encoded_input = self.generation_tokenizer(sample['answers'], add_special_tokens=True, truncation=True)
                tokenized_responses.append(encoded_input.input_ids)
        elif self.task_id == 2:
            sentences = [self.dataset.get_lang(i)[-1]['sentence'] for i in range(len(self.dataset))]
            encoded_input = self.generation_tokenizer(sentences, add_special_tokens=True, truncation=True)
            tokenized_responses = encoded_input.input_ids
        
        return tokenized_responses


    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['task_id'] = self.task_id
        if self.task_id == 2: # for caption task, use obj_locs as prompt
            prompt = data['obj_locs'][data['tgt_object_id'][0]]
            prompt_type = PromptType.LOC
        else:
            prompt = self.tokenized_step[idx]
            prompt_type = PromptType.TXT
        # data['step_prompt'] = torch.FloatTensor(prompt)
        # data['prompt'] = torch.FloatTensor(prompt)
        # data['prompt_pad_masks'] = torch.ones((len(prompt))).bool()
        data['prompt_type'] = prompt_type
        data['response'] = torch.LongTensor(self.responses[idx] if self.task_id != 1 else random.choice(self.responses[idx])) # random choose an answer as response
        if self.train:
            for k in list(data.keys()):
                if k not in self.train_keys:
                    del data[k]

            #data['tgt_object_id'] = make_bce_label(data['tgt_object_id'], num_classes=len(data['obj_labels']))
            data['tgt_object_label'] = make_bce_label(data['tgt_object_label'], num_classes=int(self.dataset.sem_type)) #607dim all 0 except index of the label is 1
        else:
            data['tgt_object_id_iou25'] = make_bce_label(data['tgt_object_id_iou25'], num_classes=len(data['obj_labels']))
            data['tgt_object_id_iou50'] = make_bce_label(data['tgt_object_id_iou50'], num_classes=len(data['obj_labels']))

        return data
        
    def collate_fn(self, batch):
        new_batch = {}

        if 'step_prompts' in batch[0].keys():
            padded_step_ids, padding_step_masks = pad_sequence_2d([sample.pop('step_prompts') for sample in batch], return_mask=True)
        new_batch['step_prompts'] = padded_step_ids
        new_batch['step_masks'] = padding_step_masks.logical_not()
        # print(new_batch['step_prompts'].shape,new_batch['step_prompts'])
        # print(new_batch['step_masks'].shape,new_batch['step_masks']) [B,max(step_len),max(token_len)]
        list_keys = [k for k, v in batch[0].items() if isinstance(v, list)]
        # print(list_keys) 
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
        
        
        padding_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor) and v.ndim > 0]
        # print(padding_keys)
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padding_value = -100 if k == 'obj_labels' or k == 'tgt_object_id' else 0
            padded_tensor = pad_sequence(tensors, pad=padding_value)
            new_batch[k] = padded_tensor
            
        new_batch.update(default_collate(batch))

        return new_batch
    
@DATASETWRAPPER_REGISTRY.register()
class UnifiedTaskDatasetWrapper_Single(ScanFamilyDatasetWrapper_Single):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset)
        self.dataset_name = dataset.__class__.__name__
        self.task_id = dataset2task_id[self.dataset_name]
        self.train = self.dataset.split == 'train'
        self.train_keys = ['task_id', 'prompt', 'prompt_pad_masks', 'prompt_type', 'response',
                           'tgt_object_id', 'tgt_object_label', 
                          'obj_fts', 'obj_mv_fts', 'obj_voxel_fts',
                          'obj_locs', 'obj_labels', 'obj_boxes', 'obj_pad_masks',
                          'voxel_seg_fts', 'mv_seg_fts', 'pc_seg_fts', 'voxel_seg_pad_masks', 'mv_seg_pad_masks',  'pc_seg_pad_masks', 'seg_center', 'seg_pad_masks', 'segment_masks', 'instance_labels',
                          'query_locs', 'query_pad_masks', 'coord_min', 'coord_max']
        if 'qa' in cfg.model.heads:
            self.train_keys.append('answer_label')
        generation_tokenizer_name = getattr(cfg.data_wrapper, 'generation_tokenizer', 't5-small')
        self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_tokenizer_name)
        self.responses = self.init_response()
        if dataset.split == 'train' and dataset.train_duplicate > 1:
            print(f'Duplicating train data {dataset.train_duplicate} times for {self.dataset_name}.')
            self.duplicate_data(dataset.train_duplicate)

    def duplicate_data(self, duplicate):
        self.dataset.lang_data = self.dataset.lang_data * duplicate
        self.tokenized_txt = self.tokenized_txt * duplicate
        self.responses = self.responses * duplicate
        
    def init_response(self):
        if self.task_id == 0:
            encoded_input = self.generation_tokenizer([''] * len(self.dataset), add_special_tokens=True, truncation=True)
            tokenized_responses = encoded_input.input_ids
        elif self.task_id == 1:
            tokenized_responses = []
            for i in range(len(self.dataset)):
                sample = self.dataset.get_lang(i)[-1]
                encoded_input = self.generation_tokenizer(sample['answers'], add_special_tokens=True, truncation=True)
                tokenized_responses.append(encoded_input.input_ids)
        elif self.task_id == 2:
            sentences = [self.dataset.get_lang(i)[-1]['sentence'] for i in range(len(self.dataset))]
            encoded_input = self.generation_tokenizer(sentences, add_special_tokens=True, truncation=True)
            tokenized_responses = encoded_input.input_ids
        
        return tokenized_responses


    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data['task_id'] = self.task_id
        if self.task_id == 2: # for caption task, use obj_locs as prompt
            prompt = data['obj_locs'][data['tgt_object_id'][0]]
            prompt_type = PromptType.LOC
        else:
            prompt = self.tokenized_txt[idx]
            prompt_type = PromptType.TXT
        data['prompt'] = torch.FloatTensor(prompt)
        data['prompt_pad_masks'] = torch.ones((len(prompt))).bool()
        data['prompt_type'] = prompt_type
        data['response'] = torch.LongTensor(self.responses[idx] if self.task_id != 1 else random.choice(self.responses[idx])) # random choose an answer as response
        if self.train:
            for k in list(data.keys()):
                if k not in self.train_keys:
                    del data[k]

            #data['tgt_object_id'] = make_bce_label(data['tgt_object_id'], num_classes=len(data['obj_labels']))
            data['tgt_object_label'] = make_bce_label(data['tgt_object_label'], num_classes=int(self.dataset.sem_type)) #607dim all 0 except index of the label is 1
        else:
            data['tgt_object_id_iou25'] = make_bce_label(data['tgt_object_id_iou25'], num_classes=len(data['obj_labels']))
            data['tgt_object_id_iou50'] = make_bce_label(data['tgt_object_id_iou50'], num_classes=len(data['obj_labels']))

        return data
        
    def collate_fn(self, batch):
        new_batch = {}
       
        if 'segment_masks' in batch[0].keys():
            # build attn masks
            padded_segment_masks, padding_mask = pad_sequence_2d([sample.pop('segment_masks') for sample in batch], return_mask=True)
            new_batch['offline_attn_mask'] = padded_segment_masks.logical_not()
            
            # build labels and masks for loss
            labels = pad_sequence([sample['instance_labels'] for sample in batch], pad=-100)
            new_batch['target_labels'] = labels
            new_batch['target_masks'] = padded_segment_masks.float()
            new_batch['target_masks_pad_masks'] = padding_mask.logical_not()
        
        if "voxel_seg_fts" in batch[0].keys():
            if isinstance(batch[0]["voxel_seg_fts"], list):
                new_voxel_seg_fts = []
                for i in range(len(batch[0]['voxel_seg_fts'])):
                    cur_level_fts = [sample['voxel_seg_fts'][i].squeeze(0) for sample in batch]
                    cur_level_fts = pad_sequence(cur_level_fts)
                    new_voxel_seg_fts.append(cur_level_fts)
                new_batch['voxel_seg_fts'] = new_voxel_seg_fts
            else:
                new_batch['voxel_seg_fts'] = pad_sequence([sample['voxel_seg_fts'] for sample in batch])
            for sample in batch:
                sample.pop('voxel_seg_fts')
        # print(len(batch))  --> Batch Size   
        # print(f"Before:{batch[0]['prompt']}")           
        list_keys = [k for k, v in batch[0].items() if isinstance(v, list)]
        # print(list_keys) tgt_obj_boxes
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
        
        
        padding_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor) and v.ndim > 0]
        # print(padding_keys)
        
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padding_value = -100 if k == 'obj_labels' else 0
            padded_tensor = pad_sequence(tensors, pad=padding_value)
            new_batch[k] = padded_tensor
            
        new_batch.update(default_collate(batch))
        
        # print(f"After:{new_batch['prompt'][0]}")
        # print(f"After:{new_batch['prompt'].shape}") -->[B,T]
        # print(new_batch['prompt'].dtype) -->torch.float32
        return new_batch
    
    
    
    
    