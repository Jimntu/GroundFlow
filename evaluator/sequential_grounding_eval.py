import json
from pathlib import Path
import random
import textdistance
import torch
import torch.nn.functional as F
import os

from common.misc import gather_dict
from evaluator.build import EVALUATOR_REGISTRY, BaseEvaluator
from datetime import datetime

@EVALUATOR_REGISTRY.register()
class SequentialGroundingEval_GroundFlow(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        if cfg.eval.get('predict_mode', False):
            self.target_metric = 'edit_distance'
        else:
            self.target_metric = 'step_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / (self.__class__.__name__ + datetime.now().strftime('%H:%M:%S:%f'))
        self.cfg = cfg
        
        self.int2cat = json.load(open(os.path.join(cfg.data.scene_verse_base,
                                            "ScanNet/annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        
        if self.cfg.eval.get('predict_mode', False):
            # calculate edit distance
            tgt_object_id = data_dict['tgt_object_id']
            if data_dict['ground_logits'] == None:
                og_pred = []
            else:
                og_pred = torch.argmax(data_dict['ground_logits'], dim=1)
            grd_batch_ind_list = data_dict['grd_batch_ind_list']
            edit_distance = 0
            for i in range(len(tgt_object_id)):
                target_sequence = list(tgt_object_id[i].cpu().numpy())
                predict_sequence = []
                if og_pred != None:
                    for j in range(len(og_pred)):
                        if grd_batch_ind_list[j] == i:
                            predict_sequence.append(og_pred[j].item())
                edit_distance += textdistance.levenshtein.distance(predict_sequence, target_sequence)
                # save for predict mode
                if self.save:
                    obj_ids = data_dict['obj_ids']
                    self.eval_results.append({'scan_id': data_dict['scan_id'][i], 
                                              'gt_object_id': [obj_ids[i][o].item() for o in target_sequence], 
                                              'predict_object_id' : [obj_ids[i][o].item() for o in predict_sequence], 
                                              'gt_task_description': data_dict['prompt_after_obj'][i],
                                              'gt_plan_text': data_dict['output_gt'][i].replace('<s>', ''), 
                                              'pred_plan_text': data_dict['output_txt'][i]})
            metrics['edit_distance'] = (edit_distance, len(tgt_object_id))
        else:
            # calculate step_acc 
            # data_dict['ground_logits'] --> [B,S,N]
            og_pred = torch.argmax(data_dict['ground_logits'], dim=2).detach()
            og_gt = data_dict['ground_label'].detach()
            og_conf = torch.max(F.softmax(data_dict['ground_logits'], dim=2), dim=2)[0].detach()
            assert og_pred.shape == og_gt.shape
            # print(og_pred.shape,og_gt.shape) [B,S]
            correct_step_count,correct_task_count = 0,0
            confidence,correctness = [],[]
            total_step_count = data_dict['step_num'].sum().item()
            total_task_count = len(data_dict['step_num'])
            results = []
            for i in range(len(og_pred)):
                valid_step = data_dict['step_num'][i].item()
                # print(valid_step,og_gt[i])
                correct_step_count += (og_pred[i][:valid_step] == og_gt[i][:valid_step]).sum().item()
                correct_task_count += int((og_pred[i][:valid_step] == og_gt[i][:valid_step]).all())
                
                if int((og_pred[i][:valid_step] == og_gt[i][:valid_step]).all())==1:
                    correct = True
                else: correct = False
                # print(data_dict['obj_labels'][i])
                if self.mode == 'test':
                    pred_name = []
                    scan_obj_labels = data_dict['obj_labels'][i].tolist()
                    pred_ids = og_pred[i][:valid_step].tolist()
                    if -100 in scan_obj_labels:
                        scan_obj_labels = scan_obj_labels[:scan_obj_labels.index(-100)]
                    for id in pred_ids:
                        pred_name.append(self.int2cat[scan_obj_labels[id]])
                    result = (data_dict['scan_id'][i], pred_name,pred_ids,og_gt[i][:valid_step].tolist(),scan_obj_labels,correct)
                    results.append(result)
                    metrics['results'] = results
            
        metrics['step_acc'] = (correct_step_count, total_step_count)
        metrics['task_acc'] = (correct_task_count, total_task_count)


        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)
                

        return metrics 
    
@EVALUATOR_REGISTRY.register()
class SequentialGroundingEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        if cfg.eval.get('predict_mode', False):
            self.target_metric = 'edit_distance'
        else:
            self.target_metric = 'step_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / (self.__class__.__name__ + datetime.now().strftime('%H:%M:%S:%f'))
        self.cfg = cfg
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False):
        metrics = {}
        
        if self.cfg.eval.get('predict_mode', False):
            # calculate edit distance
            tgt_object_id = data_dict['tgt_object_id']
            if data_dict['ground_logits'] == None:
                og_pred = []
            else:
                og_pred = torch.argmax(data_dict['ground_logits'], dim=1)
            grd_batch_ind_list = data_dict['grd_batch_ind_list']
            edit_distance = 0
            for i in range(len(tgt_object_id)):
                target_sequence = list(tgt_object_id[i].cpu().numpy())
                predict_sequence = []
                if og_pred != None:
                    for j in range(len(og_pred)):
                        if grd_batch_ind_list[j] == i:
                            predict_sequence.append(og_pred[j].item())
                edit_distance += textdistance.levenshtein.distance(predict_sequence, target_sequence)
                # save for predict mode
                if self.save:
                    obj_ids = data_dict['obj_ids']
                    self.eval_results.append({'scan_id': data_dict['scan_id'][i], 
                                              'gt_object_id': [obj_ids[i][o].item() for o in target_sequence], 
                                              'predict_object_id' : [obj_ids[i][o].item() for o in predict_sequence], 
                                              'gt_task_description': data_dict['prompt_after_obj'][i],
                                              'gt_plan_text': data_dict['output_gt'][i].replace('<s>', ''), 
                                              'pred_plan_text': data_dict['output_txt'][i]})
            metrics['edit_distance'] = (edit_distance, len(tgt_object_id))
        else:
            # calculate step_acc 
            # data_dict['ground_logits'] --> [B,S,N]
            # print(data_dict['ground_logits'].shape)
            og_pred = torch.argmax(data_dict['ground_logits'], dim=1)
            og_gt = data_dict['ground_label']
            # print(og_pred.shape,og_gt.shape) 
            # print(data_dict['step_num'],sum())
            total_count = len(og_pred)
            # print(og_gt,og_pred)
            assert og_pred.shape == og_gt.shape
            metrics['step_acc'] = ((og_pred == og_gt).sum().item(), total_count)
            # calculate task_acc
            batch_len = [len(t) for t in data_dict['tgt_object_id']]
            correct_count = 0
            total_count = 0
            
            correct_s2_count, total_wrong_s1_count, correct_s2_count_cond = 0,0,0
            for i in range(len(batch_len)):
                correct_count += int((og_pred[total_count : total_count + batch_len[i]] == og_gt[total_count : total_count + batch_len[i]]).all())
                correct_s2_count += int(og_pred[total_count+1] == og_gt[total_count+1])
                if og_pred[total_count] != og_gt[total_count]:
                    total_wrong_s1_count +=1
                    correct_s2_count_cond += int(og_pred[total_count+1] == og_gt[total_count+1])
                
                if self.save:
                    obj_ids = data_dict['obj_ids']
                    target_sequence = [t.item() for t in og_gt[total_count : total_count + batch_len[i]]] 
                    predict_sequence = [t.item() for t in og_pred[total_count : total_count + batch_len[i]]]
                    self.eval_results.append({'scan_id': data_dict['scan_id'][i], 
                                              'gt_object_id': [obj_ids[i][o].item() for o in target_sequence], 
                                              'predict_object_id' : [obj_ids[i][o].item() for o in predict_sequence], 
                                              'gt_task_description': data_dict['prompt_after_obj'][i],
                                              'gt_plan_text': data_dict['output_gt'][i].replace('<s>', ''), 
                                              'correct': target_sequence == predict_sequence})
                total_count += batch_len[i]
                
            metrics['task_acc'] = (correct_count, len(batch_len))

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)

        return metrics #the metrics contain the overall step and task acc

@EVALUATOR_REGISTRY.register()
class SequentialGroundingSingleStepEval(BaseEvaluator):
    def __init__(self, cfg, accelerator, **kwargs):
        if cfg.eval.get('predict_mode', False):
            self.target_metric = 'edit_distance'
        else:
            self.target_metric = 'step_acc'
        self.save_dir = Path(cfg.exp_dir) / "eval_results" / self.__class__.__name__
        self.cfg = cfg
        super().__init__(cfg, accelerator, **kwargs)

    def batch_metrics(self, data_dict, include_count=False): #During update,include_count=True
        metrics = {}
        results = {}
        # calculate step_acc 
        og_pred = torch.argmax(data_dict['ground_logits'], dim=1)
        og_gt = data_dict['ground_label']
        total_count = len(og_pred)
        metrics['step_acc'] = ((og_pred == og_gt).sum().item(), total_count)
        # calculate task_acc
        data_idx = data_dict['data_idx']
        # print(data_idx)
        for i, cur_idx in enumerate(data_idx):
            if cur_idx not in results.keys():
                results[cur_idx] = {'pred':[],'label':[]}
            if 'task_acc_' + cur_idx in metrics.keys():
                prev_task_acc = metrics['task_acc_' + cur_idx]
                metrics['task_acc_' + cur_idx] = ((og_pred[i] == og_gt[i]).sum().item() + prev_task_acc[0], 1 + prev_task_acc[1])
                results[cur_idx]['pred'].append(int(og_pred[i]))
                results[cur_idx]['label'].append(int(og_gt[i]))
            else:
                metrics['task_acc_' + cur_idx] = ((og_pred[i] == og_gt[i]).sum().item(), 1)
                results[cur_idx]['pred'].append(int(og_pred[i]))
                results[cur_idx]['label'].append(int(og_gt[i]))
        
        for cur_idx in data_idx:
            results[cur_idx]['correctness'] = (results[cur_idx]['pred'] == results[cur_idx]['label'])
        if self.save:
            pass

        if not include_count:
            for key, v in metrics.items():
                metrics[key] = v[0] / max(v[1], 1)
        metrics['results'] = results

        return metrics #dictionary contains overall step acc and each task's task acc
    
    def record(self):
        self.eval_dict = gather_dict(self.accelerator, self.eval_dict)
        for k, metrics in self.eval_dict.items():
            if not isinstance(metrics, list) or k == 'results':
                continue
            # metrics is a list of (value, count)
            total_value = sum(x[0] for x in metrics) #sum up each batch
            total_count = sum(x[1] for x in metrics)
            self.eval_dict[k] = total_value / max(total_count, 1)
        
        correct_count = 0
        total_count = 0
        for k in list(self.eval_dict.keys()):
            if 'task_acc' in k:
                correct_count += int(self.eval_dict[k] == 1)
                total_count += 1
                del self.eval_dict[k]
        self.eval_dict['task_acc'] = correct_count / total_count
            
        if self.save and self.accelerator.is_main_process:
            with (self.save_dir / "results.json").open("w") as f:
                json.dump(self.eval_results, f, indent=4)
        
        self.eval_dict['target_metric'] = self.eval_dict[self.target_metric]
        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        self.eval_dict['best_result'] = self.best_result
        # print(self.eval_dict)
        
        
        
        return is_best, self.eval_dict,None,None
