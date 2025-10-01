import contextlib
import copy
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer
import sys
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable the warning of tokenizers parallelism

is_interactive = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

@TRAINER_REGISTRY.register()
class MultitaskTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.dataset_name = list(self.data_proportion['test'].keys())
        self.record = {k:{'step_acc':[],
                       'task_acc':[]} for k in self.dataset_name} 
        self.record['loss'],self.record['all_step_acc'],self.record['all_task_acc'] = [],[],[]
        self.results = []
        # convert loaders and evaluators to list
        for split in ['val', 'test']:
            if split not in self.data_loaders:
                continue
            loaders = self.data_loaders[split]
            if not (isinstance(loaders, list) or isinstance(loaders, tuple)):
                self.data_loaders[split] = [loaders]
        if not isinstance(self.evaluator, list):
            self.evaluator = [self.evaluator]
            
        if cfg.get('profile'):
            def trace_handler(p):
                tb_handler = torch.profiler.tensorboard_trace_handler('./outputs/profile_trace/')
                tb_handler(p)
                output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total")
                # print(output)

            profile = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=10, warmup=10, active=10, repeat=1),
                    on_trace_ready=trace_handler,
                    profile_memory=True, record_shapes=True, with_modules=True, with_stack=True)
            profile.start()
            self.profile = profile
        else:
            self.profile = contextlib.nullcontext()
        print(self.dataset_name,self.data_proportion)

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        epoch_loss = 0
        
        for i, data_dict in enumerate(loader):
         
            
           
            
            if isinstance(self.profile, torch.profiler.profile):
                self.profile.step()
            with self.accelerator.accumulate(self.model):
                data_dict['cur_step'] = epoch * len(loader) + i
                data_dict['total_steps'] = self.total_steps
                
                data_dict = self.forward(data_dict) 
                
                loss, losses = self.loss(data_dict) #loss is the sum of the loss, losses are the dic containing each loss and total loss
                # ground_loss are registered in query3d_loss
                self.backward(loss)
                epoch_loss += loss
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step}
                log_dict.update(losses)
                self.log(log_dict, mode="train")
                # pbar.update(1)
        average_loss = epoch_loss / len(loader)
        average_loss = average_loss.detach()
        

        
        self.record['loss'].append(average_loss.item())

    @torch.no_grad()
    def eval_step(self):
        self.model.eval()
        target_metrics = []

        mode = "val" if self.mode == 'train' else "test"
        loaders = self.data_loaders[mode] 
        evaluators = self.evaluator
        dataset_index, all_step_acc, all_task_acc = 0,0,0

        for loader, evaluator in zip(loaders, evaluators):
            st = time()
            is_best, results,_,_= self.eval(loader, evaluator)

            for k,v in results.items():
                if self.mode == 'train' and k in ['step_acc','task_acc']:
                    self.record[self.dataset_name[dataset_index]][k].append(v)
                elif self.mode == 'test' and k in ['step_acc','task_acc']:
                    self.record[self.dataset_name[dataset_index]][k].append(v)
                elif self.mode == 'test' and k in ['results']:
                    self.results.append(v)
                    
            end = time()
            results['time'] = (end - st) / 60
            self.log(results, mode="val-" + loader.dataset.dataset.__class__.__name__) 
            evaluator.reset()
            target_metrics.append(results['target_metric'])
            dataset_index +=1
        for dataset in self.dataset_name:
            all_step_acc += (self.record[dataset]['step_acc'][-1] * self.data_proportion[mode][dataset])
            all_task_acc += (self.record[dataset]['task_acc'][-1] * self.data_proportion[mode][dataset])
            

        self.record['all_step_acc'].append(all_step_acc)
        self.record['all_task_acc'].append(all_task_acc)

        
        if self.record['all_task_acc'][-1] == max(self.record['all_task_acc']):
            is_best = True
        else:
            is_best = False
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loaders = self.data_loaders["test"]
        evaluators = self.evaluator
        for loader, evaluator in zip(loaders, evaluators):
            is_best, results = self.eval(loader, evaluator)
            self.log(results, mode= "test-" + loader.dataset.dataset.__class__.__name__)
            evaluator.reset()
        return results
    
    @torch.no_grad()
    def eval(self, loader, evaluator):
      
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.postprocess_for_eval(data_dict, loader)
            evaluator.update(data_dict)
  
        return evaluator.record()

    def run(self):
        path = os.path.join('Results',self.cfg.model.name,self.cfg.model.notes,'metrics','metrics.json')
        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders["train"])
            self.eval_step() #Random Model Eval
            print('-----------------START TRAINING------------------------')
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                st = time()
                self.train_step(epoch)
                epoch_time = (time() - st) / 60
                self.log({"epoch_time": epoch_time, "epoch": epoch+1, "remaining_time": epoch_time * (self.epochs - epoch) / 60}, mode="train")
            
                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step()
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False
            #Save the metrics
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(self.record, f, indent=4)
                print(f'Save the metrics in {path}')
            #Accelerator version saving model
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")
                    self.save("latest.pth")
            #nn.Module version saving model
                if is_best:
                    self.model.save_model('best')
                self.model.save_model('latest')
            self.accelerator.end_training()
            self.plot(path)
            
        if self.mode == 'test':
            self.eval_step()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.record, f,indent=4)
            
            result_path = os.path.join('Results',self.cfg.model.name,self.cfg.model.notes,'metrics','results.json')
            with open(result_path, 'w') as f:
                json.dump(self.results, f,indent=4)
        
    def plot(self,path):

        with open(path, 'r') as f:
            loaded_dict = json.load(f)


        plots_dir = os.path.join('Results', self.cfg.model.name,self.cfg.model.notes,'metrics')

 
        for key, values in loaded_dict.items():
            if key in self.dataset_name: 
                for k,v in loaded_dict[key].items(): 
                    epochs = list(range(1, len(v) + 1))  

                    plt.figure() 
                    plt.plot(epochs, v)  
                    plt.xlabel('Epoch')
                    plt.ylabel('Value')
                    plt.title(f'{key}_{k}')  
                    plt.grid(True)
                    

                    name = key.replace("SequentialGrounding","")
                    plot_filename = os.path.join(plots_dir, name, f"{name}_{k}_plot.png")
                    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
                    plt.savefig(plot_filename)
                    plt.close() 
                    
                    step_acc = loaded_dict[key].get('step_acc', [])
                    task_acc = loaded_dict[key].get('task_acc',[])
                    
                    max_step_acc = max(step_acc) if step_acc else None
                    max_task_acc = max(task_acc) if task_acc else None

           
                    last_step_acc = step_acc[-1] if step_acc else None
                    last_task_acc = task_acc[-1] if task_acc else None

   
                    message_file_path = os.path.join(plots_dir, name, 'metrics_summary.txt')
                    with open(message_file_path, 'w') as f:
                        f.write(f"In {key}\n")
                        f.write(f"Highest step accuracy: {max_step_acc}\n")
                        f.write(f"Last step accuracy: {last_step_acc}\n")
                        f.write(f"Highest task accuracy: {max_task_acc}\n")
                        f.write(f"Last task accuracy: {last_task_acc}\n")

                  
            else: 
                epochs = list(range(1, len(values) + 1)) 

                plt.figure()  
                plt.plot(epochs, values)  
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.title(f'{key}') 
                plt.grid(True)
                
          
                plot_filename = os.path.join(plots_dir, f"{key}_plot.png")
                plt.savefig(plot_filename)
                plt.close() 
                
        all_step_acc = loaded_dict.get('all_step_acc', [])
        all_task_acc = loaded_dict.get('all_task_acc', [])


        max_all_step_acc = max(all_step_acc) if all_step_acc else None
        max_all_task_acc = max(all_task_acc) if all_task_acc else None

        last_all_step_acc = all_step_acc[-1] if all_step_acc else None
        last_all_task_acc = all_task_acc[-1] if all_task_acc else None


        message_file_path = os.path.join(plots_dir, 'metrics_summary.txt')
        with open(message_file_path, 'w') as f:
            f.write(f"Highest step accuracy overall: {max_all_step_acc}\n")
            f.write(f"Last step accuracy overall: {last_all_step_acc}\n")
            f.write(f"Highest task accuracy overall: {max_all_task_acc}\n")
            f.write(f"Last task accuracy overall: {last_all_task_acc}\n")

               
        
    def postprocess_for_eval(self, data_dict, loader):
        if 'task_id' in data_dict.keys() and data_dict['task_id'][0] in [0]: 
            return
        if 'generation_logits' in data_dict.keys():
            tokenizer = loader.dataset.generation_tokenizer
            response_pred = tokenizer.batch_decode(data_dict['generation_logits'], skip_special_tokens=True)
            data_dict['caption_pred'] = response_pred
            data_dict['answer_pred'] = response_pred



