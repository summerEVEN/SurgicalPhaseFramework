
import os
import time
from tracemalloc import start
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
import re
import argparse
import numpy as np
import random
from thop import profile
from utils.utils_SAHC import  fusion,segment_bars_with_confidence_score
from utils.ribbon import visualize_predictions_and_ground_truth

from tqdm import tqdm
f_path = os.path.abspath('..')
root_path = f_path.split('surgical_code')[0]


loss_layer = nn.CrossEntropyLoss()
mse_layer = nn.MSELoss(reduction='none')


def hierarch_train(opt, model, train_loader, validation_loader, device, save_dir = 'models', debug = False):
   
    model.to(device)
    num_classes = opt.num_classes
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    best_epoch = 0
    best_acc = 0
    model.train()
    save_name = 'hier{}_msloss{}_trans{}'.format(opt.hier,opt.ms_loss,opt.trans)
    save_dir = os.path.join(save_dir, opt.model,save_name)
    for epoch in range(1, opt.epochs + 1):
        if epoch % 30 == 0:
            opt.learning_rate = opt.learning_rate * 0.5
       
        
        correct = 0
        total = 0
        loss_item = 0
        ce_item = 0 
        ms_item = 0
        lc_item = 0
        gl_item = 0
        optimizer = torch.optim.Adam(model.parameters(), opt.learning_rate, weight_decay=1e-5)
        max_seq = 0
        mean_len = 0
        ans = 0
        max_phase = 0
        for (video, labels, video_name) in (train_loader):
                labels = torch.Tensor(labels).long()        
                video, labels = video.float().to(device), labels.to(device) 
                predicted_list, feature_list, prototype = model(video)
               
                mean_len += predicted_list[0].size(-1)
                ans += 1
                all_out, resize_list, labels_list = fusion(predicted_list,labels, opt)

                max_seq = max(max_seq, video.size(1))

                loss = 0 
                
                if opt.ms_loss:
                    ms_loss = 0
                  
                    for p,l in zip(resize_list,labels_list):
                        ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, opt.num_classes), l.view(-1))
                        ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                    loss = loss + ms_loss
                    ms_item += ms_loss.item()

                optimizer.zero_grad()
                loss_item += loss.item()
             
                if opt.last:
                    all_out =  resize_list[-1]
                if opt.first:
                    all_out = resize_list[0]
                
                # print(type(all_out))
                loss.backward()

                optimizer.step()
                
                #### error
                """
                _, predicted = torch.max(all_out.data, 1)
                AttributeError: 'int' object has no attribute 'data'
                """
                _, predicted = torch.max(all_out.data, 1)
               
                # labels = labels_list[-1]
                correct += ((predicted == labels).sum()).item()
                total += labels.shape[0]
                # total +=1

        print('Train Epoch {}: Acc {}, Loss {}, ms {}'.format(epoch, correct / total, loss_item /total,  ms_item/total))
        if debug:
            test_acc, predicted, out_pro, test_video_name=hierarch_test(opt, model, validation_loader, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.state_dict(), save_dir + '/best_{}_{}.model'.format(epoch, round(best_acc, 4)))
        print('Best Test: Acc {}, Epoch {}'.format(best_acc, best_epoch))

def hierarch_test(opt, model, test_loader, device, random_mask=False):
   
    model.to(device)
    num_classes = opt.num_classes
    
    model.eval()
   
    with torch.no_grad():
        
            correct = 0
            total = 0
            loss_item = 0
            all_preds = []
            center = torch.ones((1, 64, num_classes), requires_grad=False)
            center = center.to(device)
            label_correct={}
            label_total= {}
            probabilty_list = []
            video_name_list=[]
            precision=0
            recall = 0
            ce_item = 0 
            ms_item = 0
            lc_item = 0
            gl_item = 0
            max_seq = 0 
            for n_iter,(video, labels, video_name ) in enumerate(test_loader):
                
                    
                labels = torch.Tensor(labels).long()
              
                    
                video, labels = video.float().to(device), labels.to(device)
                max_seq = max(max_seq, video.size(1))
               
                predicted_list, feature_list, _ = model(video)
                
                all_out, resize_list,labels_list = fusion(predicted_list,labels, opt)
             
                loss = 0 

              
                
                if opt.ms_loss:
                    ms_loss = 0
                    for p,l in zip(resize_list,labels_list):
                        # print(p.size())
                        ms_loss += loss_layer(p.transpose(2, 1).contiguous().view(-1, opt.num_classes), l.view(-1))
                        ms_loss += torch.mean(torch.clamp(mse_layer(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
                    loss = loss + ms_loss
                    ms_item += ms_loss.item()
               
               
                loss_item += loss.item()

                if opt.last:
                    all_out =  resize_list[-1]
                if opt.first:
                    all_out = resize_list[0]

                _, predicted = torch.max(all_out.data, 1)
                

                predicted = predicted.squeeze()

                # labels = labels_list[-1]
                correct += ((predicted == labels).sum()).item()
                total += labels.shape[0]

              
                video_name_list.append(video_name)

                all_preds.append(predicted)

                all_out = F.softmax(all_out,dim=1)

                probabilty_list.append(all_out.transpose(1,2))
            # print(max_seq)
            print('Test  Acc {}, Loss {}, ms {}'.format( correct / total, loss_item /total, ms_item/total))
            # print('BMG precision {}, BMG recall {}'.format(precision/(n_iter+1), recall/(n_iter+1) ))
            # print(len(label_total))
            for (kc, vc), (kall, vall) in zip(label_correct.items(),label_total.items()):
                print("{} acc: {}".format(kc, vc/vall))
            return correct / total, all_preds, probabilty_list, video_name_list
        
# 原来的预测和可视化函数（注释掉啦，看不懂，重写了一个）
"""
def base_predict(model, opt, device,test_loader, pki = False,split='test'):

    phase2label_dicts = {
        'cholec80':{
        'Preparation':0,
        'CalotTriangleDissection':1,
        'ClippingCutting':2,
        'GallbladderDissection':3,
        'GallbladderPackaging':4,
        'CleaningCoagulation':5,
        'GallbladderRetraction':6},
        
        'm2cai16':{
        'TrocarPlacement':0,
        'Preparation':1,
        'CalotTriangleDissection':2,
        'ClippingCutting':3,
        'GallbladderDissection':4,
        'GallbladderPackaging':5,
        'CleaningCoagulation':6,
        'GallbladderRetraction':7}
    }
    model.to(device)
    model.eval()
    save_name = '{}_hier{}_trans{}'.format(opt.sample_rate,opt.hier,opt.trans)
   

    pic_save_dir = 'results/{}/{}/vis/'.format(opt.dataset,save_name)
    results_dir = 'results/{}/{}/prediction_{}/'.format(opt.dataset,save_name,opt.sample_rate)

    gt_dir = os.path.join(root_path, '../Dataset/SAHC/{}/test_dataset/annotation_folder/'.format(opt.dataset))
  
    if not os.path.exists(pic_save_dir):
        os.makedirs(pic_save_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
   
    with torch.no_grad():
        correct =0
        total =0 
        for (video, labels, video_name) in tqdm(test_loader):
            labels = torch.Tensor(labels).long()
            print(video.size(),video_name,labels.size())
            video = video.to(device)
            labels = labels.to(device)
            predicted_list, feature_list, _ = model(video)
                
            all_out, resize_list,labels_list = fusion(predicted_list,labels, opt)
            if opt.last:
                    all_out = resize_list[-1]
            if opt.first:
                    all_out = resize_list[0]
            confidence, predicted = torch.max(F.softmax(all_out.data,1), 1)

            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]
   
            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
            
            labels = [label.item() for label in labels]
            
            pic_file = video_name[0].split('.')[0] + '-vis.png'
            pic_path = os.path.join(pic_save_dir, pic_file)
            segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])

            predicted_phases_expand = []
           
            for i in predicted:
                predicted_phases_expand = np.concatenate((predicted_phases_expand, [i]*5 )) 
                # we downsample the framerate from 25fps to 5fps         
          
            print(video_name)
         
            v_n = video_name[0]
           
            
            v_n = re.findall(r"\d+\.?\d*",v_n)
            
            v_n = float(v_n[0])
            target_video_file = "%02d_pred.txt"%(v_n)
            print(target_video_file)
           
            if opt.dataset == 'm2cai16':             
                gt_file = 'test_workflow_video_%02d.txt'%(v_n)
            else:              
                # gt_file = 'video%02d-phase.txt'%(v_n)
                gt_file = 'video%02d.txt'%(v_n)
           
            g_ptr = open(os.path.join(gt_dir, gt_file), "r")
            f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
           
 
            gt = g_ptr.readlines()[1:] ##
          
            gt = gt[::5]
            print(len(gt), len(predicted_phases_expand))
            
            if len(gt) >  len(predicted_phases_expand):
                lst = predicted_phases_expand[-1]
                print(len(gt) - len(predicted_phases_expand))
                for i in range(0,len(gt) - len(predicted_phases_expand)):
                    predicted_phases_expand=np.append(predicted_phases_expand,lst)
            else:
                predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            print(len(gt), len(predicted_phases_expand))
            assert len(predicted_phases_expand) == len(gt)
           
            # f_ptr.write("Frame\tPhase\n")
            for index, line in enumerate(predicted_phases_expand):
                # print(int(line),opt.dataset)
                phase_dict = phase2label_dicts[opt.dataset]
                p_phase = ''
                for k,v in phase_dict.items():
                    if v==int(line):
                        p_phase = k
                        break

                # line = phase2label_dicts[opt.dataset][int(line)]
                # f_ptr.write('{}\t{}\n'.format(index, int(line)))
                f_ptr.write('{}\t{}\n'.format(index, p_phase))
            f_ptr.close()

            # g_phase_ptr.write("Frame\tPhase\n")
            # for index, line in enumerate(gt):
            #     line = line.strip('\n')
            #     _, pp = line.split('\t')
            #     # print(index,pp)
            #     # pp = phase2label_dicts[opt.dataset][pp]
            #     g_phase_ptr.write('{}\t{}\n'.format(index, pp))
            # g_phase_ptr.close()
        print(correct/total)
"""

def evaluate_and_visualize(opt, model, test_loader, device):
    # 1. 加载模型参数，加载模型到GPU，设置为 eval() 模式
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
    model.to(device)
    model.eval()
    
    # if not os.path.exists(pic_save_dir):
    #     os.makedirs(pic_save_dir)
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
   
    with torch.no_grad():
        correct =0
        total =0 
        for (video, labels, video_name) in (test_loader):
            labels = torch.Tensor(labels).long()
            print("video.size(),video_name,labels.size(): ", video.size(),video_name,labels.size())
            video = video.float().to(device)
            labels = labels.to(device)
            predicted_list, feature_list, _ = model(video)
                
            all_out, resize_list,labels_list = fusion(predicted_list,labels, opt)
            if opt.last:
                    all_out = resize_list[-1]
            if opt.first:
                    all_out = resize_list[0]
            confidence, predicted = torch.max(F.softmax(all_out.data,1), 1)

            correct += ((predicted == labels).sum()).item()
            total += labels.shape[0]

            video_correct_sum = ((predicted == labels).sum()).item()
            video_label_sum = labels.shape[0]
   
            predicted = predicted.squeeze(0).tolist()
            confidence = confidence.squeeze(0).tolist()
            
            labels = [label.item() for label in labels]
            
            # pic_file = video_name[0].split('.')[0] + '-vis.png'
            # pic_path = os.path.join(pic_save_dir, pic_file)
            # segment_bars_with_confidence_score(pic_path, confidence_score=confidence, labels=[labels, predicted])
            # 这里调用了重新写的可视化函数
            visualize_predictions_and_ground_truth(predicted, labels, video_correct_sum/video_label_sum, 
                                                   video_name, opt.model_name, save_dir='./result/visualization_SAHC/')

            # predicted_phases_expand = []
           
            # for i in predicted:
            #     predicted_phases_expand = np.concatenate((predicted_phases_expand, [i]*5 )) 
            #     # we downsample the framerate from 25fps to 5fps         
          
            # print(video_name)
         
            # v_n = video_name[0]
           
            
            # v_n = re.findall(r"\d+\.?\d*",v_n)
            
            # v_n = float(v_n[0])
            # target_video_file = "%02d_pred.txt"%(v_n)
            # print(target_video_file)
           
            # if opt.dataset == 'm2cai16':             
            #     gt_file = 'test_workflow_video_%02d.txt'%(v_n)
            # else:              
            #     # gt_file = 'video%02d-phase.txt'%(v_n)
            #     gt_file = 'video%02d.txt'%(v_n)
           
            # g_ptr = open(os.path.join(gt_dir, gt_file), "r")
            # f_ptr = open(os.path.join(results_dir, target_video_file), 'w')
           
 
            # gt = g_ptr.readlines()[1:] ##
          
            # gt = gt[::5]
            # print(len(gt), len(predicted_phases_expand))
            
            # if len(gt) >  len(predicted_phases_expand):
            #     lst = predicted_phases_expand[-1]
            #     print(len(gt) - len(predicted_phases_expand))
            #     for i in range(0,len(gt) - len(predicted_phases_expand)):
            #         predicted_phases_expand=np.append(predicted_phases_expand,lst)
            # else:
            #     predicted_phases_expand = predicted_phases_expand[0:len(gt)]
            # print(len(gt), len(predicted_phases_expand))
            # assert len(predicted_phases_expand) == len(gt)
           
            # # f_ptr.write("Frame\tPhase\n")
            # for index, line in enumerate(predicted_phases_expand):
            #     # print(int(line),opt.dataset)
            #     phase_dict = phase2label_dicts[opt.dataset]
            #     p_phase = ''
            #     for k,v in phase_dict.items():
            #         if v==int(line):
            #             p_phase = k
            #             break

            #     # line = phase2label_dicts[opt.dataset][int(line)]
            #     # f_ptr.write('{}\t{}\n'.format(index, int(line)))
            #     f_ptr.write('{}\t{}\n'.format(index, p_phase))
            # f_ptr.close()

            # g_phase_ptr.write("Frame\tPhase\n")
            # for index, line in enumerate(gt):
            #     line = line.strip('\n')
            #     _, pp = line.split('\t')
            #     # print(index,pp)
            #     # pp = phase2label_dicts[opt.dataset][pp]
            #     g_phase_ptr.write('{}\t{}\n'.format(index, pp))
            # g_phase_ptr.close()
        print(correct/total)
