import os
import utils.labels
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from script.data_propre import dataset_propre
from utils.tensorboard_log import writer_log

import copy
import time
import pickle
from tqdm import tqdm

__all__ = ['train', 'test', "extract"]


"""
resnet50 好像没有处理时序信息，
但是原来的代码里面还是按照 resnet_lstm 一样使用了视频片段
应该不需要使用视频片段吧，直接对图像进行处理就行了吧

不清楚，先按照原来的写着吧
"""
def train(opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet50", debug = True):
    model.to(device)
    # writer = writer_log(opt)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    sequence_length = opt.sequence_length

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(opt.epoch):
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            torch.cuda.empty_cache()
            train_loader = dataset_propre(opt, train_dataset, True)

            # Sets the module in training mode.
            model.train()
            total = 0
            train_loss_phase = 0.0
            train_corrects_phase = 0
            batch_progress = 0.0
            running_loss_phase = 0.0
            minibatch_correct_phase = 0.0
            train_start_time = time.time()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                loss = loss_phase
                loss.backward()
                optimizer.step()

                running_loss_phase += loss_phase.data.item()
                train_loss_phase += loss_phase.data.item()

                batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
                train_corrects_phase += batch_corrects_phase
                total += len(labels_phase.data)

                progress_bar.update(len(labels_phase.data))
            progress_bar.close()


        train_elapsed_time = time.time() - train_start_time

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))

        """
        保存当前最优秀的模型
        """
        if debug:
            acc = test(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test(opt, model, test_dataset, device):
    sequence_length = opt.sequence_length

    model.to(device)
    model.eval()
    test_loader = dataset_propre(opt, test_dataset)
    
    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:

            correct = 0
            total = 0
            for data in test_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                _, preds_phase = torch.max(outputs_phase.data, 1)

                correct += torch.sum(preds_phase == labels_phase.data)

                # print(labels_phase.data.shape)
                total += len(labels_phase.data)
                progress_bar.update(len(labels_phase.data))
            progress_bar.close()
    print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

def extract(opt, model, train_dataset, test_dataset, device, save_dir = "./result/feature/resnet50"):
    """
    使用 resnet50 网络提取视频特征
    """ 
    model.load_state_dict(torch.load(opt.model_path), strict=False)
    model.to(device)
    model.eval()

    print("-----------开始运行------------")

    # resnet50 没有处理时序信息的功能，所以在这里把序列长度设置为1，直接把图片投喂进去，而不是视频片段
    opt.sequence_length = 1
    train_loader = dataset_propre(opt, train_dataset)
    test_loader = dataset_propre(opt, test_dataset)

    # Long Term Feature bank
    g_LFB_train = np.zeros(shape=(0, 2048))
    g_LFB_val = np.zeros(shape=(0, 2048))


    with torch.no_grad():
        with tqdm(total=len(train_dataset), desc="extract train", unit="batch") as progress_bar:
            for data in train_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)
                # outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    # print(save_feature)
                    # 这里reshape的 2048 和模型里面输出的大小是保持一致的
                    # 其他模型可能不一样，这里可以考虑之后，在config里面添加对应的参数，只写一个 extract函数hhh
                    # print(save_feature.shape)
                    save_feature = save_feature.reshape(1, 2048)
                    g_LFB_train = np.concatenate((g_LFB_train, save_feature),axis=0)
                progress_bar.update(len(outputs_feature))
                # print("train feature length:",len(g_LFB_train))
            progress_bar.close()
            

        with tqdm(total=len(test_dataset), desc="extract test", unit="batch") as progress_bar:
            for data in test_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    save_feature = save_feature.reshape(1, 2048)
                    g_LFB_val = np.concatenate((g_LFB_val, save_feature), axis=0)

                progress_bar.update(len(outputs_feature))
                # print("val feature length:",len(g_LFB_val))
            progress_bar.close()

    print("finish!")
    g_LFB_train = np.array(g_LFB_train)
    g_LFB_val = np.array(g_LFB_val)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "resnet50_train.pkl"), 'wb') as f:
        pickle.dump(g_LFB_train, f)

    with open(os.path.join(save_dir, "resnet50_test.pkl"), 'wb') as f:
        pickle.dump(g_LFB_val, f)


if __name__ == "__main__":
    """
    UNIT TEST
    """


