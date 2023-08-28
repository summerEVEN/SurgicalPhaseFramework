"""
可视化的模块（模型验证的一部分）
主要实现的功能：
1. 根据提供的网络模型，测试数据集的准确率，并且按照每个视频可视化结果（给出每个视频的准确率）
2. 接受多组参数，也就是多组模型
3. 保存路径：/result/visualization/model_name
    model_name 是根据多组参数来混合命名的（模型名称的拼接）
"""

import os
import matplotlib.pyplot as plt

# 设置每个阶段的颜色（这里只有7个，后续增加阶段数可能需要修改）
colors = ["#a40026", "#e75739", "#fcbd70", "#fdfdbf", "#badfec", "#6399c5", "#313594"]

def visualize_predictions_and_ground_truth(preds_phase, labels_phase, acc, video_name, model_name, save_dir='./result/visualization/'):
    """
    Args:
        preds_phase（list）: 模型预测的结果
        labels_phase（list）: ground_turth 的标签
        acc（float）: 视频的预测准确率
        video_num（int）: 视频序号
        model_name（str）: 模型的名称
    """

    fig, axs = plt.subplots(2, 1, figsize=(8, 2))  # 创建一个有两个子图的figure，这些子图是上下排列的

    # axs[0].figure(dpi=300,figsize=(30,3)) # 分辨率参数-dpi，画布大小参数-figsize
    # axs[1].figure(dpi=300,figsize=(30,3)) # 分辨率参数-dpi，画布大小参数-figsize

    axs[0].bar(range(len(preds_phase)), [1]*len(preds_phase), width=1.0, color=[colors[i] for i in preds_phase])
    axs[0].set_title(model_name + "-" + str(video_name) + "-" + str(round(acc, 4))) 
    axs[0].axis('off')  # 关闭轴
    axs[1].bar(range(len(labels_phase)), [1]*len(labels_phase), width=1.0, color=[colors[i] for i in labels_phase])
    axs[1].set_title('ground_turth') 
    axs[1].axis('off')  # 关闭轴
    
    # 调整子图之间的间距
    fig.tight_layout()

    # 去掉坐标轴
    plt.axis('off') 

    save_root_path = os.path.join(save_dir, model_name)
    if  not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    # 保存结果图
    save_path = os.path.join(save_dir, model_name, '{}.png'.format(video_name))
    plt.savefig(save_path)
    plt.close()


