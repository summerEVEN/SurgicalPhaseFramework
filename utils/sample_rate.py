"""
统一采样率
(生成指定采样率的标签文本文件)
"""
import os

def sample_txt(input_file, output_file, sample_rate):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # 通过采样率从行列表中选择行
    sampled_lines = lines[1::sample_rate]

    # 将筛选后的行写入新文件
    with open(output_file, 'w') as outfile:
        outfile.writelines(sampled_lines)

# # 输入文件路径
# input_file = 'input.txt'
# # 输出文件路径
# output_file = 'output.txt'
# # 采样率（例如每隔5行取一行）
# sample_rate = 5

# sample_txt(input_file, output_file, sample_rate)

video_label_folder ="/home/ubuntu/disk1/Even/Dataset/cholec80_full/phase_annotations"
for v_f in os.listdir(video_label_folder):
    print("标签数据路径： ", v_f)
    video_num = os.path.basename(v_f)[5:7]
    video_num_int = int(video_num)

    if(video_num_int > 40):
        output_file = "../../Dataset/SAHC/even/test_dataset/annotation_folder/video" + video_num + ".txt"
        sample_txt(os.path.join(video_label_folder, v_f), output_file, 25)
    else: 
        output_file = "../../Dataset/SAHC/even/train_dataset/annotation_folder/video" + video_num + ".txt"
        sample_txt(os.path.join(video_label_folder, v_f), output_file, 25)