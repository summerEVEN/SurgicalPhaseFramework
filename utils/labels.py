"""
用来存放一些关于标签处理的函数

"""

# 定义 阶段 和 label 之间的字典
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

def phase2label(phases, phase2label_dict):
    """
    返回一段 phase 对应的 label
    phases 里面包含多个 phase
    """
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases

def read_labels(opt, label_file):
    """
    读取 标签文件 里面的 label 信息，转化为 0 -（num-1）之间的数字

    opt: 参数的dict
    label_file: label 的文件路径
    """
    with open(label_file,'r') as f:
        phases = [line.strip().split('\t')[1] for line in f.readlines()]
        labels = phase2label(phases, phase2label_dicts[opt.dataset])
    return labels



if __name__ == "__main__":
    """
    UNIT TEST
    """
    test_list = [
        'Preparation',
        'CalotTriangleDissection',
        'ClippingCutting',
        'GallbladderDissection',
        'GallbladderPackaging',
        'CleaningCoagulation',
        'GallbladderRetraction'
    ]

    label_list = [1,2,3,4,5,1,2]

    print(phase2label(test_list, phase2label_dicts["cholec80"]))

    print(phase2label_dicts["cholec80"][test_list[1]])

    print(label2phase(label_list, phase2label_dicts['cholec80']))
