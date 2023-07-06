# 封装一个简单的tensorboard的小函数
# 就是为了方便管理，不用每次修改tensorboard的event文件的保存路径（只是想实现这个功能）
from torch.utils.tensorboard import SummaryWriter
import datetime
import os.path

__all__ = ["writer_log"]

def writer_log(opt):
    TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
    writer_log_dir = os.path.join('./result/tensorboard/{}'.format(opt.model_name), TIMESTAMP)
    writer = SummaryWriter(writer_log_dir)

    return writer
