"""
用来读取 yml 配置文件里面的参数的函数

参考 https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/captioning/utils/opts.py 这个代码
"""

import argparse

def if_use_feat(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_att, use_fc = False, True
    elif caption_model == 'language_model':
        use_att, use_fc = False, False
    elif caption_model in ['updown', 'topdown']:
        use_fc, use_att = True, True
    else:
        use_att, use_fc = True, False
    return use_fc, use_att


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings


    # Model settings
    parser.add_argument('--id', type=str, default=None, help="随便写点关于这次训练的一些信息？")
    parser.add_argument('--model_name', type=str, default=None, help='模型的名称')
    parser.add_argument('--epoch', type=int, default=3, help="训练轮次")

    parser.add_argument('--batch_size', type=int, default=60, help="批大小")
    parser.add_argument('--sequence_length', type=int, default=4, help="视频片段的长度")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="学习率")
    parser.add_argument('--workers', type=int, default=4, help="^")
    parser.add_argument('--model_path', type=str, help="模型的路径")

    parser.add_argument('--train_feature_path', type=str, default=None, help="训练集的特征")
    parser.add_argument('--test_faeture_path', type=str, default=None, help="测试集的特征文件")

    # TMR
    parser.add_argument('--is_time_conv', type=bool, default=False, help="是否使用时间卷积")
    parser.add_argument('--LFB_length', type=int, default=20, help="使用memory bank时候的查找特征的长度")

    # TCN
    parser.add_argument('--out_features', type=int, default=7, help="手术阶段的类别数,TCN输出的特征大小")
    parser.add_argument('--mstcn_causal_conv', type=bool, default=True, help="True：使用因果卷积；False：不使用因果卷积")
    parser.add_argument('--mstcn_layers', type=int, default=8, help="一个TCN的层数")
    parser.add_argument('--mstcn_f_maps', type=int, default=32, help="TCN特征的大小??")
    parser.add_argument('--mstcn_f_dim', type=int, default=2048, help="TCN特征的大小??")
    parser.add_argument('--mstcn_stages', type=int, default=2, help="几个TCN")

    # parser.add_argument('--')


    # add_diversity_opts(parser)

    """
    前面的部分，按照模型的需要，设计参数的名称然后添加进来
    可以按照里面的方法，按照功能分成多个函数，然后调用函数，添加参数

    重点是借鉴后面的 从命令行里面读取参数配置的文件，然后覆盖前面的参数
    
    """
    # config
    parser.add_argument('--cfg', type=str, default="",
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args, _ = parser.parse_known_args()
    print("-------------")
    # args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from .config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        # args = parser.parse_args(namespace=args)
        args, _ = parser.parse_known_args(namespace=args)

    # 读取完 configs 文件后，需要保证参数的正确性（排除掉不符合模型结构的数据）
    """
    这个部分后期可以补上(在模型训练前保证参数设置合理)
    """
    # Check if args are valid
    # assert args.rnn_size > 0, "rnn_size should be greater than 0"
    # assert args.num_layers > 0, "num_layers should be greater than 0"
    # assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    # assert args.batch_size > 0, "batch_size should be greater than 0"
    # assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    # assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    # assert args.beam_size > 0, "beam_size should be greater than 0"
    # assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    # assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    # assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    # assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    # assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    # # default value for start_from and checkpoint_path
    # args.checkpoint_path = args.checkpoint_path or './log_%s' %args.id
    # args.start_from = args.start_from or args.checkpoint_path

    # # Deal with feature things before anything
    # args.use_fc, args.use_att = if_use_feat(args.caption_model)
    # if args.use_box: args.att_feat_size = args.att_feat_size + 5

    return args


def add_eval_options(parser):
    # Basic options
    parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
    parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')

    # Sampling options
    add_eval_sample_opts(parser)

    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='', 
                    help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--image_root', type=str, default='', 
                    help='In case the image paths have to be preprended with a root path to an image folder')
    # For evaluation on MSCOCO images from some split:
    parser.add_argument('--input_fc_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_att_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_box_dir', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_label_h5', type=str, default='',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_json', type=str, default='', 
                    help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add_argument('--split', type=str, default='test', 
                    help='if running on MSCOCO images, which split to use: val|test|train')
    parser.add_argument('--coco_json', type=str, default='', 
                    help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
    # misc
    parser.add_argument('--id', type=str, default='', 
                    help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
    parser.add_argument('--verbose_beam', type=int, default=1, 
                    help='if we need to print out all beam search beams.')
    parser.add_argument('--verbose_loss', type=int, default=0, 
                    help='If calculate loss using ground truth during evaluation')

def add_diversity_opts(parser):
    parser.add_argument('--sample_n', type=int, default=1,
                    help='Diverse sampling')
    parser.add_argument('--sample_n_method', type=str, default='sample',
                    help='sample, bs, dbs, gumbel, topk, dgreedy, dsample, dtopk, dtopp')
    parser.add_argument('--eval_oracle', type=int, default=1, 
                    help='if we need to calculate loss.')


# Sampling related options
def add_eval_sample_opts(parser):
    parser.add_argument('--sample_method', type=str, default='greedy',
                    help='greedy; sample; gumbel; top<int>, top<0-1>')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_method = greedy, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--max_length', type=int, default=20,
                    help='Maximum length during sampling')
    parser.add_argument('--length_penalty', type=str, default='',
                    help='wu_X or avg_X, X is the alpha')
    parser.add_argument('--group_size', type=int, default=1,
                    help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
    parser.add_argument('--diversity_lambda', type=float, default=0.5,
                    help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
    parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature when sampling from distributions (i.e. when sample_method = sample). Lower = "safer" predictions.')
    parser.add_argument('--decoding_constraint', type=int, default=0,
                    help='If 1, not allowing same word in a row')
    parser.add_argument('--block_trigrams', type=int, default=0,
                    help='block repeated trigram.')
    parser.add_argument('--remove_bad_endings', type=int, default=0,
                    help='Remove bad endings')
    parser.add_argument('--suppress_UNK', type=int, default=1,
                    help='Not predicting UNK')


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_opt()
    print(args)
    print()
    sys.argv = [sys.argv[0], '--cfg', 'configs/updown_long.yml']
    args1 = parse_opt()
    print(dict(set(vars(args1).items()) - set(vars(args).items())))
    print()
    sys.argv = [sys.argv[0], '--cfg', 'configs/updown_long.yml', '--caption_model', 'att2in2']
    args2 = parse_opt()
    print(dict(set(vars(args2).items()) - set(vars(args1).items())))