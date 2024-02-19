import argparse
import os
import torch


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')
        # pre-exploration
        self.parser.add_argument("--pre_explore", default=False, type=bool)
        self.parser.add_argument("--env_based", default=False, type=bool)

        # fedavg
        self.parser.add_argument("--if_fed", default=False, type=bool)
        self.parser.add_argument("--seed", default=10, type=int)
        self.parser.add_argument("--fed_alg", default='fedavg', type=str, help='fedavg, simi_sum, moon')
        self.parser.add_argument("--n_parties", default=61, type=int, help='total number of parties')
        self.parser.add_argument("--sample_fraction", default=0.2, type=float, help='training traction per round')
        self.parser.add_argument("--comm_round", default=85, type=int,
                                 help='local_rank for distributed training on gpus')
        self.parser.add_argument("--local_epoches", default=5, type=int, help='batch or total')
        self.parser.add_argument("--fedavg_epoch", default=-1, type=int)
        self.parser.add_argument("--load_fedavg", default=False)
        self.parser.add_argument("--global_lr", default=1, type=float)
        self.parser.add_argument("--unseen_only", default=False)
        self.parser.add_argument("--part_unseen", default=False)
        
        #Attack
        self.parser.add_argument("--attack_type", default=0,type=int,help="0: No Attack. 1: LabelFlipped. 2: BackDoor Attack, 3: Finegrained Backdoor Attack")
        self.parser.add_argument("--malicious_fraction", default=0.1,type=float)
        self.parser.add_argument("--defense_method", default="mean",type=str,help="mean, median, tr_mean, multi_krum, bulyan, multi_krum_cos, bulyan_cos")
        self.parser.add_argument("--backdoor_valid", default=False,type=bool)
        self.parser.add_argument("--no_train", default=False,type=bool)
        self.parser.add_argument("--compare", default=False,type=bool)
        self.parser.add_argument("--minus", default=0,type=int)
        self.parser.add_argument("--backdoor_val_rate", default=1.0,type=float)
        self.parser.add_argument("--malicious_rate", default=1.0,type=float)
        self.parser.add_argument("--backdoor_train_rate", default=0.5,type=float)
        self.parser.add_argument("--scaled_factor", default=0.1,type=float)
        self.parser.add_argument("--do_bulyan", default=False,type=bool)
        self.parser.add_argument("--do_mask", default=False,type=bool)
        self.parser.add_argument("--do_mean", default=False,type=bool)
        self.parser.add_argument("--do_resample", default=False,type=bool)
        self.parser.add_argument("--backdoor_multiple_val", default=False,type=bool)
        self.parser.add_argument("--return_empty", default=False,type=bool)
        self.parser.add_argument("--use_dba", default=False,type=bool)

        #Generalize
        self.parser.add_argument("--generalize", default=False,type=bool)
        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')

