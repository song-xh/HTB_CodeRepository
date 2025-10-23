# arguments.py

import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # 环境
    parser.add_argument('--map', type=str, default='boatschedule')
    parser.add_argument('--n_agents', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--alg', type=str, default='qmix')
    parser.add_argument('--last_action', type=bool, default=True)
    parser.add_argument('--reuse_network', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--optimizer', type=str, default="RMS")
    parser.add_argument('--evaluate_epoch', type=int, default=20)
    parser.add_argument('--model_dir', type=str, default='./MARL/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--result_name', type=str, default='test')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)

    # —— 新增：先验与观测尾部填充 —— #
    parser.add_argument('--replay_dir', type=str, default='')
    parser.add_argument('--use_prior', action='store_true', default=True)
    parser.add_argument('--prior_dim_site', type=int, default=8)
    parser.add_argument('--prior_dim_plane', type=int, default=3)
    parser.add_argument('--obs_pad', type=int, default=32)

    # —— 新增：训练超参（设为 None，便于 mixin 时不覆盖 CLI） —— #
    parser.add_argument('--n_epoch', type=int, default=None)
    parser.add_argument('--n_episodes', type=int, default=None)
    parser.add_argument('--train_steps', type=int, default=None)
    parser.add_argument('--evaluate_cycle', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--save_cycle', type=int, default=None)
    parser.add_argument('--target_update_cycle', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)

    # —— 新增：探索率（兼容你传的名字） —— #
    parser.add_argument('--epsilon_start', type=float, default=None)
    parser.add_argument('--epsilon_end', type=float, default=None)
    parser.add_argument('--epsilon_anneal_steps', type=int, default=None)
    parser.add_argument('--epsilon_anneal_scale', type=str,
                        default=None, choices=['step', 'episode', 'epoch'])

    args = parser.parse_args()
    return args


def get_mixer_args(args):
    # 只在为 None 时设置默认
    def _setdefault(name, value):
        if getattr(args, name, None) is None:
            setattr(args, name, value)

    # network
    _setdefault('rnn_hidden_dim', 64)
    _setdefault('qmix_hidden_dim', 32)
    _setdefault('two_hyper_layers', False)
    _setdefault('hyper_hidden_dim', 64)
    _setdefault('qtran_hidden_dim', 64)
    _setdefault('lr', 5e-4)

    # epsilon (兼容 CLI 的 epsilon_* 映射)
    if args.epsilon_start is not None or args.epsilon_end is not None or args.epsilon_anneal_steps is not None:
        eps = 1.0 if args.epsilon_start is None else args.epsilon_start
        min_eps = 0.05 if args.epsilon_end is None else args.epsilon_end
        steps = 50000 if args.epsilon_anneal_steps is None else args.epsilon_anneal_steps
        args.epsilon = eps
        args.min_epsilon = min_eps
        args.anneal_epsilon = (eps - min_eps) / float(max(1, steps))
        if getattr(args, 'epsilon_anneal_scale', None) is None:
            args.epsilon_anneal_scale = 'step'
    else:
        _setdefault('epsilon', 1.0)
        _setdefault('min_epsilon', 0.05)
        # 若用户没给 anneal_epsilon，就用 50000 步退火
        if getattr(args, 'anneal_epsilon', None) is None:
            args.anneal_epsilon = (args.epsilon - args.min_epsilon) / 50000.0
        _setdefault('epsilon_anneal_scale', 'step')

    # loop 相关
    _setdefault('n_epoch', 15000)
    _setdefault('n_episodes', 5)
    _setdefault('train_steps', 2)
    _setdefault('evaluate_cycle', 50)
    _setdefault('batch_size', 32)
    _setdefault('buffer_size', int(5e3))
    _setdefault('save_cycle', 50)
    _setdefault('target_update_cycle', 200)

    # QTRAN 与其余（保留）
    _setdefault('lambda_opt', 1)
    _setdefault('lambda_nopt', 1)
    _setdefault('grad_norm_clip', 10)
    _setdefault('noise_dim', 16)
    _setdefault('lambda_mi', 0.001)
    _setdefault('lambda_ql', 1)
    _setdefault('entropy_coefficient', 0.001)

    return args



# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 30000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args

