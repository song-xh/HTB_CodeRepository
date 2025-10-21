import numpy as np
import torch
from MARL.policy.vdn import VDN
from MARL.policy.qmix import QMIX
from MARL.policy.coma import COMA
from MARL.policy.reinforce import Reinforce
from MARL.policy.central_v import CentralV
from MARL.policy.qtran_alt import QtranAlt
from MARL.policy.qtran_base import QtranBase
from MARL.policy.maven import MAVEN
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')

    # 有可能分布选出来不符合要求的动作,目前貌似用不到这个东西
    def random_choice_with_mask(self, avail_actions):
        temp = []
        wait = self.n_actions-3
        for i, eve in enumerate(avail_actions):
            if eve == 1:
                temp.append(i)  # 可选站位
        if temp[0] == wait:  # 没有可选站位，动作为等待
            return wait
        else:
            if wait in temp:
                temp.remove(wait)
            return np.random.choice(temp, 1, False)[0]

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # [[]]  -> []
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
        else:
            # print(avail_actions.shape, q_value.shape)
            q_value[avail_actions == 0.0] = - float("inf")
            if np.random.uniform() < epsilon:
                # action = np.random.choice(avail_actions_ind)  # action是一个整数
                action = self.random_choice_with_mask(avail_actions[0])
                if action == self.n_actions-3:
                    # print(avail_actions[0])
                    pass
            else:
                action = torch.argmax(q_value).cpu()  # 此处应该判断一下是不是都是-inf
                # print(66666,action,q_value)

        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        
        prob = prob / prob.sum()
        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            # action = Categorical(prob.squeeze(0)).sample().long()
            action = torch.multinomial(prob, num_samples=1)
            while prob[0][action] == 0:
                action = torch.multinomial(prob, num_samples=1)
        return action

    # 获取bach中最大的结束step数
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 获取这些episode中最大的结束step数
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:  # 如果episodelimit的长度内没有terminal==1，导致max_episode_len == 0
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break   # 若已经终止则跳出这个episode检查下一个episode
            
            if terminated[episode_idx, transition_idx, 0] == 0:
                max_episode_len = self.args.episode_limit   # 若该episode没有在episode_limit步前完成调度，设max_episode_len为episode_limit
                break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]    # 取所有episode前max_episode_len个step的信息
        loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)

        if train_step > 0 and train_step % self.args.save_cycle == 0:
            print("\n开始保存模型", train_step, self.args.save_cycle)
            self.policy.save_model(train_step)
        
        return loss


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon, evaluate=False):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        prob = prob / prob.sum()
        if epsilon == 0 and evaluate:
            # 测试时直接选最大的
            action = torch.argmax(prob)
        else:
            # action = Categorical(prob.squeeze(0)).sample().long()
            action = torch.multinomial(prob, num_samples=1)
            while prob[0][action] == 0:
                action = torch.multinomial(prob, num_samples=1)
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        # 获取这些episode中最大的结束step数
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:  # 如果episodelimit的长度内没有terminal==1，导致max_episode_len == 0
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break   # 若已经终止则跳出这个episode检查下一个episode
            if terminated[episode_idx, transition_idx, 0] == 0:
                max_episode_len = self.args.episode_limit   # 若该episode没有在episode_limit步前完成调度，设max_episode_len为episode_limit
                break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        loss = self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
        return loss
