"""
@author: Daniel Shani
"""
import numpy as np
from scipy.special import softmax


class BasisLearner:
    """
    RL agent class. Uses Q-learning with linear function approximation to learn
    the value function. The weight vector is updated using the Q-learning
    update, and is dot producted with the feature vector to get the value.
    """

    def __init__(self, env, pars):

        """
        Args:
        env (Environment): environment object
        pars (DotDict): parameters file
        """

        self.transitions = env.transitions

        self.allo_SR = SR(
            pars.SR_lr_a, pars.gamma, env.allo_M, env.allo_Q,
            self.transitions, pars.lesionMEC)
        self.ego_SR = SR(
            pars.SR_lr_e, pars.gamma, env.ego_M, env.ego_Q,
            env.ego_transitions, pars.lesionLEC)

        self.num_actions = self.allo_SR.SR_sas.shape[1]
        self.env = env
        self.allo_dim = self.allo_SR.SR_sas.shape[0]
        self.ego_dim = self.ego_SR.SR_sas.shape[0]
        self.weight = np.zeros(self.allo_dim + self.ego_dim + 1)
        self.current_state, self.current_ego, self.current_direction = \
            self.env.reset()
        self.reward = 0
        self.pars = pars

        self.m = np.zeros((self.allo_dim + self.ego_dim + 1))
        self.v = np.zeros((self.allo_dim + self.ego_dim + 1))
        self.t = 1
        self.grad = None

        self.heatmap = None
        self.collect_paths = False
        self.paths = [[]]
        self.path = []

    def set_heatmap(self, size):
        self.heatmap = np.zeros((size, size))

    def clear_heatmap(self, size=None):
        self.heatmap = None
        if size is not None:
            self.heatmap = np.zeros((size, size))

    def get_phi(self, state, ego, direction):
        """
        Gets the feature at the current location. This is used to calculate the
        value function by dot product
        with the weight vector.

        Args:
            state (int): allocentric state
        Returns:
            phi (D x num_actions array): [phi_1(a),...,phi_D(a)]
            basis values at that point
        """
        phi_allo = np.roll(self.allo_SR.SR_sas[state], shift=-direction, axis=0)
        phi_ego = self.ego_SR.SR_sas[ego]
        full_phi = np.concatenate(
            (np.ones(self.num_actions)[:, np.newaxis],
             phi_allo, phi_ego), axis=1)
        return full_phi

    def q_w(self, weight, state, ego, direction, action=None):

        """
        Calculates Q-value.
        Args:
            weight (self.allo_dim + self.ego_dim array): weight vector
            state (int < self.allo_dim)
            ego (int < self.ego_dim)
            action (None, int < self.num_actions, or "all"): if None,
            returns V(s) = max_a Q(s,a), if int a, returns Q(s,a), if "all",
            returns Q(s,a) for all actions.
        """

        phi = self.get_phi(state, ego, direction)
        values = [np.dot(phi[a, :], weight)
                  for a in range(self.num_actions)]
        if action is None:
            value = np.max(values)
        elif isinstance(action, str):
            if action == "all":
                value = values
            else:
                return ValueError("Invalid string")
        else:
            value = values[action]
        return value

    def choose_action(self, state, ego, direction):

        """
        Chooses an action using epsilon-softmax policy.
        Args:
            state (int): allocentric state
            ego (int): egocentric state
        """

        epsilon = np.random.uniform(low=0.0, high=1.0)
        if epsilon < self.pars.explore_param:
            return np.random.choice(np.arange(self.num_actions))
        return np.random.choice(
            np.arange(self.num_actions),
            p=self.softmax_policy(state, ego, direction))

    def update_SR(self):
        self.allo_SR.update_SR()
        self.ego_SR.update_SR()

    def update_weight(
            self, next_state, next_ego, action, reward,
            update_SR=True):

        """
        Updates the weight vector by using Q-learning update with linear
        function approximation.

        Args:
            next_state (int): allocentric state
            next_ego (int): egocentric state
            action (int): action
            reward (float): reward
        """

        next_direction = (self.current_direction + action) % 4
        if update_SR:
            self.allo_SR.td_learn(
                self.current_state, next_direction, next_state)  # update SR in
            # allo coords
            self.ego_SR.td_learn(self.current_ego, action, next_ego)

        if next_state is None:
            y = reward
        else:
            y = reward \
                + self.pars.gamma * self.q_w(
                self.weight, next_state,
                next_ego, next_direction)
        y_prime = self.q_w(
            self.weight, self.current_state, self.current_ego,
            self.current_direction, action)

        feature = self.get_phi(
            self.current_state, self.current_ego,
            self.current_direction)[action, :]
        grad = 2 * (y - y_prime) * feature
        
        if self.pars.beta1 == 0 and self.pars.beta2 == 0:
            self.weight += self.pars.eta * grad
            self.grad = grad
        
        else:

            ## adaptive LR ###
            m = self.pars.beta1 * self.m + (1 - self.pars.beta1) * grad
            v = self.pars.beta2 * self.v + (1 - self.pars.beta2) * (grad ** 2)
            mhat = m / (1 - self.pars.beta1 ** self.t)
            vhat = v / (1 - self.pars.beta2 ** self.t)
            self.t += 1
            self.weight += self.pars.eta * mhat / (np.sqrt(vhat) + self.pars.eps)
            self.m = m
            self.v = v
            self.grad = grad

    def softmax_policy(self, state, ego, direction):
        action_values = np.array(
            self.q_w(
                self.weight, state, ego, direction,
                action="all"))
        policy = softmax(self.pars.temperature * action_values)
        return policy

    def train_step(self, update_SR=True):

        """
        Performs a training step.
        First chooses action in current state, using softmax policy.
        Environment step is performed using action.
        Agent then updates its weight vector using the new state and reward.
        """
        action = self.choose_action(
            self.current_state, self.current_ego,
            self.current_direction)

        self.path.append(
            [self.current_state, self.current_ego,
             self.current_direction, action])

        next_state, next_ego, direction, reward, done = self.env.step(action)

        self.update_weight(next_state, next_ego, action, reward, update_SR)
        self.current_direction = direction

        self.current_state = next_state
        self.current_ego = next_ego

        if self.collect_paths:
            self.paths[-1].append(self.env.position_2d)

        if done:
            self.path.append(
                [self.current_state, self.current_ego,
                 self.current_direction, action])

        if self.heatmap is not None:
            self.heatmap[self.env.position_2d[0], self.env.position_2d[1]] += 1

        return next_state, next_ego, direction, reward, done, action

    def clear_paths(self):
        self.paths = [[]]

    def reset(self):
        """
        Initializes the environment and the agent.
        """
        self.current_state, self.current_ego, self.current_direction = \
            self.env.reset()

        self.allo_SR.reset_SR()
        self.ego_SR.reset_SR()

        self.path = []

    def truncated_gradient(self, grav):
        """Method to induce sparsity in the weights - currently not used"""
        return self._truncated_gradient(
            self.weight, grav * self.pars.eta,
            self.pars.theta)

    @staticmethod
    def _truncated_gradient(v, alpha, theta):
        """
        Applies truncated gradient function T_1 as described in Langford et al.
        2009 pp.4.
        Args:
            v: vector of values to be truncated
            alpha: non-negative scalar
            theta: positive scalar scalar

        Returns: T_1(v,alpha,theta) as defined in the paper.
        """
        cond_list = [(np.less_equal(np.zeros(v.shape), v))
                     & (np.less_equal(v, theta * np.ones(v.shape))),
                     (np.greater_equal(np.zeros(v.shape), v))
                     & (np.greater_equal(v, -theta * np.ones(v.shape))),
                     abs(v) > theta]

        choice_list = [np.maximum(0, v - alpha), np.minimum(0, alpha - v), v]
        return np.select(cond_list, choice_list, 0)

    def save_weights(self, path, episode):
        np.save(path + '/w_' + str(episode), self.weight)

    def save_SR(self, path, episode):
        np.save(path + '/allo_SR_sas_' + str(episode), self.allo_SR.SR_sas)
        np.save(path + '/allo_SR_ss_' + str(episode), self.allo_SR.SR_ss)

        np.save(path + '/ego_SR_sas_' + str(episode), self.ego_SR.SR_sas)
        np.save(path + '/ego_SR_ss_' + str(episode), self.ego_SR.SR_ss)

    def save_path(self, filepath):
        np.save(filepath, self.path)
        self.path = []

    def switch_world(self, gridworld, tangible=True, switch_SRs=True):


        (allo_dim, ego_dim, allo_Q, allo_M, ego_Q, ego_M, allo_indices,
         ego_indices) \
            = self.env.switch_world(
            gridworld, tangible=tangible, switch_SRs=switch_SRs)
        self.update_sizes_new_world(allo_dim, ego_dim)
        self.update_SR_new_world(
            allo_Q, allo_M, ego_Q, ego_M,
            allo_indices, ego_indices)


        self.current_state, self.current_ego, self.current_direction = \
            self.env.reset()

    def update_SR_new_world(
            self, allo_Q, allo_M, ego_Q, ego_M, allo_indices,
            ego_indices):
        self.allo_SR.re_init_SR(allo_M, allo_Q, allo_indices)
        self.ego_SR.re_init_SR(ego_M, ego_Q, ego_indices)

    def update_sizes_new_world(self, allo_dim, ego_dim):

        # update weight vector sizes

        if self.allo_dim < allo_dim or self.ego_dim < ego_dim:
            w_0 = self.weight.copy()[0]
            w_allo = self.weight.copy()[1:self.allo_dim + 1]
            w_ego = self.weight.copy()[self.allo_dim + 1:]

            m_0 = self.m.copy()[0]
            v_0 = self.v.copy()[0]
            m_allo = self.m.copy()[1:self.allo_dim + 1]
            m_ego = self.m.copy()[self.allo_dim + 1:]
            v_allo = self.v.copy()[1:self.allo_dim + 1]
            v_ego = self.v.copy()[self.allo_dim + 1:]

            m_allo = np.pad(
                m_allo, (0, allo_dim - self.allo_dim),
                'constant', constant_values=(0, 0))
            m_ego = np.pad(
                m_ego, (0, ego_dim - self.ego_dim),
                'constant', constant_values=(0, 0))

            # print(m_ego.shape)
            # print(m_allo.shape)
            # print(m_0.shape)
            self.m = np.concatenate((m_0[np.newaxis], m_allo, m_ego))

            v_allo = np.pad(
                v_allo, (0, allo_dim - self.allo_dim),
                'constant', constant_values=(0, 0))
            v_ego = np.pad(
                v_ego, (0, ego_dim - self.ego_dim),
                'constant', constant_values=(0, 0))
            self.v = np.concatenate((v_0[np.newaxis], v_allo, v_ego))

            w_allo = np.pad(
                w_allo, (0, allo_dim - self.allo_dim),
                'constant', constant_values=(np.mean(w_allo), np.mean(w_allo)))
            w_ego = np.pad(
                w_ego, (0, ego_dim - self.ego_dim),
                'constant')
            self.weight = np.concatenate((w_0[np.newaxis], w_allo, w_ego))
            self.allo_dim = allo_dim
            self.ego_dim = ego_dim

            # adaptive lr
            # self.v = np.zeros((self.allo_dim + self.ego_dim))
            # self.t = 1

        # if self.allo_dim > self.allo_SR.num_states or self.ego_dim > \
        #         self.ego_SR.num_states:
        #     # M_allo = np.pad(
        #     #     self.allo_SR.M_new, ((0, self.allo_dim -
        #     #                           self.allo_SR.num_states),
        #     #                          (0,
        #     #                           self.allo_dim - self.allo_SR.num_states)),
        #     #     'constant', constant_values=0)
        #     Q_allo = np.pad(
        #         self.allo_SR.SR_sas_new, ((0, self.allo_dim -
        #                               self.allo_SR.num_states),
        #                              (0, 0),
        #                              (0,
        #                               self.allo_dim - self.allo_SR.num_states)),
        #         'constant', constant_values=0.)
        #
        #     M_allo = np.eye(self.allo_dim)
        #     # Q_allo = np.zeros((self.allo_dim, 4, self.allo_dim))
        #
        #     self.allo_SR.re_init_SR(SR_ss=M_allo, SR_sas=Q_allo)
        #
        #     # M_ego = np.pad(
        #     #     self.ego_SR.SR_ss_new, ((0, self.ego_dim -
        #     #                          self.ego_SR.num_states),
        #     #                         (0, self.ego_dim - self.ego_SR.num_states)),
        #     #     'constant', constant_values=0.)
        #     # Q_ego = np.pad(
        #     #     self.ego_SR.SR_sas_new, ((0, self.ego_dim -
        #     #                          self.ego_SR.num_states),
        #     #                         (0, 0),
        #     #                         (0, self.ego_dim - self.ego_SR.num_states)),
        #     #     'constant', constant_values=0.)
        #     #
        #     # self.ego_SR.re_init_SR(M_ego, Q_ego)


class SR:
    """
    Class for storing and updating the state-action successor representation.
    Learns the SR using TD-learning.
    Initialise with M and Q SRS and then update during training.
    """

    def __init__(self, lr, gamma, SR_ss, SR_sas, transitions, lesion=False):
        """
        Args:
            lr: learning rate for updating the SR
            gamma: discount factor for TD update
            SR_ss: initial state-state SR
            SR_sas: initial state-action SR
        """
        self.lr = lr
        self.gamma = gamma
        self.lesion = lesion

        # just testing out normalisation - i know this might not be a good
        # correspondence between the two
        SR_ss = SR_ss / np.mean(SR_ss, axis=1, keepdims=True)
        SR_sas = SR_sas / np.mean(SR_sas, axis=(1, 2), keepdims=True)

        if lesion:
            self.SR_ss = np.zeros_like(SR_ss)
            self.SR_sas = np.zeros_like(SR_sas)
        else:
            self.SR_ss = SR_ss
            self.SR_sas = SR_sas

        self.SR_ss_new = np.copy(self.SR_ss)
        self.SR_sas_new = np.copy(self.SR_sas)
        self.num_states = SR_sas.shape[0]
        self.num_actions = SR_sas.shape[1]
        self.identity = np.eye(self.num_states)
        self.transitions = transitions
        self.protected_indices = np.sum(self.transitions, axis=(1, 2)) != 0

    def re_init_SR(self, SR_ss=None, SR_sas=None, protected_indices=None):

        """
        Re-initializes the SR with new M and Q.
        Args:
            M: new state-state SR
            Q: new state-action SR
        """

        if SR_sas is not None:

            # # just testing out normalisation - i know this might not be a good
            # # correspondence between the two
            # SR_sas = SR_sas / np.mean(SR_sas, axis=(1, 2), keepdims=True)

            if SR_sas.shape[0] > self.SR_sas.shape[0]:
                if self.lesion:
                    self.SR_sas = np.zeros_like(SR_sas)
                else:

                    new_protected_indices = np.pad(
                        self.protected_indices,
                        (0, SR_sas.shape[0] -
                         self.protected_indices.shape[0]), 'constant',
                        constant_values=(False, False))

                    SAS_new = np.zeros_like(SR_sas)
                    SAS_new[np.where(new_protected_indices), :,
                    :self.SR_sas.shape[0]] \
                        = (
                        self.SR_sas)[
                        np.where(self.protected_indices)]

                    SAS_new[~new_protected_indices] = SR_sas[
                        ~new_protected_indices]

                    self.SR_sas = SAS_new

                    self.protected_indices = np.logical_or(
                        new_protected_indices, protected_indices)

                    # print(self.protected_indices)


            else:
                if self.lesion:
                    self.SR_sas = np.zeros_like(SR_sas)
                else:
                    self.SR_sas[:SR_sas.shape[0], :SR_sas.shape[1],
                    :SR_sas.shape[2]] = SR_sas


        # if SR_ss is not None:
        #
        #     # # just testing out normalisation - i know this might not be a good
        #     # # corresoondence between the two
        #     # SR_ss = SR_ss / np.mean(SR_ss, axis=1, keepdims=True)
        #
        #     if SR_ss.shape[0] > self.SR_ss.shape[0]:
        #         if self.lesion:
        #             self.SR_ss = np.zeros_like(SR_ss)
        #         else:
        #             self.SR_ss = SR_ss
        #     else:
        #         if self.lesion:
        #             self.SR_ss = np.zeros_like(SR_ss)
        #         else:
        #             self.SR_ss[:SR_ss.shape[0], :SR_ss.shape[1]] = SR_ss

        self.SR_ss = np.mean(self.SR_sas, axis=1)

        self.SR_ss_new = np.copy(self.SR_ss)
        self.SR_sas_new = np.copy(self.SR_sas)
        self.num_states = self.SR_sas.shape[0]
        self.num_actions = self.SR_sas.shape[1]
        self.identity = np.eye(self.num_states)

    def td_learn(self, s0, a0, s1):

        self.learn_ss(s0, s1)
        self.learn_sas(s0, a0, s1)

    def learn_ss(self, s0, s1):
        if s1 is not None:
            if not self.lesion:
                update = self.lr * (self.identity[s1] +
                                    self.gamma * self.SR_ss_new[s1,
                                                 :] - self.SR_ss_new[s0,
                                                      :])
                # print(update)
                self.SR_ss_new[s0, :] += update
        else:
            if not self.lesion:
                self.SR_ss_new[s0, :] += self.lr * (
                        self.identity[s0] - self.SR_ss_new[s0, :])

    def learn_sas(self, s0, a0, s1):
        if s1 is not None:
            if not self.lesion:
                self.SR_sas_new[s0, a0, :] += self.lr * (
                        self.identity[s1] + self.gamma * self.SR_ss_new[s1,
                                                         :] - self.SR_sas_new[
                                                              s0,
                                                              a0, :])
        else:
            if not self.lesion:
                self.SR_sas_new[s0, a0, :] += self.lr * (self.identity[s0]
                                                         - self.SR_sas_new[s0,
                                                           a0, :])
        # for s, a in itertools.product(
        #         range(self.num_states), range(self.num_actions)):
        #     if s0 == np.argmax(self.transitions[s, a, :]):
        #         self.Q_new[s, a, :] = np.eye(self.num_states)[s0] \
        #                          + self.gamma * self.M_new[s0, :]

    def update_SR(self):
        self.SR_ss = self.SR_ss_new
        self.SR_sas = self.SR_sas_new
        self.SR_ss_new = np.copy(self.SR_ss)
        self.SR_sas_new = np.copy(self.SR_sas)

    def reset_SR(self):
        self.SR_ss_new = np.copy(self.SR_ss)
        self.SR_sas_new = np.copy(self.SR_sas)

    # def new_world(self, env):
    #
    #     new_size = transitions.shape[0]
    #     if new_size > self.num_states:
    #         M = np.pad(
    #             self.M_new, ((0, new_size - self.num_states),
    #                          (0, new_size - self.num_states)),
    #             'constant', constant_values=0)
    #         Q = np.pad(
    #             self.Q_new, ((0, new_size - self.num_states),
    #                          (0, 0),
    #                          (0, new_size - self.num_states)),
    #             'constant', constant_values=0)
    #
    #         self.__init__(self.lr, self.gamma, M, Q)
