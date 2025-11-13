"""
@author: Daniel Shani
"""
import itertools
from collections import deque

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from helper_functions_ import plain_world_like, check_inside_


class Environment:
    """
    Environment class. Contains the gridworld, the transitions, the basis,
    the dimensions, and the goal position.
    """

    def __init__(self, pars, gridworld):

        """
        Initialize the environment. Calculates the allocentric and egocentric
        transition matrices and uses these to calculate the successor
        representations which are used as features by the agent.
        State-Action successor representations are used - defined as
        M[s'|s,a] = I[s'=s] + gamma M[s'|z] where z is the state that follows s
        after action a.

        Args:
            pars: parameters object specified in parameters.py
            gridworld: binary gridworld matrix (size x size)
        """

        self.ego_bins = []
        self.allo_bins = []

        self.allo_to_ego = [{}]
        self.ego_to_allo = [{}]

        self.directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.pars = pars
        self.world = gridworld

        self.size = self.world.shape[0]

        self.transparent = self.pars.transparent
        self.opacity = self.pars.opacity
        self.tangible = True

        self.SR_init = self.pars.SR_init
        #
        # if pars.reward_location == "middle":
        #     self.goal_position_2d = \
        #         [int((self.size - 2) / 2), int((self.size - 2) / 2)]
        # else:
        #     self.goal_position_2d = \
        #         [self.size - 2, self.size - 2]

        self.goal_position_2d = np.argwhere(self.world == -1)[0]

        self.ego_transitions = self.create_transitions_ego(gridworld)
        self.transitions = self.create_transitions_allo(gridworld)

        if self.SR_init == 'empty':
            plain_world = plain_world_like(gridworld)

            self.allo_Q, self.allo_M = get_successor_representation(
                self.create_transitions_allo(plain_world),
                pars.agamma, pars.normalize)
        elif self.SR_init == 'full':
            self.allo_Q, self.allo_M = get_successor_representation(
                self.transitions, pars.agamma, pars.normalize)
        elif self.SR_init == 'blind':
            self.allo_M = np.eye(self.transitions.shape[0])
            self.allo_Q = np.repeat(self.allo_M[:,np.newaxis,:], 4, axis=1)

        # self.allo_Q, self.allo_M = get_successor_representation(
        #     self.transitions, pars.agamma)
        # self.allo_basis = self.allo_Q

        # self.allo_basis = get_successor_representation(self.transitions,
        # pars.agamma) # basis_size x allo_state_size x num_actions
        if self.SR_init == 'blind':
            self.ego_M = np.eye(self.ego_transitions.shape[0])
            self.ego_Q = np.repeat(self.ego_M[:,np.newaxis,:], 4, axis=1)
        else:
            self.ego_Q, self.ego_M = get_successor_representation(
            self.ego_transitions, pars.egamma, pars.normalize)

        self.allo_dim = len(self.allo_bins)
        self.ego_dim = len(self.ego_bins)

        self.position_1d, self.position_2d, self.direction = \
            self.choose_positions()

        # Optional queue of predefined reset positions/directions. If present,
        # `reset()` will consume entries from this deque instead of sampling.
        # This attribute is simple data (tuples, ints) and is fully picklable.
        self.reset_queue = None

    def switch_world(self, gridworld, tangible=True, switch_SRs=True):


        """
        Switches the gridworld.
        """

        self.tangible = tangible
        self.allo_to_ego.append({})
        self.ego_to_allo.append({})
        self.world = gridworld
        self.transitions = self.create_transitions_allo(gridworld)
        SR_transitions_ego = self.create_transitions_ego(gridworld)
        self.ego_transitions = SR_transitions_ego

        if switch_SRs:
            if self.SR_init == 'empty':
                plain_world = plain_world_like(gridworld)
                SR_transitions_allo = self.create_transitions_allo(plain_world)
                self.allo_Q, self.allo_M = get_successor_representation(
                    SR_transitions_allo,
                    self.pars.agamma, self.pars.normalize)
            elif self.SR_init == 'full':
                SR_transitions_allo = self.transitions
                self.allo_Q, self.allo_M = get_successor_representation(
                SR_transitions_allo,
                self.pars.agamma, self.pars.normalize)
            elif self.SR_init == 'blind':
                SR_transitions_allo = self.transitions
                self.allo_M = np.eye(self.transitions.shape[0])
                self.allo_Q = np.repeat(self.allo_M[:, np.newaxis, :], 4, axis=1)




            # self.allo_Q, self.allo_M = get_successor_representation(
            #     self.transitions, pars.agamma)

            allo_indices = np.sum(SR_transitions_allo, axis=(1, 2)) != 0
            ego_indices = np.sum(SR_transitions_ego, axis=(1, 2)) != 0
            if self.SR_init == 'blind':
                self.ego_M = np.eye(self.ego_transitions.shape[0])
                self.ego_Q = np.repeat(self.ego_M[:, np.newaxis, :], 4, axis=1)
            else:

                self.ego_Q, self.ego_M = get_successor_representation(
                SR_transitions_ego, self.pars.egamma, self.pars.normalize)
        else:
            allo_indices = np.sum(self.transitions, axis=(1, 2)) != 0
            ego_indices = np.sum(self.ego_transitions, axis=(1, 2)) != 0


        self.allo_dim = len(self.allo_bins)
        self.ego_dim = len(self.ego_bins)

        # print(self.ego_dim)

        self.size = self.world.shape[0]
        # if self.pars.reward_location == "middle":
        #     self.goal_position_2d = \
        #         [int((self.size - 2) / 2), int((self.size - 2) / 2)]
        # else:
        #     self.goal_position_2d = \
        #         [self.size - 2, self.size - 2]

        self.goal_position_2d = np.argwhere(self.world == -1)[0].tolist()

        return self.allo_dim, self.ego_dim, self.allo_Q, self.allo_M, \
            self.ego_Q, self.ego_M, allo_indices, ego_indices

    def change_reward_location(self, position_2d=None):
        """
        Changes the reward location to a random unoccupied location in the
        gridworld.
        """
        if position_2d is None:
            x = np.random.choice(self.size)
            y = np.random.choice(self.size)
            if self.world[x, y] == 0:
                self.goal_position_2d = [x, y]
            else:
                self.change_reward_location()
        else:
            self.goal_position_2d = position_2d

    def choose_positions(self):
        """
        Chooses a random unoccupied position and direction for the agent.
        """
        if self.pars.random_starts:
            position_1d = np.random.choice(self.size ** 2)
            position_2d = self.allo_bins[position_1d]
            direction = np.random.choice(4)
        else:
            position_2d = [1, 1]
            direction = 0
            position_1d = self.get_1d_pos(position_2d)
        if self.world[position_2d[0], position_2d[1]] != 0:
            return self.choose_positions()

        return position_1d, position_2d, direction

    def get_1d_pos(self, position_2d):
        """
        Args:
            position_2d: [x, y] position of the agent

        Returns: 1d_pos (int): allocentric state of the agent
        """
        # if direction is None:
        #     direction = self.direction

        return self.allo_bins.index(tuple(position_2d))

    def get_2d_pos(self, position_1d):
        """
        Args:
            position_1d: allocentric state of the agent

        Returns: [x, y] position of the agent
        """

        return self.allo_bins[position_1d]

    def get_ego_obs(self, position_2d, direction):
        """
        Centers the world about the agent and returns the egocentric
        observation.
        Args:
            position_2d: [x, y] position of the agent
            direction: direction of the agent

        Returns: ego_obs (int): egocentric state of the agent at the given
        position when facing the given direction
        """
        return self.get_egocentric_view(self.world, position_2d, direction)[0]

    def step(self, action):

        """
        Step in the environment.
        If the agent is in the reward location then any action leads to reward
        and the episode is done.
        If the agent is not in the reward location then the action is taken and
        the agent moves to the next state.
        If the next state is the same as the previous then this is interpreted
        as hitting a wall and a wall cost is induced.
        Args:
            action: action taken
        Returns:
            position: current allocentric state
            observation: current egocentric state
            direction: current direction
            reward: reward for taking action
            done: True if the agent is in the reward location
        """

        done = bool(np.all(self.position_2d == self.goal_position_2d))
        if done:
            return None, None, None, self.pars.end_reward, done

        self.direction = (self.direction + action) % 4
        new_position_option = np.copy(self.position_2d)
        if action == 0:
            new_position_option += self.directions[self.direction]


        if self.world[new_position_option[0],
        new_position_option[1]] <= (0 if self.tangible else 1):
            self.position_2d = new_position_option
            # print(self.directions[self.direction])
            reward = self.pars.step_cost
        else:
            reward = self.pars.wall_cost

        self.position_1d = \
            self.get_1d_pos(self.position_2d)
        # done = bool(np.all(self.position_2d == self.goal_position_2d))
        #
        # if done:
        #     reward = self.pars.end_reward
        return self.position_1d, self.get_egocentric_view(
            self.world, self.position_2d, self.direction)[0], \
            self.direction, reward, done

    def get_direction(self):
        return self.direction

    def reset(self):
        """
        Initialises agent. Returns a tuple of allocentric state, egocentric
        state and direction.
        """
        if getattr(self, 'reset_queue', None):
            try:
                position_2d, direction = self.reset_queue.popleft()
                self.position_2d = [int(position_2d[0]), int(position_2d[1])]
                self.direction = int(direction)
                self.position_1d = self.get_1d_pos(self.position_2d)
            except IndexError:
                # Fallback to normal sampling if queue is empty
                self.position_1d, self.position_2d, self.direction = \
                    self.choose_positions()
        else:
            self.position_1d, self.position_2d, self.direction = \
                self.choose_positions()
        return self.position_1d, self.get_egocentric_view(
            self.world, self.position_2d, self.direction)[0], self.direction

    def show_world(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.world, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def produce_schematic(self, x=3, y=4, d=2, save=True, ax=None):
        if ax is None:
            ax = plt.gca()
        from matplotlib import patches

        # allocentric subfigure

        ax3 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=1)

        square = np.zeros_like(self.world)
        # add cross at position (x,y)

        square[y, x] = 1

        # plt.gca().add_patch(
        #     patches.Circle(
        #         (x, y), 1, edgecolor='r', facecolor='r', fill=True))

        ax3.imshow(square, cmap='Greys')
        # add dotted arrows in directions North, East, South, West from
        # position (x,y) or length 2
        plt.arrow(
            x, y + 0.5, 0, 1., color='b', linewidth=1,
            head_width=0.5, head_length=0.5)
        plt.arrow(
            x + 0.5, y, 1., 0, color='b', linewidth=1,
            head_width=0.5, head_length=0.5)
        plt.arrow(
            x, y - 0.5, 0, -1., color='b', linewidth=1,
            head_width=0.5, head_length=0.5)
        plt.arrow(
            x - 0.5, y, -1., 0, color='b', linewidth=1,
            head_width=0.5, head_length=0.5)

        ax3.set_title('$s^A$')
        ax3.set_ylabel('$a^A$', color='b')
        plt.xticks([])
        plt.yticks([])
        # ax3.set_anchor('W')

        # egocentric subfigure

        ax2 = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=1)
        ego = self.get_egocentric_view(self.world, (y, x), d, display=True)
        ax2.set_ylabel('$a^E$', color='b')
        # add up arrow and curly clockwise and anticlockwise arrows
        plt.gca().add_patch(
            patches.FancyArrowPatch(
                (self.pars.horizon, self.pars.horizon), (self.pars.horizon,
                                                         self.pars.horizon - 1),
                arrowstyle='Simple, '
                           'tail_width=0.1, '
                           'head_width=4, head_length=8',
                color='b'))
        plt.gca().add_patch(
            patches.FancyArrowPatch(
                (self.pars.horizon, self.pars.horizon), (self.pars.horizon + .5,
                                                         self.pars.horizon),
                arrowstyle='Simple, '
                           'tail_width=0.1, '
                           'head_width=4, head_length=8',
                color='b',
                connectionstyle="arc3,rad=-1"))
        plt.gca().add_patch(
            patches.FancyArrowPatch(
                (self.pars.horizon, self.pars.horizon), (self.pars.horizon - .5,
                                                         self.pars.horizon),
                arrowstyle='Simple, '
                           'tail_width=0.1, '
                           'head_width=4, head_length=8',
                color='b',
                connectionstyle="arc3,rad=1"))
        plt.gca().add_patch(
            patches.FancyArrowPatch(
                (self.pars.horizon, self.pars.horizon - .75), (self.pars.horizon,
                                                               self.pars.horizon + .4),
                arrowstyle='Simple, '
                           'tail_width=0.1, '
                           'head_width=4, head_length=8',
                color='b'))

        ax2.set_title('$s^E$')
        # ax2.set_anchor('W')

        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3)
        masked_array = np.ma.masked_where(self.world == -1, self.world)
        cmap = cm.Greys
        cmap.set_bad(color='green')

        ax1.imshow(masked_array, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        # add agent at position (x,y) which consists of a small yellow
        #  CHARACTER with a white border in direction d

        plt.gca().add_patch(
            patches.Circle(
                (x, y), 0.5, edgecolor='w', facecolor='y', fill=True))
        plt.arrow(
            x, y, 0.5 * np.sin(np.pi * d / 2),
                  -0.5 * np.cos(np.pi * d / 2), facecolor='y', edgecolor='w',
            linewidth=1,
            head_width=0.5, head_length=0.5)

        # plt.gca().add_patch(
        #     patches.RegularPolygon(
        #         (x, y), 3, 0.5, orientation=np.pi + np.pi * d / 2,
        #         edgecolor='w', facecolor='y', fill=True))

        # add red rectangle of width 5 and height 3 around location (x,
        # y) rotated in direction d around point (x, y)
        plt.gca().add_patch(
            patches.Rectangle(
                (x - self.pars.horizon - .5, y - self.pars.horizon - .5),
                2 * self.pars.horizon + 1,
                self.pars.horizon + 1,
                edgecolor='r',
                facecolor='none',
                fill=False, angle=90 * d, rotation_point=(x, y)))

        aliases = self.ego_to_allo[-1].get(ego[0])
        for y_, x_, d_ in aliases:
            plt.gca().add_patch(
                patches.Rectangle(
                    (x_ - self.pars.horizon - .5, y_ - self.pars.horizon - .5),
                    2 * self.pars.horizon + 1,
                    self.pars.horizon + 1,
                    edgecolor='lightblue',
                    linestyle='dashed',
                    facecolor='none',
                    fill=False, angle=90 * d_, rotation_point=(x_, y_)))

        # ax1.set_anchor('W')
        plt.tight_layout()

        if save:
            plt.savefig('schematic.png')
        plt.show()

    # def get_optimal_value_function(self, gamma, N):
    #     """
    #     Get the optimal value function for the agent. Used in value function
    #     regression. Uses dynamic programming to get the optimal value function.
    #     """

    #     optimal_values = np.zeros((4, self.size, self.size, 4))
    #     for _, d, x, y in itertools.product(
    #             range(N), range(4),
    #             range(self.size),
    #             range(self.size)):
    #         if self.world[x, y] == 0:
    #             for a in range(4):
    #                 d_, x_, y_ = \
    #                     np.unravel_index(
    #                         np.argmax(
    #                             self.transitions_[d, x, y, a, :, :, :]),
    #                         (4, self.size, self.size))
    #                 if np.all([x_, y_] == self.goal_position_2d):
    #                     reward = self.pars.end_reward
    #                     optimal_values[d, x, y, a] = reward
    #                 else:
    #                     if np.all([d, x, y] == [d_, x_, y_]):
    #                         reward = self.pars.wall_cost
    #                     else:
    #                         reward = self.pars.step_cost

    #                     optimal_values[d, x, y, a] = reward + \
    #                                                  gamma * np.max(
    #                         optimal_values[d_, x_, y_, :])

    #     show_4x4(optimal_values)

    #     return optimal_values

    # def get_full_theta(self):
    #     """
    #     Get the full feature vector. Used for regression.
    #     """
    #     theta = np.zeros(
    #         (self.allo_dim + self.ego_dim, 4,
    #          self.size, self.size, 4))
    #     for d in range(4):
    #         for x in range(self.size):
    #             for y in range(self.size):
    #                 if self.world[x, y] == 0:
    #                     state = self.get_1d_pos([x, y])
    #                     ego = self.get_egocentric_view(self.world, [x, y], d)[0]
    #                     theta_allo = self.allo_basis[state]
    #                     theta_ego = self.ego_basis[ego]
    #                     theta[:, d, x, y, :] = \
    #                         np.concatenate((theta_allo, theta_ego), axis=1)
    #     return theta

    def create_transitions_ego(self, world=None):
        """
        Creates the transitions matrix for the head-dependent agent. Cycles
        through possible positions (x,y) and directions (N,E,S,W) in the world
        and possible actions (fwd, turn right, turn 180, turn left) and creates
        the allocentric and egocentric state-action adjacency matrices.
        Args:
            world: binary matrix representing the world. 1 represents a wall, 0
            represents a free space.

        Returns:
            transition_matrix: 4xsizexsizex4x4xsizexsize array representing the
            state-action transitions. transitions[d,x,y,a,d_,x_,y_]=1 if a takes
            agent from (d,x,y) to (d_,x_,y_).
            ego_transitions: num_ego_states x 4 x num_ego_states array
            representing transitions between ego states.
        """

        if world is None:
            world = self.world
        direction_vectors = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        size = world.shape[0]
        ego_transitions = np.zeros((3 ** 8, 4, 3 ** 8))
        for direction, i, j, action in itertools.product(
                range(4), range(size),
                range(size), range(4)):
            if world[i, j] <= (0 if self.tangible else 1):
                current_view = \
                    self.get_egocentric_view(world, [i, j], direction)[0]
                direction_new = (direction + action) % 4
                v_new = direction_vectors[direction_new]
                if world[i, j] == -1:
                    ego_transitions[current_view, action, current_view] += 1
                else:
                    if action == 0:  # forward
                        if world[i + v_new[0], j + v_new[1]] <= 0:
                            new_view = self.get_egocentric_view(
                                world, [i + v_new[0], j + v_new[1]],
                                direction_new)[0]
                            ego_transitions[current_view, action, new_view] += 1
                        else:
                            ego_transitions[current_view, action, current_view] += 1
                    elif action == 2:
                        new_view = self.get_egocentric_view(
                            world, [i, j],
                            direction_new)[0]
                        ego_transitions[current_view, action, new_view] += 1
                    elif action % 2 == 1:  # rotation
                        new_view = self.get_egocentric_view(
                            world, [i, j],
                            direction_new)[0]
                        ego_transitions[current_view, action, new_view] += 1
        return ego_transitions[:len(self.ego_bins), :, :len(self.ego_bins)]

    def create_transitions_allo(self, gridworld=None):
        """
        Creates the transition matrix for the allocentric agent. Returns the
        un-normalised transition matrix T[s,a,s_] for s = (x,y) and a = (N,E,
        S,W).

        Returns:
            transitions: (allo_dim x 4 x allo_dim) array representing P(s,a,s_)
        """
        if gridworld is None:
            gridworld: np.ndarray = self.world
        size = gridworld.shape[0]
        transitions = np.zeros((size ** 2, 4, size ** 2))
        for x, y, a in itertools.product(
                range(size), range(size),
                range(4)):
            if (x, y) in self.allo_bins:
                i = self.allo_bins.index((x, y))
            else:
                self.allo_bins.append((x, y))
                i = len(self.allo_bins) - 1

            if gridworld[x,y] == -1:
                transitions[i, :, i] = 1
                continue
            else:
                if gridworld[x, y] == 0:
                    v = self.directions[a]
                    x_new = x + v[0]
                    y_new = y + v[1]
                    if (x_new, y_new) in self.allo_bins:
                        j = self.allo_bins.index((x_new, y_new))
                    else:
                        self.allo_bins.append((x_new, y_new))
                        j = len(self.allo_bins) - 1

                    if gridworld[x + v[0], y + v[1]] <= 0:
                        transitions[i, a, j] += 1
                    else:
                        transitions[i, a, i] += 1

        return transitions

    def get_egocentric_view(self, world, pos, k, display=False, ax=None):
        """
        Get the view of the world from the perspective of the agent.
        Args:
            world: binary matrix representing the world. 1 represents a wall, 0
            represents a free space.
            pos: position of the agent in the world. [i,j]
            k: direction of the agent. 0=N, 1=E, 2=S, 3=W

        Returns:
            int: integer representing the view of the world (which is originally
            a binary vector).
        """

        assert len(pos) == 2
        i, j = pos
        if self.pars.local:
            centered_world = world[max(0, i - self.pars.horizon):min(
                i +
                self.pars.horizon +
                1,
                world.shape[0]),
                             max(j - self.pars.horizon, 0):min(
                                 j +
                                 self.pars.horizon
                                 + 1,
                                 world.shape[1])]

            if i - self.pars.horizon < 0:
                centered_world = np.concatenate(
                    [np.inf * np.ones(
                        (self.pars.horizon - i,
                         centered_world.shape[1])),
                     centered_world], axis=0)
            if i + self.pars.horizon + 1 > world.shape[0]:
                centered_world = np.concatenate(
                    [centered_world,
                     np.inf * np.ones(
                         (i + self.pars.horizon + 1 - world.shape[0],
                          centered_world.shape[1]))], axis=0)
            if j - self.pars.horizon < 0:
                centered_world = np.concatenate(
                    [np.inf * np.ones(
                        (centered_world.shape[0],
                         self.pars.horizon - j)),
                     centered_world], axis=1)
            if j + self.pars.horizon + 1 > world.shape[1]:
                centered_world = np.concatenate(
                    [centered_world,
                     np.inf * np.ones(
                         (centered_world.shape[0],
                          j + self.pars.horizon + 1 - world.shape[1]))],
                    axis=1)

            current_view = np.rot90(centered_world, k)

            current_view = current_view[
                           0:current_view.shape[0] - self.pars.horizon, :]
            current_view = np.maximum(current_view, 0)
            current_view = current_view.astype(float)

            if self.opacity:
                to_mask = [[np.zeros(4) for _ in range(current_view.shape[1])]
                           for _ in
                           range(current_view.shape[0])]
                mask = np.ones_like(current_view)

                # attempt at finding cone: difference between to left and to
                # right will be which lines go to top vs bottom

                for i, j in itertools.product(
                        range(current_view.shape[0]), range(
                            current_view.shape[1])):
                    if current_view[i, j] > 0:
                        # location of block
                        x = j - self.pars.horizon
                        y = current_view.shape[0] - (i + 1)
                        # cone = find_cone(x,y)

                        # mask out areas behind block

                        for i_, j_ in itertools.product(
                                range(current_view.shape[0]),
                                range(current_view.shape[1])):
                            if (i, j) != (i_, j_):
                                y_ = current_view.shape[0] - (i_ + 1)
                                x_ = j_ - self.pars.horizon
                                if (abs(x_) >= abs(x)) and (abs(y_) >= abs(
                                        y)) and (x_ * x >= 0) and (y_ * y >= 0):
                                    points = [(x_ + 1 / 2, y_ + 1 / 2),
                                              (x_ + 1 / 2, y_ - 1 / 2),
                                              (x_ - 1 / 2, y_ - 1 / 2),
                                              (x_ - 1 / 2, y_ + 1 / 2)]

                                    # points = [(y_ + 1 / 2) / (x_ - 1 / 2),
                                    #       max(y_ - 1 / 2,0) / (x_ + 1 / 2),
                                    #       max(y_ - 1 / 2,0) / (x_ - 1 / 2),
                                    #       (y_ + 1 / 2) / (x_ + 1 / 2)]
                                    for index, p_ in enumerate(points):
                                        if check_inside_(p_, (x, y)):
                                            to_mask[i_][j_][index] = 1

                for i, j in itertools.product(
                        range(current_view.shape[0]), range(
                            current_view.shape[1])):
                    if np.sum(to_mask[i][j] == 1) > 4 - self.opacity:
                        mask[i][j] = 0

                # for i, j in itertools.product(
                #         range(current_view.shape[0]), range(
                #             current_view.shape[1])):
                #     if current_view[i, j] > 0:
                #         print(i, j)
                #
                #         x = j - self.pars.horizon
                #         y = current_view.shape[0] - (i + 1)
                #
                #         cone = find_cone(
                #             [(y + 1 / 2) / (x - 1 / 2),
                #              (y - 1 / 2) / (x + 1 / 2),
                #              (y - 1 / 2) / (x - 1 / 2),
                #              (y + 1 / 2) / (x + 1 / 2)])
                #
                #         for i_ in range(i + 1):
                #             if j > self.pars.horizon:
                #                 for j_ in range(j, current_view.shape[1], 1):
                #                     if (i, j) != (i_, j_):
                #                         y_ = current_view.shape[0] - (i_ + 1)
                #                         x_ = j_ - self.pars.horizon
                #                         points = [(y_ + 1 / 2) / (x_ - 1 / 2),
                #                                   (y_ - 1 / 2) / (x_ + 1 / 2),
                #                                   (y_ - 1 / 2) / (x_ - 1 / 2),
                #                                   (y_ + 1 / 2) / (x_ + 1 / 2)]
                #                         # points = [y_/x_ if x_ != 0 else np.inf]
                #                         if np.all(
                #                                 [check_inside(p, cone) for p in
                #                                  points]) and current_view[i_,
                #                         j_] != np.inf:
                #                             current_view[i_, j_] = -np.inf
                #             else:
                #                 for j_ in range(j + 1):
                #                     if (i, j) != (i_, j_):
                #                         y_ = current_view.shape[0] - (i_ + 1)
                #                         x_ = j_ - self.pars.horizon
                #                         points = [(y_ + 1 / 2) / (x_ - 1 / 2),
                #                                   (y_ - 1 / 2) / (x_ + 1 / 2),
                #                                   (y_ - 1 / 2) / (x_ - 1 / 2),
                #                                   (y_ + 1 / 2) / (x_ + 1 / 2)]
                #                         # points = [y_/x_ if x_ != 0 else np.inf]
                #                         if np.all(
                #                                 [check_inside(p, cone) for p in
                #                                  points]) and current_view[i_,
                #                         j_] != np.inf:
                #                             current_view[i_, j_] = -np.inf

                # # left:
                # k = 0
                # print(k)
                # while (current_view[-1, self.pars.horizon - k] <= 0 <
                #        self.pars.horizon - k):
                #     k += 1
                # current_view[-1, :self.pars.horizon - k] = np.inf
                # # right
                # k = 0
                # print(k)
                # while (k < self.pars.horizon + 1 and current_view[
                #     -1][self.pars.horizon + k] <= 0):
                #     k += 1
                # current_view[-1, self.pars.horizon + k + 1:] = np.inf
                #
                # # forward
                # k = 0
                # print(k)
                # while (current_view[-1 - k][self.pars.horizon] <= 0 and
                #        k < self.pars.horizon):
                #     k += 1
                # current_view[:-1 - k, self.pars.horizon] = np.inf
                #
                # # right diag
                # k = 0
                # while (current_view[-1 - k, self.pars.horizon + k] <= 0 and
                #        k < self.pars.horizon):
                #     k += 1
                # for j in range(self.pars.horizon, k, -1):
                #     current_view[-1 - j, self.pars.horizon + j] = np.inf
                # # current_view[:-1-k,self.pars.horizon+k:] = np.inf
                #
                # # left diag
                # k = 0
                # while current_view[
                #     -1 - k, self.pars.horizon - k] <= 0 < self.pars.horizon - k:
                #     k += 1
                # for j in range(self.pars.horizon, k, -1):
                #     current_view[-1 - j, self.pars.horizon - j] = np.inf
                #
                #
                # for i in range(current_view.shape[0]):
                #     for j in range(current_view.shape[1]):
                #         if current_view[i,j]>0:
                #             line_x = j-self.pars.horizon
                #             line_y = current_view.shape[0]-(i+1)
                #             m = line_y/line_x
                #             for y in range(i):
                #                 for x in range(current_view.shape[0]):
                #                     y_ = current_view.shape[0]-(y+1)
                #                     x_ = x-self.pars.horizon
                #                     if (y_ == m*x_ and \
                #                             current_view[-y,x]!= np.inf):
                #                         current_view[-y,x] = np.inf
                #
                #
                #
                # changed = True
                # while changed:
                #     changed = False
                #     j = 1  # row from bottom
                #     k = 0  # column
                #     while j < self.pars.horizon:
                #         while k < 2 * self.pars.horizon:
                #             if np.all(current_view[-1 - j, k:k + 2] > 0):
                #                 if (k < self.pars.horizon and
                #                         current_view[
                #                             -1 - j - 1, k] != np.inf):
                #                     current_view[-1 - j - 1, k] = np.inf
                #                     changed = True
                #                 elif (k >= self.pars.horizon and
                #                       current_view[
                #                           -1 - j - 1, k + 1] != np.inf):
                #                     current_view[-1 - j - 1, k + 1] = np.inf
                #                     changed = True
                #
                #             k += 1
                #         j += 1
                #         k = 0
                #
                # # loop
                # # two - to - one - fills
                current_view[mask == 0] = -np.inf
            # plt.imshow(current_view)
            # plt.show()
            # print(current_view)
            if display:
                if ax:
                    ax.imshow(
                        current_view, cmap='Greys', vmin=np.min(world),
                        vmax=np.max(world))
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:

                    plt.imshow(
                        current_view, cmap='Greys', vmin=np.min(world),
                        vmax=np.max(world))
                    plt.xticks([])
                    plt.yticks([])

            current_view = current_view.flatten()
        else:
            current_view = []
            for d in range(4):
                direction = self.directions[d]
                dist = 0
                while world[
                    i + dist * direction[0], j + dist * direction[1]] <= 0 \
                        and dist < self.pars.horizon:
                    dist += 1
                # trying to remove the reward from the view
                color = max(
                    [0., world[i + dist * direction[0], j + dist *
                               direction[1]]])
                current_view.append((dist, color))

            current_view = deque(current_view)
            current_view.rotate(-k)
            del current_view[2]
            current_view = np.array(current_view).flatten()

        if tuple(current_view) in self.ego_bins:
            num = self.ego_bins.index(tuple(current_view))
            self.allo_to_ego[-1][(pos[0], pos[1], k)] = num
            if num not in self.ego_to_allo[-1].keys():
                self.ego_to_allo[-1][num] = []
            if (pos[0], pos[1], k) in self.ego_to_allo[-1].get(num, []):
                pass
            else:
                # if num not in self.ego_to_allo[-1].keys():
                #     self.ego_to_allo[-1][num] = []
                self.ego_to_allo[-1].get(num, []).append(
                    (pos[0], pos[1],
                     k))

            return num, tuple(current_view)
        else:
            self.ego_bins.append(tuple(current_view))
            return self.get_egocentric_view(world, pos, k)

    def numpy2int(self, array):
        """Convert egocentric numpy array into integer."""
        array = tuple(array)
        return self.ego_bins.index(array)

    def int2numpy(self, num):
        num = int(num)
        """Convert a positive integer num into an m-bit bit vector"""
        return np.array(self.ego_bins[num])

# def show_predictions(clf, theta, q):
#     """Used for value function regression."""
#     q_ = np.reshape(q, (-1,))
#     theta_ = np.reshape(theta, (-1, q_.shape[0]))

#     clf.fit(theta_.T, q_)
#     weight = clf.coef_
#     q_fit = clf.predict(theta_.T).reshape(q.shape)

#     show_4x4(q_fit)
#     return weight


def get_successor_representation(transitions, gamma, normalize=False):
    """
    Calculate the state-action successor representations,
    defined as M[s'|s,a] = I[s'=s] + gamma M[s'|z],
    where z is the state that comes after state s and action a.
    Args:
        transitions: flattened array representing the state-action
        transitions. transitions[s,a,s_]=1 if a takes agent from s to s_.
        gamma: discount factor.
    Returns:
        Q: state-action successor representation.
        M: state-state successor representation.
    """
    for i in range(transitions.shape[0]):
        for j in range(transitions.shape[1]):
            if transitions[i, j, :].sum():
                transitions[i, j, :] = transitions[i, j, :] / transitions[i, j,
                                                              :].sum()
    P = np.mean(transitions, 1)
    M = np.linalg.inv(np.eye(P.shape[0]) - gamma * P)  # M(s, sbar)

    Q = np.zeros((M.shape[0], 4, M.shape[0]))
    for s_0, a_0 in itertools.product(range(transitions.shape[0]), range(4)):
        s_1 = np.argmax(transitions[s_0, a_0, :])
        Q[s_0, a_0, :] = np.eye(transitions.shape[0])[s_0] \
                         + gamma * M[s_1, :]

    if normalize:
        M /= M.sum(axis=0)
        Q /= Q.sum(axis=0)

    return Q, M
