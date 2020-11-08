import sim
import numpy as np


class Model:
    def __init__(self, map):
        # map object as defined in sim.py
        self.map = map

    def step(self, state, action):
        """
        calculate the transition and expected rewards from given state and action pair
        :param state: tuple containing three elements. The first element is another tuple represent the current
        coordinate of the car (we have to use tuple here instead of list to make it hashable so we can use state as
        keys in the dictionary representing transition function); the second element is a boolean, whether the traffic
        has cleared or not; the third element is a float, the current time
        :param action: int, {0, 1, 2, 3}, the action to take
        :return:
        transition: dictionary, transition function. the keys are states, and the values are the probability of
        transitioning to that state after taking the given action at given state
        reward: float, expected reward for taking given action at given state. For this problem, we are using the
        negative of travel time as rewards
        """
        # unpack the state
        coord = state[0]
        cleared = state[1]
        time = state[2]
        transition = {}
        reward = 0
        if np.linalg.norm(coord - self.map.goal) < 1e-5:
            # the goal is an absorbing state, the state will not change once we reach goal
            transition[state] = 1
            reward = 0
        else:
            # calculate next coordinate given action
            next_coord = coord + self.map.action_map[action]
            # make sure the action does not go off the map
            next_coord = np.maximum(np.minimum(next_coord, self.map.n - 1), 0)
            if cleared:
                # if the traffic is cleared, it will remain cleared
                next_state = (tuple(next_coord), cleared, time + 1)
                transition[next_state] = 1
                reward += -1
            else:
                # possible outcome: the traffic clears
                p_clear = min(self.map.f_slope * time, 1)
                next_state = (tuple(next_coord), True, time + 1)
                transition[next_state] = p_clear
                reward += - p_clear
                # possible outcome: the traffic does not clear
                # find the travel time
                travel_time = 1
                if np.linalg.norm(coord - self.map.traffic) < 1e-5 or \
                        np.linalg.norm(next_coord - self.map.traffic) < 1e-5:
                    travel_time = 2
                elif np.linalg.norm(coord - self.map.traffic) < 1 + 1e-5 or \
                        np.linalg.norm(next_coord - self.map.traffic) < 1 + 1e-5:
                    travel_time = 1.5
                next_state = (tuple(next_coord), False, time + travel_time)
                transition[next_state] = 1 - p_clear
                reward += - (1 - p_clear) * travel_time
        return transition, reward

    def u_lower_bound(self, state):
        """
        estimate lower bound for value function. the lower bound is estimated by assuming all edges have travel time 2,
        which is the max in this problem
        :param state: tuple, details please see docstring for function step
        :return: u, float, lower bound for value function
        """
        coord = state[0]
        dist = abs(self.map.goal[0] - coord[0]) + abs(self.map.goal[1] - coord[1])
        u = - 2 * dist
        return u

    def Q_upper_bound(self, state, action):
        """
        estimate the upper bound for action value function. the upper bound is estimated by assuming the traffic has
        cleared, and all edges have travel time 1
        :param state: tuple, details please see docstring for function step
        :param action: int, {0, 1, 2, 3}, details please see docstring for function step
        :return: q, float, upper bound for action value function
        """
        coord = state[0]
        # calculate next coordinate given action
        next_coord = coord + self.map.action_map[action]
        # make sure the action does not go off the map
        next_coord = np.maximum(np.minimum(next_coord, self.map.n - 1), 0)
        dist = abs(self.map.goal[0] - next_coord[0]) + abs(self.map.goal[1] - next_coord[1])
        q = -dist
        return q


def branch_and_bound(model, state, depth):
    """
    use branch and bound to calculate the best action to take
    :param model: model object, defines the problem we are solving
    :param state: tuple, detail please docstring for function model.step
    :param depth: int, depth for the branch and bound algorithm
    :return:
    a_best: int, {0, 1, 2, 3}, the best action to take at current state
    u_best: float, the best value function at current state (achieved by taking action a_best)
    """
    if depth <= 0:
        # base case, we have reached the bottom of the branching
        a_best = None
        u_best = model.u_lower_bound(state)
    else:
        # init variables
        a_best = None
        u_best = -np.inf
        # iterate over all possible actions
        for a in range(len(model.map.action_map)):
            if model.Q_upper_bound(state, a) < u_best:
                # if the upper bound is less than the best we have seen, prune it
                # note that we are not pruning all subsequent branches because our actions are not sorted according to
                # descending upper bounds
                continue
            # calculate transition and rewards
            t, r = model.step(state, a)
            # calculate value function using lookahead
            u = r
            for s in t.keys():
                _, u_next = branch_and_bound(model, s, depth - 1)
                u += t[s] * u_next
            # update if the action is better than what we have seen so far
            if u > u_best:
                u_best = u
                a_best = a
    return a_best, u_best


if __name__ == '__main__':
    depth = 3
    map = sim.Map(5, np.array([2, 0]), np.array([4, 4]), np.array([3, 2]))
    model = Model(map)
    actions = []
    # run branch and bound until we reach goal
    while np.linalg.norm(map.state - map.goal) > 1e-5:
        a, _ = branch_and_bound(model, (map.state, map.cleared, map.time), depth)
        print('taking action {0} from state {1} at time {2}'.format(a, map.state, map.time))
        actions.append(a)
        map.step(a)
    print('total cost {}'.format(map.time))
