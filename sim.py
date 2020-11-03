import numpy as np


class Map:
    def __init__(self, n, start, goal, traffic):
        # number of streets in each direction
        self.n = n
        # np array, coordinate of starting point
        self.start = start
        # np array, coordinate of goal
        self.goal = goal
        # np array, coordinate of traffic jam
        # the travel of streets directly connected to the traffic  will have travel time 2,
        # and streets that are one block away from the traffic will have travel time of 1.5
        #   |          |
        #  1.5         1
        #   |          |
        #   *---1.5----*---1----*
        #   |          |
        #   2         1.5
        #   |          |
        # traffic --2--*---1.5--*
        self.traffic = traffic
        # current time
        self.time = 0
        # current location of the car
        self.state = start
        # four actions are available, 0:left, 1: right, 2: down, 3: up
        self.action_map = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]
        # slope of sigma/time, where sigma is the standard deviation of the noise
        self.sigma_slope = 0.025
        # minimum travel time for one block, used when adding noise to travel time
        self.min_travel_time = 0.5
        # penalty for not reaching the goal
        self.goal_penalty = 10

    def step(self, action):
        """
        run the simulation for one step
        :param action: int, from {0, 1, 2, 3}
        :return: None
        """
        current = self.state
        # calculate next state given action
        next = self.state + self.action_map[action]
        # make sure the action does not go off the map
        next = np.maximum(np.minimum(next, self.n - 1), 0)
        # get the travel time for the action
        travel_time = 1
        if np.linalg.norm(current - self.traffic) < 1e-5 or np.linalg.norm(next - self.traffic) < 1e-5:
            travel_time = 2
        elif np.linalg.norm(current - self.traffic) < 1 + 1e-5 or np.linalg.norm(next - self.traffic) < 1 + 1e-5:
            travel_time = 1.5
        # add noise to the travel time, standard deviation of noise increase with time
        travel_time += np.random.normal(0, self.sigma_slope * self.time)
        # make sure the travel time is above a minimum to be realistic
        travel_time = max(travel_time, self.min_travel_time)
        # update state and time
        self.state = next
        self.time = self.time + travel_time
        return

    def run(self, actions):
        """
        run the simulation given a sequence of actions
        :param actions: list of int, from {0, 1, 2, 3}
        :return: cost, total cost, including travel time and penalty for final state
        """
        # run the simulation
        for a in actions:
            self.step(a)
            print('at state {0}, time {1}'.format(self.state, self.time))
        # add penalty for not reaching goal
        cost = self.time + self.goal_penalty * np.linalg.norm(self.state - self.goal)
        print('total cost {}'.format(cost))
        return cost


if __name__ == '__main__':
    sim_map = Map(5, np.array([0, 0]), np.array([4, 4]), np.array([3, 2]))
    ref_path = [3, 3, 3, 3, 1, 1, 1, 1]
    cost = sim_map.run(ref_path)
