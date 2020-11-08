import numpy as np
from sim import *
import time

class Offline():
    """Solves for optimal route"""
    def __init__(self,map,k_max,discount_factor=1,max_t=45):
        self.map = map 
        self.num_a = len(map.action_map)
        self.DF = discount_factor
        self.k_max = k_max
        self.all_coord = []
        self.max_t = max_t
        for i in range(map.n):
            for j in range(map.n):
                self.all_coord.append((i,j)) 
        self.state_space = []
        for x in self.all_coord:
            for t in np.arange(0,self.max_t,.5):
                self.state_space.append((x,t,True))
                self.state_space.append((x,t,False))
        self.U_mapping = {}
        i = 0
        for s in self.state_space:
            self.U_mapping[s] = i
            i += 1

    def step(self, state, action):
        """
        calculate the transition and expected rewards from given state and action pair
        :param state: tuple containing three elements. The first element is another tuple represent the current
        coordinate of the car (we have to use tuple here instead of list to make it hashable so we can use state as
        keys in the dictionary representing transition function); the second element is a boolean, whether the traffic
        has cleared or not; the third element is a float, the current time
        :param action: 
        :return:
        transition: dictionary, transition function. the keys are states, and the values are the probability of
        transitioning to that state after taking the given action at given state
        reward: float, expected reward for taking given action at given state. For this problem, we are using the
        negative of travel time as rewards
        """
        # unpack the state
        coord = state[0]
        cleared = state[2]
        time = state[1]
        transition = {}
        reward = 0
        if np.linalg.norm(coord - self.map.goal) < 1e-5:
            # the goal is an absorbing state, the state will not change once we reach goal
            transition[state] = 1
            reward = 0
            next_state_ret = [state]
        else:
            # calculate next coordinate given action
            next_coord = coord + action
            # make sure the action does not go off the map
            next_coord = np.maximum(np.minimum(next_coord, self.map.n - 1), 0)
            if cleared:
                # if the traffic is cleared, it will remain cleared
                next_state = (tuple(next_coord), time + 1,cleared)
                transition[next_state] = 1
                reward += -1
                next_state_ret = [next_state]
            else:
                # possible outcome: the traffic clears
                p_clear = min(self.map.f_slope * time, 1)
                next_state = (tuple(next_coord), time + 1,True)
                transition[next_state] = p_clear
                reward += - p_clear
                next_state_c = next_state
                # possible outcome: the traffic does not clear
                # find the travel time
                travel_time = 1
                if np.linalg.norm(coord - self.map.traffic) < 1e-5 or \
                        np.linalg.norm(next_coord - self.map.traffic) < 1e-5:
                    travel_time = 2
                elif np.linalg.norm(coord - self.map.traffic) < 1 + 1e-5 or \
                        np.linalg.norm(next_coord - self.map.traffic) < 1 + 1e-5:
                    travel_time = 1.5
                next_state = (tuple(next_coord), time + travel_time, False)
                transition[next_state] = 1 - p_clear
                reward += - (1 - p_clear) * travel_time
                next_state_nc = next_state
                next_state_ret = [next_state_nc,next_state_c]
        return transition, reward, next_state_ret

    def lookahead(self,U,s,a):
        term = 0
        t,r,new_s = self.step(s,a)
        for sp in new_s:
            if sp[1] < self.max_t:
                term += t[sp]*U[self.U_mapping[sp]]
        term *= self.DF
        return r + term

    def backup(self, U, s):
        arg = []
        for a in self.map.action_map:
            arg.append(self.lookahead(U,s,a))
        u = np.amax(np.array(arg))
        return u

    def ValFPoly(self,U): # page 134
        num = len(self.state_space)
        policy = np.zeros((num,))

        for s in self.U_mapping.keys():
            term_u = np.zeros((self.num_a,))
            term_a = np.zeros((self.num_a,))
            i_a = 0
            for a in [0,1,2,3]:
                action = self.map.action_map[a]
                term_u[i_a] = self.lookahead(U,s,action)
                term_a[i_a] = a
                i_a += 1
            i = np.argmax(term_u)
            policy[self.U_mapping[s]] = term_a[i]
        return policy

    def compute(self):
        num = len(self.state_space)
        U = np.zeros((num,))
        for k in range(self.k_max): # 139
            for s in self.state_space:
                u = self.backup(U,s)
                U[self.U_mapping[s]] = u
        self.policy = self.ValFPoly(U)        

    def comp_path(self):
        path = []
        states = [(tuple(self.map.state),self.map.time,self.map.cleared)]
        while True:
            #print("intersection", states[-1][0])
            i = self.U_mapping[states[-1]]
            action = int(self.policy[i])
            path.append(action)
            self.map.step(action)
            states.append((tuple(self.map.state),self.map.time,self.map.cleared))
            if self.map.time >=self.max_t:
                print("error, max time reached")
                break
            if (self.map.state == self.map.goal).all():
                break
        return path
        
def main():
    """
    Map(n, start, goal, traffic)
    ref_path = [1, 1, 3, 3, 3, 3]
    """
    start = time.time()
    sim_map = Map(5, np.array([2, 0]), np.array([4, 4]), np.array([3, 2]))
    Value_it = Offline(sim_map,10)
    Value_it.compute()
    path = Value_it.comp_path()
    print("final path",path)
    print('total cost {}'.format(Value_it.map.time))
    endings = time.time()
    print("final time:", endings-start)

if __name__ == '__main__':
    main()