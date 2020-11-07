import sim
import numpy as np

class Model:
    def __init__(self, map):
        # map object as defined in sim.py
        self.map = map

    def step(self, state, action):
        current = state[0]
        cleared = state[1]
        time = state[2]

        # Build TR matrices
        transition = {}
        reward = 0

        if np.linalg.norm(current - self.map.goal) < 1e-5:
            transition[state] = 1
            reward = 0
        else:
            next = current + self.map.action_map[action]
            next = np.maximum(np.minimum(next, self.map.n), 0)
            if cleared:
                nextS = (tuple(next), cleared, time + 1)
                transition[nextS] = 1
                reward += -1
            else:
                probClear = min(self.map.f_slope * time, 1)
                nextS = (tuple(next),1, time + 1)
                transition[nextS] = probClear
                reward += - probClear

                # Ripped from sim>step
                travTime = 1
                if np.linalg.norm(current - self.map.traffic) < 1e-5 or \
                        np.linalg.norm(next - self.map.traffic) < 1e-5:
                    travTime = 2
                elif np.linalg.norm(current - self.map.traffic) < 1 + 1e-5 or \
                        np.linalg.norm(next - self.map.traffic) < 1 + 1e-5:
                    travTime = 1.5
                nextS = (tuple(next),0,time + travTime)
                transition[nextS] = 1 - probClear
                reward += - (1 - probClear) * travTime
        return transition, reward

def fwdSearch(model, state, d):
	current = state[0];
	# Note to self: start from 0 for action numbering
	if d <= 0:
		aBest = None
		uBest = abs(self.map.goal[0] - current[0]) + abs(self.map.goal[1] - current[1])
	else:
	    aBest = None
	    uBest = -np.inf
	    for a in range(len(model.map.action_map)):
            t,r = model.step(state,a)
			u = r

			for s in t.keys():
			# Recursive calling for specified depth
				_, uNext = fwdSearch(model,s,d-1)
				u += t[s] * uNext

			if u > uBest:
				uBest = u
				aBest = a
	return aBest, uBest


if __name__ == '__main__':
    map = sim.Map(5, np.array([2, 0]), np.array([4, 4]), np.array([3, 2]))
    model = Model(map)

    actions = []
    d = 3
    count = 1
    maxIter = 7
    while np.linalg.norm(map.state - map.goal) > 1e-5 and \
    		count <= maxIter:
        a, _ = fwdSearch(model,(map.state, map.cleared, map.time), d)
        print('Taking action {0} from state {1} at time {2}'.format(a, map.state, map.time))
        actions.append(a)
        map.step(a)
        count = count+1
    print('Total cost {}'.format(map.time))