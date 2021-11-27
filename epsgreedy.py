import numpy as np
import gym

def epsilongreedy(env, maxep = 200, epsilon = 0.1, decay = 0, isdecaylinear = True, rng = np.random.RandomState(1111)):
    env.reset()
    e = 0
    a = 0
    Q = np.zeros(env.length)
    N = np.zeros(env.length)
    qkeep = np.zeros([maxep, env.length])
    r_dict = np.zeros(maxep)
    regret = np.zeros(maxep)
    optimal_action = np.zeros(maxep)
    optim = 0
    action = np.zeros(maxep)
    while(e < maxep):
        ac = np.argmax(env.reward_space)
        v_star = np.max(env.reward_space)
        if rng.uniform(0, 1) >= epsilon:
            a = np.argmax(Q)
        else:
            a = rng.randint(0, env.length)
        action[e] = a
        if e==0:
            regret[e] = v_star - Q[a]
        else:
            regret[e] = regret[e-1] + v_star - Q[a]
        if a == ac:
            optim += 1
        optimal_action[e] = (optim/(e+1))*100
        if isdecaylinear:
            epsilon -= decay
        else:
            epsilon *= decay
        _, R, done,_ = env.step(a, rend = False, rng = rng)
        r_dict[e] = R
        N[a] += 1
        Q[a] = Q[a] + (R - Q[a])/N[a]
        qkeep[e] = Q
        e += 1
    return optimal_action, regret, r_dict, action, qkeep