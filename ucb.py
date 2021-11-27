import numpy as np
import gym

def ucbfunc(env, maxep = 1000, rng = np.random.RandomState(1111)):
    env.reset()
    e = 0
    c = 100
    Q = np.zeros(env.length)
    N = np.zeros(env.length)
    qkeep = np.zeros([maxep, env.length])
    r_dict = np.zeros(maxep)
    regret = np.zeros(maxep)
    optimal_action = np.zeros(maxep)
    optim = 0
    while(e < maxep):
        ac = np.argmax(env.reward_space)
        v_star = np.max(env.reward_space)
        if e < env.length :
            a = e
        else:
            U = c * np.sqrt(np.log(e)/N)
            a = np.argmax(Q + U)
        if e==0:
            regret[e] = v_star - Q[a]
        else:
            regret[e] = regret[e-1] + v_star - Q[a]
        if ac == a:
            optim += 1
        optimal_action[e] = (optim/(e+1))*100
        _, R, done, _ = env.step(a, rend = False,rng = rng)
        r_dict[e] = R
        N[a] += 1
        Q[a] = Q[a] + (R - Q[a])/N[a]
        qkeep[e] = Q
        e += 1
    return optimal_action, regret, r_dict, qkeep