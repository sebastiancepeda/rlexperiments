import gym
import utils
import theano.tensor as T
import theano
import lasagne
import numpy

env = gym.make('CartPole-v0')
#env = gym.make('MsPacman-v0')
#env = gym.make('Breakout-v0')

print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

n0 = env.observation_space.shape[0]
#n1 = env.observation_space.shape[1]
#n2 = env.observation_space.shape[2]

n_action = env.action_space.n
#n_features = n0*n1*n2
n_features = n0*2
print('n_features: {}'.format(n_features))

print("* Building model and compiling functions...")
state_var = T.matrix('state')
action_var = T.ivector('action')

network = utils.build_custom_mlp(n_features, n_action, state_var, depth=1, width=5, drop_input=0.0, drop_hidden=0.0)
policy = lasagne.layers.get_output(network)

action_one_hot = T.extra_ops.to_one_hot(action_var, n_action, dtype='int32')
#policy_action = T.log((policy*action_one_hot).sum())
#policy_action = T.log(policy.max())+0*action_one_hot.sum()
policy_error = policy - action_one_hot
policy_action = ((policy_error*policy_error).mean())
#policy_action = T.log((policy * action_one_hot).sum())

params = lasagne.layers.get_all_params(network, trainable=True)
train_fn = theano.function([state_var, action_var], [policy_action])
output_model = theano.function([state_var], policy)

# compilation
comp_grads = []
comp_params_giver = []
comp_params_updater = []
for w in params:
    grad = T.grad(policy_action,  w)
    grad_fn = theano.function([state_var,  action_var], grad)
    comp_grads = comp_grads + [grad_fn]
    params_fn = theano.function([],  w)
    comp_params_giver = comp_params_giver + [params_fn]
    w_in = T.matrix()
    if(w_in.type != w.type):
        w_in = T.vector()
    w_update = theano.function([w_in], updates=[(w, w_in)])
    comp_params_updater = comp_params_updater + [w_update]

def params_giver():
    ws = []
    for param_fn in comp_params_giver:
        ws = numpy.append(ws, param_fn())
    return ws

def grad_giver(state, action):
    gs = []
    for grad_fn in comp_grads:
        gs = numpy.append(gs, grad_fn(state, action))
    return gs

''' Updates the weights (w) in the net. '''
def params_updater(all_w):
    idx_init = 0
    params_idx = 0
    for w_updater in comp_params_updater:
        w = params[params_idx]
        params_idx += 1
        w_value_pre = w.get_value()
        w_act = all_w[idx_init:idx_init+w_value_pre.size]
        w_value = w_act.reshape(w_value_pre.shape)
        idx_init += w_value_pre.size
        w_updater(w_value)
    return

''' Returns the loss, given (x,t,w). '''
def func(params, *args):
    state_t = args[0]
    action_t = args[1]
    params_updater(params)
    return train_fn(state_t.astype('float32'), action_t.astype(numpy.int32))[0]

''' Returns the derivative of the loss with respect to the weights, given (x, t, w). '''
def fprime(params, *args):
    state_t = args[0]
    action_t = args[1]
    params_updater(params)
    return grad_giver(state_t.astype('float32'), action_t.astype(numpy.int32))

print("* Iterations...")

learning_rate = 10**-6
gamma = 1-10**-3
m_r = 0.99
m_t = 0
l_r = learning_rate
w_t = params_giver()
best_w = 0
best_r = 0
freq = 1
#previous_w = w_t
rand_prob = 0.0
reward_mean = 0
cum_gradient = 0
state_mean = 0
debug_level = 1
for i in range(10000000):
    observation = env.reset()
    state = numpy.concatenate((observation,observation))
    previous_state = state
    state_mean = 0.99*state_mean + 0.01*observation
    #state = observation - state_mean
    episode_reward = 0
    cum_log_probability_gradient = 0
    for step in range(1000000):
        if(step >= 1 and (i % freq) == 0):
            env.render()
        
        # FeedForward
        params_updater(w_t.astype('float32'))
        policy = output_model(state.reshape(1, n_features))
        if(numpy.random.rand() < rand_prob):
            action = numpy.random.randint(n_action)
            #rand_prob *= 0.99
        else:
            #action = policy.argmax()
            action = numpy.random.choice(n_action, 1, p=policy[0])[0]
            #print('action: {}'.format(action))
        previous_obs = observation
        observation, reward, bDone, info = env.step(action)
        if(policy.max() > 0.9):
            print('policy: {}'.format(policy))
        previous_state = state
        state = numpy.concatenate((observation,previous_obs))
        state_mean = 0.99*state_mean + 0.01*0
        #state = observation - state_mean
        if bDone:
            reward -= 10
        episode_reward += reward
        
        # FeedBack
        action_aux = numpy.zeros(1)
        action_aux[0] = action
        train_data = (state.reshape(1, n_features), action_aux.astype(int))
        log_policy_action = func(w_t.astype('float32'), *train_data)
        dlogPdw = fprime(w_t.astype('float32') , *train_data)
        cum_log_probability_gradient += dlogPdw
        cum_gradient += (gamma**step)*reward*cum_log_probability_gradient
        
        if bDone:
            break
    
    # Parameters update
    if((i % freq) == 0):
        g_t = cum_gradient
        m_t = m_r*m_t + g_t*l_r
        dw_t  = m_r*m_t + g_t*l_r
        w2 = w_t + dw_t
        params_updater(w2.astype('float32'))
        policy2 = output_model(previous_state.reshape(1, n_features))
        policy_action_diff = policy2[0][action] - policy[0][action]
        print('policy_action_diff: {}'.format(policy_action_diff))
        #if(policy_action_diff >= 0):
        w_t = w_t + dw_t
        print('dw_t: {}, \t'.format(abs(dw_t).mean()))
        print('cum_gradient: {}, \t'.format(abs(cum_gradient).mean()))
        cum_gradient = 0
    
    reward_mean = 0.99*reward_mean + 0.01*episode_reward
    print('episode_reward: {}'.format(episode_reward))
    print('reward_mean: {}'.format(reward_mean))
    corrected_reward = (episode_reward - reward_mean)
    print('corrected_reward: {}'.format(corrected_reward))
