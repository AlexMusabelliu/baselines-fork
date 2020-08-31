import numpy as np
from baselines.common.runners import AbstractEnvRunner
import random
import tensorflow as tf

def occlude(data, percent=.5, attention=None, sess=None, fd=None):
    '''
    Args:
        data - a numpy array to be occluded
        percent (default=.5) - the percent of pixels to occlude as a decimal
        attention (default=None) - the tensor of attention for attention-based occlusion
        sess (default=None) - the session used to calculate the occlusion threshold value
        fd (default=None) - the feed dictionary used to calculate the occlusion threshold value

    Returns:
        result - occluded version of the numpy array, data
    '''
    def recursive_map(inputs):
        # inputs = tf.convert_to_tensor(inputs)
        # if inputs.get_shape().ndims > 0:
        #     return tf.map_fn(recursive_map, inputs)
        # else:
        #     return tf.cond(tf.reduce_mean(inputs - ma) < 0, lambda: tf.constant(0.0), lambda: inputs)
        def _map(inputs):
            r = []
            for cur in inputs:
                if cur.ndim > 0:
                    r.append(_map(cur))
                else:
                    e = tf.cond(tf.reduce_mean(cur - ma) < 0, lambda: 0.0, lambda: float(cur))
                    r.append(e)
            return r

        return np.array(_map(inputs))        
        
    if percent > 1:
        percent = 1

    if attention != None:
        # print(f"Size of attention tensor: {tf.size(attention)}\nShape of attnetion tensor: {tf.shape(attention)}\nSize/Shape of data: {tf.size(data)} / {tf.shape(data)}")
        # print(type(attention))
        aflat = tf.reshape(attention, [-1])
        m = tf.gather(aflat, tf.nn.top_k(aflat, k=tf.size(aflat)).indices)
        #m = tf .sort(aflat, axis=-1, direction="ASCENDING").eval()
        msize = m.get_shape().as_list()[0]
        # print(type(msize))
        ma = tf.slice(m, [int(msize * percent)], [1]).eval(feed_dict=fd, session=sess)
        # print(tf.size(ma), tf.shape(ma), ma)
        # print(type(data), data.shape)
        result = np.ma.filled(np.ma.masked_where(data < ma, data), fill_value=0)
        
        # print(f"---*****Size/shape of mod tensor: {tf.size(result)} / {tf.shape(result)}")
        
    return result

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self, occlude=False):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for iter_step in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            if occlude:
                qs, qfd = self.model.act_model.qeval(self.obs, S=self.states, M=self.dones)
                self.obs = occlude(self.obs, percent=self.model.act_model.__dict__.get("percent"), attention=self.model.act_model.__dict__.get("extra"), sess=qs, fd=qfd)

            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


