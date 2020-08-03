import numpy as np
from baselines.common.runners import AbstractEnvRunner
import random
import tensorflow as tf

def occlude(data, percent=.5, height=84, width=84, gen=None, attention=None):
    '''
    Args:
        data - the tensor / matrix to occlude
        percent (default=.5) - the percent of pixels to occlude as a decimal
        height (default=84) - the height of the tensor
        width (default=84) - the width of the tensor
        gen (default=None) - what to use for the mask's data. can be an object or a function.
        attention (default=None) - the tensor of attention for attention-based occlusion
        TODO: change height and width to take from data

    Returns:
        mod - modified tensor with occlusion
    '''

    if percent > 1:
        percent = 1

    if attention == None:
        m = gen() if callable(gen) else gen if type(gen) == list else None if gen == None else False
        
        if not m:
            if m == False:
                print("Warning! Invalid gen. specified, defaulting to random data generation...")
            m = np.array([0 if random.random() <= percent else 1 for x in range(height * width)])

        m = np.reshape(np.dstack([m] * 4), [4, height, width])

        mask = tf.convert_to_tensor(mask_np, dtype=tf.bool)
        mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=len(mask.shape))
        data = tf.convert_to_tensor(data_np, dtype=tf.float32)

        result = mask * data

    else:
        print(f"Size of attention tensor: {tf.size(attention)}\nShape of attnetion tensor: {tf.shape(attention)}\nSize/Shape of data: {tf.size(data)} / {tf.shape(data)}")
        with tf.Session() as sess:
            aflat = tf.reshape(attention, [-1])
            m = tf.sort(aflat, axis=-1, direction="ASCENDING").eval(session=sess)
            ma = m[m.size * percent // 1]

            result = tf.map_fn(lambda x: 0 if x < ma else x, attention)
        
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

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            self.obs = occlude(self.obs, percent=self.model.act_model.__dict__.get("percent"), attention=self.model.act_model.__dict__.get("extra"))

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


