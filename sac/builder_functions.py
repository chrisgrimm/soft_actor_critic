from networks.network_interface import AbstractSoftActorCritic
from networks.policy_mixins import GaussianPolicy, Categorical_X_GaussianPolicy, MLPPolicy
from networks.value_function_mixins import MLPValueFunc

def build_high_level_agent(env, name='SAC_high_level', learning_rate=1*10**-4, width=128, random_goal=False, network_depth=2,
                           grad_clip_magnitude=1000, accept_discrete_and_gaussian=False, reward_scaling=0.01):
    PolicyType = GaussianPolicy if not accept_discrete_and_gaussian else Categorical_X_GaussianPolicy(env.num_columns, 1)
    class Agent(
        PolicyType,
        MLPPolicy(width, network_depth),
        MLPValueFunc(width, network_depth),
        AbstractSoftActorCritic):
        def __init__(self, s_shape, a_shape):
            super(Agent, self).__init__(s_shape, a_shape, global_name=name, learning_rate=learning_rate, inject_goal_randomness=random_goal,
                                        grad_clip_magnitude=grad_clip_magnitude, alpha=reward_scaling)
    return Agent(env.observation_space.shape, env.action_space.shape)