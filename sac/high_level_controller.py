import numpy as np
import tensorflow as tf
import tqdm

from networks.policy_mixins import GaussianPolicy, CNN_Power2_Policy, MLPPolicy
from networks.value_function_mixins import MLPValueFunc, CNN_Power2_ValueFunc
from networks.network_interface import AbstractSoftActorCritic
from builder_functions import build_high_level_agent
from high_level_environment import DummyHighLevelEnv

class HighLevelController(object):

    def __init__(self, use_encoding=False, naive_policy=True):
        self.use_encoding = use_encoding
        self.naive_policy = naive_policy
        self.sess = None
        # it shouldnt matter if the environment uses column 0, we arent computing the reward function directly.
        self.env_single_column_list = [DummyHighLevelEnv(
            sparse_reward=True,
            distance_mode='max',
            use_encoding=use_encoding,
            single_column=column
        ) for column in range(8)]

        self.env_all_columns = DummyHighLevelEnv(
            sparse_reward=True,
            distance_mode='max',
            use_encoding=use_encoding,
            single_column=-1
        )
        self.env = self.env_single_column_list[0]
        self.num_cols = 2
        # map from the column number into the run that was used.
        self.agent_runs = [0, 0, 0, 1, 0, 0, 0, 0][:self.num_cols]
        self.agent_names = [f'column{col_num}' for col_num in range(8)][:self.num_cols]
        # TODO implement the getter for the agent paths.
        path_root = 'column_factor_agents/runs/column{col_num}_{run_num}/weights/sac.ckpt'
        self.agent_paths = [path_root.format_map(locals()) for col_num, run_num in enumerate(self.agent_runs)]

        self.agents = [self.load_agent(path, name) for path, name in zip(self.agent_paths, self.agent_names)]

    def load_agent(self, option_path, global_scope):
        print(f'loading {global_scope}...')
        print(global_scope, option_path)
        agent = build_high_level_agent(self.env_single_column_list[0], global_scope)
        #if self.sess is None:
        #    self.sess = agent.sess
        agent.restore(option_path)
        return agent

    def achieve_high_level_configuration(self, goal):
        if self.naive_policy:
            self.achieve_high_level_configuration_naive(goal)
        else:
            raise NotImplemented

    def achieve_high_level_configuration_naive(self, goal):
        state = self.env.get_observation()
        for i in range(self.num_cols):
            num_steps = 0
            while True:
                num_steps += 1
                action = self.agents[i].get_actions([state])[0]
                state, _, _, _ = self.env.step(action)
                # hax to prevent the episode from ending, since we arent really using these environments properly.
                self.env.num_steps = 0
                if self.env_single_column_list[i].get_terminal(state, goal=goal):
                    break
                if num_steps >= 100:
                    break
                self.env.render()
        state = self.env.get_observation()
        at_goal = self.env_all_columns.get_terminal(state, goal=goal)
        return at_goal

    def assess_performance(self, num_runs=100):
        num_successes = 0
        for i in tqdm.tqdm(range(num_runs)):
            #goal = np.random.uniform(0, 1, size=8)
            goal = np.zeros(shape=[8])
            if self.achieve_high_level_configuration(goal):
                num_successes += 1
        print(f'Success Rate: {str(num_successes / num_runs*100)[:5]}%\t ({num_successes}/{num_runs})')



if __name__ == '__main__':
    controller = HighLevelController()
    controller.assess_performance()


