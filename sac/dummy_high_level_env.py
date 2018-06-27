import cv2, numpy as np
#from indep_control2.vae_network import VAE_Network
from gym.spaces import Box

def make_n_columns(height_percents, spacing=2, size=32):
    background = np.zeros((size, size, 3), dtype=np.float32)
    num_columns = len(height_percents)
    column_width = (size - (num_columns-1)*spacing) // num_columns
    assert column_width > 0
    start_pos = 0
    for i in range(num_columns):
        height = int(height_percents[i]*size)
        background[size-height:size, start_pos:start_pos+column_width, :] = 1
        start_pos += (column_width + spacing)
    return background


class DummyHighLevelEnv(object):

    def __init__(self, sparse_reward=False, goal_reward=10, no_goal_penalty=-0.1, goal_threshold=0.1, buffer=None, use_encoding=False, distance_mode='mean'):
        # environment hyperparameters
        self.sparse_reward = sparse_reward
        self.goal_reward = goal_reward
        self.no_goal_penalty = no_goal_penalty
        self.goal_threshold = goal_threshold
        self.obs_size = 8
        self.spacing = 2
        self.image_size = 128
        self.possible_distance_modes = ['mean', 'sum']
        try:
            assert distance_mode in self.possible_distance_modes
        except AssertionError:
            raise Exception(f'Distance mode must be in list: {self.possible_distance_modes}')
        self.distance_mode = distance_mode




        self.num_steps = 0
        self.max_steps = 100

        # hindsight stuff
        self.current_trajectory = []
        self.buffer = buffer

        # encoding stuff
        self.use_encoding = use_encoding
        self.num_factors = 20
        if use_encoding:
            self.obs_size = 20
            self.nn = VAE_Network(self.num_factors, 10 * 10, mode='image')
            self.nn.restore('./indep_control2/vae_network.ckpt')


        self.observation_space = Box(-3, 3, shape=[2*self.obs_size], dtype=np.float32)
        self.action_space = Box(0, 1, shape=[2], dtype=np.float32)
        # initialize the environment
        self.column_position = self.new_column_position()
        self.goal = self.new_goal()


    def add_hindsight_experience(self):
        if len(self.current_trajectory) == 0:
            return
        _, _, _, last_sp, _ = self.current_trajectory[-1]
        goal = np.copy(last_sp[:self.obs_size])
        for s, a, r, sp, t in self.current_trajectory:
            new_s = np.copy(np.concatenate([s[:self.obs_size], goal], axis=0))
            new_a = np.copy(a)
            new_sp = np.copy(np.concatenate([sp[:self.obs_size], goal], axis=0))
            new_r = self.get_reward(new_sp)
            new_t = self.get_terminal(new_sp)
            self.buffer.append(new_s, new_a, new_r, new_sp, new_t)

    def step(self, raw_action, action_converter):
        action = action_converter(raw_action)
        (column_index, parameter) = action

        old_obs = self.get_observation()

        self.column_position[column_index] = parameter
        self.num_steps += 1

        obs = self.get_observation()

        reward = self.get_reward(obs)

        terminal = self.get_terminal(obs) or self.num_steps >= self.max_steps
        if terminal:
            print('dist_to_goal', self.dist_to_goal(obs[:self.obs_size], obs[self.obs_size:]))

        if self.buffer is not None:
            self.current_trajectory.append((old_obs, raw_action, reward, obs, terminal))

        return obs, reward, terminal, {}

    def new_column_position(self):
        return np.random.uniform(0, 1, size=[8])

    def new_goal(self):
        if self.use_encoding:
            column_image = make_n_columns(self.new_column_position(), spacing=self.spacing, size=self.image_size)
            encoding = self.nn.encode_deterministic([column_image])[0]
            return encoding
        else:
            return self.new_column_position()

    def get_observation(self, goal=None):
        goal = np.copy(self.goal) if goal is None else goal
        if self.use_encoding:
            column_image = make_n_columns(self.column_position, spacing=self.spacing, size=self.image_size)
            encoding = self.nn.encode_deterministic([column_image])[0]
            return np.concatenate([encoding, goal], axis=0)
        else:
            return np.concatenate([np.copy(self.column_position), goal], axis=0)

    def get_reward(self, obs, goal=None):
        goal = obs[self.obs_size:] if goal is None else goal
        obs_part = obs[:self.obs_size]
        if self.sparse_reward:
            return self.goal_reward if self.get_terminal(obs, goal) else self.no_goal_penalty
        else:
            new_dist = self.dist_to_goal(obs_part, goal)
            return -20*new_dist

    def get_terminal(self, obs, goal=None):
        goal = obs[self.obs_size:] if goal is None else goal
        obs_part = obs[:self.obs_size]
        return self.dist_to_goal(obs_part, goal) < self.goal_threshold


    def dist_to_goal(self, obs_part, goal):
        if self.distance_mode == 'sum':
            return np.sum(np.abs(obs_part - goal))
        elif self.distance_mode == 'mean':
            return np.mean(np.abs(obs_part - goal))
        else:
            raise Exception('If youre getting this exception, something is wrong with the code')


    def reset(self):
        if self.buffer is not None:
            self.add_hindsight_experience()
            self.current_trajectory = []
        self.column_position = self.new_column_position()
        self.goal = self.new_goal()
        self.num_steps = 0
        return self.get_observation()

    def render(self):
        cv2.imshow('game', 255 * make_n_columns(self.column_position, spacing=2, size=128))
        cv2.waitKey(1)


if __name__ == '__main__':
    env = DummyHighLevelEnv(sparse_reward=True, distance_mode='sum')