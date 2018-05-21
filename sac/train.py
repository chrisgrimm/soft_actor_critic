import itertools
import pickle
import time
from typing import Callable, Tuple, Union, Iterable, Iterator

import gym
import numpy as np
import tensorflow as tf
from collections import Counter
from gym import spaces

from environments.hindsight_wrapper import HindsightWrapper
from environments.unsupervised import UnsupervisedEnv
from sac.agent import AbstractAgent, PropagationAgent
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import PropStep, Step, State

LOGGER_VALUES = """\
entropy
V loss
Q loss
pi loss
V grad
Q grad
pi grad\
""".split('\n')


def inject_mimic_experiences(mimic_file, buffer, N=1):
    with open(mimic_file, 'rb') as f:
        mimic_trajectories = [pickle.load(f)]
    for trajectory in mimic_trajectories:
        for (s1, a, r, s2, t) in trajectory:
            for _ in range(N):
                buffer.append(Step(s1=s1, a=a, r=r, s2=s2, t=t))


class Trainer:
    def __init__(self, env: gym.Env, seed: int, buffer_size: int,
                 activation: Callable, n_layers: int, layer_size: int,
                 learning_rate: float, reward_scale: float, grad_clip: float,
                 batch_size: int, num_train_steps: int,
                 logdir: str, save_path: str, load_path: str,
                 render: bool):

        if seed is not None:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            env.seed(seed)

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.env = env
        self.buffer = ReplayBuffer(buffer_size)
        self.reward_scale = reward_scale

        s1 = self.reset()

        self.agent = agent = self.build_agent(
            activation=activation,
            n_layers=n_layers,
            layer_size=layer_size,
            learning_rate=learning_rate,
            grad_clip=grad_clip)

        if isinstance(env.unwrapped, UnsupervisedEnv):
            # noinspection PyUnresolvedReferences
            env.unwrapped.initialize(agent.sess, self.buffer)

        saver = tf.train.Saver()
        tb_writer = None
        if load_path:
            saver.restore(agent.sess, load_path)
            print("Model restored from", load_path)
        if logdir:
            tb_writer = tf.summary.FileWriter(
                logdir=logdir, graph=agent.sess.graph)

        count = Counter(reward=0, episode=0)
        episode_count = Counter()
        info_counter = Counter()
        evaluation_period = 10

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % evaluation_period == evaluation_period - 1
            a = agent.get_actions(
                [self.vectorize_state(s1)], sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(a)
            if 'print' in info:
                print('time-step:', time_steps, info['print'])

            tick = time.time()

            episode_count += Counter(reward=r, timesteps=1)
            if save_path and time_steps % 5000 == 0:
                print("model saved in path:",
                      saver.save(agent.sess, save_path=save_path))
            if not is_eval_period:
                self.add_to_buffer(s1=s1, a=a, r=r, s2=s2, t=t)
                if len(self.buffer) >= self.batch_size:
                    for i in range(self.num_train_steps):
                        sample_steps = self.sample_buffer()
                        # noinspection PyProtectedMember
                        step = self.agent.train_step(sample_steps._replace(
                            s1=list(map(self.vectorize_state, sample_steps.s1)),
                            s2=list(map(self.vectorize_state, sample_steps.s2)),
                        ))
                        episode_count += Counter({k: getattr(step, k.replace(' ', '_'))
                                                  for k in LOGGER_VALUES})
            s1 = s2
            if t:
                s1 = self.reset()
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', count['episode'],
                    time_steps, episode_count['reward']))
                count += Counter(reward=(episode_count['reward']), episode=1)
                fps = int(episode_count['timesteps'] / (time.time() - tick))
                if logdir:
                    summary = tf.Summary()
                    if is_eval_period:
                        summary.value.add(
                            tag='eval reward',
                            simple_value=(episode_count['reward']))
                    summary.value.add(
                        tag='average reward',
                        simple_value=(
                            count['reward'] / float(count['episode'])))
                    summary.value.add(tag='fps', simple_value=fps)
                    summary.value.add(tag='reward', simple_value=episode_count['reward'])
                    for k in LOGGER_VALUES:
                        summary.value.add(tag=k, simple_value=episode_count[k] / float(episode_count['timesteps']))
                    tb_writer.add_summary(summary, count['episode'])
                    tb_writer.flush()

                for k in episode_count:
                    episode_count[k] = 0

    def build_agent(self,
                    activation: Callable,
                    n_layers: int,
                    layer_size: int,
                    learning_rate: float,
                    grad_clip: float,
                    base_agent: AbstractAgent = AbstractAgent) -> AbstractAgent:
        state_shape = self.env.observation_space.shape
        if isinstance(self.env.action_space, spaces.Discrete):
            action_shape = [self.env.action_space.n]
            PolicyType = CategoricalPolicy
        else:
            action_shape = self.env.action_space.shape
            PolicyType = GaussianPolicy

        class Agent(PolicyType, base_agent):
            def __init__(self, s_shape, a_shape):
                super(Agent, self).__init__(
                    s_shape=s_shape,
                    a_shape=a_shape,
                    activation=activation,
                    n_layers=n_layers,
                    layer_size=layer_size,
                    learning_rate=learning_rate,
                    grad_clip=grad_clip)

        return Agent(state_shape, action_shape)

    def reset(self) -> State:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, dict]:
        """ Preprocess action before feeding to env """
        if type(self.env.action_space) is spaces.Discrete:
            # noinspection PyTypeChecker
            return self.env.step(np.argmax(action))
        else:
            action = np.tanh(action)
            hi, lo = self.env.action_space.high, self.env.action_space.low
            # noinspection PyTypeChecker
            return self.env.step((action + 1) / 2 * (hi - lo) + lo)

    def vectorize_state(self, state: State) -> np.ndarray:
        """ Preprocess state before feeding to network """
        return state

    def add_to_buffer(self, s1: State, a: Union[float, np.ndarray], r: float, s2: State, t: bool) -> None:
        self.buffer.append(
            Step(s1=s1, a=a, r=r * self.reward_scale, s2=s2, t=t))

    def sample_buffer(self):
        return Step(*self.buffer.sample(self.batch_size))


class TrajectoryTrainer(Trainer):
    def __init__(self, **kwargs):
        self.trajectory = []
        super().__init__(**kwargs)
        self.s1 = self.reset()

    def step(self, action: np.ndarray) -> Tuple[State, float, bool, dict]:
        s2, r, t, i = super().step(action)
        self.trajectory.append(Step(s1=self.s1, a=action, r=r, s2=s2, t=t))
        self.s1 = s2
        return s2, r, t, i

    def reset(self) -> State:
        self.trajectory = []
        self.s1 = super().reset()
        return self.s1


class HindsightTrainer(TrajectoryTrainer):
    def __init__(self, env, **kwargs):
        assert isinstance(env, HindsightWrapper)
        super().__init__(env=env, **kwargs)

    def reset(self) -> State:
        assert isinstance(self.env, HindsightWrapper)
        for s1, a, r, s2, t in self.env.recompute_trajectory(self.trajectory):
            self.buffer.append((s1, a, r * self.reward_scale, s2, t))
        return super().reset()

    def vectorize_state(self, state: State) -> np.ndarray:
        assert isinstance(self.env, HindsightWrapper)
        return self.env.vectorize_state(state)


class PropagationTrainer(TrajectoryTrainer):
    def add_to_buffer(self, **_):
        pass

    def build_agent(self,
                    activation: Callable,
                    n_layers: int,
                    layer_size: int,
                    learning_rate: float,
                    grad_clip: float,
                    base_agent: AbstractAgent = AbstractAgent,
                    **kwargs) -> AbstractAgent:
        return super().build_agent(
            activation=activation,
            n_layers=n_layers,
            layer_size=layer_size,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            base_agent=PropagationAgent)

    def reset(self) -> State:
        self.buffer.extend(self.step_generator(self.trajectory))
        return super().reset()

    def step_generator(self, trajectory: Iterable[Step]) -> Iterator[PropStep]:
        v2 = 0
        for step in reversed(trajectory):
            v2 = .99 * v2 + step.r
            # noinspection PyProtectedMember
            prop_step = PropStep(v2=v2, **step._asdict())
            # noinspection PyProtectedMember
            yield prop_step._replace(r=step.r * self.reward_scale)

    def sample_buffer(self):
        return PropStep(*self.buffer.sample(self.batch_size))


class HindsightPropagationTrainer(HindsightTrainer, PropagationTrainer):
    def reset(self) -> State:
        assert isinstance(self.env, HindsightWrapper)
        trajectory = list(self.env.recompute_trajectory(self.trajectory))
        self.buffer.extend(self.step_generator(trajectory))
        return PropagationTrainer.reset(self)


