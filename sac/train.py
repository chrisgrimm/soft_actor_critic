import argparse
import itertools
import pickle
import time
from typing import Callable, Tuple, Union, Generator, Iterable, Iterator

import gym
import numpy as np
import tensorflow as tf
from collections import Counter
from gym import spaces

from environment.hindsight_wrapper import HindsightWrapper
from sac.agent import AbstractAgent, PropagationAgent, TrainStep
from sac.policies import CategoricalPolicy, GaussianPolicy
from sac.replay_buffer import ReplayBuffer
from sac.utils import PropStep, Step, State


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
                 learning_rate: float, reward_scale: float, batch_size: int,
                 num_train_steps: int, logdir: str, save_path: str, load_path: str,
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
            learning_rate=learning_rate)

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
        evaluation_period = 10

        for time_steps in itertools.count():
            is_eval_period = count['episode'] % evaluation_period == evaluation_period - 1
            a = agent.get_actions(
                [self.vectorize_state(s1)], sample=(not is_eval_period))
            if render:
                env.render()
            s2, r, t, info = self.step(a)
            if t:
                print('reward:', r)

            tick = time.time()

            episode_count += Counter(reward=r, timesteps=1)
            if save_path and time_steps % 5000 == 0:
                print("model saved in path:",
                      saver.save(agent.sess, save_path=save_path))
            if not is_eval_period:
                self.process_step(s1=s1, a=a, r=r, s2=s2, t=t)
                if len(self.buffer) >= self.batch_size:
                    for i in range(self.num_train_steps):
                        s1_sample, a_sample, r_sample, s2_sample, t_sample = self.buffer.sample(
                            self.batch_size)
                        s1_sample = list(map(self.vectorize_state, s1_sample))
                        s2_sample = list(map(self.vectorize_state, s2_sample))
                        step = self.agent.train_step(Step(
                                s1=s1_sample,
                                a=a_sample,
                                r=r_sample,
                                s2=s2_sample,
                                t=t_sample))
                        episode_count += Counter({
                            'V loss': step.V_loss,
                            'Q loss': step.Q_loss,
                            'pi loss': step.pi_loss,
                            'V grad': np.max(step.V_grad),
                            'Q grad': np.max(step.Q_grad),
                            'pi grad': np.max(step.pi_grad),
                            'entropy': step.entropy
                        })
            s1 = s2
            if t:
                s1 = self.reset()
                print('({}) Episode {}\t Time Steps: {}\t Reward: {}\t'
                      'V Loss: {}\t Q Loss: {}'.format(
                    'EVAL' if is_eval_period else 'TRAIN', (count['episode']),
                    time_steps, episode_count['reward'], episode_count['V loss'], episode_count['Q loss']))
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
                    for k in ['entropy', 'reward'] + ['{} {}'.format(v, w)
                                                      for v in ('V', 'Q', 'pi')
                                                      for w in ('loss', 'grad')]:
                        print(k, episode_count[k])
                        summary.value.add(tag=k, simple_value=episode_count[k])
                    tb_writer.add_summary(summary, count['episode'])
                    tb_writer.flush()

                for k in episode_count:
                    episode_count[k] = 0

    def build_agent(self,
                    activation: Callable,
                    n_layers: int,
                    layer_size: int,
                    learning_rate: float,
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
                    learning_rate=learning_rate)

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
            # noinspection PyUnresolvedReferences
            # noinspection PyTypeChecker
            return self.env.step((action + 1) / 2 * (hi - lo) + lo)

    def vectorize_state(self, state: State) -> np.ndarray:
        """ Preprocess state before feeding to network """
        return state

    def process_step(self, s1: State, a: Union[float, np.ndarray], r: float, s2: State, t: bool) -> None:
        self.buffer.append(
            Step(s1=s1, a=a, r=r * self.reward_scale, s2=s2, t=t))


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
    def process_step(self, s1: State, a: Union[float, np.ndarray], r: float, s2: State, t: bool) -> None:
        if len(self.buffer) >= self.batch_size:
            for i in range(self.num_train_steps):
                sample = self.buffer.sample(self.batch_size)
                s1_sample, a_sample, r_sample, s2_sample, t_sample, v2_sample = sample
                s1_sample = list(map(self.vectorize_state, s1_sample))
                s2_sample = list(map(self.vectorize_state, s2_sample))
                return self.agent.train_step(
                    PropStep(
                        s1=s1_sample,
                        a=a_sample,
                        r=r_sample,
                        s2=s2_sample,
                        t=t_sample,
                        v2=v2_sample))

    def build_agent(self,
                    activation: Callable,
                    n_layers: int,
                    layer_size: int,
                    learning_rate: float,
                    base_agent: AbstractAgent = AbstractAgent) -> AbstractAgent:
        return super().build_agent(
            activation=activation,
            n_layers=n_layers,
            layer_size=layer_size,
            learning_rate=learning_rate,
            base_agent=PropagationAgent)

    def reset(self) -> State:
        self.buffer.extend(self.step_generator(self.trajectory))
        return super().reset()

    def step_generator(self, trajectory: Iterable[Step]) -> Iterator[PropStep]:
        v2 = 0
        for step in reversed(trajectory):
            v2 = .99 * v2 + step.r
            # noinspection PyProtectedMember
            yield step._replace(r=step.r * self.reward_scale)


class HindsightPropagationTrainer(HindsightTrainer, PropagationTrainer):
    def reset(self) -> State:
        assert isinstance(self.env, HindsightWrapper)
        trajectory = list(self.env.recompute_trajectory(self.trajectory))
        self.buffer.extend(self.step_generator(trajectory))
        return PropagationTrainer.reset(self)


def activation_type(name: str) -> Callable:
    activations = dict(
        relu=tf.nn.relu,
        crelu=tf.nn.crelu,
        selu=tf.nn.selu,
        elu=tf.nn.elu,
        leaky=tf.nn.leaky_relu,
        leaky_relu=tf.nn.leaky_relu,
        tanh=tf.nn.tanh,
    )
    try:
        return activations[name]
    except KeyError:
        raise argparse.ArgumentTypeError(
            "Activation name must be one of the following:", '\n'.join(
                activations.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='HalfCheetah-v2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--activation', default=tf.nn.relu, type=activation_type)
    parser.add_argument('--n-layers', default=3, type=int)
    parser.add_argument('--layer-size', default=256, type=int)
    parser.add_argument('--learning-rate', default=3e-4, type=float)
    parser.add_argument('--buffer-size', default=int(10 ** 7), type=int)
    parser.add_argument('--num-train-steps', default=1, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--reward-scale', default=1., type=float)
    parser.add_argument('--mimic-file', default=None, type=str)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--reward-prop', action='store_true')
    args = parser.parse_args()

    # if args.mimic_file is not None:
    #     inject_mimic_experiences(args.mimic_file, buffer, N=10)

    trainer = PropagationTrainer if args.reward_prop else Trainer
    trainer(
        env=gym.make(args.env),
        seed=args.seed,
        buffer_size=args.buffer_size,
        activation=args.activation,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        logdir=args.logdir,
        render=args.render)
