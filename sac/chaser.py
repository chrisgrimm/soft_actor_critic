import cv2
import numpy as np
import pygame
from gym import Env, spaces


class ChaserEnv(Env):

    def __init__(self, visual=False, no_prey=False, max_steps=1000):
        self.size = 20
        self.block_pixels = 2
        self.visual = visual
        self.no_prey = no_prey
        self.agent_pos = self.get_position()
        self.prey_pos = self.get_position(positions_to_avoid=set(self.agent_pos))
        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Box(-1, 1, shape=[2])
        self.agent_color = (255, 0, 0)
        self.prey_color = (0, 255, 0)
        self.overlap_color = (0, 0, 255)
        self.background_color = (255,255,255)
        self.max_steps = max_steps
        self.step_num = 0

        self.action_dict = {
            #0: (0, 0),
            0: (1, 0),
            1: (-1, 0),
            2: (0, 1),
            3: (0, -1)
        }
        self.surface = pygame.Surface((self.size * self.block_pixels, self.size * self.block_pixels))

        if self.visual:
            self.observation_space = spaces.Box(0, 1, shape=[self.size * self.block_pixels, self.size * self.block_pixels, 3])
        else:
            # position of agent and position of prey.
            self.observation_space = spaces.Box(0, 1, shape=[4])


    def get_position(self, positions_to_avoid=None):
        while True:
            position = tuple(np.random.randint(0, self.size, 2))
            if (positions_to_avoid is None) or (position not in positions_to_avoid):
                return position


    def get_obs(self, agent_pos, prey_pos, visual):
        if visual:
            self.surface.fill(self.background_color)
            agent_pos = (agent_pos[0]*self.block_pixels, agent_pos[1]*self.block_pixels)

            prey_pos = (prey_pos[0]*self.block_pixels, prey_pos[1]*self.block_pixels)
            agent_rect = pygame.Rect(agent_pos, (self.block_pixels, self.block_pixels))
            if self.no_prey:
                pygame.draw.rect(self.surface, self.agent_color, agent_rect)
            else:
                prey_rect = pygame.Rect(prey_pos, (self.block_pixels, self.block_pixels))
                if self.agent_pos == self.prey_pos:
                    pygame.draw.rect(self.surface, self.overlap_color, agent_rect)
                else:
                    pygame.draw.rect(self.surface, self.agent_color, agent_rect)
                    pygame.draw.rect(self.surface, self.prey_color, prey_rect)
            return pygame.surfarray.array3d(self.surface)
        else:
            if self.no_prey:
                return np.array(self.agent_pos)
            else:
                return np.concatenate([np.array(self.agent_pos) / self.size, np.array(self.prey_pos) / self.size], axis=0)

    def update_position(self, old_pos, delta):
        new_pos_x = np.clip((old_pos[0] + delta[0]), 0, self.size)
        new_pos_y = np.clip((old_pos[1] + delta[1]), 0, self.size)
        return (new_pos_x, new_pos_y)


    def reset(self):
        self.agent_pos = self.get_position()
        self.step_num = 0
        self.prey_pos = self.get_position(positions_to_avoid=set(self.agent_pos))
        return self.get_obs(self.agent_pos, self.prey_pos, self.visual)


    def step(self, action):
        #dx, dy = action[0], action[1]
        #if dx >= 0 and dy >= 0:
        #    action = 0
        #elif dx >= 0 and dy < 0:
        #    action = 1
        #elif dx < 0 and dy < 0:
        #    action = 2
        #else:
        #    action = 3
        #action = np.argmax(action)
        terminal = False
        reward = -0.01
        self.step_num += 1
        self.agent_pos = self.update_position(self.agent_pos, self.action_dict[action])

        norm_agent, norm_prey = np.array(self.agent_pos) / self.size, np.array(self.prey_pos) / self.size
        dist = np.sqrt(np.sum(np.square(norm_agent - norm_prey)))
        reward = -dist

        if self.agent_pos == self.prey_pos and not self.no_prey:
            terminal = True
            #reward = 1
            return self.get_obs(self.agent_pos, self.prey_pos, self.visual), reward, terminal, {}


        #self.prey_pos = self.update_position(self.prey_pos, np.random.randint(-1, 2, size=2))

        #reward = 1 if self.prey_pos == self.agent_pos and not self.no_prey else -0.01
        terminal = ((self.prey_pos == self.agent_pos) and (not self.no_prey)) or (self.step_num >= self.max_steps)
        return self.get_obs(self.agent_pos, self.prey_pos, self.visual), reward, terminal, {}


    def get_random_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            sample = self.get_obs(self.get_position(), self.get_position(), self.visual)
            batch.append(sample)
        return np.array(batch)

    def render(self):
        s = self.get_obs(self.agent_pos, self.prey_pos, True)
        cv2.imshow('game', s)
        cv2.waitKey(1)


env = ChaserEnv()

def get_batch_chaser(batch_size):
    return env.get_random_batch(batch_size)



if __name__ == '__main__':
    env = ChaserEnv()
    s = env.reset()
    while True:
        action = np.random.uniform(-1, 1, size=2)
        #onehot_action = utils.onehot(action, 4)
        s, r, t, info = env.step(action)
        if t:
            print(r)
            env.reset()
        print(s)
        #cv2.imshow('game', s)
        #cv2.waitKey(1)




