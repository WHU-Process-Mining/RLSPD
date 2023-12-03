import numpy as np


class BaseAgent(object):
    def __init__(self, env, device, state_dim, action_dim):
        self.env = env
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train(self):
        """
        Implement your training algorithm
        """
        raise NotImplementedError("Subclasses should implement this!")

    def make_action(self, state):
        """
        Return predicted action of your agent based on observation
        """
        raise NotImplementedError("Subclasses should implement this!")

    # test agent in the env
    def test(self, total_episodes=5):
        rewards = []
        for i in range(total_episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            # one episode
            while True:
                action = self.make_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
        self.env.close()
        print('Run %d episodes' % total_episodes, 'Reward:', rewards, 'Mean:', np.mean(rewards))
