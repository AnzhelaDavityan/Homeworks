

from Bandit import Bandit  # Import Bandit class from Bandit.py
import numpy as np
import pandas as pd
from loguru import logger

class UCB1(Bandit):
    """
    Upper Confidence Bound (UCB1) Bandit Algorithm.
    Selects the arm with the highest upper confidence bound at each step.
    """
    def __init__(self, p):
        """Initialize UCB1 bandit algorithm.

        Args:
            p (list[float]): True mean reward values for each arm.
        """
        self.p = p
        self.n = len(p)
        self.counts = np.zeros(self.n)
        self.values = np.zeros(self.n)
        self.rewards = []
        self.cumulative_rewards = []
        self.trials = 20000
        self.data = []

    def __repr__(self):
        """Return a string representation of the UCB1 bandit with its parameters."""
        return f"UCB1(p={self.p})"

    def pull(self, t):
        """
        Pull an arm based on the UCB1 algorithm:
        Selects the arm with the highest upper confidence bound.

        UCB1 formula:
        UCB_i(t) = mean_i(t) + sqrt((2 * log(t)) / N_i(t))
        Where:
            mean_i(t) = estimated mean reward for arm i
            N_i(t) = number of times arm i has been pulled
            t = current time step (trial)
        """
        ucb_values = self.values + np.sqrt((2 * np.log(t)) / (self.counts + 1e-5))  # Avoid division by zero
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        Update the estimated value of the chosen arm based on the received reward.

        Args:
            chosen_arm (int): The index of the arm selected.
            reward (float): The reward received from pulling the arm.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = (1 - 1/n) * value + (1/n) * reward

    def experiment(self):
        """Run the UCB1 algorithm for a specified number of trials."""
        for t in range(1, self.trials + 1):
            arm = self.pull(t)  # Select arm based on UCB1 logic
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.cumulative_rewards.append(np.sum(self.rewards))
            self.data.append([arm, reward, 'UCB1'])

    def report(self):
        """Store results in CSV and log final average reward and regret."""
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('rewards1.csv', mode='a', header=False, index=False)  # Save to rewards1.csv
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        cumulative_reward = np.sum(self.rewards)
        logger.info(f"[UCB1] Cumulative reward: {cumulative_reward:.4f}")
        logger.info(f"[UCB1] Average reward: {avg_reward:.4f}")
        logger.info(f"[UCB1] Cumulative regret: {regret:.4f}")



if __name__ == "__main__":
    Bandit_Reward = [1, 2, 3, 4]

    ucb1 = UCB1(Bandit_Reward)

    ucb1.experiment()

    ucb1.report()
