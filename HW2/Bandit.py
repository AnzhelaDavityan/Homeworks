# """
#   Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
# """
# ############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



np.random.seed(42)

class Bandit(ABC):
    """Abstract base class for multi-armed bandit algorithms.

    Defines the interface for any bandit strategy. All subclasses must implement
    initialization, arm selection, reward update, experiment execution, and reporting.
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Initialize the bandit with true mean rewards.

        Args:
            p (list[float]): True mean reward values for each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the bandit instance."""
        pass

    @abstractmethod
    def pull(self):
        """Select an arm to pull based on the current strategy logic.

        Returns:
            int: The index of the selected arm.
        """
        pass

    @abstractmethod
    def update(self, chosen_arm, reward):
        """Update the strategy's internal state based on received reward.

        Args:
            chosen_arm (int): The index of the arm that was selected.
            reward (float): The reward received from pulling the arm.
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the bandit strategy over a predefined number of trials,
        selecting arms, observing rewards, and updating estimates.
        """
        pass

    @abstractmethod
    def report(self):
        """Print average reward, cumulative regret, and save results to CSV file.
        Useful for comparing algorithm performance.
        """
        pass


class Visualization():
    """Class responsible for generating visual comparisons of algorithms.

    Includes reward progression and cumulative reward plots.
    """

# Set Seaborn style for better aesthetics
    sns.set(style="whitegrid", palette="muted", font_scale=1.5)  # Customizing the style

    def plot1(self, rewards_eg, rewards_ts):
        """Plot per-trial reward comparison between Epsilon-Greedy and Thompson Sampling.

        Args:
            rewards_eg (list[float]): List of rewards from Epsilon-Greedy strategy.
            rewards_ts (list[float]): List of rewards from Thompson Sampling strategy.
        """
        plt.figure(figsize=(10, 6))  # Make the figure larger for better readability

        # Use Light Blue for Epsilon-Greedy and Violet for Thompson Sampling
        plt.plot(rewards_eg, label='Epsilon-Greedy', linestyle='-', color='#ADD8E6', marker='o', markersize=1, linewidth=1)
        plt.plot(rewards_ts, label='Thompson Sampling', linestyle='--', color='#8A2BE2', marker='s', markersize=1, linewidth=1)

        # Adding titles and labels with custom fonts
        plt.title('Reward over Time', fontsize=15, weight='bold')
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Reward', fontsize=12)

        # Customize the legend
        plt.legend(fontsize=10, loc='upper left', title="Algorithms", title_fontsize=12)

        # Adding grid lines with slight customization
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tick_params(axis='both', which='major', labelsize=10)  # Make tick labels smaller

        # Save the plot as a PNG file
        plt.savefig('plot1.png', dpi=300, bbox_inches='tight')

        # Show the plot


    def plot2(self, cum_eg, cum_ts):
        """Plot cumulative reward comparison for both algorithms over time.

        Args:
            cum_eg (list[float]): Cumulative rewards for Epsilon-Greedy.
            cum_ts (list[float]): Cumulative rewards for Thompson Sampling.
        """
        plt.figure(figsize=(10, 6))  # Make the figure larger for better readability

        # Use Light Blue for Epsilon-Greedy and Violet for Thompson Sampling
        plt.plot(cum_eg, label='Epsilon-Greedy', linestyle='-', color='#ADD8E6', marker='o', markersize=1, linewidth=1)
        plt.plot(cum_ts, label='Thompson Sampling', linestyle='--', color='#8A2BE2', marker='s', markersize=1, linewidth=1)

        # Adding titles and labels with custom fonts
        plt.title('Cumulative Rewards', fontsize=15, weight='bold')
        plt.xlabel('Trial', fontsize=12)
        plt.ylabel('Cumulative Reward', fontsize=12)

        # Customize the legend
        plt.legend(fontsize=10, loc='upper left', title="Algorithms", title_fontsize=12)

        # Adding grid lines with slight customization
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plt.tick_params(axis='both', which='major', labelsize=10)  # Make tick labels smaller

        # Save the plot as a PNG file
        plt.savefig('plot2.png', dpi=300, bbox_inches='tight')



class EpsilonGreedy(Bandit):
    """Implementation of Epsilon-Greedy bandit algorithm.

    Selects a random arm with small probability (epsilon), and otherwise
    exploits the arm with the highest estimated average reward.
    """
    def __init__(self, p):
        """Initialize the Epsilon-Greedy bandit.

        Args:
            p (list[float]): True mean rewards for each bandit arm.
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
        """Return a string representation of the bandit with its parameters."""
        return f"EpsilonGreedy(p={self.p})"

    def pull(self):
        """Select an arm using epsilon-greedy logic: explore or exploit.

        Returns:
            int: Index of the selected arm.
        """
        t = len(self.rewards) + 1
        epsilon = 1 / t
        if np.random.rand() < epsilon:
            return np.random.randint(self.n)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """Update the running average reward of the selected arm.

        Args:
            chosen_arm (int): The selected arm.
            reward (float): Observed reward from the environment.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = (1 - 1/n) * value + (1/n) * reward

    def experiment(self):
        """Run the epsilon-greedy algorithm over 20,000 trials,
        collecting reward data and updating estimates.
        """
        for t in range(self.trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.cumulative_rewards.append(np.sum(self.rewards))
            self.data.append([arm, reward, 'EpsilonGreedy'])

    def report(self):
        """Store results in CSV and log final average reward and regret."""
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('rewards.csv', mode='a', header=False, index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        cumulative_reward = np.sum(self.rewards)  # Sum of all rewards gives cumulative reward
        logger.info(f"[EpsilonGreedy] Cumulative reward: {cumulative_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Average reward: {avg_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative regret: {regret:.4f}")


class ThompsonSampling(Bandit):
    """Thompson Sampling algorithm with Gaussian likelihoods.

    Selects arms by sampling from the posterior distributions of their mean rewards.
    """
    def __init__(self, p):
        """Initialize Thompson Sampling parameters and reward tracking.

        Args:
            p (list[float]): True mean rewards for each arm.
        """
        self.p = p
        self.n = len(p)
        self.trials = 20000
        self.rewards = []
        self.cumulative_rewards = []
        self.data = []
        self.mu = np.zeros(self.n)
        self.lambda_ = np.ones(self.n)

    def __repr__(self):
        """Return a string representation of the bandit with its parameters."""
        return f"ThompsonSampling(p={self.p})"

    def pull(self):
        """Sample from the posterior distributions to choose an arm.

        Returns:
            int: The index of the arm with the highest sample.
        """
        sampled = np.random.normal(self.mu, 1/np.sqrt(self.lambda_))
        return np.argmax(sampled)

    def update(self, chosen_arm, reward):
        """Update the posterior distribution of the selected arm.

        Args:
            chosen_arm (int): Arm that was pulled.
            reward (float): Reward received.
        """
        self.lambda_[chosen_arm] += 1
        self.mu[chosen_arm] = (self.lambda_[chosen_arm] * self.mu[chosen_arm] + reward) / (self.lambda_[chosen_arm] + 1)

    def experiment(self):
        """Run Thompson Sampling for 20,000 trials and record rewards.
        """
        for t in range(self.trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.cumulative_rewards.append(np.sum(self.rewards))
            self.data.append([arm, reward, 'ThompsonSampling'])

    def report(self):
        """Store results in CSV and log final average reward and regret."""
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('rewards.csv', mode='a', header=False, index=False)
        avg_reward = np.mean(self.rewards)
        cumulative_reward = np.sum(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"[ThompsonSampling] Cumulative reward: {cumulative_reward:.4f}")
        logger.info(f"[ThompsonSampling] Average reward: {avg_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative regret: {regret:.4f}")



def comparison():
    """Run both bandit algorithms, log their results, and create visual comparisons.

    Executes Epsilon-Greedy and Thompson Sampling strategies side by side and plots their
    performance in terms of rewards and cumulative gains.
    """
    # think of a way to compare the performances of the two algorithms VISUALLY and
    Bandit_Reward = [1, 2, 3, 4]
    eg = EpsilonGreedy(Bandit_Reward)
    ts = ThompsonSampling(Bandit_Reward)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    viz = Visualization()
    viz.plot1(eg.rewards, ts.rewards)
    viz.plot2(eg.cumulative_rewards, ts.cumulative_rewards)


if __name__=='__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()
