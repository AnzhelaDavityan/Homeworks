# Bandit Algorithm Implementations

## Project Overview
This repository contains implementations of multi-armed bandit algorithms: Epsilon-Greedy, Thompson Sampling, and UCB1 (Upper Confidence Bound 1). The aim is to evaluate and compare these algorithms based on cumulative reward, average reward, and cumulative regret to identify the most effective strategy for maximizing rewards over time.

---

## Algorithms Implemented

### 1. Epsilon-Greedy
The Epsilon-Greedy strategy balances exploration and exploitation by choosing a random arm with probability epsilon (which decays over time) or exploiting the best-known arm.

### 2. Thompson Sampling
A Bayesian approach selecting arms by sampling from their posterior distributions, adjusting exploration based on uncertainty.

### 3. UCB1 (Upper Confidence Bound 1) – Suggested Implementation
UCB1 selects arms based on an optimistic estimate balancing mean rewards and uncertainty:

\[
UCB_i(t) = \text{mean}_i(t) + \sqrt{\frac{2 \log(t)}{N_i(t)}}
\]

- \( \text{mean}_i(t) \) = estimated mean reward of arm \(i\)  
- \( N_i(t) \) = number of pulls of arm \(i\)  
- \( t \) = current trial

UCB1 offers strong theoretical guarantees and performed best in our experiments.

---

## Results Summary

| Algorithm          | Cumulative Reward | Average Reward | Cumulative Regret   |
|--------------------|-------------------|----------------|--------------------|
| UCB1               | 80,079.9807       | 4.0040         | -79.9807           |
| Epsilon-Greedy     | 79,734.4694       | 3.9867         | 265.5306           |
| Thompson Sampling  | 79,335.1285       | 3.9668         | 664.8715           |

UCB1 outperforms the others in maximizing reward and minimizing regret.

---

## Requirements

This project depends on the following Python packages:

- loguru==0.7.2
- numpy
- pandas
- scipy
- seaborn
- matplotlib

Please ensure these are installed to run the code smoothly.

---

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo

### 2. Create and activate a virtual environment

- On macOS/Linux:
python3 -m venv env
source env/bin/activate
- On Windows:
python -m venv env
env\Scripts\activate
### 3. Install dependencies
pip install -r requirements.txt


