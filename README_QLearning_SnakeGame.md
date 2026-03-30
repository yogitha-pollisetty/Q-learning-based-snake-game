# 🐍 Q-Learning Based Snake Game — Reinforcement Learning Project

## 📌 Project Overview
This project implements a **Reinforcement Learning agent** using the **Q-Learning algorithm** to train an AI to play the classic Snake game. The agent learns optimal strategies through thousands of episodes of gameplay, improving its performance over time without any human intervention.

---

## 🎯 Objective
To train an AI agent that can play Snake autonomously by learning from its own experience — navigating toward food while avoiding walls and itself — using Q-Learning reward-based decision making.

---

## 🛠️ Tools & Technologies Used

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Q-Learning Algorithm | Reinforcement learning technique |
| Q-Table | Storing and updating learned strategies |
| Custom Game Environment | Grid-based state space built in Python |
| Matplotlib | Visualizing agent performance |

---

## 🧠 How It Works

### 1. Environment
- Built a custom Python game environment with a **grid-based state space**
- The agent observes the current state of the game at each step
- Actions available: Move Up, Down, Left, Right

### 2. Q-Learning Algorithm
- The agent uses a **Q-Table** to store the value of each action in each state
- Q-Table is updated dynamically using the Q-Learning update rule:
  - **Q(s,a) = Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]**
  - Where: α = learning rate, γ = discount factor, r = reward

### 3. Reward Function
- **Positive reward** — agent moves toward food
- **Negative reward** — agent hits a wall or itself
- **Large positive reward** — agent eats food successfully

### 4. Exploration vs Exploitation
- Used **Epsilon-Greedy strategy** to balance:
  - **Exploration** — trying new actions to discover better strategies
  - **Exploitation** — using learned strategies to maximize reward
- Epsilon decreases over time as the agent becomes more confident

---

## 📈 Results

- Agent performance improved significantly over thousands of episodes
- Cumulative rewards increased as the agent learned optimal navigation strategies
- Average episode score improved consistently across training runs

---

## 📊 Visualizations

- **Cumulative Rewards Plot** — shows total reward earned per episode over time
- **Average Episode Metrics** — tracks improvement in agent performance across training

---

## 🔑 Key Learnings

- How reinforcement learning differs from supervised and unsupervised learning
- Practical implementation of Q-Learning from scratch in Python
- Importance of reward function design in shaping agent behavior
- Balancing exploration vs exploitation for optimal learning convergence

---

## 📁 Project Structure

```
q-learning-snake-game/
│
├── snake_game.py          # Custom game environment
├── q_learning_agent.py    # Q-Learning agent implementation
├── train.py               # Training loop and episode management
├── visualize.py           # Performance visualization
├── q_table.pkl            # Saved Q-Table after training
└── README.md              # Project documentation
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yogitha-pollisetty/q-learning-snake-game.git

# Navigate to project folder
cd q-learning-snake-game

# Install dependencies
pip install matplotlib numpy

# Train the agent
python train.py

# Visualize performance
python visualize.py
```

---

## 👩‍💻 About Me
**Yogitha Pollisetty** — Data Analyst with 2+ years of experience and a Master's in Data Science from DePaul University. Skilled in SQL, Python, Power BI, and Machine Learning.

🔗 [LinkedIn](https://linkedin.com/in/yogithapollisetty) | 📧 yogithapollisetty@gmail.com

---

*This project was completed as part of my Master's in Data Science at DePaul University.*
