Installation and Usage
Requirements

The project was developed in Python and uses PyTorch, Gymnasium, and additional libraries for poker simulation and Reinforcement Learning training.

Recommended environment:

Python 3.10+
Linux / Ubuntu (recommended)
Windows with WSL also works

GPU support with CUDA is recommended for faster training, although the project can also run on CPU.

Project Setup
1. Clone the Repository
git clone <repository_url>
cd <project_folder>
2. Install the clubs Poker Engine

The project depends on an external poker engine called clubs.

This step is important and must be done correctly.

Navigate into the clubs directory and install it in editable mode:

cd clubs
pip install -e .
cd ..

The command must be executed from inside the clubs directory itself.

If this step is skipped, the simulator will fail to import the poker engine correctly.

3. Install Python Dependencies

If a requirements.txt file is available:

pip install -r requirements.txt

Otherwise, install the main dependencies manually:

pip install torch gymnasium numpy matplotlib tensorboard
Running Training

To start training the DQN agent:

python main.py

The system will start running self-play poker tournaments and training the agent.

During training:

model checkpoints are periodically saved,
TensorBoard logs are generated,
training statistics are tracked.
TensorBoard Monitoring

To monitor training metrics:

tensorboard --logdir logs

Then open:

http://localhost:6006

in your browser.

Remote Server Usage

If running on a remote machine or cluster, you will need port forwarding to access TensorBoard locally.

Example using SSH:

ssh -L 6006:localhost:6006 user@remote_server

Then open:

http://localhost:6006

on your local machine.

Evaluation

The project includes evaluation scripts for running automated tournaments between agents.

The evaluation system supports:

large tournament batches,
average placement statistics,
cumulative reward tracking,
win-rate analysis,
policy comparison.
Visualization Tools

The project also includes visualization tools for analyzing learned Push/Fold strategies.

These tools generate:

hand-range heatmaps,
policy visualizations,
tournament behavior analysis.
Main Project Files
simulation.py	Tournament simulation environment
dqn_agent.py	DQN agent implementation
qagent.py	Classical Q-Learning agent
poker_agents.py	Heuristic baseline agents
evaluator.py	Evaluation and tournament analysis
main.py	Main training entry point
simconfig.py	Simulation configuration and hyperparameters
