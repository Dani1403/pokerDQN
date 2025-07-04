import multiprocessing as mp
import random
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gymnasium as gym
from gymnasium.vector import make as make_vec_env
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    NUM_ENVS     = 8
    OBS_DIM      = 4          
    N_ACTIONS    = 2
    GAMMA        = 0.99
    LR           = 5e-4
    BATCH_SIZE   = 128
    BUFFER_SIZE  = 50_000
    TARGET_SYNC  = 500        
    EPS_START    = 1.0
    EPS_END      = 0.05
    EPS_DECAY    = 5_000      
    WARM_UP      = 2_000
    MAX_EPISODES = 600
    MAX_GRAD_NORM = 10.0
    MAX_EVAL_STEPS = 500
    TAU          = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = f"runs/cartpole_dqn_{datetime.now():%Y%m%d-%H%M%S}"
    writer  = SummaryWriter(log_dir)
    print("TensorBoard logs in:", log_dir)

    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(OBS_DIM, 256), nn.ReLU(),
                nn.Linear(256, 256),     nn.ReLU(),
                nn.Linear(256, N_ACTIONS)
            )
        def forward(self, x):
            return self.net(x)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer  = Adam(policy_net.parameters(), lr=LR)

    Transition = collections.namedtuple(
        "Transition",
        ("state", "action", "reward", "next_state", "done")
    )
    replay = collections.deque(maxlen=BUFFER_SIZE)

    def sample_batch():
        batch = random.sample(replay, BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(s,  dtype=torch.float32, device=device),
                torch.tensor(a,  dtype=torch.int64,  device=device),
                torch.tensor(r,  dtype=torch.float32, device=device),
                torch.tensor(s2, dtype=torch.float32, device=device),
                torch.tensor(d,  dtype=torch.float32, device=device))

    def compute_loss(batch):
        s, a, r, s2, d = batch
        q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a  = policy_net(s2).argmax(1, keepdim=True)
            q_next  = target_net(s2).gather(1, best_a).squeeze(1)
            q_tgt   = r + GAMMA * q_next * (1 - d)
        return nn.functional.mse_loss(q_pred, q_tgt)


    eval_env = gym.make("CartPole-v1")     
    def greedy_eval(net, episodes=10):
        scores = []
        for _ in tqdm(range(episodes)):
            s, _ = eval_env.reset()
            done, total = False, 0
            for _ in range(MAX_EVAL_STEPS):
                if done: 
                    break
                with torch.no_grad():
                    a = int(net(torch.as_tensor(s,
                                                dtype=torch.float32,
                                                device=device)).argmax())
                s, r, done, trunc, _ = eval_env.step(a)
                total += r
                done = done or trunc
            scores.append(total)

        return scores

    def show_results(scores):
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.title(" Cart Pole Rewards over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()
        

    env = gym.vector.make("CartPole-v1",
                          num_envs=NUM_ENVS,
                          asynchronous=True)
    states, _ = env.reset()
    episode_returns = np.zeros(NUM_ENVS, dtype=np.float32)
    completed_eps   = 0
    global_step     = 0

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    pbar = tqdm(total=MAX_EPISODES, desc="Episodes finished")

    while completed_eps < MAX_EPISODES:
        global_step += NUM_ENVS

        # epsilon-greedy actions for the entire batch
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-global_step / EPS_DECAY)
        writer.add_scalar("policy/epsilon", eps, global_step)

        with torch.no_grad():
            q_values = policy_net(torch.as_tensor(states,
                                                  dtype=torch.float32,
                                                  device=device))
            greedy_actions = q_values.argmax(1).cpu().numpy()

        random_mask   = np.random.rand(NUM_ENVS) < eps
        random_actions = np.random.randint(0, N_ACTIONS, size=NUM_ENVS)
        actions = np.where(random_mask, random_actions, greedy_actions)

        # step all environments in parallel
        next_states, rewards, dones, truncs, _ = env.step(actions)
        terminals = np.logical_or(dones, truncs)

        # store in replay
        for i in range(NUM_ENVS):
            replay.append(
                Transition(states[i],
                           actions[i],
                           rewards[i],
                           next_states[i],
                           terminals[i])
            )

        # per-environment episode accounting
        episode_returns += rewards
        for i in range(NUM_ENVS):
            if terminals[i]:
                writer.add_scalar("episode/return",
                                  episode_returns[i],
                                  completed_eps)
                episode_returns[i] = 0.0
                completed_eps += 1
                pbar.update(1)

        states = next_states

        # optimise after warm-up
        if len(replay) >= WARM_UP:
            loss = compute_loss(sample_batch())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy_net.parameters(),
                MAX_GRAD_NORM
            )
            optimizer.step()
            writer.add_scalar("loss/td_error", loss.item(), global_step)

        # # target sync
        # if global_step % TARGET_SYNC == 0:
        #     target_net.load_state_dict(policy_net.state_dict())
        with torch.no_grad():
             for tgt, src in zip(target_net.parameters(), policy_net.parameters()):
                tgt.data.mul_(1.0 - TAU)
                tgt.data.add_(TAU * src.data)

    writer.close()
    env.close()
    print("Training finished.")

    show_results(greedy_eval(policy_net, 100))
    

if __name__ == "__main__":
    mp.freeze_support()
    main()