import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dqn_agent import DQNAgent

H_PARAMS_DQN = {
    'N_ACTIONS': 2,
    'STATE_DIM': 8,
    'HIDDEN_DIM': 128,
    'GAMMA': 0.99,
    'LR': 1e-4,
    'BATCH_SIZE': 64,
    'BUFFER_SIZE': 500_000,
    'TARGET_SYNC': 2000,
    'FREQ_TRAIN': 32,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 3_000,
}

H_PARAMS_ICM = {
    # Hyperparameters for ICM can be defined here
    'N_INPUTS': 8,
    'N_OUTPUTS': 4,
    'HIDDEN_DIM': 128,
    'LR': 1e-4,
}

class ICMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["icm_net"])

class Poker_DQN():
    def __init__(self, 
                 env, 
                 hparams_icm=H_PARAMS_ICM, 
                 hparams_dqn=H_PARAMS_DQN, 
                 name="PokerDQN", 
                 device=(torch.device("cuda" if torch.cuda.is_available() else "cpu")), 
                 enable_tb=True):

        self.name = name
        self.env = env
        self.device = device
        self.enable_tb = enable_tb
        self.prize_pool = np.array(env._prize_pool, dtype=np.float32)
        s = float(np.sum(np.abs(self.prize_pool)))
        if s > 0:
            self.prize_pool /= s

        #icm module is a fully connected component
        self.icm = ICMNet(
            hparams_icm['N_INPUTS'], 
            hparams_icm['HIDDEN_DIM'], 
            hparams_icm['N_OUTPUTS'],
            ).to(device)

        self.icm_optimizer = optim.Adam(
            self.icm.parameters(), lr=hparams_icm['LR'])


        #DQN module 

        # state dim for DQN agent
        self.base_state_dim = hparams_dqn['STATE_DIM']
        self.icm_out_dim = hparams_icm['N_OUTPUTS']
        hparams_dqn_meta = dict(hparams_dqn)
        hparams_dqn_meta['STATE_DIM'] = self.base_state_dim + self.icm_out_dim

        #create DQN agent
        self.state_dqn = DQNAgent(
            env=env,
            name=name+"_state_DQN",
            hparams=hparams_dqn_meta,
            device=self.device,
            enable_tb=enable_tb,
        )

        # training hyperparameters
        self.gamma = self.state_dqn.gamma
        self.batch_size = self.state_dqn.batch_size
        self.buffer = deque(maxlen=self.state_dqn.buffer_size)
        # state_dqn buffer IS NOT USED - we use our own buffer for the composite states
        self.freq_train = self.state_dqn.freq_train
        self.target_sync = self.state_dqn.target_sync

        # counters
        self.global_step = 0
        self.train_steps = 0

        # Tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if enable_tb:
            self.writer = SummaryWriter(f"logs/run_{self.name}_{timestamp}")

    # icm state encoder
    def _state_icm(self, obs):
        bb = self.env.table.dealer.blinds[1]
        stacks_bb = np.array([
            min(s // bb, self.env.max_stack_bb - 1) for s in obs['stacks']
            ], dtype=np.float32,
        )
        stacks_norm = stacks_bb / (self.env.max_stack_bb - 1)
        state_icm = np.concatenate([
            stacks_norm,
            self.prize_pool,
        ]).astype(np.float32)
        return state_icm

    def _preprocess_state(self, obs):
        # split in two states : one for ICM, one for DQN
        state_icm = self._state_icm(obs)
        state_dqn = self.state_dqn._preprocess_state(obs)
        return state_icm, state_dqn

    # NEVER store icm_out in replay buffer
    # Always recompute icm_out during training
    def act(self, state):
        state_icm, state_dqn_base = state
        icm_out = self.icm(torch.from_numpy(
            state_icm).to(self.device).unsqueeze(0)[0])
        state_dqn_full = np.concatenate([
            state_dqn_base, 
            icm_out.detach().cpu().numpy().astype(np.float32)
        ]).astype(np.float32)
        action = self.state_dqn.act(state_dqn_full)
        return action

    def _remember(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    def update_parameters(self, state, action, reward, next_state, done):
        
        if state is None:
            return

        action_idx = 0 if action == self.env.actions_to_env[0] else 1

        self._remember(state, action_idx, reward, next_state, done)
        self.global_step += 1

        if self.global_step % self.freq_train != 0:
            return

        if len(self.buffer) < self.batch_size:
            return

        # sample from the replay buffer
        idxs = np.random.randint(len(self.buffer), size=self.batch_size)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        # unpack stored composite states
        state_icm_np = np.stack([s[0] for s in states]).astype(np.float32)         # [B, 8]
        state_dqn_np = np.stack([s[1] for s in states]).astype(np.float32)         # [B, 8]
        next_icm_np  = np.stack([ns[0] for ns in next_states]).astype(np.float32)  # [B, 8]
        next_dqn_np  = np.stack([ns[1] for ns in next_states]).astype(np.float32)  # [B, 8]

        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)         # [B,1]
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)        # [B,1]
        dones_t   = torch.FloatTensor(dones).unsqueeze(1).to(self.device)          # [B,1]

        # NEVER store icm_out in replay buffer
        # Always recompute icm_out during training
        icm_out = self.icm(torch.from_numpy(state_icm_np).to(self.device))         # [B,4]
        with torch.no_grad():
            icm_out_next = self.icm(torch.from_numpy(next_icm_np).to(self.device)) # [B,4] 

        # build full DQN states
        state_full_t = torch.cat(
            [torch.from_numpy(state_dqn_np).to(self.device), icm_out],
            dim=1
        )  # [B,12]

        next_full_t = torch.cat(
            [torch.from_numpy(next_dqn_np).to(self.device), icm_out_next],
            dim=1
        )  # [B,12]

        # ---- Double DQN target 
        q_values = self.state_dqn.net(state_full_t).gather(1, actions_t)  # [B,1]

        with torch.no_grad():
            next_q_online = self.state_dqn.net(next_full_t)               # [B,2]
            next_actions = next_q_online.argmax(dim=1, keepdim=True)      # [B,1]
            next_q_target = self.state_dqn.target_net(next_full_t).gather(1, next_actions)  # [B,1]
            target_q = rewards_t + self.gamma * (1 - dones_t) * next_q_target

        loss = nn.MSELoss()(q_values, target_q)

        # ---- backward ----
        self.state_dqn.optimizer.zero_grad()
        self.icm_optimizer.zero_grad()

        loss.backward()

        #test : gradients of icm components
        assert icm_out.requires_grad, "ICM output detached gradients will not flow"

        # clip gradients
        nn.utils.clip_grad_norm_(self.state_dqn.net.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)

        self.state_dqn.optimizer.step()
        self.icm_optimizer.step()

        # ---- TB logging (DQN + ICM) ----
        if self.enable_tb and (self.global_step % 100 == 0):
            # DQN-style logs
            self.writer.add_scalar("Loss/TD_Error", loss.item(), self.global_step)
            self.writer.add_scalar("Policy/Epsilon", self.state_dqn.epsilon, self.global_step)
            self.writer.add_scalar("Q_values/avg_q", q_values.mean().item(), self.global_step)

            self.writer.add_scalar(
                "Loss/TD_Error_log", torch.log(loss + 1e-6).item(), self.global_step)
            self.writer.add_scalar(
                " mean batch rewards", rewards_t.mean().item(), self.global_step)
            # ICM diagnostics
            icm_entropy = -(icm_out * torch.log(icm_out + 1e-8)).sum(dim=1).mean()
            self.writer.add_scalar("ICM/entropy", icm_entropy.item(), self.global_step)

            gn = 0.0
            if self.icm.model[0].weight.grad is not None:
                gn = self.icm.model[0].weight.grad.norm().item()
            self.writer.add_scalar("ICM/grad_norm", gn, self.global_step)

        # ---- target sync
        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.state_dqn.target_net.load_state_dict(self.state_dqn.net.state_dict())

        # Decay epsilon
        self.state_dqn._update_epsilon()


    def save(self, path):
        checkpoint = {
            # ---- DQN ----
            "dqn_net": self.state_dqn.net.state_dict(),
            "dqn_target_net": self.state_dqn.target_net.state_dict(),
            "dqn_optimizer": self.state_dqn.optimizer.state_dict(),
            "epsilon": self.state_dqn.epsilon,

            # ---- ICM ----
            "icm_net": self.icm.state_dict(),
            "icm_optimizer": self.icm_optimizer.state_dict(),

            # ---- bookkeeping ----
            "global_step": self.global_step,

        }
        torch.save(checkpoint, path)
        print(f"[{self.name}] Model saved at {path}")


    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)

        # ---- DQN ----
        self.state_dqn.net.load_state_dict(checkpoint["dqn_net"])
        self.state_dqn.target_net.load_state_dict(checkpoint["dqn_target_net"])
        self.state_dqn.optimizer.load_state_dict(checkpoint["dqn_optimizer"])
        self.state_dqn.epsilon = checkpoint["epsilon"]

        # ---- ICM ----
        self.icm.load_state_dict(checkpoint["icm_net"])
        self.icm_optimizer.load_state_dict(checkpoint["icm_optimizer"])

        # ---- bookkeeping ----
        self.global_step = checkpoint.get("global_step", 0)


    def __str__(self):
        return self.name