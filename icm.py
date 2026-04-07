from poker_dqn import ICMNet, H_PARAMS_ICM
from simconfig import PRIZE_POOL
import numpy as np
import torch
icm = ICMNet(H_PARAMS_ICM['N_INPUTS'],
             H_PARAMS_ICM['HIDDEN_DIM'], 
             H_PARAMS_ICM['N_OUTPUTS'])

icm.load('checkpoints/poker_dqn_2_20260107_173558_609746/final.pt')

print("ICM loaded successfully")

# Output the ICM's prediction for a sample vector of prize pool and stacks
stacks = np.array([10, 9, 8, 7])

prize_pool = np.array([50,30,20,10])

input_vector = np.concatenate((stacks, prize_pool)).astype(np.float32)

input_tensor = torch.from_numpy(input_vector)

with torch.no_grad():
    predicted_rewards = icm(input_tensor)
    print("Probabilities:", predicted_rewards.numpy())
