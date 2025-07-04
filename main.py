import time
import simulation                      
import gymnasium as gym      
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from poker_agents import *


def run_tournament(env, agents):
    obs, _ = env.reset(options={"reset_button": True, "reset_stacks": True})

    all_in_count = [0] * env.num_players
    fold_count = [0] * env.num_players

    done = False
    while not done:

        curr_agent = agents[env.table.dealer.action]

        action = curr_agent.act(obs)

        #update stats
        if action == env.all_in:
            all_in_count[env.table.dealer.action] += 1
        elif action == env.fold:
            fold_count[env.table.dealer.action] += 1

        next_obs, reward, done, _, _ = env.step(action)

        #env.render()
        
        for i, agent in enumerate(agents):
            agent.update_parameters(obs, action, reward[i], next_obs)

        obs = next_obs

    return reward, all_in_count, fold_count



# Helper: moving average smoothing
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def main():
    prize_pool = (450, 270, 180)
    env = simulation.PokerTournament(
        players=6,
        prize_pool=prize_pool, 
        render_mode="ascii"                
    )

    agents = [AllInAgent(env), FoldAgent(env), RandomAllInFoldAgent(env), AllInPairAgent(env), SuitedAgent(env), TwoHighAgent(env)]

    n_tournaments = 1000

    rewards_per_tournament = []
    hands_played_per_tournament = []
    blinds_end_per_tournament = []
    num_all_ins_per_player = []
    num_fold_per_player = []

    for tournament in tqdm(range(n_tournaments), desc="Running Tournament"):

        reward, all_in_count, fold_count = run_tournament(env, agents)

        rewards_per_tournament.append(reward)
        hands_played_per_tournament.append(env._hands_played)
        blinds_end_per_tournament.append(env.table.dealer.blinds)
        num_all_ins_per_player.append(all_in_count)
        num_fold_per_player.append(fold_count)


    window_size = 50
    reward_per_agent = np.array(rewards_per_tournament).T
    #num_all_ins_per_player = np.array(num_all_ins_per_player).T

    plt.figure(figsize=(50, 12))


    # plot the rewards per episode
    #plt.subplot(2, 2, 1)
    for i, rewards in enumerate(reward_per_agent):
        smoothed = moving_average(rewards, window_size)
        plt.plot(range(window_size - 1, n_tournaments), smoothed, linewidth=2, label=f"Player {i + 1}")
    plt.title("Rewards per Tournament in Poker Simulation")
    plt.xlabel("Tournament")
    plt.ylabel("Rewards")
    x_ticks_spacing = max(1, n_tournaments // 10)
    plt.xticks(ticks = range(0,n_tournaments,x_ticks_spacing), rotation=45)


    avg_rewards = np.mean(rewards_per_tournament, axis=0)
    win_count = [np.count_nonzero(reward_per_agent[i] == prize_pool[0]) for i in range(len(agents))]
    win_rate = [(win / n_tournaments) * 100 for win in win_count]

    plt.legend([f"Player {i + 1} : {agent} \n \
                  Average reward: {avg_rewards[i]:.2f} \n \
                  Win rate: {win_rate[i]:.2f}%" \
               for i, agent in enumerate(agents)])    
    plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.4)


    #plot tournament statistics
    # plt.subplot(2, 2, 2)
    # plt.plot(range(n_tournaments), hands_played_per_tournament, marker='o')
    # plt.title("Hands Played per Tournament in Poker Simulation")
    # plt.xlabel("Tournament")
    # plt.ylabel("Hands Played")
    # plt.xticks(range(n_tournaments))
    # plt.grid()

    # plot blinds at the end of each tournament
    # plt.subplot(2, 2, 3)
    # plt.plot(range(n_tournaments), [
    #          blinds[0] for blinds in blinds_end_per_tournament], marker='o', label='Small Blind')
    # plt.plot(range(n_tournaments), [
    #     blinds[1] for blinds in blinds_end_per_tournament], marker='o', label='Big Blind')
    # plt.title("Blinds at the End of Each Tournament in Poker Simulation")
    # plt.xlabel("Tournament")
    # plt.ylabel("Blinds")
    # plt.xticks(range(n_tournaments))
    # plt.legend()
    # plt.grid()

    # plot all-in counts per player
    # plt.subplot(2, 2, 4)
    # for i, rewards in enumerate(reward_per_agent):
    #     smoothed = moving_average(rewards, window_size)
    #     plt.plot(range(window_size - 1, n_tournaments), smoothed, linewidth=2, label=f"Player {i + 1}")
    # plt.title("All-In Counts per Player in Poker Simulation")
    # plt.xlabel("Tournament")
    # plt.ylabel("All-In Count")
    # plt.xticks(range(n_tournaments))
    # plt.legend([f"Player {i + 1} : {agent}" for i, agent in enumerate(agents)])
    # plt.grid()
    # plt.tight_layout()


    plt.show()

    env.close()
    
if __name__ == "__main__":
    main()
