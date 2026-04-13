from eval import eval_checkpoint_dir
from training import train_and_save
from dqn_agent import DQNAgent
from poker_agents import *
import os 
import simulation
from poker_dqn import Poker_DQN

def main():
    #env
    env = simulation.PokerTournament()

    # # agents
    # dqn1 = DQNAgent(env, name="dqn1")
    # dqn2 = DQNAgent(env, name="dqn2")
    # dqn3 = DQNAgent(env, name="dqn3")
    # dqn4 = DQNAgent(env, name="dqn4")

    # SELF_PLAY_LINEUP = [dqn1, dqn2, dqn3, dqn4]
    # N_total = 20_000
    # learn_size = 5_000

    # #checkpoint_dirs = train_and_save(env, N_total=N_total, learn_size=learn_size, training_lineup=SELF_PLAY_LINEUP, checkpoint_root="checkpoints")
    # env.close()
    # checkpoint_dirs = {
    #     "dqn1": "checkpoints/dqn1_20251222_163157_707965",
    #     "dqn2": "checkpoints/dqn2_20251222_163157_707965",
    #     "dqn3": "checkpoints/dqn3_20251222_163157_707965",
    #     "dqn4": "checkpoints/dqn4_20251222_163157_707965",
    # }
    # for agent_name, ckpt_dir in checkpoint_dirs.items():
    #     dir = {agent_name: ckpt_dir}
    #     eval_checkpoint_dir(checkpoint_dirs=dir, n_workers_per_lineup=8, n_tournaments_per_worker=625)


    # agents 
    poker_dqn_1 = Poker_DQN(env, name="poker_dqn_1")
    poker_dqn_2 = Poker_DQN(env, name="poker_dqn_2")
    poker_dqn_3 = Poker_DQN(env, name="poker_dqn_3")
    poker_dqn_4 = Poker_DQN(env, name="poker_dqn_4")
    SELF_PLAY_LINEUP = [poker_dqn_1, poker_dqn_2, poker_dqn_3, poker_dqn_4]
    RANDOM_LINEUP = [poker_dqn_1, RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    ALL_IN_PAIR_LINEUP = [poker_dqn_1, AllInPairAgent(
        env), AllInPairAgent(env), AllInPairAgent(env)]
    N_total = 1_000_000
    learn_size = 50_000
    checkpoint_dirs = train_and_save(env, N_total=N_total, learn_size=learn_size,training_lineup=SELF_PLAY_LINEUP, checkpoint_root="checkpoints")

    # checkpoint_dirs = {
    #     "poker_dqn_1": "checkpoints/poker_dqn_1_20260107_173558_609746",
    #     "poker_dqn_2": "checkpoints/poker_dqn_2_20260107_173558_609746",
    #     "poker_dqn_3": "checkpoints/poker_dqn_3_20260107_173558_609746",
    #     "poker_dqn_4": "checkpoints/poker_dqn_4_20260107_173558_609746",
    # }
    env.close()
    
    eval_checkpoint_dir(checkpoint_dirs=checkpoint_dirs,
                            n_workers_per_lineup=10, n_tournaments_per_worker=10_000)

if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
