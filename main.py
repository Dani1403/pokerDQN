from eval import eval_checkpoint_dir
from training import train_and_save
from dqn_agent import DQNAgent
from poker_agents import *
import os 
import simulation
import cProfile, pstats, io
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
    poker_dqn = Poker_DQN(env, name="poker_dqn")
    RANDOM_LINEUP = [poker_dqn, RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    N_total = 10_000
    learn_size = 2_000
    checkpoint_dir = train_and_save(env, N_total=N_total, learn_size=learn_size,
                                    training_lineup=RANDOM_LINEUP, checkpoint_root="checkpoints")
    env.close()
    for agent_name, ckpt_dir in checkpoint_dir.items():
        dir = {agent_name: ckpt_dir}
        eval_checkpoint_dir(checkpoint_dirs=dir,
                            n_workers_per_lineup=2, n_tournaments_per_worker=500)

if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
