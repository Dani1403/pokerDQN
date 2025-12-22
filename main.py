from eval import eval_checkpoint_dir
from training import train_and_save
from dqn_agent import DQNAgent
from poker_agents import *
import os 
import simulation
import cProfile, pstats, io

def main():
    #env
    env = simulation.PokerTournament()

    # agents
    dqn1 = DQNAgent(env, name="dqn1")
    dqn2 = DQNAgent(env, name="dqn2")

    training_lineup = [dqn1,dqn2, RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]

    N_total = 5_000
    learn_size = 1_000

    checkpoint_dirs = train_and_save(env, N_total=N_total, learn_size=learn_size, training_lineup=training_lineup, checkpoint_root="checkpoints")
    env.close()
    eval_checkpoint_dir(checkpoint_dirs=checkpoint_dirs, n_workers_per_lineup=4, n_tournaments_per_worker=500)


if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
