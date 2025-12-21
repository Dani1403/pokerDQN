from eval import eval_checkpoint_dir
import cProfile, pstats, io

def main():
    # env = simulation.PokerTournament()
    # dqn = DQNAgent(env, "dqn")
    # if os.path.exists(f"checkpoints/{dqn}/final.pt"):
    #     dqn.load(f"checkpoints/{dqn}/final.pt")
    #     print(f"Loaded pretrained DQNAgent {dqn}")
    # dqn2 = DQNAgent(env, "dqn2")
    # if os.path.exists(f"checkpoints/{dqn2}/final.pt"):
    #     dqn2.load(f"checkpoints/{dqn2}/final.pt")
    #     print("Loaded pretrained DQNAgent dqn2")
    # RANDOM_LINEUP = [dqn,RandomAllInFoldAgent(env), RandomAllInFoldAgent(env), RandomAllInFoldAgent(env)]
    # ALL_IN_PAIR_LINEUP = [dqn, AllInPairAgent(env), AllInPairAgent(env), AllInPairAgent(env)]
    # TWO_HIGH_LINEUP = [dqn, TwoHighAgent(env), TwoHighAgent(env), TwoHighAgent(env)]
    # POOL = [RandomAllInFoldAgent, AllInPairAgent, TwoHighAgent, SuitedAgent]

    # train_and_evaluate(env, N_total=100_000, learn_size=20_000, eval_size=1_000,
    #                    training_lineup=ALL_IN_PAIR_LINEUP,
    #                    evaluation_lineups=[ALL_IN_PAIR_LINEUP])
    # env.close()
    eval_checkpoint_dir("checkpoints/dqn/20251217_205057_434015") #continuing of best model
    #eval_checkpoint_dir("checkpoints/dqn/20251218_085359_791915") #best model so far at the end


if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(40)
    #print(s.getvalue())
