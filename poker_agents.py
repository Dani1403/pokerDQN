import random
from clubs.poker.card import CHAR_RANK_TO_INT_RANK

class BaseAgent:
    def __init__(self, env):
        self.env = env

    def act(self, observation) -> int:
        raise NotImplementedError(
            "This method should be overridden by subclasses.")

    def update_parameters(self, obs, action, reward, next_obs):
        pass

    def __str__(self):
        return self.__class__.__name__

class RandomAllInFoldAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        return random.choice([self.env.all_in, self.env.fold])

    def update_parameters(self, obs, action, reward, next_obs):
        pass


class AllInAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        return self.env.all_in

    def update_parameters(self, obs, action, reward, next_obs):
        pass


class FoldAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        return self.env.fold

    def update_parameters(self, obs, action, reward, next_obs):
        pass


class AllInPairAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        hole_cards = observation['hole_cards']
        if hole_cards[0].rank == hole_cards[1].rank:
            #print("[AGENT] Going all-in with a pair:", hole_cards)
            return self.env.all_in
        else:
            return self.env.fold

    def update_parameters(self, obs, action, reward, next_obs):
        pass


class AllInHighCardAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        hole_cards = observation['hole_cards']
        card_1 = CHAR_RANK_TO_INT_RANK[hole_cards[0].rank]
        card_2 = CHAR_RANK_TO_INT_RANK[hole_cards[1].rank]
        if card_1 >= 10 or card_2 >= 10:
            #print("[AGENT] Going all-in with a high card:", hole_cards)
            return self.env.all_in
        else:
            return self.env.fold

    def update_parameters(self, obs, action, reward, next_obs):
        pass



class SuitedAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        card1, card2 = observation['hole_cards']
        if card1.suit == card2.suit:
            # print("[AGENT] Going all-in with suited cards:", card1, card2)
            return self.env.all_in
        return self.env.fold


class TwoHighAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def act(self, observation) -> int:
        card1, card2 = observation['hole_cards']
        r1 = CHAR_RANK_TO_INT_RANK[card1.rank]
        r2 = CHAR_RANK_TO_INT_RANK[card2.rank]
        if r1 >= 10 and r2 >= 10:
            # print("[AGENT] Going all-in with two high cards:", card1, card2)
            return self.env.all_in
        return self.env.fold

