## 🚀 How to Run
Clone the repository:

```bash
git clone https://github.com/Dani1403/pokerDQN.git
cd poker-tournament-sim
```
Install dependencies
```bash
pip install -r requirements.txt
```

Run 
```bash
python main.py
```
You can tweak the simulation settings (e.g. blinds, number of players, prize structure) by editing inside **simconfig.py**:
```python
NUM_PLAYERS = 6
PRIZE_POOL = (450, 270, 180)
BLIND_SCHEDULE = (1, 2, 4, 6, 8, 12, 16, 24)
START_STACK = 150
```

**IMPORTANT** 
If you change the number of players, please make sure you add enough agents to the agents list

## 🐞 Known Bug in Clubs

There is a known issue inside the `clubs` engine — likely in **engine.py**.  

For now, here is the way to correct the bug :
  - go to **clubs/poker/engine.py**
  - in the end of the function **_eval_round**, change :

  ```python
  if remainder:
    for player_idx in range(1, self.num_players + 1):
        if payouts[player_idx + self.button % self.num_players]:  # ← ERROR
            payouts[player_idx] += remainder
            break
  ```
  by 
  ```python       
    #TODO : CORRECTED BUG IN LIB
    if remainder:
        # worst player is first player after button involved in pot
        for player_idx in range(self.num_players):
            player_idx = (self.button + player_idx + 1) % self.num_players
            if payouts[player_idx]:
                payouts[player_idx] += remainder
                break
  ```

---

## 🧪 Dependencies
clubs
clubs_gym
gymnasium
numpy
matplotlib
tqdm


