# MonopolySimulator

German-language Monopoly simulation with pluggable player strategies and a DQN reinforcement-learning agent.

## Project layout

```
monosim/
  player.py       # Base Player class — all game logic lives here
  players.py      # Strategy subclasses (PlayerNoBrown, PlayerGreedy, PlayerCautious)
  rl_player.py    # DQN agent (RLAgent, RLPlayer)
  board.py        # Board/card data
  simulator.py    # (legacy) single-game runner
run.py            # CLI: two-player simulation, all types
train.py          # RL training loop
test/
  test_players.py # pytest suite
```

## Running

```bash
# Two-player simulation (1000 games)
python run.py dummy greedy --spiele 1000

# Available player types: dummy, no_brown, greedy, cautious, rl
python run.py rl dummy --modell rl_model.pt --spiele 500

# RL training
python train.py --gegner dummy --episoden 10000 --speichern rl_model.pt
python train.py --gegner greedy --episoden 20000
```

## Tests

```bash
pytest test/
```

## Player strategies

| Type | Behavior |
|------|----------|
| `dummy` | Buys everything it can afford |
| `no_brown` | Like dummy, skips brown streets |
| `greedy` | Mortgages own properties (own turn only) to fund every purchase; never sells houses |
| `cautious` | Only starts auctions when richer than all opponents; always bids up to normal price |
| `rl` | DQN agent — 10-dim state, 5 bid fractions (0%, 50%, 100%, 150%, 200% of list price) |

## Key design notes

### Auction (second-price / Vickrey)

Winner pays `max(AUCTION_MINIMUM_BID, second_highest_bid + 1)`, capped at their own bid. Only the auction initiator can mortgage during the auction (`_prepare_to_pay` hook).

### Hotel data model

A hotel is stored as `(4, 1)` in `_dict_owned_houses_hotels`, not `(0, 1)`. This is because rent lookup uses the key `rent_with_4_houses_1_hotels`. The 4 physical house tokens are returned to the bank even though the tuple still shows `4`.

### RL reward structure

- **Intermediate** (between bids): `(cash_delta + mortgage_value * 3) / 500`, clipped to `[-3, 3]`
- **Terminal**: `±5` for win/loss, `0` for draw
- At list price (~2× mortgage): buying yields `−2m + 3m = +m` → positive reward
- At 150% list: `−3m + 3m = 0` → neutral
- At 200% list: `−4m + 3m = −m` → negative (discourages overbidding)

### Greedy mortgaging

`PlayerGreedy` sets `_is_my_turn = True` only during `play()`. `bid()` and `_prepare_to_pay()` check this flag so mortgages are only raised when it's genuinely the player's own turn.
