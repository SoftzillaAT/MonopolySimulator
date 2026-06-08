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

# With per-game purchase logs written to logs/
python run.py dummy greedy --spiele 10 --logs logs/

# RL training (auto-resumes from checkpoint if file exists)
python train.py --gegner dummy --episoden 10000 --speichern rl_model.pt
python train.py --selfplay --episoden 20000 --speichern rl_model.pt
```

## Tests

```bash
pytest test/
```

## Player strategies

| Type | Behavior |
|------|----------|
| `dummy` | Always starts auctions, bids up to 80% of cash capped at list price |
| `no_brown` | Like dummy, skips brown streets |
| `greedy` | Mortgages own properties (own turn only) to fund every purchase; never sells houses |
| `cautious` | Only starts auctions when richer than all opponents; always bids up to normal price |
| `rl` | DQN agent — 14-dim state, 5 bid fractions (0%, 50%, 100%, 150%, 200% of list price) |

## Key design notes

### Auction (second-price / Vickrey)

Winner pays `max(AUCTION_MINIMUM_BID, second_highest_bid + 1)`, capped at their own bid. Only the auction initiator can mortgage during the auction (`_prepare_to_pay` hook). All property acquisitions — including from chance/community cards — go through `run_auction`.

### want_to_auction

Base player always returns `True`. The RL player uses the Q-network to decide: if the selected action is 0 (no bid), the auction is skipped and a skip penalty is pushed to the replay buffer.

### Hotel data model

A hotel is stored as `(4, 1)` in `_dict_owned_houses_hotels`, not `(0, 1)`. This is because rent lookup uses the key `rent_with_4_houses_1_hotels`. The 4 physical house tokens are returned to the bank even though the tuple still shows `4`.

### RL reward structure

- **Intermediate** (between bids): `(cash_delta + value_acquired) / 500`, clipped to `[-3, 3]`
- **value_acquired** = `mortgage_value × 3` + color bonuses (pair: +200, complete: +500)
- **Skip penalty**: `-(mortgage_value) / 500` when auction is skipped
- **Terminal**: `±10` for win/loss, `0` for draw
- At list price (~2× mortgage): buying yields `−2m + 3m = +m` → positive reward
- At 150% list: `−3m + 3m = 0` → neutral
- At 200% list: `−4m + 3m = −m` → negative (discourages overbidding)

### RL state (14 features)

`my_cash, opp_cash, prop_price, my_props, opp_props, my_mortgaged, my_dev, rel_wealth, affordability, my_colors, my_near_complete, opp_near_complete, this_color_progress, completes_color`

### Greedy mortgaging

`PlayerGreedy` sets `_is_my_turn = True` only during `play()`. `bid()` and `_prepare_to_pay()` check this flag so mortgages are only raised when it's genuinely the player's own turn.

### Game logging

Purchase events are written to `bank['_game_log']` in `run_auction`. Pass `--logs <dir>` to `run.py` to write one `spiel_NNNN.txt` per game. Logging is off by default.
