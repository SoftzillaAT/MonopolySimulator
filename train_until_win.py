#!/usr/bin/env python3
"""Trainiert rl_v7.pt so lange bis es dummy mit >50% Siegrate schlägt."""

import random
from monosim.board import (get_board, get_roads, get_properties,
                           get_community_chest_cards, get_chance_cards, get_bank)
from monosim.players import PLAYER_TYPES
from monosim.rl_player import RLAgent, RLPlayer
from train import run_episode
from run import run_simulation

SAVE_PATH    = 'rl_v7.pt'
BATCH        = 5000
TEST_GAMES   = 500
TARGET_RATE  = 0.50
MAX_EPISODES = 500000

import os
agent = RLAgent()
if os.path.exists(SAVE_PATH):
    agent.load(SAVE_PATH)
    print(f'Checkpoint geladen (train_steps={agent.train_steps}, ε={agent.epsilon:.3f})\n')

OpponentClass = PLAYER_TYPES['dummy']
total_ep = 0
round_n  = 0

while total_ep < MAX_EPISODES:
    round_n += 1
    wins = losses = draws = 0
    for i in range(BATCH):
        won, lost, draw = run_episode(agent, OpponentClass, seed=total_ep + i, training=True)
        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1
    total_ep += BATCH

    decided  = wins + losses
    train_wr = wins / decided if decided else 0.0
    print(f'[Runde {round_n:>3} | Ep {total_ep:>6}]  '
          f'Train-Siegrate {train_wr:.1%} | ε={agent.epsilon:.3f} | '
          f'Buffer {len(agent.replay_buffer):>6}')

    agent.save(SAVE_PATH)

    # Test gegen dummy (deterministisch, seed 90000+)
    test_wins = run_simulation('rl', 'dummy', TEST_GAMES, start_seed=90000, model_path=SAVE_PATH)
    rl_key    = 'rl_1'
    dum_key   = 'dummy_2'
    decided_t = test_wins[rl_key] + test_wins[dum_key]
    test_wr   = test_wins[rl_key] / decided_t if decided_t else 0.0
    print(f'              Test  vs dummy:  {test_wr:.1%} ({test_wins[rl_key]}/{TEST_GAMES})\n')

    if test_wr >= TARGET_RATE:
        print(f'Ziel erreicht! Siegrate {test_wr:.1%} >= {TARGET_RATE:.0%}')
        break
else:
    print('Maximale Episodenzahl erreicht.')
