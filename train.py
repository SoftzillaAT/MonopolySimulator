#!/usr/bin/env python3
"""
RL-Spieler trainieren.

Verwendung:
    python train.py                          # gegen dummy, 10000 Episoden
    python train.py --gegner greedy --episoden 20000
    python train.py --gegner dummy --speichern rl_vs_dummy.pt
"""

import argparse
import random

from monosim.board import (get_board, get_roads, get_properties,
                           get_community_chest_cards, get_chance_cards, get_bank)
from monosim.players import PLAYER_TYPES
from monosim.rl_player import RLAgent, RLPlayer


def run_episode(agent, opponent_class, seed, training=True, max_turns=2000):
    random.seed(seed)
    bank             = get_bank()
    list_board       = get_board()
    dict_roads       = get_roads()
    dict_properties  = get_properties()
    community_deck   = list(get_community_chest_cards().keys())
    chance_deck      = list(get_chance_cards().keys())

    rl = RLPlayer('rl', 1, bank, list_board, dict_roads, dict_properties,
                  community_deck, chance_deck, agent=agent, training=training)
    opp = opponent_class('gegner', 2, bank, list_board, dict_roads, dict_properties,
                         community_deck, chance_deck)

    rl.meet_other_players([opp])
    opp.meet_other_players([rl])

    players = [rl, opp]
    random.shuffle(players)

    turns = 0
    while not rl.has_lost() and not opp.has_lost() and turns < max_turns:
        for p in players:
            p.play()
        turns += 1

    won  = opp.has_lost()
    lost = rl.has_lost()
    draw = not won and not lost

    if training:
        rl.end_episode(won=won, draw=draw)

    return won, lost, draw


def run_episode_selfplay(agent, seed, max_turns=2000):
    random.seed(seed)
    bank            = get_bank()
    list_board      = get_board()
    dict_roads      = get_roads()
    dict_properties = get_properties()
    community_deck  = list(get_community_chest_cards().keys())
    chance_deck     = list(get_chance_cards().keys())

    rl1 = RLPlayer('rl1', 1, bank, list_board, dict_roads, dict_properties,
                   community_deck, chance_deck, agent=agent, training=True)
    rl2 = RLPlayer('rl2', 2, bank, list_board, dict_roads, dict_properties,
                   community_deck, chance_deck, agent=agent, training=True)

    rl1.meet_other_players([rl2])
    rl2.meet_other_players([rl1])

    players = [rl1, rl2]
    random.shuffle(players)

    turns = 0
    while not rl1.has_lost() and not rl2.has_lost() and turns < max_turns:
        for p in players:
            p.play()
        turns += 1

    won1 = rl2.has_lost()
    won2 = rl1.has_lost()
    draw = not won1 and not won2

    # Both push final transitions; only one triggers train_step + decay
    rl1.end_episode(won=won1, draw=draw, do_train=False)
    rl2.end_episode(won=won2, draw=draw, do_train=True)

    return won1, won2, draw


def train_selfplay(num_episodes=10000, save_path='rl_model.pt', device='cpu'):
    import os
    agent = RLAgent(device=device)

    if os.path.exists(save_path):
        agent.load(save_path)
        print(f'Checkpoint geladen: {save_path} (train_steps={agent.train_steps}, ε={agent.epsilon:.3f})')

    print(f'Self-play Training — {num_episodes} Episoden\n')

    interval = 500
    wins = losses = draws = 0

    for ep in range(num_episodes):
        won, lost, draw = run_episode_selfplay(agent, seed=ep)

        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1

        if (ep + 1) % interval == 0:
            decided  = wins + losses
            win_rate = wins / decided if decided else 0.0
            print(f'[{ep+1:>6}/{num_episodes}]  '
                  f'Siege {wins:>4} | Niederlagen {losses:>4} | Unentschieden {draws:>3} | '
                  f'Siegrate {win_rate:.1%} | ε={agent.epsilon:.3f} | '
                  f'Buffer {len(agent.replay_buffer):>6}')
            wins = losses = draws = 0

    agent.save(save_path)
    return agent


def train(opponent_type='dummy', num_episodes=10000, save_path='rl_model.pt', device='cpu'):
    import os
    agent        = RLAgent(device=device)
    OpponentClass = PLAYER_TYPES[opponent_type]

    if os.path.exists(save_path):
        agent.load(save_path)
        print(f'Checkpoint geladen: {save_path} (train_steps={agent.train_steps}, ε={agent.epsilon:.3f})')

    print(f'Training RL vs {opponent_type} — {num_episodes} Episoden\n')

    interval = 500
    wins = losses = draws = 0

    for ep in range(num_episodes):
        won, lost, draw = run_episode(agent, OpponentClass, seed=ep, training=True)

        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1

        if (ep + 1) % interval == 0:
            decided  = wins + losses
            win_rate = wins / decided if decided else 0.0
            print(f'[{ep+1:>6}/{num_episodes}]  '
                  f'Siege {wins:>4} | Niederlagen {losses:>4} | Unentschieden {draws:>3} | '
                  f'Siegrate {win_rate:.1%} | ε={agent.epsilon:.3f} | '
                  f'Buffer {len(agent.replay_buffer):>6}')
            wins = losses = draws = 0

    agent.save(save_path)
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gegner',    default='dummy',       choices=list(PLAYER_TYPES.keys()))
    parser.add_argument('--episoden',  type=int, default=10000)
    parser.add_argument('--speichern', default='rl_model.pt')
    parser.add_argument('--selfplay',  action='store_true')
    parser.add_argument('--device',    default='cpu', metavar='GERÄT',
                        help='PyTorch-Device (cpu, cuda, mps) — Standard: cpu')
    args = parser.parse_args()

    if args.selfplay:
        train_selfplay(args.episoden, args.speichern, args.device)
    else:
        train(args.gegner, args.episoden, args.speichern, args.device)
