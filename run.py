#!/usr/bin/env python3
"""
Monopoly Simulator – Einstiegspunkt

Verwendung:
    python run.py <spieler1> <spieler1> [--spiele N] [--seed S]
    python run.py rl dummy --modell rl_model.pt   # trainiertes RL-Modell laden

Verfügbare Spielertypen:
    dummy       Standardspieler – kauft alles was er kann
    no_brown    Kauft nie braune Straßen
    greedy      Nimmt Hypotheken auf um alles zu kaufen, verkauft keine Häuser
    cautious    Startet Auktion nur wenn er mehr Geld hat; bietet immer bis Normalpreis
    rl          Reinforcement-Learning Spieler (DQN)

Beispiele:
    python run.py dummy dummy
    python run.py dummy greedy --spiele 2000
    python run.py rl dummy --modell rl_model.pt --spiele 500
"""

import argparse
import random
import os

from monosim.board import get_board, get_roads, get_properties, get_community_chest_cards, get_chance_cards, get_bank
from monosim.players import PLAYER_TYPES

ALL_TYPES = list(PLAYER_TYPES.keys()) + ['rl']


def make_player(player_type, name, number, bank, list_board, dict_roads, dict_properties,
                community_deck, chance_deck, model_path=None, device='cpu'):
    if player_type == 'rl':
        from monosim.rl_player import RLAgent, RLPlayer
        agent = RLAgent(device=device)
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
        agent.epsilon = 0.0  # kein Exploration beim Testen
        return RLPlayer(name, number, bank, list_board, dict_roads, dict_properties,
                        community_deck, chance_deck, agent=agent, training=False)
    PlayerClass = PLAYER_TYPES[player_type]
    return PlayerClass(name, number, bank, list_board, dict_roads, dict_properties,
                       community_deck, chance_deck)


def run_simulation(player1_type, player2_type, num_games, start_seed, model_path=None, log_dir=None, device='cpu'):
    wins = {player1_type + '_1': 0, player2_type + '_2': 0, 'niemand': 0}

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    for seed in range(start_seed, start_seed + num_games):
        random.seed(seed)

        bank             = get_bank()
        list_board       = get_board()
        dict_roads       = get_roads()
        dict_properties  = get_properties()
        community_deck   = list(get_community_chest_cards().keys())
        chance_deck      = list(get_chance_cards().keys())

        p1 = make_player(player1_type, player1_type, 1, bank, list_board, dict_roads,
                         dict_properties, community_deck, chance_deck, model_path, device)
        p2 = make_player(player2_type, player2_type, 2, bank, list_board, dict_roads,
                         dict_properties, community_deck, chance_deck, model_path, device)

        p1.meet_other_players([p2])
        p2.meet_other_players([p1])

        list_players = [p1, p2]
        random.shuffle(list_players)

        turns = 0
        while not p1.has_lost() and not p2.has_lost() and turns < 2000:
            for player in list_players:
                player.play()
            turns += 1

        if p1.has_lost():
            wins[player2_type + '_2'] += 1
            result = f'{player2_type} gewonnen'
        elif p2.has_lost():
            wins[player1_type + '_1'] += 1
            result = f'{player1_type} gewonnen'
        else:
            wins['niemand'] += 1
            result = 'Unentschieden (Zeitlimit)'

        if log_dir:
            lines = [
                f'Spiel {seed}: {player1_type} vs {player2_type}',
                f'Runden: {turns} | Ergebnis: {result}',
                '-' * 60,
            ]
            lines += bank.get('_game_log', [])
            log_path = os.path.join(log_dir, f'spiel_{seed:04d}.txt')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')

    return wins


def main():
    parser = argparse.ArgumentParser(
        description='Monopoly Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Verfügbare Spielertypen: ' + ', '.join(ALL_TYPES)
    )
    parser.add_argument('spieler1', choices=ALL_TYPES)
    parser.add_argument('spieler2', choices=ALL_TYPES)
    parser.add_argument('--spiele', type=int, default=1000, metavar='N')
    parser.add_argument('--seed',   type=int, default=0,    metavar='S')
    parser.add_argument('--modell', default='rl_model.pt', metavar='PFAD',
                        help='Pfad zum gespeicherten RL-Modell (Standard: rl_model.pt)')
    parser.add_argument('--logs', default=None, metavar='VERZEICHNIS',
                        help='Verzeichnis für Spiel-Logs (z.B. logs/)')
    parser.add_argument('--device', default='cpu', metavar='GERÄT',
                        help='PyTorch-Device (cpu, cuda, mps) — Standard: cpu')

    args = parser.parse_args()

    print(f'Simuliere {args.spiele} Spiele: {args.spieler1} vs {args.spieler2} ...')
    wins = run_simulation(args.spieler1, args.spieler2, args.spiele, args.seed, args.modell, args.logs, args.device)

    p1_key = args.spieler1 + '_1'
    p2_key = args.spieler2 + '_2'
    total_decided = wins[p1_key] + wins[p2_key]

    print(f'\nErgebnis nach {args.spiele} Spielen:')
    print(f'  Spieler 1 ({args.spieler1}) gewonnen: {wins[p1_key]}')
    print(f'  Spieler 2 ({args.spieler2}) gewonnen: {wins[p2_key]}')
    print(f'  Unentschieden (Zeitlimit):       {wins["niemand"]}')
    if total_decided > 0:
        print(f'\n  Siegrate Spieler 1: {wins[p1_key] / total_decided * 100:.1f}%')
        print(f'  Siegrate Spieler 2: {wins[p2_key] / total_decided * 100:.1f}%')


if __name__ == '__main__':
    main()
