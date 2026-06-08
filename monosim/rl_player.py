import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from monosim.player import Player


# ------------------------------------------------------------------ device

def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ------------------------------------------------------------------ network

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ replay buffer

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------ agent (shared across episodes)

class RLAgent:
    """Enthält Netzwerk, Replay-Buffer und Trainingslogik. Wird über Episoden hinweg geteilt."""

    STATE_DIM = 14
    ACTION_DIM = 5
    # 0=nicht bieten, dann 50/100/150/200 % des Normalpreises (gedeckelt durch Bargeld)
    BID_FRACTIONS = [0.0, 0.5, 1.0, 1.5, 2.0]

    def __init__(self, lr=1e-3, gamma=0.99, buffer_size=200000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9998):
        self.device = _get_device()
        self.q_network      = QNetwork(self.STATE_DIM, self.ACTION_DIM).to(self.device)
        self.target_network = QNetwork(self.STATE_DIM, self.ACTION_DIM).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size         = 64
        self.train_steps        = 0
        self.target_update_freq = 200

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.ACTION_DIM - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_t).argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q  = self.target_network(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({'q_network':   self.q_network.state_dict(),
                    'epsilon':     self.epsilon,
                    'train_steps': self.train_steps}, path)
        print(f'Modell gespeichert: {path}')

    def load(self, path):
        data = torch.load(path, weights_only=True, map_location=self.device)
        self.q_network.load_state_dict(data['q_network'])
        self.target_network.load_state_dict(data['q_network'])
        self.epsilon     = data.get('epsilon', self.epsilon_end)
        self.train_steps = data.get('train_steps', 0)


# ------------------------------------------------------------------ player

class RLPlayer(Player):
    """Monopoly-Spieler der DQN mit dichtem Reward nutzt.

    Reward zwischen zwei Geboten:
        Δbargeld  + grundstückswert_erworben * 3  (normalisiert)
        + Bonus für 2 Grundstücke einer Farbe / komplette Farbgruppe
    Terminal:
        +5 Sieg / -5 Niederlage
    """

    REWARD_SCALE             = 500.0
    TERMINAL_REWARD          = 10.0
    COLOR_PAIR_BONUS         = 200.0   # Eigenes erstes Paar in einer 3er-Gruppe
    COLOR_COMPLETE_BONUS     = 500.0   # Eigene Farbgruppe vervollständigt
    OPP_COLOR_COMPLETE_PENALTY = 500.0 # Gegner vervollständigt eine Farbgruppe

    def __init__(self, name, number, bank, list_board, dict_roads, dict_properties,
                 community_cards_deck, chance_cards_deck=None, agent=None, training=True):
        super().__init__(name, number, bank, list_board, dict_roads, dict_properties,
                         community_cards_deck, chance_cards_deck)
        self.agent    = agent or RLAgent()
        self.training = training
        # Anzahl Straßen je Farbe (unveränderlich, einmal berechnen)
        self._color_totals = {}
        for info in dict_roads.values():
            c = info['color']
            self._color_totals[c] = self._color_totals.get(c, 0) + 1
        self._reset_tracking()

    def _reset_tracking(self):
        self._pending_state          = None
        self._pending_action         = None
        self._value_acquired         = 0.0   # Farbboni für eigene Gruppen
        self._opp_complete_snapshot  = 0     # komplette Gegner-Farbgruppen beim letzten Snapshot
        self._cached_action          = None  # (property_info, state, action) — von want_to_auction gesetzt

    # ---------------------------------------------------------------- reward helpers

    def _count_opp_complete_groups(self):
        opponents = [p for p in self._dict_players.values() if not p.has_lost()]
        return sum(
            1 for opp in opponents
            for color, total in self._color_totals.items()
            if opp._dict_owned_colors.get(color, 0) == total
        )

    def _compute_reward(self):
        opp_new_complete = self._count_opp_complete_groups() - self._opp_complete_snapshot
        opp_penalty = -opp_new_complete * self.OPP_COLOR_COMPLETE_PENALTY
        return (self._value_acquired + opp_penalty) / self.REWARD_SCALE

    def _color_bonus(self, color):
        if not color or color not in self._color_totals:
            return 0.0
        owned = self._dict_owned_colors.get(color, 0)
        total = self._color_totals[color]
        if owned == total:
            return self.COLOR_COMPLETE_BONUS
        if owned == 2 and total > 2:
            return self.COLOR_PAIR_BONUS
        return 0.0

    # ---------------------------------------------------------------- buy hooks

    def buy(self, dict_road_info, road_name):
        super().buy(dict_road_info, road_name)
        if self.training:
            self._value_acquired += self._color_bonus(dict_road_info.get('color'))

    def buy_property(self, dict_property_info):
        super().buy_property(dict_property_info)

    # ---------------------------------------------------------------- state (14 features)

    def _get_state(self, dict_property_info):
        all_players = [self] + list(self._dict_players.values())
        opponents   = [p for p in all_players if p is not self and not p.has_lost()]

        my_cash    = min(self._cash / 3000.0, 2.0)
        opp_cash   = min(max((p._cash for p in opponents), default=0) / 3000.0, 2.0)
        prop_price = min(dict_property_info['price'] / 400.0, 2.0)

        my_props  = (len(self._list_owned_roads) + len(self._list_owned_stations)
                     + len(self._list_owned_utilities)) / 28.0
        opp_props = (max((len(p._list_owned_roads) + len(p._list_owned_stations)
                          + len(p._list_owned_utilities)) for p in opponents)
                     / 28.0 if opponents else 0.0)

        my_mortgaged = (len(self._list_mortgaged_roads) + len(self._list_mortgaged_stations)
                        + len(self._list_mortgaged_utilities)) / 28.0
        my_dev       = sum(v[0] + v[1] * 5 for v in self._dict_owned_houses_hotels.values()) / 40.0
        rel_wealth   = float(np.clip(
            (self._cash - (opponents[0]._cash if opponents else 0)) / 3000.0, -2.0, 2.0))
        affordability = min(self._cash / max(dict_property_info['price'], 1), 3.0) / 3.0
        my_colors    = sum(self._dict_owned_colors.values()) / 8.0

        # Farbgruppen-Features
        num_colors = max(len(self._color_totals), 1)

        # Anteil Farbgruppen wo ich genau noch 1 Grundstück zur Vervollständigung fehle
        my_near = sum(
            1 for color, total in self._color_totals.items()
            if self._dict_owned_colors.get(color, 0) == total - 1
        ) / num_colors

        # Gleiches für den stärksten Gegner
        best_opp = max(opponents, key=lambda p: len(p._list_owned_roads), default=None)
        opp_near = (sum(
            1 for color, total in self._color_totals.items()
            if best_opp._dict_owned_colors.get(color, 0) == total - 1
        ) / num_colors) if best_opp else 0.0

        # Wie weit brächte mich dieser Kauf in der jeweiligen Farbgruppe?
        this_color = dict_property_info.get('color')
        if this_color and this_color in self._color_totals:
            my_count           = self._dict_owned_colors.get(this_color, 0)
            total_in_color     = self._color_totals[this_color]
            this_color_prog    = (my_count + 1) / total_in_color
            completes_color    = 1.0 if (my_count + 1) == total_in_color else 0.0
        else:
            this_color_prog = 0.0
            completes_color = 0.0

        return np.array([my_cash, opp_cash, prop_price, my_props, opp_props,
                         my_mortgaged, my_dev, rel_wealth, affordability, my_colors,
                         my_near, opp_near, this_color_prog, completes_color],
                        dtype=np.float32)

    # ---------------------------------------------------------------- auction gate + bid

    def want_to_auction(self, dict_property_info):
        if self._cash < self.AUCTION_MINIMUM_BID:
            return False
        state  = self._get_state(dict_property_info)
        action = self.agent.select_action(state)
        self._cached_action = (dict_property_info, state, action)

        if RLAgent.BID_FRACTIONS[action] > 0.0:
            return True

        # Aktion 0: Auktion überspringen — fixe Strafe (= Kaufbonus negativ)
        if self.training:
            skip_penalty = -20.0 / self.REWARD_SCALE  # -0.04 normalisiert
            next_state   = state  # kein Zustandswechsel
            if self._pending_state is not None:
                reward = self._compute_reward() + skip_penalty
                self.agent.replay_buffer.push(
                    self._pending_state, self._pending_action, reward, next_state, False)
                self._pending_state  = None
                self._pending_action = None
                self._value_acquired = 0.0
        return False

    def bid(self, dict_property_info, player_offer):
        if self._cash < self.AUCTION_MINIMUM_BID:
            return 0

        cached = self._cached_action
        if cached is not None and cached[0] is dict_property_info:
            state, action = cached[1], cached[2]
            self._cached_action = None
        else:
            state  = self._get_state(dict_property_info)
            action = self.agent.select_action(state)

        if self.training:
            if self._pending_state is not None:
                reward = self._compute_reward()
                self.agent.replay_buffer.push(
                    self._pending_state, self._pending_action, reward, state, False)

            self._pending_state         = state
            self._pending_action        = action
            self._value_acquired        = 0.0
            self._opp_complete_snapshot = self._count_opp_complete_groups()

        fraction = RLAgent.BID_FRACTIONS[action]
        if fraction == 0.0:
            return 0
        return min(int(fraction * dict_property_info['price']), self._cash)

    # ---------------------------------------------------------------- episode end

    def end_episode(self, won, draw=False, do_train=True):
        if not self.training:
            return

        if self._pending_state is not None:
            intermediate = self._compute_reward()
            terminal     = self.TERMINAL_REWARD if won else (-self.TERMINAL_REWARD if not draw else 0.0)
            final_reward = terminal + intermediate
            zero_state   = np.zeros(RLAgent.STATE_DIM, dtype=np.float32)
            self.agent.replay_buffer.push(
                self._pending_state, self._pending_action, final_reward, zero_state, True)

        self._reset_tracking()
        if do_train:
            self.agent.train_step()
            self.agent.decay_epsilon()
