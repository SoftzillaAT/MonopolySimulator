from monosim.player import Player


def _modify_buy(buy):
    def _wrapper(self, dict_road_info, road_name):
        if dict_road_info['color'] == 'brown':
            pass
        else:
            buy(self, dict_road_info, road_name)
    return _wrapper


class PlayerNoBrown(Player):
    """Never buys brown roads."""
    buy = _modify_buy(Player.buy)


class PlayerGreedy(Player):
    """Kauft alles was möglich ist. Im eigenen Zug nimmt er Hypotheken auf um zu kaufen.
    Verkauft nie Häuser."""

    def _mortgageable_now(self):
        """Tatsächlich hypothezierbares Kapital (Straßen ohne Häuser, Bahnhöfe, Werke)."""
        roads = sum(
            self._dict_roads[r]['mortgage_value']
            for r in self._list_owned_roads
            if not self._dict_roads[r]['is_mortgaged']
            and self._dict_owned_houses_hotels.get(r, (0, 0)) == (0, 0)
        )
        stations = sum(
            self._dict_properties[s]['mortgage_value']
            for s in self._list_owned_stations
            if not self._dict_properties[s]['is_mortgaged']
        )
        utilities = sum(
            self._dict_properties[u]['mortgage_value']
            for u in self._list_owned_utilities
            if not self._dict_properties[u]['is_mortgaged']
        )
        return roads + stations + utilities

    def play(self):
        self._is_my_turn = True
        try:
            super().play()
        finally:
            self._is_my_turn = False

    def bid(self, dict_property_info, player_offer):
        """Im eigenen Zug: bis Bargeld + Hypotheken. Im fremden Zug: nur Bargeld."""
        if getattr(self, '_is_my_turn', False):
            max_bid = self._cash + self._mortgageable_now()
        else:
            max_bid = self._cash
        if max_bid < self.AUCTION_MINIMUM_BID:
            return 0
        return min(max_bid, dict_property_info['price'])

    def _prepare_to_pay(self, amount):
        """Hypotheken aufnehmen um den Auktionspreis zu bezahlen (nur im eigenen Zug augerufen)."""
        if amount <= self._cash:
            return
        needed = amount - self._cash
        try:
            list_props = self.choose_mortgage_properties(needed)
            for prop in list_props:
                self.mortgage(prop[1], prop[0])
        except Exception:
            pass

    def have_enough_money(self, amount, plus_mortgageable=False):
        """Nur tatsächlich hypothezierbares Kapital zählen (keine Straßen mit Häusern)."""
        if plus_mortgageable:
            return amount <= self._cash + self._mortgageable_now()
        return amount <= self._cash

    def is_bankrupt(self, value_to_pay, creditor=None):
        """Bankrott wenn Bargeld + reales Hypothekenpotenzial nicht reicht."""
        if self._cash + self._mortgageable_now() < value_to_pay:
            self.declare_bankruptcy(creditor)
        return self._has_lost

    def get_money_from_mortgages(self, amount_required):
        """Nur Hypotheken aufnehmen, keine Häuser verkaufen.
        amount_required ist der Fehlbetrag (nicht der Gesamtbetrag)."""
        if amount_required <= 0:
            return
        list_properties = self.choose_mortgage_properties(amount_required)
        for prop in list_properties:
            self.mortgage(prop[1], prop[0])


class PlayerCautious(Player):
    """Startet eine Auktion nur wenn er mehr Bargeld hat als alle Gegner.
    In fremden Auktionen bietet er immer bis zum Normalpreis mit."""

    def want_to_auction(self, dict_property_info):
        all_players = [self] + list(self._dict_players.values())
        opponent_cash = [p._cash for p in all_players if p is not self and not p.has_lost()]
        if opponent_cash and self._cash <= max(opponent_cash):
            return False
        return super().want_to_auction(dict_property_info)

    def bid(self, dict_property_info, player_offer):
        if self._cash < self.AUCTION_MINIMUM_BID:
            return 0
        return min(self._cash, dict_property_info['price'])


PLAYER_TYPES = {
    'dummy': Player,
    'no_brown': PlayerNoBrown,
    'greedy': PlayerGreedy,
    'cautious': PlayerCautious,
}
