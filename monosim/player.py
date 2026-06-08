from monosim.board import get_color_to_house_mapping
import random
#  TODO allow user to set verbosity. Text should be printed only if verbosity=1 is set.
#   Add paramenter to the constructor and a ad-hoc function set_verbosity().
#   Do this first otherwise it's impossible to run the simulations with jupyter notebooks because of the excessive
#   amount of text in the output.
# TODO make test function to check if the _list_mortgaged_roads is properly used. DO not mortgage properties
#  from this list Check that attribute _properties_total_mortgageable_amount and _list_mortgaged_roads always sum up
#  to the same value
#  TODO check if a road is mortgaged in the function choose_house_hotel_to_buy. If the road is mortgaged is probably
#   not possible to buy a house (check the rules)
#  TODO Implement cards functionality (opportunity, etc.)


class Player:
    def __init__(self, name, number, bank, list_board, dict_roads, dict_properties, community_cards_deck, chance_cards_deck=None):
        self._name = name
        self._number = number
        self._list_board = list_board
        self._dict_roads = dict_roads
        self._dict_properties = dict_properties

        self._position = 0
        self._dice_value = 0
        self._cash = 1500
        self._properties_total_mortgageable_amount = 0
        self._exit_jail = False  # holds card to exit jail
        self._jail_count = 0
        self._free_visit = False
        self._bank = bank
        self._list_players = None
        self._dict_players = None
        self._list_owned_roads = []
        self._list_owned_stations = []
        self._list_owned_utilities = []
        self._list_mortgaged_roads = []
        self._list_mortgaged_stations = []
        self._list_mortgaged_utilities = []
        self._dict_owned_colors = {'brown': False, 'light_blue': False, 'purple': False, 'orange': False,
                                   'red': False, 'yellow': False, 'green': False, 'blue': False}
        self._dict_owned_houses_hotels = {}
        self._has_lost = False
        self.color_to_house_mapping = get_color_to_house_mapping()
        self.community_cards_deck = community_cards_deck
        self.chance_cards_deck = chance_cards_deck if chance_cards_deck is not None else []

    def roll_dice(self):
        """ Simulate the roll of two dice. Returns two int values between 1 and 6.

        :return: (tuple) two int values between 1 and 6 drawn from a uniform distribution.
        """

        x = random.randint(1, 6)
        y = random.randint(1, 6)
        return x, y

    def get_state(self):
        """ Get the player's state. The state contains information such as position, roads owned, money, mortgaged
            properties, etc.

        :return: (dictionary) key: property, value: property value (e.g. {'cash': 100, 'position': 10, ...})
        """

        return {'name': self._name, 'number': self._number, 'position': self._position,
                'dice_value': self._dice_value, 'cash': self._cash,
                'mortgageable_amount': self._properties_total_mortgageable_amount, 'jail_count': self._jail_count,
                'exit_jail': self._exit_jail, 'free_visit': self._free_visit, 'owned_roads': self._list_owned_roads,
                'owned_stations': self._list_owned_stations, 'owned_utilities': self._list_owned_utilities,
                'mortgaged_roads': self._list_mortgaged_roads, 'mortgaged_stations': self._list_mortgaged_stations,
                'mortgaged_utilities': self._list_mortgaged_utilities, 'owned_colors': self._dict_owned_colors,
                'owned_houses_hotels': self._dict_owned_houses_hotels, 'has_lost': self._has_lost,
                'bank_cash': self._bank['cash']}

    def set_cash(self, amount):
        self._cash = amount

    def meet_other_players(self, list_players):
        """ Get opponent players. Create dict {'name': Player} of players.

        :param list_players: (list) players objects of the other opponents
        :return:
        """
        self._list_players = list_players

        # dict of players easier/faster to use later
        self._dict_players = {player._name: player for player in list_players}

    def have_enough_money(self, amount, plus_mortgageable=False):
        """ Determine if the player has enough money. The required amount is passed as parameter (amount).
            The function can compare the amount against cash only or cash+mortgageable amount.

        :param amount: (int) Cash required
        :param plus_mortgageable: (bool) Compare against cash only (if False) or cash + mortgageable (if True)
        :return: (bool) if True, player has enough money, False otherwise
        """
        if plus_mortgageable:
            return amount <= self.cash + self._properties_total_mortgageable_amount
        else:
            return amount <= self._cash

    AUCTION_MINIMUM_BID = 10

    def want_to_auction(self, dict_property_info):
        return True

    def pay_bank(self, amount):
        """ Pay amount to the bank. Money are subtracted from player's cash and added to bank's total cash.

        :param amount: (int) Amount of money to pay
        :return: None
        """

        if amount > self._cash:
            raise Exception('Player {} does not have enough money to pay the bank'.format(self._name))
        # pay bank
        self._bank['cash'] += amount
        self._cash -= amount

    def pay_tax(self, tax_amount):
        """ Pay tax. This is used when the player ends in a tax cell (income tax or super tax) or when the player
            picks a fee card (e.g., doctor_fees).

        :param tax_amount: (int) Amount of money to pay
        :return: None
        """

        if self.have_enough_money(tax_amount):
            self.pay_bank(tax_amount)
        else:
            if not self.is_bankrupt(tax_amount):
                amount_required = tax_amount - self._cash
                self.get_money_from_mortgages(amount_required)
                self.pay_bank(tax_amount)

    def pay_opponent(self, opponent_name, amount):
        """ Pay another player.

        :param opponent_name: (String) name of the other opponent
        :param amount: (int) Amount to pay
        :return:
        """
        opponent = self._dict_players[opponent_name]
        assert self._cash >= amount
        opponent._cash += amount
        self._cash -= amount

    def buy(self, dict_road_info, road_name):
        """ Buy road

        :param dict_road_info: (dictionary) Road information
        :param road_name: (String) Road name
        :return:
        """

        road_price = dict_road_info['price']
        # enough money?
        if self._cash < road_price:
            raise Exception('Player {} does not have enough money'.format(self._name))
        # pay bank
        self._bank['cash'] += road_price
        self._cash -= road_price
        # exchange ownership
        dict_road_info['belongs_to'] = self._name
        self._list_owned_roads.append(road_name)
        self._dict_owned_houses_hotels[road_name] = (0, 0)
        # mortgage value
        self._properties_total_mortgageable_amount += dict_road_info['mortgage_value']

        color = dict_road_info['color']
        count_roads_of_color = 0
        for road in self._list_owned_roads:
            if color == self._dict_roads[road]['color']:
                count_roads_of_color += 1
        if color == 'brown' and count_roads_of_color == 2:
            self._dict_owned_colors[color] = True
        elif color == 'light_blue' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'purple' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'orange' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'red' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'yellow' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'green' and count_roads_of_color == 3:
            self._dict_owned_colors[color] = True
        elif color == 'blue' and count_roads_of_color == 2:
            self._dict_owned_colors[color] = True

    def buy_property(self, dict_property_info):
        """ Buy property (station or utility)

        :param dict_property_info: (dictionary) Property information
        :return:
        """
        property_price = dict_property_info['price']
        property_name = dict_property_info['name']
        property_type = dict_property_info['type']

        # enough money?
        if self._cash < property_price:
            raise Exception('Player {} does not have enough money'.format(self._name))
        # pay bank
        self._bank['cash'] += property_price
        self._cash -= property_price
        # exchange ownership
        dict_property_info['belongs_to'] = self._name
        if property_type == 'station':
            self._list_owned_stations.append(property_name)
        elif property_type == 'utility':
            self._list_owned_utilities.append(property_name)
        else:
            raise Exception('Property type {} does not exist'.format(property_type))
        # mortgage value
        self._properties_total_mortgageable_amount += dict_property_info['mortgage_value']

    def pay_rent(self, dict_property_info, amount):
        """ Pay the rent to the owner of the property.

        :param dict_property_info: (dict) Property information
        :param amount: (int) Rent amount
        :return:
        """

        opponent_name = dict_property_info['belongs_to']
        self.pay_opponent(opponent_name, amount)

    def mortgage_or_bid(self, dict_road_info):
        """ Determine whether to mortgage (to buy) or bid"""
        # This function is incomplete. In reality, the player should decide whether to mortgage or bid to try buuying
        # the road at a lower (available) price.
        return 'mortgage'

    def mortgage(self, property_name, property_type):

        if property_type == 'station':
            if property_name not in self._list_owned_stations:
                raise Exception('Station {} not owned by player {}'.format(property_name, self._name))
            if self._dict_properties[property_name]['is_mortgaged']:
                raise Exception('Station {} is already mortgaged'.format(property_name))
            self._dict_properties[property_name]['is_mortgaged'] = True
            mortgage_value = self._dict_properties[property_name]['mortgage_value']
            self._list_mortgaged_stations.append(property_name)
        elif property_type == 'utility':
            if property_name not in self._list_owned_utilities:
                raise Exception('Utility {} not owned by player {}'.format(property_name, self._name))
            if self._dict_properties[property_name]['is_mortgaged']:
                raise Exception('Utility {} is already mortgaged'.format(property_name))
            self._dict_properties[property_name]['is_mortgaged'] = True
            mortgage_value = self._dict_properties[property_name]['mortgage_value']
            self._list_mortgaged_utilities.append(property_name)
        elif property_type == 'road':
            if property_name not in self._list_owned_roads:
                raise Exception('Road {} not owned by player {}'.format(property_name, self._name))
            if self._dict_roads[property_name]['is_mortgaged']:
                raise Exception('Road {} is already mortgaged'.format(property_name))
            houses, hotels = self._dict_owned_houses_hotels.get(property_name, (0, 0))
            if houses > 0 or hotels > 0:
                raise Exception('Road {} has houses/hotels and cannot be mortgaged'.format(property_name))
            self._dict_roads[property_name]['is_mortgaged'] = True
            mortgage_value = self._dict_roads[property_name]['mortgage_value']
            self._list_mortgaged_roads.append(property_name)
        else:
            raise Exception('Property type {} unknown'.format(property_type))

        self._properties_total_mortgageable_amount -= mortgage_value
        self._bank['cash'] -= mortgage_value
        self._cash += mortgage_value

    def unmortgage(self, property_name, property_type):
        """ Unmortgage property.

        :param property_name: (String) Property name
        :param property_type: (String) Property type (station or utility)
        :return: None
        """

        if property_type == 'road':
            if property_name not in self._list_mortgaged_roads:
                raise Exception('Road {} not mortgaged by player {}'.format(property_name, self._name))
            self._dict_roads[property_name]['is_mortgaged'] = False
            unmortgage_value = self._dict_roads[property_name]['unmortgage_value']
            mortgage_value = self._dict_roads[property_name]['mortgage_value']
            self._list_mortgaged_roads.remove(property_name)
        elif property_type == 'utility':
            if property_name not in self._list_mortgaged_utilities:
                raise Exception('Utility {} not mortgaged by player {}'.format(property_name, self._name))
            self._dict_properties[property_name]['is_mortgaged'] = False
            unmortgage_value = self._dict_properties[property_name]['unmortgage_value']
            mortgage_value = self._dict_properties[property_name]['mortgage_value']
            self._list_mortgaged_utilities.remove(property_name)
        elif property_type == 'station':
            if property_name not in self._list_mortgaged_stations:
                raise Exception('Station {} not mortgaged by player {}'.format(property_name, self._name))
            self._dict_properties[property_name]['is_mortgaged'] = False
            unmortgage_value = self._dict_properties[property_name]['unmortgage_value']
            mortgage_value = self._dict_properties[property_name]['mortgage_value']
            self._list_mortgaged_stations.remove(property_name)
        else:
            raise Exception('Property type {} unknown'.format(property_type))

        if self._cash < unmortgage_value:
            raise Exception('Player {} has insufficient cash ({}) to unmortgage {} (costs {})'.format(
                self._name, self._cash, property_name, unmortgage_value))

        self._properties_total_mortgageable_amount += mortgage_value
        self._cash -= unmortgage_value
        self._bank['cash'] += unmortgage_value

    def choose_mortgage_properties(self, amount):
        """ Return a list of properties to mortgage given a required amount. This function
        doesn't take into account the cash available to the player. Example: if player owns 100$
        and the required amount is 150, the returned properties have a mortgage value >= 150.

        :param amount: (int) Amount required
        :return: (list) list of tuples (property_type, property_name)
        """
        # Note: this function is in reality more complex. This is just a temporary logic
        # until a more "intelligent" logic is built.

        list_mortgageable_roads = [('road', road_name, self._dict_roads[road_name]['mortgage_value'])
                                   for road_name in self._list_owned_roads
                                   if self._dict_roads[road_name]['is_mortgaged'] is False
                                   and self._dict_owned_houses_hotels.get(road_name, (0, 0)) == (0, 0)]
        list_mortgageable_stations = [('station', station_name, self._dict_properties[station_name]['mortgage_value'])
                                      for station_name in self._list_owned_stations
                                      if self._dict_properties[station_name]['is_mortgaged'] is False]
        list_mortgageable_utilities = [('utility', utility_name, self._dict_properties[utility_name]['mortgage_value'])
                                       for utility_name in self._list_owned_utilities
                                       if self._dict_properties[utility_name]['is_mortgaged'] is False]

        list_mortgageable_properties = list_mortgageable_roads + list_mortgageable_stations + list_mortgageable_utilities

        if len(list_mortgageable_properties) == 0:
            raise Exception('player {} has no properties to mortgage'.format(self._name))

        idx_property = 1
        mortgage_value = 0
        list_properties = []
        while (mortgage_value < amount) and (idx_property <= len(list_mortgageable_properties)):
            tuple_tmp = (list_mortgageable_properties[idx_property - 1][0], list_mortgageable_properties[idx_property - 1][1])
            list_properties.append(tuple_tmp)
            mortgage_value += list_mortgageable_properties[idx_property - 1][2]
            idx_property += 1

        if mortgage_value < amount:
            raise Exception('player {} total mortgageable amount is insufficient'.format(self._name))

        return list_properties

    def choose_unmortgage_properties(self):
        """ Return a list of properties to unmortgage given the available cash.

        Example: if player owns 130 and owns 'old kent road' (unmortgage value 33) and 'kings cross station'
        (unmortgage value 110), the returned property can be either one or the other (not both).
        In this function the properties are scanned in the following order: roads, stations, utilities. For this reason
        in the example, 'old kent road' would be returned.
        Better (more complex) logic should be implemented.

        :return: (list) list of tuples (property_type, property_name)
        """

        total_unmortgage_value = 0
        list_unmortgage_properties = []
        for road_name in self._list_mortgaged_roads:
            unmortgage_value = self._dict_roads[road_name]['unmortgage_value']
            if unmortgage_value + total_unmortgage_value <= self._cash:
                list_unmortgage_properties.append(('road', road_name))
                total_unmortgage_value += unmortgage_value
        for station_name in self._list_mortgaged_stations:
            unmortgage_value = self._dict_properties[station_name]['unmortgage_value']
            if unmortgage_value + total_unmortgage_value <= self._cash:
                list_unmortgage_properties.append(('station', station_name))
                total_unmortgage_value += unmortgage_value
        for utility_name in self._list_mortgaged_utilities:
            unmortgage_value = self._dict_properties[utility_name]['unmortgage_value']
            if unmortgage_value + total_unmortgage_value <= self._cash:
                list_unmortgage_properties.append(('utility', utility_name))
                total_unmortgage_value += unmortgage_value

        return list_unmortgage_properties

    def sell_house(self, road):
        """Sell one house at half price. House token is returned to the bank."""
        houses, hotels = self._dict_owned_houses_hotels.get(road, (0, 0))
        if houses == 0:
            raise Exception('No houses on road {} to sell'.format(road))
        house_value = self._dict_roads[road]['houses_cost'] // 2
        self._bank['cash'] -= house_value
        self._cash += house_value
        self._bank['houses'] += 1
        self._dict_owned_houses_hotels[road] = (houses - 1, hotels)

    def sell_hotel(self, road):
        """Sell a hotel at half price. Hotel token is returned to the bank."""
        houses, hotels = self._dict_owned_houses_hotels.get(road, (0, 0))
        if hotels == 0:
            raise Exception('No hotel on road {} to sell'.format(road))
        hotel_value = self._dict_roads[road]['hotels_cost'] // 2
        self._bank['cash'] -= hotel_value
        self._cash += hotel_value
        self._bank['hotels'] += 1
        self._dict_owned_houses_hotels[road] = (houses, hotels - 1)

    def get_money_from_mortgages(self, amount_required):
        """Raise at least amount_required by selling houses/hotels first, then mortgaging.

        Monopoly rule: houses must be sold before a road can be mortgaged.
        Houses and hotels are sold at half their purchase price.
        """
        if self._cash + self._properties_total_mortgageable_amount < amount_required:
            raise Exception('player {} has insufficient funds'.format(self._name))

        # Sell houses/hotels until enough has been raised or all are sold
        raised = 0
        for road_name in list(self._list_owned_roads):
            if raised >= amount_required:
                break
            if self._dict_roads[road_name]['is_mortgaged']:
                continue
            houses, hotels = self._dict_owned_houses_hotels.get(road_name, (0, 0))
            if hotels > 0:
                raised += self._dict_roads[road_name]['hotels_cost'] // 2
                self.sell_hotel(road_name)
            houses = self._dict_owned_houses_hotels.get(road_name, (0, 0))[0]
            while houses > 0 and raised < amount_required:
                raised += self._dict_roads[road_name]['houses_cost'] // 2
                self.sell_house(road_name)
                houses -= 1

        # Mortgage properties for any remaining shortfall
        if raised < amount_required:
            list_properties = self.choose_mortgage_properties(amount_required - raised)
            for property in list_properties:
                self.mortgage(property[1], property[0])

    def mortgage_and_pay_rent(self, dict_property_info):
        """ Mortgage the necessary properties to pay the rent for the given property.

        :param dict_property_info: (Dictionary) information of the property to pey rent
        :return:
        """

        property_rent = self.estimate_rent(dict_property_info)
        amount_required = property_rent - self._cash

        if self._cash >= property_rent:
            raise Exception('player {} has money and should not mortgage'.format(self._name))
        if self._properties_total_mortgageable_amount + self._cash < property_rent:
            raise Exception('player {} cannot rent (insufficient funds)'.format(self._name))

        self.get_money_from_mortgages(amount_required)
        self.pay_rent(dict_property_info, property_rent)

    def mortgage_and_buy(self, dict_property_info, property_name, property_type):
        """ Mortgage the necessary properties to buy the given property

            :param dict_property_info: (Dictionary) information of the property to buy
            :param property_name: (String) Property name
            :param property_type: (String) Property type (road or station or utility)
            :return:
        """

        property_price = dict_property_info['price']
        amount_required = property_price - self._cash
        self.get_money_from_mortgages(amount_required)

        # self.get_money_from_mortgages(dict_property_info)
        if property_type == 'road':
            self.buy(dict_property_info, property_name)
        elif property_type == 'station' or property_type == 'utility':
            self.buy_property(dict_property_info)
        else:
            raise Exception('Property type {} does not exist'.format(property_type))

    def make_offer(self, opponent):
        """ Make an offer to the owner of the road"""
        # TODO placeholder. To implement.
        return None

    def has_all_roads_of_color(self, color):
        """ Returns True if player owns all the roads of a given color. Return False otherwise. It's use to calculate
            the rent amount that the other player needs to pay when ends in a road cell.
            Example: if player owns 'old kent road' and 'whitechapel road', has_all_road_of_color('brown') -> True

        :param color: (String) Road's color
        :return: (Boolean) True if player owns all the roads of a given color.
        """
        return self._dict_owned_colors[color]

    def get_houses_hotel_count(self, road_name):
        """ Given a road name, returns the number of houses or hotel owned.
            Example: if player owns 2 houses in 'old kent road' returns (2, 0).
            Example: if player owns 1 hotel in 'old kent road' returns (0, 1).

        :param road_name: (String) Name of the road
        :return: (tuple) number of houses, number of hotels
        """
        return self._dict_owned_houses_hotels[road_name]

    def get_owned_stations_count(self):
        """ Return the number of stations owned.

        :return: (int) Number of stations
        """

        return len(self._list_owned_stations)

    def get_owned_utilities_count(self):
        """ Return the number of utilities owned.

        :return: (int) Number of utilities
        """

        return len(self._list_owned_utilities)

    def get_owned_colors(self):
        """ Return a list of colors owned by the player. Note: a color is owned when the player has all the road
            of that specific color.

        :return: (list) List of strings. Example: ['brown', 'blue']
        """

        list_colors = [color for color in self.color_to_house_mapping.keys() if self.has_all_roads_of_color(color)]
        return list_colors

    def estimate_rent_road(self, dict_property_info):
        """ Given a road, estimate how much rent needs to be paid based on the other player's owned properties.
            For example: if player 2 owns all the roads of a color, return rent 'rent_with_color_set'.

        :param dict_property_info:  (dict) Road information
        :return: (int) Rent amount
        """

        opponent_name = dict_property_info['belongs_to']
        if opponent_name is None:
            raise Exception('{} does not belong to anyone'.format(dict_property_info['name']))
        opponent = self._dict_players[opponent_name]
        property_name = dict_property_info['name']

        road_color = dict_property_info['color']
        if opponent.has_all_roads_of_color(road_color):
            houses, hotel = opponent.get_houses_hotel_count(property_name)
            if hotel == 0 and houses == 0:
                rent = dict_property_info['rent_with_color_set']
            else:
                rent = dict_property_info['rent_with_{}_houses_{}_hotels'.format(houses, hotel)]
        else:
            rent = dict_property_info['rent']

        return rent

    def estimate_rent_station(self, dict_property_info):
        """ Given a station, estimate how much rent needs to be paid based on the other player's owned properties.
            Example: if player owns two stations, return rent = 50.

        :param dict_property_info:  (dict) Station information
        :return: (int) Rent amount
        """

        opponent_name, property_name = dict_property_info['belongs_to'], dict_property_info['name']
        if opponent_name is None:
            raise Exception('{} does not belong to anyone'.format(dict_property_info['name']))
        opponent = self._dict_players[opponent_name]
        num_of_stations = opponent.get_owned_stations_count()

        if dict_property_info['type'] != 'station':
            raise Exception('Property type must be of type "station"')

        if num_of_stations == 1:
            return 25
        elif num_of_stations == 2:
            return 50
        elif num_of_stations == 3:
            return 100
        elif num_of_stations == 4:
            return 200
        else:
            raise Exception("The maximum number of stations is 4.")

    def estimate_rent_utility(self, dict_property_info):
        """ Given a utility, estimate how much rent needs to be paid based on the other player's owned properties.
            Example: if player owns the Electric company, return rent = 4 * dice_value.
            Example: if player owns the Electric company and Water work, return rent = 10 * dice_value.

        :param dict_property_info:  (dict) Utility information
        :return: (int) Rent amount
        """

        opponent_name, property_name = dict_property_info['belongs_to'], dict_property_info['name']
        if opponent_name is None:
            raise Exception('{} does not belong to anyone'.format(dict_property_info['name']))
        opponent = self._dict_players[opponent_name]
        num_of_utilities = opponent.get_owned_utilities_count()

        if dict_property_info['type'] != 'utility':
            raise Exception('Property type must be of type "utility"')

        if num_of_utilities == 1:
            return self._dice_value * 4
        elif num_of_utilities == 2:
            return self._dice_value * 10
        else:
            raise Exception("The maximum number of utilities is 2.")

    def estimate_rent(self, dict_property_info):
        """ Given a property or road, estimate how much rent needs to be paid based on the other player's owned
            properties.

        :param dict_property_info:  (dict) property information
        :return: (int) Rent amount
        """
        if dict_property_info['type'] == 'road':
            property_rent = self.estimate_rent_road(dict_property_info)
        elif dict_property_info['type'] == 'station':
            property_rent = self.estimate_rent_station(dict_property_info)
        elif dict_property_info['type'] == 'utility':
            property_rent = self.estimate_rent_utility(dict_property_info)
        else:
            raise Exception('Property type not recognized')

        return property_rent

    def is_bankrupt(self, value_to_pay, creditor=None):
        """ Check if player has enough money to pay, or if it needs to declare bankruptcy.

        :param value_to_pay:  (int) Amount of money to pay.
        :param creditor: (Player) Player owed the money, or None if owed to bank.
        :return: (bool) True if players is bankrupt, False if it's not.
        """
        if self._properties_total_mortgageable_amount + self._cash < value_to_pay:
            self.declare_bankruptcy(creditor)
        return self._has_lost

    def get_tax_value(self, tax_type):
        """ Return the amount of money to pay for a given tax type (income tax or super tax).

        :param tax_type: (str) income tax or super tax
        :return: (int) amount of money to pay
        """
        if tax_type == 'income tax':
            return 200
        elif tax_type == 'super tax':
            return 100
        else:
            raise Exception('Tax type {} not recognized'.format(tax_type))

    def owns_all_roads_of_a_color(self):
        """ Return True if the players owns at least all the roads of one color, False otherwhise.

        :return: (bool) True if players owns at least one color.
        """
        if self._dict_owned_colors['brown'] or self._dict_owned_colors['light_blue'] or \
                self._dict_owned_colors['purple'] or self._dict_owned_colors['orange'] or \
                self._dict_owned_colors['red'] or self._dict_owned_colors['yellow'] or \
                self._dict_owned_colors['green'] or self._dict_owned_colors['blue']:
                    return True
        else:
            return False

    def want_to_buy_house_hotel(self):
        """ Determine if the player wants to buy a house or a hotel. This function should be used only if the player
            owns at least all the roads of one color (e.g. player owns all the roads with the brown color).

            :return: (bool) True if player decides to buy a house or a hotel.
        """
        # Note: this function is in reality more complex. This is just a temporary logic
        # until a more "intelligent" logic is built. For now we let the player decide to buy a house/hotel if it
        # falls in a cell multiple of 5.
        return self._dice_value % 5 == 0

    def want_to_unmortgage(self):
        """ Determine if the player wants to unmortgage a road or a property.

        :return: (bool) True if the player decides to unmortgage roads or properties
        """
        return self._dice_value % 2 == 0

    def choose_house_hotel_to_buy(self):
        """ Select where to buy a house or a hotel. It returns the road name and whether the player should buy a house
            or a hotel. In case the player cannot buy any house or hotel (cause of lack of money or he has
            all houses/hotels already), then None is returned.
            The function checks the following:
                1) which color does the player own
                2) does the player have the maximum amount of houses/hotels for that color (4 houses + 1 hotel in each road)
            The function makes sure that a the road name with the minimum amount of houses is returned.

        :return: (tuple) (road_name, 'house/hotel') The first element indicates the road where to buy, while the second
                                                    one whether to buy a house or a hotel. If player can't buy either,
                                                    (None, None) is returned.
        """
        # Note: this function is in reality more complex. This is just a temporary logic
        # until a more "intelligent" logic is built. If the players owns more than one color, the function should
        # more intelligently choose the color where to buy a house or hotel.

        for color in ['blue', 'green', 'yellow', 'red', 'orange', 'purple', 'light_blue', 'brown']:
            if self._dict_owned_colors[color]:
                min_count_houses_hotels_in_road = 5

                for i, road in enumerate(self.color_to_house_mapping[color]):
                    houses, hotels = self._dict_owned_houses_hotels[road]
                    count_houses_hotels_in_road = houses + hotels
                    if count_houses_hotels_in_road < min_count_houses_hotels_in_road:
                        min_count_houses_hotels_in_road = count_houses_hotels_in_road
                        road_selected = road

                if min_count_houses_hotels_in_road == 5:
                    # players has all possible roads and hotels of that color
                    pass
                else:
                    if min_count_houses_hotels_in_road == 4:
                        return road_selected, 'hotel'
                    else:
                        return road_selected, 'house'
        return None, None

    def buy_house(self, road):
        """ Buy a house in the given road.

        :param road: (str) road in where to buy the house
        """
        if self._dict_owned_houses_hotels[road][0] == 4:
            raise Exception("Player {} already owns 4 houses in the road {}. "
                            "No more houses can be bought.".format(self._name, road))

        if self._bank['houses'] > 0:
            house_price = self._dict_roads[road]['houses_cost']
            if self._cash < house_price:
                raise Exception("Player {} has not enough money to buy a house in road {}".format(self._name, road))

            self.pay_bank(house_price)
            houses_owned = self._dict_owned_houses_hotels[road][0]
            self._dict_owned_houses_hotels[road] = (houses_owned + 1, 0)
            self._bank['houses'] -= 1
        else:
            raise Exception('The bank has no more houses to sell.')

    def buy_hotel(self, road):
        """ Buy a hotel in the given road.

        :param road: (str) road in where to buy the hotel
        """

        if self._dict_owned_houses_hotels[road][1] == 1:
            raise Exception("Player {} already owns 1 hotel in the road {}. "
                            "No more hotels can be bought.".format(self._name, road))

        if self._bank['hotels'] > 0:
            houses_owned, hotels_owned = self._dict_owned_houses_hotels[road]
            if houses_owned != 4:
                raise Exception("Player {} cannot buy a hotel in road {} if "
                                "4 houses are not owned first".format(self._name, road))
            hotel_price = self._dict_roads[road]['hotels_cost']
            if self._cash < hotel_price:
                raise Exception("Player {} has not enough money to buy a hotel in road {}".format(self._name, road))

            self.pay_bank(hotel_price)
            self._dict_owned_houses_hotels[road] = (houses_owned, 1)
            self._bank['hotels'] -= 1
            self._bank['houses'] += 4  # 4 houses are returned to the bank when buying a hotel
        else:
            raise Exception('The bank has no more hotels to sell.')

    def want_to_mortgage_to_buy_house(self):
        """ Determine if player wants to mortgage properties to gain money to buy a house, when the player doesn't
            have enough cash to buy.

        :return: (bool) if True the player mortgages properties to buy a house. False otherwise
        """
        # Note: For now we do not allow the player to mortgage properties to buy houses. This is just a placeholder
        # function for now. We will consider more complex logics in the future.
        return False

    def want_to_mortgage_to_buy_hotel(self):
        """ Determine if player wants to mortgage properties to gain money to buy a hotel, when the player doesn't
            have enough cash to buy.

        :return: (bool) if True the player mortgages properties to buy a hotel. False otherwise
        """
        # Note: For now we do not allow the player to mortgage properties to buy hotels. This is just a placeholder
        # function for now. We will consider more complex logics in the future.
        return False

    def go_to_jail(self):
        """ Move the player in the jail cell. Change player position to 10. Used when the player ends in the cell
            30 (go to jail) or when chances and opportunity cards say to do so."""
        self._position = 10

    def get_out_of_jail(self):
        """ Player leaves the jail. Jail count is set to zero and the position updated with the latest dices values"""
        self._jail_count = 0
        new_position = (self._position + self._dice_value) % len(self._list_board)
        if new_position < self._position:
            self._cash += 200
        self._position = new_position

    def pay_jail_or_wait(self):
        """ Determine whether the player wants to wait the next turn or pay to get out of the jail. This is used
            only when the jail_counter is lower than three and the dices didn't return equal values ("roll a double").
        :return: (str) 'wait' if the players decides to wait, 'pay otherwise'
        """

        if self._jail_count == 3:
            raise Exception('Player {} has been in jail for three consecutive rounds. Player has to pay and leave the'
                            'jail.')

        return 'wait'

    def community_chest_street_repair(self):
        """ Implement logic for the community chest card "street repair", where the player has to pay 40 for each house
            and 115 for each hotel. The player looses the game if there are no enough money available (bankrupt).

        :return:
        """
        count_houses = 0
        count_hotels = 0
        for color in self.get_owned_colors():
            for road in self.color_to_house_mapping[color]:
                n_houses, n_hotels = self.get_houses_hotel_count(road)
                count_houses += n_houses
                count_hotels += n_hotels

        # 40 per house and 115 per hotel
        total_required_amount = 40 * count_houses + 115 * count_hotels

        if self.have_enough_money(total_required_amount):
            self.pay_bank(total_required_amount)
        elif self.have_enough_money(total_required_amount, plus_mortgageable=True):
            residual_amount_required = total_required_amount - self.cash
            self.get_money_from_mortgages(residual_amount_required)
        else:
            self.is_bankrupt(total_required_amount)

    def play_community_chest(self, board_cell_name):
        """ Execute logic of the chosen community chest card.

        :param board_cell_name: (string) name of the community chest card
        :return:
        """

        if board_cell_name == 'street_repair':
            self.community_chest_street_repair()
        elif board_cell_name == 'stock_sale':
            self._cash += 50
        elif board_cell_name == 'holiday_fund':
            self._cash += 100
        elif board_cell_name == 'second_price':
            self._cash += 100
        elif board_cell_name == 'inherit':
            self._cash += 100
        elif board_cell_name == 'consultancy':
            self._cash += 25
        elif board_cell_name == 'income_tax':
            self._cash += 20
        elif board_cell_name == 'insurance':
            self._cash += 100
        elif board_cell_name == 'bank_error':
            self._cash += 200
        elif board_cell_name == 'hospital_fees' or board_cell_name == 'school_fees' or board_cell_name == 'doctor_fees':
            amount_to_pay = 100 if board_cell_name == 'hospital_fees' else 50
            self.pay_tax(amount_to_pay)
        elif board_cell_name == 'jail':
            self.go_to_jail()
        elif board_cell_name == 'to_go':
            self._position = 0
            self._cash += 200
        elif board_cell_name == 'out_of_jail':
            self._exit_jail = True
        elif board_cell_name == 'birthday':
            for player in self._dict_players.values():
                if not player.has_lost() and player._cash >= 10:
                    player.pay_opponent(self._name, 10)


    # ------------------------------------------------------------------ helpers

    def _advance_to(self, target_position):
        """Move to target_position, collecting 200 if passing Go."""
        if target_position <= self._position:
            self._cash += 200
        self._position = target_position

    def _get_nearest_station_position(self):
        for pos in [5, 15, 25, 35]:
            if pos > self._position:
                return pos
        return 5  # wrap around

    def _get_nearest_utility_position(self):
        for pos in [12, 28]:
            if pos > self._position:
                return pos
        return 12  # wrap around

    def _update_color_ownership(self, color):
        """Recalculate whether this player owns all roads of a color."""
        if all(self._dict_roads[r]['belongs_to'] == self._name
               for r in self.color_to_house_mapping[color]):
            self._dict_owned_colors[color] = True

    def _chance_general_repairs(self):
        count_houses = count_hotels = 0
        for color in self.get_owned_colors():
            for road in self.color_to_house_mapping[color]:
                h, ho = self.get_houses_hotel_count(road)
                count_houses += h
                count_hotels += ho
        total = 25 * count_houses + 100 * count_hotels
        if total > 0:
            self.pay_tax(total)

    # ------------------------------------------------------------------ chance

    def play_chance(self, card_name):
        """Execute logic of the chosen chance card."""
        station_names = {5: 'kings cross station', 15: 'marylebone station',
                         25: 'Fenchurch st. station', 35: 'liverpool st station'}

        if card_name == 'advance_to_go':
            self._cash += 200
            self._position = 0
        elif card_name in ('advance_to_trafalgar', 'advance_to_mayfair',
                           'advance_to_pall_mall', 'advance_to_kings_cross'):
            targets = {'advance_to_trafalgar': 24, 'advance_to_mayfair': 39,
                       'advance_to_pall_mall': 11, 'advance_to_kings_cross': 5}
            target = targets[card_name]
            self._advance_to(target)
            cell = self._list_board[target]
            if cell['type'] == 'road':
                road_name = cell['name']
                info = self._dict_roads[road_name]
                owner = info['belongs_to']
                if owner is None:
                    if self.want_to_auction(info):
                        self.run_auction(info, road_name, 'road')
                elif owner != self._name and not info['is_mortgaged']:
                    rent = self.estimate_rent_road(info)
                    if self.have_enough_money(rent):
                        self.pay_rent(info, rent)
            elif cell['type'] == 'station':
                station_name = cell['name']
                info = self._dict_properties[station_name]
                owner = info['belongs_to']
                if owner is None:
                    if self.want_to_auction(info):
                        self.run_auction(info, station_name, 'station')
                elif owner != self._name and not info['is_mortgaged']:
                    rent = self.estimate_rent_station(info)
                    if self.have_enough_money(rent):
                        self.pay_rent(info, rent)
        elif card_name == 'advance_to_station':
            target = self._get_nearest_station_position()
            self._advance_to(target)
            station_name = station_names[target]
            info = self._dict_properties[station_name]
            owner = info['belongs_to']
            if owner is None:
                if self.want_to_auction(info):
                    self.run_auction(info, station_name, 'station')
            elif owner != self._name and not info['is_mortgaged']:
                rent = self.estimate_rent_station(info) * 2  # double rent for chance card
                if self.have_enough_money(rent):
                    self.pay_rent(info, rent)
        elif card_name == 'advance_to_utility':
            utility_names = {12: 'Electric company', 28: 'water works'}
            target = self._get_nearest_utility_position()
            self._advance_to(target)
            utility_name = utility_names[target]
            info = self._dict_properties[utility_name]
            owner = info['belongs_to']
            if owner is None:
                if self.want_to_auction(info):
                    self.run_auction(info, utility_name, 'utility')
            elif owner != self._name and not info['is_mortgaged']:
                rent = self._dice_value * 10  # always 10x for chance card
                if self.have_enough_money(rent):
                    self.pay_rent(info, rent)
        elif card_name == 'bank_dividend':
            self._cash += 50
        elif card_name == 'out_of_jail':
            self._exit_jail = True
        elif card_name == 'go_back_3':
            self._position = (self._position - 3) % len(self._list_board)
        elif card_name == 'go_to_jail':
            self.go_to_jail()
        elif card_name == 'general_repairs':
            self._chance_general_repairs()
        elif card_name == 'pay_poor_tax':
            self.pay_tax(15)
        elif card_name == 'elected_chairman':
            for player in self._dict_players.values():
                if not player.has_lost() and self._cash >= 50:
                    self.pay_opponent(player._name, 50)
        elif card_name == 'building_loan':
            self._cash += 150

    # ------------------------------------------------------------------ auction

    def bid(self, dict_property_info, player_offer):
        """Return bid amount (up to 80% of cash, capped at property price).
        Returns 0 if the player cannot meet the minimum bid of AUCTION_MINIMUM_BID.
        """
        if self._cash < self.AUCTION_MINIMUM_BID:
            return 0
        return min(int(self._cash * 0.8), dict_property_info['price'])

    def _prepare_to_pay(self, amount):
        """Hook called before a winning auction purchase. Override to raise funds."""
        pass

    def run_auction(self, dict_property_info, property_name, property_type):
        """Auction an unowned property among all active players.
        The highest bidder wins. They pay second-highest bid + 1 (min AUCTION_MINIMUM_BID).
        If no one meets the minimum, the property stays unowned.
        """
        all_players = [self] + list(self._dict_players.values())
        active_players = [p for p in all_players if not p.has_lost()]

        bids = sorted(
            ((player.bid(dict_property_info, None), player) for player in active_players),
            key=lambda x: x[0], reverse=True
        )

        if not bids or bids[0][0] < self.AUCTION_MINIMUM_BID:
            return  # no valid bid — property stays unowned

        best_bid, best_bidder = bids[0]
        second_bid = bids[1][0] if len(bids) > 1 else 0
        winning_price = max(self.AUCTION_MINIMUM_BID, second_bid + 1)
        winning_price = min(winning_price, best_bid)  # never pay more than own bid

        if best_bidder is self:  # nur der Auktionsinitiator ist im eigenen Zug
            best_bidder._prepare_to_pay(winning_price)
        orig_price = dict_property_info['price']
        dict_property_info['price'] = winning_price
        if property_type == 'road':
            best_bidder.buy(dict_property_info, property_name)
        else:
            best_bidder.buy_property(dict_property_info)
        dict_property_info['price'] = orig_price

        balances = ', '.join(f"{p._name}: {p._cash}" for p in all_players)
        self._bank.setdefault('_game_log', []).append(
            f"{best_bidder._name} kauft '{property_name}' — gezahlt: {winning_price} (Normalpreis: {orig_price}) | Kontostand: {balances}"
        )

    # ------------------------------------------------------------------ bankruptcy

    def declare_bankruptcy(self, creditor=None):
        """Transfer all assets to creditor (or back to bank) and mark as lost."""
        self._has_lost = True

        # Return houses/hotels to bank
        for road_name, (houses, hotels) in self._dict_owned_houses_hotels.items():
            self._bank['houses'] += houses
            self._bank['hotels'] += hotels

        if creditor is not None:
            creditor._cash += self._cash
            for road_name in list(self._list_owned_roads):
                self._dict_roads[road_name]['belongs_to'] = creditor._name
                self._dict_roads[road_name]['is_mortgaged'] = False
                creditor._list_owned_roads.append(road_name)
                creditor._dict_owned_houses_hotels[road_name] = (0, 0)
                creditor._properties_total_mortgageable_amount += self._dict_roads[road_name]['mortgage_value']
                creditor._update_color_ownership(self._dict_roads[road_name]['color'])
            for station_name in list(self._list_owned_stations):
                self._dict_properties[station_name]['belongs_to'] = creditor._name
                self._dict_properties[station_name]['is_mortgaged'] = False
                creditor._list_owned_stations.append(station_name)
                creditor._properties_total_mortgageable_amount += self._dict_properties[station_name]['mortgage_value']
            for utility_name in list(self._list_owned_utilities):
                self._dict_properties[utility_name]['belongs_to'] = creditor._name
                self._dict_properties[utility_name]['is_mortgaged'] = False
                creditor._list_owned_utilities.append(utility_name)
                creditor._properties_total_mortgageable_amount += self._dict_properties[utility_name]['mortgage_value']
        else:
            self._bank['cash'] += self._cash
            for road_name in self._list_owned_roads:
                self._dict_roads[road_name]['belongs_to'] = None
                self._dict_roads[road_name]['is_mortgaged'] = False
            for station_name in self._list_owned_stations:
                self._dict_properties[station_name]['belongs_to'] = None
                self._dict_properties[station_name]['is_mortgaged'] = False
            for utility_name in self._list_owned_utilities:
                self._dict_properties[utility_name]['belongs_to'] = None
                self._dict_properties[utility_name]['is_mortgaged'] = False

        self._cash = 0
        self._list_owned_roads = []
        self._list_owned_stations = []
        self._list_owned_utilities = []
        self._list_mortgaged_roads = []
        self._list_mortgaged_stations = []
        self._list_mortgaged_utilities = []
        self._dict_owned_houses_hotels = {}
        self._properties_total_mortgageable_amount = 0
        self._dict_owned_colors = {color: False for color in self._dict_owned_colors}

    def _on_turn_start(self):
        """Hook am Anfang jedes Zuges. Subklassen können hier Hypotheken aufnehmen o.ä."""
        pass

    def play(self):
        self._on_turn_start()
        roll_dices = True
        num_of_rolls = 0

        while (roll_dices):
            roll_dices = False
            move_position = True
            tuple_dices = self.roll_dice()
            num_of_rolls = num_of_rolls + 1
            # On double roll play again
            if tuple_dices[0] == tuple_dices[1]:
                roll_dices = True
                if num_of_rolls == 3:
                    self.go_to_jail()
                    move_position = False

            if (move_position):
                self._dice_value = tuple_dices[0] + tuple_dices[1]
                if self._position != 10 or (self._position == 10 and self._free_visit):  # if player is not in jail
                    self._position = (self._position + self._dice_value) % len(self._list_board)
                    self._free_visit = True if self._position == 10 else False

                    # check if player passed Go. If yes, get 200 $
                    if self._position - self._dice_value < 0:
                        self._cash += 200

            board_cell = self._list_board[self._position]
            board_cell_type = board_cell['type']

            # Buy a house or hotel
            if self.owns_all_roads_of_a_color() and self.want_to_buy_house_hotel():

                road, house_or_hotel = self.choose_house_hotel_to_buy()
                if house_or_hotel == 'house' and self._bank['houses'] > 0 and self._dict_owned_houses_hotels[road][0] < 4:
                    house_price = self._dict_roads[road]['houses_cost']
                    if self.have_enough_money(house_price):
                        self.buy_house(road)
                    else:
                        if self.want_to_mortgage_to_buy_house():
                            if self._properties_total_mortgageable_amount + self._cash >= house_price:
                                self.get_money_from_mortgages(house_price)
                                self.buy_house(road)
                elif house_or_hotel == 'hotel' and self._bank['hotels'] > 0 and self._dict_owned_houses_hotels[road][1] == 0:
                    hotel_price = self._dict_roads[road]['hotels_cost']
                    if self.have_enough_money(hotel_price):
                        self.buy_hotel(road)
                    else:
                        if self.want_to_mortgage_to_buy_hotel():
                            if self._properties_total_mortgageable_amount + self._cash >= hotel_price:
                                self.get_money_from_mortgages(hotel_price)
                                self.buy_hotel(road)

            # Unmortgage property
            if self._properties_total_mortgageable_amount > 0 and self.want_to_unmortgage():
                list_unmortgage_properties = self.choose_unmortgage_properties()
                for property_type, property_name in list_unmortgage_properties:
                    self.unmortgage(property_name, property_type)

            if board_cell_type == 'jail' and not self._free_visit:
                # In jail we do not play again
                roll_dices = False
                # Double roll → free
                if tuple_dices[0] == tuple_dices[1]:
                    self.get_out_of_jail()

                # Use get-out-of-jail-free card
                elif self._exit_jail:
                    self._exit_jail = False
                    self.get_out_of_jail()

                # Player has been 3 rounds in jail → must pay
                elif self._jail_count == 3:
                    if self.have_enough_money(50):
                        self.pay_bank(50)
                        self.get_out_of_jail()
                    else:
                        if not self.is_bankrupt(50):
                            amount_to_mortgage = 50 - self._cash
                            self.get_money_from_mortgages(amount_to_mortgage)
                            self.pay_bank(50)
                            self.get_out_of_jail()

                # Player decides to pay or wait in jail
                else:
                    if self.pay_jail_or_wait() == 'wait':
                        self._jail_count += 1
                    else:
                        if self.have_enough_money(50):
                            self.pay_bank(50)
                            self.get_out_of_jail()
                        else:
                            if not self.is_bankrupt(50):
                                self.get_money_from_mortgages(50)
                                self.pay_bank(50)
                                self.get_out_of_jail()

            elif board_cell_type == 'road' or board_cell_type == 'station' or board_cell_type == 'utility':
                property_name = board_cell['name']
                dict_property_info = self._dict_roads[property_name] if board_cell_type == 'road' else self._dict_properties[property_name]
                property_owner = dict_property_info['belongs_to']
                if property_name in self._list_owned_roads or property_name in self._list_owned_stations or property_name in self._list_owned_utilities:
                    pass
                elif property_owner is None:
                    if self.want_to_auction(dict_property_info):
                        self.run_auction(dict_property_info, property_name, board_cell_type)
                    # else: skip — property stays unowned
                elif property_owner is not None and dict_property_info['is_mortgaged'] is False:
                    rent = self.estimate_rent(dict_property_info)
                    creditor = self._dict_players.get(property_owner)
                    if self.have_enough_money(rent):
                        self.pay_rent(dict_property_info, rent)
                    else:
                        if not self.is_bankrupt(rent, creditor):
                            self.mortgage_and_pay_rent(dict_property_info)

            elif board_cell_type == 'go' or board_cell_type == 'free parking':
                pass

            elif board_cell_type == 'tax':
                tax_amount = self.get_tax_value(board_cell['name'])
                self.pay_tax(tax_amount)

            elif board_cell_type == 'go to jail':
                self.go_to_jail()

            elif board_cell_type == 'community chest':
                card_name = self.community_cards_deck[0]
                self.community_cards_deck.append(self.community_cards_deck.pop(0))
                self.play_community_chest(card_name)

            elif board_cell_type == 'chance':
                if self.chance_cards_deck:
                    card_name = self.chance_cards_deck[0]
                    self.chance_cards_deck.append(self.chance_cards_deck.pop(0))
                    self.play_chance(card_name)

    @property
    def cash(self):
        return self._cash

    def has_lost(self):
        return self._has_lost
