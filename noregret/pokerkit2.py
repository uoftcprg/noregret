"""Module for PokerKit."""
from collections.abc import Iterable, Mapping
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from itertools import accumulate, combinations, starmap
from math import comb
from operator import mul
from typing import Any

from ordered_set import OrderedSet
from pokerkit import Automation, BettingStructure, Card, Opening, Poker
from scipy.sparse import lil_array
from tqdm import tqdm

from noregret.games.black_box import BlackBoxGame
from noregret.games.extensive_form import TwoPlayerZeroSumExtensiveFormGame
from noregret.games.utilities import to_extensive_form_game
from noregret.sequence_form_polytopes import SequenceFormPolytope


@dataclass
class PokerKitGame(BlackBoxGame):
    """Class for PokerKit games."""
    AUTOMATIONS = (
        Automation.ANTE_POSTING,
        Automation.BET_COLLECTION,
        Automation.BLIND_OR_STRADDLE_POSTING,
        Automation.RUNOUT_COUNT_SELECTION,
        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
        Automation.HAND_KILLING,
        Automation.CHIPS_PUSHING,
        Automation.CHIPS_PULLING,
    )
    """Automations."""
    game_type: type[Poker]
    """Game type."""
    args: list[Any] = field(default_factory=list)
    """Arguments."""
    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments."""

    def __post_init__(self):
        if (
                not self.is_two_player
                or not self.is_fixed_limit
                or not self.is_holdem
                or self.is_split
        ):
            raise NotImplementedError

    @property
    def initial_state(self):
        state = self.game_type.create_state(
            self.AUTOMATIONS,
            *self.args,
            **self.kwargs,
        )

        while state.can_burn_card():
            state.burn_card('??')

        state.deck_cards.clear()
        state.deck_cards.extend(state.deck)

        return state

    @property
    def player_count(self):
        return self.initial_state.player_count

    @property
    def is_zero_sum(self):
        return True

    @property
    def is_fixed_limit(self):
        """Return whether the game is fixed-limit.

        :return: Whether the game is fixed-limit.
        """
        return (
            self.initial_state.betting_structure
            == BettingStructure.FIXED_LIMIT
        )

    @property
    def is_holdem(self):
        """Return whether the game is hold'em.

        :return: Whether the game is hold'em.
        """
        for i, street in enumerate(self.initial_state.streets):
            if (
                    (street.hole_dealing_statuses and i)
                    or any(street.hole_dealing_statuses)
                    or street.draw_status
                    or street.opening != Opening.POSITION
            ):
                return False

        return True

    @property
    def is_split(self):
        """Return whether the game is split.

        :return: Whether the game is split.
        """
        return len(self.initial_state.hand_types) > 1

    @property
    def root_node(self):
        return self.initial_state, ('',) * self.player_count, '', ''

    def _chance(self, state):
        if state.can_deal_hole():
            index = state.hole_dealee_index
            deal_count = len(state.hole_dealing_statuses[index])
            action_count = comb(len(state.deck_cards), deal_count)
        elif state.can_deal_board():
            index = None
            deal_count = state.board_dealing_count
            action_count = comb(len(state.deck_cards), deal_count)
        else:
            index = None
            deal_count = 0
            action_count = None

        return index, deal_count, action_count

    def actions(self, node):
        state, *_ = node
        _, count, _ = self._chance(state)
        actions = OrderedSet()

        if count:
            for cards in combinations(state.deck_cards, count):
                actions.add(''.join(map(repr, cards)))
        else:
            if state.can_fold():
                actions.add('f')

            if state.can_check_or_call():
                actions.add('c')

            if state.can_complete_bet_or_raise_to():
                actions.add('r')

        return actions

    def apply(self, node, action):
        state, hole_cards, board_cards, actions = node
        state = deepcopy(state)
        index, count, _ = self._chance(state)

        if count:
            if index is None:
                state.deal_board(action)

                board_cards += action
            else:
                state.deal_hole(action)

                hole_cards = (
                    hole_cards[:index]
                    + (hole_cards[index] + action,)
                    + hole_cards[index + 1:]
                )
        else:
            match action:
                case 'f':
                    state.fold()
                case 'c':
                    state.check_or_call()
                case 'r':
                    state.complete_bet_or_raise_to()
                case _:
                    raise NotImplementedError

            actions += action

        while state.can_burn_card():
            state.burn_card('??')

        return state, hole_cards, board_cards, actions

    def player(self, node):
        state, *_ = node

        return state.actor_index

    def utility(self, node, player):
        state, *_ = node

        if state.status:
            utility = 0
        else:
            starting_stack = state.starting_stacks[player]
            stack = state.stacks[player]
            utility = stack - starting_stack

        return utility

    def utilities(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        state, *_ = node

        if state.status:
            utilities = np.zeros(state.player_count, dtype)
        else:
            starting_stacks = np.array(state.starting_stacks, dtype)
            stacks = np.array(state.stacks, dtype)
            utilities = stacks - starting_stacks

        return utilities

    def information_set(self, node):
        state, hole_cards, board_cards, actions = node

        if state.actor_index is None:
            raise ValueError('chance node')

        tokens = hole_cards[state.actor_index], board_cards, actions

        return ':'.join(tokens)

    def chance_probability(self, node, action):
        state, *_ = node
        _, _, count = self._chance(state)

        return 1 / count

    def chance_probabilities(self, node):
        np = self.kernel.numpy
        dtype = self.kernel.data_type
        state, *_ = node
        _, _, count = self._chance(state)

        return np.full(count, 1 / count, dtype)

    def to_extensive_form(self, kernel=None, progress_bar=True):
        """Convert to extensive-form.

        :param kernel: The optional kernel.
        :param progress_bar: Whether to show a progress bar.
        :return: The converted game.
        """
        if kernel is None:
            kernel = self.kernel

        pbar_args = ()
        pbar_kwargs = {}

        if progress_bar is True:
            pbar_status = True
        elif isinstance(progress_bar, Mapping):
            pbar_status = True
            pbar_kwargs = progress_bar
        elif isinstance(progress_bar, Iterable):
            pbar_status = True
            pbar_args = progress_bar
        else:
            pbar_status = False

        def pbar(it, *args, **kwargs):
            if pbar_status:
                it = tqdm(it, *pbar_args, *args, **pbar_kwargs, **kwargs)

            return it

        scipy = kernel.scipy
        dtype = kernel.data_type
        chances = OrderedSet()

        @dataclass
        class Game(PokerKitGame):
            AUTOMATIONS = (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.RUNOUT_COUNT_SELECTION,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            )

            def actions(self, node):
                actions = super().actions(node)

                if actions and self.player(node) is None:
                    action = actions[0]
                    actions = (action,)

                    chances.add(action)

                return actions

            def apply(self, node, action):
                node = super().apply(node, action)
                state, *_ = node
                status = False

                while state.can_show_or_muck_hole_cards():
                    state.show_or_muck_hole_cards(status)

                    status = True

                return node

            def chance_probability(self, node, action):
                return 1

            def chance_probabilities(self, node):
                np = self.kernel.numpy
                dtype = self.kernel.data_type

                return np.ones(1, dtype)

        game = Game(self.kernel, self.game_type, self.args, self.kwargs)
        game = to_extensive_form_game(kernel, game)

        def deal(deck, index):
            if index == len(chances):
                yield ()
            else:
                count = len(tuple(Card.parse(chances[index])))
                kwargs = {'leave': False} if index else {}

                for cards in pbar(
                        combinations(deck, count),
                        total=comb(len(deck), count),
                        **kwargs,
                ):
                    raw_cards = ''.join(map(repr, cards))

                    for dealings in deal(deck - set(cards), index + 1):
                        yield raw_cards, *dealings

        state, *_ = self.root_node
        dealings = tuple(deal(OrderedSet(state.deck), 0))
        n = self.player_count
        decision_points = [defaultdict(OrderedSet) for _ in range(n)]
        actions = [defaultdict(OrderedSet) for _ in range(n)]
        parent_sequences = [{} for _ in range(n)]

        def replace(ps, rs, s):
            tokens = []

            while s:
                status = False

                for p, r in zip(ps, rs):
                    if s.startswith(p):
                        tokens.append(r)

                        s = s[len(p):]
                        status = True

                if not status:
                    tokens.append(s[0])

                    s = s[1:]

            return ''.join(tokens)

        for i, sfp in enumerate(pbar(game.sequence_form_polytopes)):
            for j, A_j in pbar(sfp.actions.items(), leave=False):
                for chances2 in pbar(dealings, leave=False):
                    j2 = replace(chances, chances2, j)

                    decision_points[i][j].add(j2)
                    actions[i][j2].update(A_j)

                    p_j = sfp.parent_sequences[j]

                    if p_j is None:
                        parent_sequences[i][j2] = None
                    else:
                        j3, a = p_j
                        j3 = replace(chances, chances2, j3)
                        parent_sequences[i][j2] = j3, a

        sfps = tuple(
            starmap(
                partial(SequenceFormPolytope, kernel),
                zip(actions, parent_sequences),
            ),
        )
        payoffs = lil_array(
            tuple(sfp.column_count for sfp in sfps),
            dtype=dtype,
        )
        row_decision_points, column_decision_points = decision_points
        row_sfp, column_sfp = game.sequence_form_polytopes
        row_sfp2, column_sfp2 = sfps
        (hand_type,) = state.hand_types
        ks = list(map(len, map(tuple, map(Card.parse, chances))))
        ps = []
        n = len(state.deck)

        for k in ks:
            ps.append(1 / comb(n, k))

            n -= k

        ps = dict(zip(accumulate(ks), accumulate(ps, mul)))
        rs, cs = game.payoffs.nonzero()

        for r, c in zip(pbar(rs), cs):
            u = game.payoffs[r, c]
            row_seq = row_sfp.sequence(r)
            column_seq = column_sfp.sequence(c)
            row_j, row_a = row_seq
            column_j, column_a = column_seq

            for row_j2 in pbar(row_decision_points[row_j], leave=False):
                for column_j2 in pbar(
                        column_decision_points[column_j],
                        leave=False,
                ):
                    row_seq2 = row_j2, row_a
                    column_seq2 = column_j2, column_a
                    r2 = row_sfp2.column(row_seq2)
                    c2 = column_sfp2.column(column_seq2)
                    row_hole_cards = OrderedSet(
                        Card.parse(row_j2.split(':')[0]),
                    )
                    board_cards = OrderedSet(Card.parse(row_j2.split(':')[1]))
                    column_hole_cards = OrderedSet(
                        Card.parse(column_j2.split(':')[0]),
                    )
                    board_cards2 = OrderedSet(
                        Card.parse(column_j2.split(':')[1]),
                    )

                    assert not row_hole_cards & board_cards
                    assert not column_hole_cards & board_cards2

                    if (
                            row_hole_cards & column_hole_cards
                            or board_cards != board_cards2
                    ):
                        continue

                    row_cards = row_hole_cards | board_cards
                    column_cards = column_hole_cards | board_cards
                    k = len(row_cards | column_cards)
                    p = ps[k]

                    assert not payoffs[r2, c2]

                    pu = p * u

                    if row_a != 'f' and column_a != 'f':
                        pu = abs(pu)
                        row_hand = hand_type(row_cards)
                        column_hand = hand_type(column_cards)

                        if row_hand > column_hand:
                            payoffs[r2, c2] = pu
                        elif row_hand < column_hand:
                            payoffs[r2, c2] = -pu
                        else:
                            payoffs[r2, c2] = 0
                    else:
                        payoffs[r2, c2] = pu

        payoffs = scipy.sparse.csr_array(payoffs)

        return TwoPlayerZeroSumExtensiveFormGame(kernel, payoffs, sfps)
