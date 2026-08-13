from hypothesis import given, settings
from hypothesis import strategies as st

from quoridor_rl.game import Player, Position


@given(st.lists(st.integers(min_value=0), max_size=25))
@settings(max_examples=30, deadline=None)
def test_legal_play_preserves_public_position_invariants(selectors: list[int]) -> None:
    position = Position.initial()

    for selector in selectors:
        actions = position.legal_actions()
        if not actions:
            break
        parent = position
        parent_facts = (
            parent.pawns,
            parent.walls_remaining,
            parent.placed_walls_by_player,
            parent.to_move,
            parent.winner,
        )

        position = parent.play(actions[selector % len(actions)])

        assert (
            parent.pawns,
            parent.walls_remaining,
            parent.placed_walls_by_player,
            parent.to_move,
            parent.winner,
        ) == parent_facts
        assert position.pawns[0] != position.pawns[1]
        assert all(0 <= count <= 10 for count in position.walls_remaining)
        for player in Player:
            assert (
                len(position.placed_walls_by_player[player])
                + position.walls_remaining[player]
                == 10
            )
        assert hash(position) == hash(position)
        if position.winner is None:
            assert position.to_move in Player
        else:
            assert position.to_move is None
