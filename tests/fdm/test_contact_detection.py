"""Physics-step collision labels and their one-policy-step termination latch."""

import pytest
import torch

from unitree_rl_lab.fdm.utils.contact import (
    advance_collision_delay,
    any_body_contact,
    clear_collision_delay,
    recent_body_contacts,
)


@pytest.mark.parametrize("history_index", range(4))
def test_any_physics_substep_in_current_policy_step_counts(history_index: int):
    history = torch.zeros(1, 6, 3, 3)
    history[0, history_index, 1, 0] = 1.1
    contacts = recent_body_contacts(history, physics_steps=4, threshold=1.0)
    assert contacts.tolist() == [[False, True, False]]
    assert any_body_contact(contacts, [1]).tolist() == [True]


@pytest.mark.parametrize("history_index", [4, 5])
def test_previous_policy_step_does_not_count(history_index: int):
    history = torch.zeros(1, 6, 3, 3)
    history[0, history_index, 1, 0] = 2.0
    contacts = recent_body_contacts(history, physics_steps=4, threshold=1.0)
    assert not any_body_contact(contacts, [1]).item()


def test_force_norm_threshold_and_navigation_body_groups():
    history = torch.zeros(2, 6, 5, 3)
    history[0, 3, 0, 0] = 1.1  # torso
    history[0, 1, 1, 0] = 1.0  # left wrist: exactly 1 N, below strict threshold
    history[0, 2, 3, 0] = 0.9  # right hand: each component <1 N, vector norm >1 N
    history[0, 2, 3, 1] = 0.9
    history[1, 0, 4, 0] = 2.0  # foot is excluded from navigation links
    history[1, 4, 2, 0] = 2.0  # old left-hand contact is outside this policy step

    contacts = recent_body_contacts(history, physics_steps=4, threshold=1.0)
    navigation_ids = torch.tensor([0, 1, 2, 3])
    groups = torch.stack(
        [any_body_contact(contacts, ids) for ids in ([0], [1, 2], [3])], dim=-1
    )
    assert any_body_contact(contacts, navigation_ids).tolist() == [True, False]
    assert groups.tolist() == [[True, False, True], [False, False, False]]


def test_short_history_fails_instead_of_missing_physics_substeps():
    with pytest.raises(ValueError, match="policy step needs 4"):
        recent_body_contacts(torch.zeros(1, 3, 2, 3), physics_steps=4, threshold=1.0)


def test_collision_terminates_one_step_later_and_reset_clears_only_selected_envs():
    latch = torch.zeros(2, dtype=torch.bool)
    assert advance_collision_delay(latch, torch.tensor([True, True])).tolist() == [False, False]
    clear_collision_delay(latch, torch.tensor([0]))
    assert latch.tolist() == [False, True]
    assert advance_collision_delay(latch, torch.tensor([False, False])).tolist() == [False, True]
    assert not latch.any()
