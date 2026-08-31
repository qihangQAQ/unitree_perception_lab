"""Terrain-logging environment with training-only imagined-foothold state."""

from __future__ import annotations

import torch

from .terrain_logging_env import TerrainLoggingManagerBasedRLEnv


class FootholdTerrainLoggingManagerBasedRLEnv(TerrainLoggingManagerBasedRLEnv):
    """Cache action-time foothold predictions for reward and touchdown labeling."""

    _FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(self, *args, **kwargs):
        self._foothold_tracking_initialized = False
        self._foothold_prediction_valid = False
        super().__init__(*args, **kwargs)

    def _initialize_foothold_tracking(self) -> None:
        if self._foothold_tracking_initialized:
            return

        robot = self.scene["robot"]
        body_ids, body_names = robot.find_bodies(self._FOOT_BODY_NAMES, preserve_order=True)
        if tuple(body_names) != self._FOOT_BODY_NAMES:
            raise RuntimeError(
                "Failed to resolve G1 feet in left/right order: "
                f"expected {self._FOOT_BODY_NAMES}, got {tuple(body_names)}."
            )

        contact_sensor = self.scene.sensors["contact_forces"]
        contact_body_ids, contact_body_names = contact_sensor.find_bodies(
            self._FOOT_BODY_NAMES,
            preserve_order=True,
        )
        if tuple(contact_body_names) != self._FOOT_BODY_NAMES:
            raise RuntimeError(
                "Failed to resolve contact-sensor feet in left/right order: "
                f"expected {self._FOOT_BODY_NAMES}, got {tuple(contact_body_names)}."
            )

        self._foothold_body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self._foothold_contact_body_ids = torch.tensor(
            contact_body_ids,
            device=self.device,
            dtype=torch.long,
        )
        self._imagined_foothold_mu = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._imagined_foothold_sigma = torch.full((self.num_envs, 2), 0.15, device=self.device)
        self._foothold_decision_base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._foothold_decision_base_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._foothold_decision_contact = torch.ones(
            self.num_envs,
            2,
            device=self.device,
            dtype=torch.bool,
        )
        self._foothold_decision_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self._foothold_tracking_initialized = True

    def set_imagined_footholds(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Cache predictions and the matching pre-action contact state."""

        self._initialize_foothold_tracking()
        if mu.shape != self._imagined_foothold_mu.shape:
            raise ValueError(
                f"Expected foothold mu shape {tuple(self._imagined_foothold_mu.shape)}, "
                f"got {tuple(mu.shape)}."
            )
        if sigma.shape != self._imagined_foothold_sigma.shape:
            raise ValueError(
                f"Expected foothold sigma shape {tuple(self._imagined_foothold_sigma.shape)}, "
                f"got {tuple(sigma.shape)}."
            )

        self._imagined_foothold_mu.copy_(mu.detach().to(self.device))
        self._imagined_foothold_sigma.copy_(sigma.detach().to(self.device))

        robot = self.scene["robot"]
        self._foothold_decision_base_pos_w.copy_(robot.data.root_pos_w)
        self._foothold_decision_base_quat_w.copy_(robot.data.root_quat_w)
        decision_contact, _ = self._foothold_contact_state()
        self._foothold_decision_contact.copy_(decision_contact)
        contact_sensor = self.scene.sensors["contact_forces"]
        self._foothold_decision_air_time.copy_(
            contact_sensor.data.current_air_time[:, self._foothold_contact_body_ids]
        )
        self._foothold_prediction_valid = True

    def _foothold_contact_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_sensor = self.scene.sensors["contact_forces"]
        force_history = contact_sensor.data.net_forces_w_history[
            :,
            :,
            self._foothold_contact_body_ids,
            :,
        ]
        contact = torch.amax(torch.linalg.vector_norm(force_history, dim=-1), dim=1) > 0.5
        current_forces = contact_sensor.data.net_forces_w[
            :,
            self._foothold_contact_body_ids,
            :,
        ]
        return contact, current_forces

    @staticmethod
    def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_w.unbind(dim=-1)
        return torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
