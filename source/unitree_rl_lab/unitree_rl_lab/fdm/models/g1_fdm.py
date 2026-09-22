"""All-at-once multi-step height-map FDM adapted to the 29-DoF G1."""

from __future__ import annotations

import torch
from torch import nn

from ..config import FDMModelCfg
from ..utils.se2 import integrate_body_twists


class HeightMapEncoder(nn.Module):
    """Encode one 60x46 robot-yaw-aligned height map to a 512-D latent."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim != 512:
            raise ValueError("The reference 60x46 CNN has a fixed 512-D flattened output.")
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
        )

    def forward(self, height_map: torch.Tensor) -> torch.Tensor:
        if tuple(height_map.shape[-3:]) != (1, 60, 46):
            raise ValueError(f"Expected height maps [B, 1, 60, 46], got {tuple(height_map.shape)}.")
        latent = self.features(height_map)
        if latent.shape[-1] != 512:
            raise RuntimeError(f"Reference height encoder produced {latent.shape[-1]} features instead of 512.")
        return latent


class G1HeightFDM(nn.Module):
    """Predict all ten motion corrections and collision logits jointly.

    The command GRU emits one hidden state per future command, but the complete
    hidden sequence is flattened before both decoder heads. Consequently every
    prediction step can depend on the complete proposed command plan.
    """

    def __init__(self, cfg: FDMModelCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FDMModelCfg()
        self.cfg.validate()
        self.state_encoder = nn.GRU(
            input_size=self.cfg.state_dim + self.cfg.proprio_dim,
            hidden_size=self.cfg.state_hidden_dim,
            num_layers=self.cfg.state_gru_layers,
            batch_first=True,
            dropout=0.2 if self.cfg.state_gru_layers > 1 else 0.0,
        )
        self.height_encoder = HeightMapEncoder(self.cfg.height_latent_dim)
        self.command_encoder = nn.Sequential(
            nn.Linear(self.cfg.command_dim, self.cfg.command_latent_dim),
            nn.LeakyReLU(0.1),
        )
        context_dim = self.cfg.state_hidden_dim + self.cfg.height_latent_dim
        self.command_gru = nn.GRU(
            input_size=self.cfg.command_latent_dim + context_dim,
            hidden_size=self.cfg.command_hidden_dim,
            num_layers=self.cfg.command_gru_layers,
            batch_first=True,
            dropout=0.2 if self.cfg.command_gru_layers > 1 else 0.0,
        )
        flattened_dim = self.cfg.horizon * self.cfg.command_hidden_dim
        self.motion_decoder = nn.Sequential(
            nn.Linear(flattened_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.cfg.horizon * 3),
        )
        self.collision_decoder = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.cfg.horizon),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2 and "weight" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        # Start close to ideal command integration while the correction head learns.
        nn.init.zeros_(self.motion_decoder[-1].weight)
        nn.init.zeros_(self.motion_decoder[-1].bias)

    def forward(
        self,
        relative_state_history: torch.Tensor,
        proprio_history: torch.Tensor,
        height_map: torch.Tensor,
        future_commands: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected_history = (self.cfg.history_length, self.cfg.state_dim)
        if tuple(relative_state_history.shape[-2:]) != expected_history:
            raise ValueError(f"Expected state history [..., {expected_history}], got {relative_state_history.shape}.")
        if tuple(proprio_history.shape[-2:]) != (self.cfg.history_length, self.cfg.proprio_dim):
            raise ValueError("Unexpected proprioception history shape.")
        if tuple(future_commands.shape[-2:]) != (self.cfg.horizon, self.cfg.command_dim):
            raise ValueError("Unexpected command plan shape.")

        state_input = torch.cat((relative_state_history, proprio_history), dim=-1)
        _, state_hidden = self.state_encoder(state_input)
        state_latent = state_hidden[-1]
        height_latent = self.height_encoder(height_map)
        context = torch.cat((state_latent, height_latent), dim=-1)
        command_latent = self.command_encoder(future_commands)
        recurrent_input = torch.cat(
            (command_latent, context.unsqueeze(1).expand(-1, self.cfg.horizon, -1)), dim=-1
        )
        recurrent_output, _ = self.command_gru(recurrent_input)
        joint_latent = recurrent_output.flatten(start_dim=1)
        correction_twist = self.motion_decoder(joint_latent).view(-1, self.cfg.horizon, 3)
        corrected_twist = future_commands + correction_twist
        future_pose = integrate_body_twists(corrected_twist, self.cfg.command_timestep)
        collision_logits = self.collision_decoder(joint_latent)
        return {
            "future_pose": future_pose,
            "collision_logits": collision_logits,
            "correction_twist": correction_twist,
            "corrected_twist": corrected_twist,
        }

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self(
            batch["relative_state_history"],
            batch["proprio_history"],
            batch["height_map"],
            batch["future_commands"],
        )
