"""Manager-based RL environment with terrain-specific TensorBoard metrics."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class TerrainLoggingManagerBasedRLEnv(ManagerBasedRLEnv):
    """Add per-terrain curriculum and termination metrics to ``extras["log"]``.

    Terrain assignments and termination results are captured before reset so a
    curriculum update cannot attribute a termination to the post-reset terrain.
    The environment's original reset and curriculum behavior remains unchanged.
    """

    _TERRAIN_TERMINATION_PREFIX = "Termination_"

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Run one environment step and append the current terrain metrics."""
        self._clear_previous_terrain_termination_metrics()
        obs, reward, terminated, time_out, extras = super().step(action)

        self._write_terrain_level_metrics()
        # Keep every active term/terrain tag present on steps without resets as
        # well. This avoids stale reset values and gives RSL-RL stable log keys.
        self._write_zero_terrain_termination_metrics()

        return obs, reward, terminated, time_out, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Capture pre-reset termination attribution, then run the native reset."""
        termination_snapshot = self._capture_terrain_terminations(env_ids)
        super()._reset_idx(env_ids)

        self._write_terrain_termination_metrics(termination_snapshot)
        self._write_zero_terrain_termination_metrics()

    def _capture_terrain_terminations(
        self, env_ids: Sequence[int]
    ) -> tuple[tuple[str, ...], tuple[str, ...], torch.Tensor, torch.Tensor] | None:
        """Snapshot terrain types and per-term done values before reset."""
        layout = self._get_terrain_layout()
        if layout is None:
            return None

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        term_names = tuple(self.termination_manager.active_terms)
        if env_ids_tensor.numel() == 0 or not term_names:
            return None

        terrain = self.scene.terrain
        terrain_type_indices = self._map_columns_to_terrain_types(
            terrain.terrain_types[env_ids_tensor], layout[1]
        ).clone()
        term_dones = torch.stack(
            [self.termination_manager.get_term(term_name)[env_ids_tensor] for term_name in term_names], dim=-1
        ).clone()

        return layout[0], term_names, terrain_type_indices, term_dones

    def _write_terrain_level_metrics(self):
        """Write mean level, max level, and environment count for each terrain."""
        layout = self._get_terrain_layout()
        if layout is None:
            return

        terrain = self.scene.terrain
        terrain_levels = terrain.terrain_levels.float()
        terrain_type_indices = self._map_columns_to_terrain_types(terrain.terrain_types, layout[1])
        log = self.extras.setdefault("log", {})

        for terrain_type_idx, terrain_type_name in enumerate(layout[0]):
            type_mask = terrain_type_indices == terrain_type_idx
            env_count = int(type_mask.sum().item())
            if env_count > 0:
                type_levels = terrain_levels[type_mask]
                mean_level = type_levels.mean().item()
                max_level = type_levels.max().item()
            else:
                mean_level = 0.0
                max_level = 0.0

            log[f"Terrain_Level_Mean/{terrain_type_name}"] = mean_level
            log[f"Terrain_Level_Max/{terrain_type_name}"] = max_level
            log[f"Terrain_Num_Envs/{terrain_type_name}"] = float(env_count)

    def _write_terrain_termination_metrics(
        self,
        snapshot: tuple[tuple[str, ...], tuple[str, ...], torch.Tensor, torch.Tensor] | None,
    ):
        """Write each termination's rate among reset environments of each terrain."""
        if snapshot is None:
            return

        terrain_type_names, term_names, terrain_type_indices, term_dones = snapshot
        log = self.extras.setdefault("log", {})

        for terrain_type_idx, terrain_type_name in enumerate(terrain_type_names):
            type_mask = terrain_type_indices == terrain_type_idx
            reset_count = int(type_mask.sum().item())

            for term_idx, term_name in enumerate(term_names):
                if reset_count > 0:
                    rate = term_dones[type_mask, term_idx].float().mean().item()
                else:
                    rate = 0.0
                log[f"{self._TERRAIN_TERMINATION_PREFIX}{term_name}/{terrain_type_name}"] = rate

    def _write_zero_terrain_termination_metrics(self):
        """Ensure stable TensorBoard keys without overwriting this step's rates."""
        layout = self._get_terrain_layout()
        if layout is None:
            return

        log = self.extras.setdefault("log", {})
        for term_name in self.termination_manager.active_terms:
            for terrain_type_name in layout[0]:
                log.setdefault(f"{self._TERRAIN_TERMINATION_PREFIX}{term_name}/{terrain_type_name}", 0.0)

    def _clear_previous_terrain_termination_metrics(self):
        """Remove rates from the preceding step before the base environment runs."""
        log = self.extras.get("log")
        if not isinstance(log, dict):
            return

        for key in tuple(log):
            if key.startswith(self._TERRAIN_TERMINATION_PREFIX):
                del log[key]

    def _get_terrain_layout(self) -> tuple[tuple[str, ...], torch.Tensor] | None:
        """Return terrain names and a column-to-name-index tensor.

        Isaac Lab assigns terrain types by column in curriculum mode, so the
        mapping mirrors ``TerrainGenerator._generate_curriculum_terrains``.
        A random multi-terrain generator does not retain the sampled type for
        each tile, and therefore cannot be grouped exactly from importer data.
        """
        terrain = getattr(self.scene, "terrain", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        terrain_cfg = getattr(terrain, "cfg", None)
        generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(generator_cfg, "sub_terrains", None)
        num_cols = getattr(generator_cfg, "num_cols", None)

        if terrain_types is None or terrain_levels is None or not sub_terrains or not num_cols:
            return None

        terrain_type_names = tuple(sub_terrains.keys())
        proportions = tuple(float(sub_cfg.proportion) for sub_cfg in sub_terrains.values())
        curriculum = bool(getattr(generator_cfg, "curriculum", False))
        cache_key = (id(generator_cfg), int(num_cols), curriculum, terrain_type_names, proportions)

        if getattr(self, "_terrain_logging_layout_cache_key", None) == cache_key:
            return self._terrain_logging_layout_cache

        if curriculum:
            total_proportion = sum(proportions)
            if total_proportion <= 0.0:
                return None

            cumulative_proportions = []
            cumulative = 0.0
            for proportion in proportions:
                cumulative += proportion / total_proportion
                cumulative_proportions.append(cumulative)

            column_type_indices = []
            for column_idx in range(int(num_cols)):
                column_position = column_idx / int(num_cols) + 0.001
                terrain_type_idx = next(
                    (
                        idx
                        for idx, cumulative_proportion in enumerate(cumulative_proportions)
                        if column_position < cumulative_proportion
                    ),
                    len(terrain_type_names) - 1,
                )
                column_type_indices.append(terrain_type_idx)
        elif len(terrain_type_names) == 1:
            column_type_indices = [0] * int(num_cols)
        else:
            # In random mode each tile is sampled independently. Isaac Lab's
            # TerrainImporter stores only row/column indices, not the sampled
            # sub-terrain name, so guessing here would misattribute metrics.
            return None

        column_type_indices_tensor = torch.tensor(column_type_indices, device=self.device, dtype=torch.long)
        layout = (terrain_type_names, column_type_indices_tensor)
        self._terrain_logging_layout_cache_key = cache_key
        self._terrain_logging_layout_cache = layout
        return layout

    @staticmethod
    def _map_columns_to_terrain_types(terrain_columns: torch.Tensor, column_type_indices: torch.Tensor) -> torch.Tensor:
        """Map importer column indices to configured sub-terrain indices."""
        mapped_types = torch.full_like(terrain_columns, -1, dtype=torch.long)
        valid_columns = (terrain_columns >= 0) & (terrain_columns < column_type_indices.numel())
        if valid_columns.any():
            mapped_types[valid_columns] = column_type_indices[terrain_columns[valid_columns]]
        return mapped_types
