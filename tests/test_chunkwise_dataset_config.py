import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from imagewam.chunkwise import resolve_chunkwise_geometry


class ChunkwiseDatasetConfigTest(unittest.TestCase):
    def test_flux_tasks_use_chunkwise_only_overlays(self):
        root = Path(__file__).parents[1]
        expected = {
            "libero_flux2_klein_4b_base_imagewam.yaml": "libero_flux2_chunkwise",
            "libero_flux2_klein_9b_base_imagewam.yaml": "libero_flux2_chunkwise",
            "robotwin_flux2_klein_4b_base_imagewam.yaml": "robotwin_flux2_chunkwise",
            "robotwin_flux2_klein_9b_base_imagewam.yaml": "robotwin_flux2_chunkwise",
            "robotwin_flux2_klein_4b_base_clean_imagewam.yaml": "robotwin_flux2_clean_chunkwise",
            "robotwin_flux2_klein_9b_base_clean_imagewam.yaml": "robotwin_flux2_clean_chunkwise",
            "interndata_a1_ee_v3_flux2_klein_4b_base_imagewam.yaml": "interndata_a1_ee_v3_flux2_chunkwise",
        }
        for task_name, overlay in expected.items():
            with self.subTest(task=task_name):
                text = (root / "configs" / "task" / task_name).read_text()
                self.assertIn(f"override /data: {overlay}", text)

        non_flux = (root / "configs" / "task" / "libero_omnigen2_imagewam.yaml").read_text()
        self.assertNotIn("flux2_chunkwise", non_flux)

    def test_overlays_derive_geometry_from_model_source_of_truth(self):
        root = Path(__file__).parents[1]
        for name in (
            "libero_flux2_chunkwise.yaml",
            "robotwin_flux2_chunkwise.yaml",
            "interndata_a1_ee_v3_flux2_chunkwise.yaml",
        ):
            with self.subTest(overlay=name):
                text = (root / "configs" / "data" / name).read_text()
                self.assertIn("num_frames: null", text)
                self.assertIn("observation_chunk_count: ${model.chunkwise_causal.num_chunks}", text)
                self.assertIn("actions_per_chunk: ${model.chunkwise_causal.actions_per_chunk}", text)

    def test_flux_task_composition_resolves_k4_and_k1_geometry(self):
        config_dir = str((Path(__file__).parents[1] / "configs").resolve())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            default_cfg = compose(
                config_name="train",
                overrides=["task=libero_flux2_klein_4b_base_imagewam"],
            )
            k1_cfg = compose(
                config_name="train",
                overrides=[
                    "task=libero_flux2_klein_4b_base_imagewam",
                    "model.chunkwise_causal.num_chunks=1",
                ],
            )
            legacy_cfg = compose(
                config_name="train",
                overrides=["task=libero_omnigen2_imagewam"],
            )

        default_geometry = resolve_chunkwise_geometry(
            default_cfg.data.train.observation_chunk_count,
            default_cfg.data.train.actions_per_chunk,
            num_frames=default_cfg.data.train.num_frames,
        )
        k1_geometry = resolve_chunkwise_geometry(
            k1_cfg.data.train.observation_chunk_count,
            k1_cfg.data.train.actions_per_chunk,
            num_frames=k1_cfg.data.train.num_frames,
        )
        self.assertEqual(
            (default_geometry.num_frames, default_geometry.total_action_horizon, default_geometry.observation_indices),
            (65, 64, (0, 16, 32, 48, 64)),
        )
        self.assertEqual(
            (k1_geometry.num_frames, k1_geometry.total_action_horizon, k1_geometry.observation_indices),
            (17, 16, (0, 16)),
        )
        self.assertEqual(legacy_cfg.data.train.num_frames, 17)
        self.assertNotIn("observation_chunk_count", legacy_cfg.data.train)


if __name__ == "__main__":
    unittest.main()
