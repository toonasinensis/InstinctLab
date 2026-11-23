"""CPU contracts for reversible single-motion and terrain selection."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "pgmt_selection", Path(__file__).parents[1] / "instinctlab/tasks/shadowing/pgmt/selection.py"
)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class TestSelection(unittest.TestCase):
    def test_motion_subset_and_restore(self):
        files = ["dance1_subject1.named.npz", "dance1_subject2.named.npz", "walk1_subject1.named.npz"]
        self.assertEqual(selection.select_motion_files(files, ["dance1_subject2"]), files[1:2])
        self.assertEqual(selection.select_motion_files(files, [files[2], files[0]]), [files[0], files[2]])
        self.assertEqual(selection.select_motion_files(files, None), files)
        for names in ([], ["missing"], ["dance1_subject2", files[1]]):
            with self.assertRaises(ValueError):
                selection.select_motion_files(files, names)

    def test_column_family_identity_survives_removal_and_restore(self):
        names = ["flat", "slope_up", "slope_down", "stairs_up", "stairs_down", "boxes", "rough"]
        for weights, expected in (
            ([0.2, 0.1, 0.1, 0.2, 0.2, 0, 0.2], [0, 0, 1, 1, 2, 2, 2, 2, 4, 4]),
            ([0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        ):
            cfg = {name: SimpleNamespace(proportion=weight) for name, weight in zip(names, weights)}
            columns = selection.terrain_column_names(cfg, 10)
            self.assertEqual([selection.TERRAIN_FAMILIES[name] for name in columns], expected)
            if weights[5] == 0:
                self.assertEqual(columns.count("stairs_up"), 2)
                self.assertEqual(columns.count("stairs_down"), 2)

    def test_legacy_stair_names_preserve_geometry_and_column_order(self):
        prefix = "instinctlab.tasks.shadowing.pgmt.terrain_cfg:"
        old_up = {"function": prefix + "level_stairs", "step_height_range": (0.02, 0.25)}
        old_down = {"function": prefix + "level_inverted_stairs", "step_height_range": (0.02, 0.20)}
        sub_terrains = {"flat": {}, "stairs_up": old_up, "stairs_down": old_down, "rough": {}}
        generator = {"sub_terrains": sub_terrains}
        motion = {"terrain_sampling_weights": {"stairs_up": 1, "stairs_down": 2}}
        saved = {"scene": {"terrain": {"terrain_generator": generator}}, "commands": {"motion": motion}}

        restored = selection.rename_legacy_stair_settings(saved)
        terrains = restored["scene"]["terrain"]["terrain_generator"]["sub_terrains"]
        self.assertEqual(list(terrains), ["flat", "stairs_down", "stairs_up", "rough"])
        self.assertIs(terrains["stairs_down"], old_up)
        self.assertIs(terrains["stairs_up"], old_down)
        self.assertEqual(restored["commands"]["motion"]["terrain_sampling_weights"], {"stairs_down": 1, "stairs_up": 2})
        self.assertIs(selection.rename_legacy_stair_settings(restored), restored)
        self.assertIs(sub_terrains["stairs_up"], old_up)

        previous = {"terrain_columns": list(sub_terrains)}
        current = {"terrain_columns": list(terrains), "stair_names_from_center": True}
        self.assertEqual(selection.rename_legacy_stair_columns(previous, current), current)
        self.assertIs(selection.rename_legacy_stair_columns(current, current), current)

    def test_invalid_terrain_proportions_are_not_silently_accepted(self):
        for weights in ((0, 0), (-1, 2), (float("nan"), 1), (0.99999, 0.00001)):
            cfg = {name: SimpleNamespace(proportion=w) for name, w in zip(("flat", "rough"), weights)}
            with self.assertRaises(ValueError):
                selection.terrain_column_names(cfg, 10)


if __name__ == "__main__":
    unittest.main()
