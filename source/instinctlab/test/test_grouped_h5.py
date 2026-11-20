import h5py
import importlib.util
import numpy as np
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "grouped_h5", Path(__file__).parents[1] / "instinctlab/motion_reference/motion_files/grouped_h5.py"
)
grouped_h5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouped_h5)


class TestGroupedH5(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "motion.h5"
        self.cache = root / "cache"
        self.joints = ("left", "right")
        self.bodies = ("pelvis",)
        self.calls = []
        with h5py.File(self.source, "w") as file:
            self.write_group(file, "any/second", 3, 1.0)
            self.write_group(file, "first", 2, 5.0)

    def write_group(self, file, name, frames, value):
        group = file.create_group(name)
        group["fps"] = 50.0
        group["joint_names"] = ["right", "left"]
        group["root_trans"] = np.full((frames, 3), value, dtype=np.float32)
        group["root_wxyz"] = np.tile([0.0, 0.0, 0.0, 2.0], (frames, 1))
        group["dof_pos"] = np.tile([value, -value], (frames, 1))

    def preprocess(self, positions, quaternions, joints, fps):
        self.calls.append(len(joints))
        shape = (len(joints), 1)
        return {
            "joint_pos": joints,
            "joint_vel": np.zeros_like(joints),
            "body_pos_w": positions[:, None],
            "body_quat_w": quaternions[:, None],
            "body_lin_vel_w": np.zeros_like(positions[:, None]),
            "body_ang_vel_w": np.zeros_like(positions[:, None]),
            "contact": np.ones(shape, dtype=bool),
            "valid": np.ones(shape, dtype=bool),
            "bottom": np.full(shape, np.nan, dtype=np.float32),
        }

    def open_cache(self, *, motion_names=None, identity="model"):
        return grouped_h5.open_grouped_motion_cache(
            self.source, self.cache, self.joints, self.bodies, 50, identity, self.preprocess, motion_names=motion_names
        )

    def test_cache_keeps_clip_boundaries_and_only_preprocesses_once(self):
        names, lengths = grouped_h5.read_grouped_motion_index(self.source, 50, self.joints)
        self.assertEqual((names, lengths), (("any/second", "first"), (3, 2)))
        with self.open_cache() as cache:
            self.assertEqual(tuple(cache["motion_names"].asstr()[:]), names)
            self.assertEqual(tuple(cache["starts"][:]), (0, 3))
            self.assertEqual(tuple(cache["lengths"][:]), lengths)
            np.testing.assert_array_equal(cache["joint_pos"][[0, 3]], [[-1, 1], [-5, 5]])
            np.testing.assert_array_equal(cache["body_quat_w"][0, 0], [0, 0, 0, 1])
            key = cache.attrs["cache_key"]
            data_key = cache.attrs["data_key"]
            path = cache.filename
        with patch.object(grouped_h5.hashlib, "file_digest", side_effect=AssertionError("Unexpected source hash")):
            with self.open_cache() as cache:
                self.assertEqual(cache.attrs["cache_key"], key)
                self.assertEqual(cache.attrs["data_key"], data_key)
        self.assertEqual(self.calls, [3, 2])

        with h5py.File(path, "r+") as cache:
            cache["starts"][1] = 2
        with self.open_cache() as cache:
            self.assertEqual(tuple(cache["starts"][:]), (0, 3))
        self.assertEqual(self.calls, [3, 2, 3, 2])

    def test_selection_precedes_payload_reads_and_partitions_cache(self):
        with h5py.File(self.source, "r+") as file:
            file["first/root_wxyz"][0] = 0
        names = ["any/second"]
        self.assertEqual(
            grouped_h5.read_grouped_motion_index(self.source, 50, self.joints, motion_names=names),
            (("any/second",), (3,)),
        )
        with self.open_cache(motion_names=names) as cache:
            self.assertEqual(tuple(cache["motion_names"].asstr()[:]), tuple(names))
            key = cache.attrs["cache_key"]
        with self.open_cache(motion_names=names, identity="different model") as cache:
            self.assertNotEqual(cache.attrs["cache_key"], key)
        for invalid in ([], ["second"], names * 2, ["/any/second"]):
            with self.assertRaises(ValueError), self.open_cache(motion_names=invalid):
                self.fail("Invalid motion selection opened a cache")
        with self.assertRaisesRegex(ValueError, "invalid root quaternion"), self.open_cache():
            self.fail("Invalid raw clip opened a cache")
        self.assertEqual(len(list(self.cache.glob("*.h5"))), 2)
        self.assertEqual(list(self.cache.glob("*.tmp")), [])

        with h5py.File(self.source, "r+") as file:
            file["any/second/dof_pos"][0] = [7, 8]
        with self.open_cache(motion_names=names) as cache:
            self.assertNotEqual(cache.attrs["cache_key"], key)
            np.testing.assert_array_equal(cache["joint_pos"][0], [8, 7])

    def test_data_key_matches_byte_copy_and_distinguishes_selection(self):
        with self.open_cache() as cache:
            key = cache.attrs["cache_key"]
            data_key = cache.attrs["data_key"]
        copy = self.source.with_name("copy.h5")
        shutil.copyfile(self.source, copy)
        self.source = copy
        with self.open_cache() as cache:
            self.assertNotEqual(cache.attrs["cache_key"], key)
            self.assertEqual(cache.attrs["data_key"], data_key)
        with self.open_cache(motion_names=["first"]) as cache:
            self.assertNotEqual(cache.attrs["data_key"], data_key)


if __name__ == "__main__":
    unittest.main()
