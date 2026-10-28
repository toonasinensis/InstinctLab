"""One terrain query convention for PGMT resets, targets, contacts and metrics."""

import torch

from isaaclab.utils.warp import raycast_mesh


class TerrainContext:
    def __init__(self, command):
        self.command, self.env = command, command.env
        n, device = self.env.num_envs, command.device
        self.family = torch.zeros(n, dtype=torch.long, device=device)
        self.level = self.family.clone()
        self.outside = torch.zeros(n, dtype=torch.bool, device=device)
        self.support_z = torch.zeros(n, device=device)
        self.off_platform_steps = torch.zeros(n, device=device)
        self.actual_tile = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)

    def query(self, xy):
        """World xy of ANY shape [...,2] -> height, upward normal, validity."""
        shape = xy.shape[:-1]
        if self.env.scene.terrain.cfg.terrain_type == "plane":
            h = xy.new_zeros(shape)
            normal = xy.new_zeros(*shape, 3)
            normal[..., 2] = 1
            return h, normal, torch.ones(shape, dtype=torch.bool, device=xy.device)
        if not hasattr(self, "_mesh"):
            import numpy as np

            from pxr import UsdGeom

            import isaaclab.sim as sim_utils
            from isaaclab.utils.warp import convert_to_warp_mesh

            prim = sim_utils.get_first_matching_child_prim(
                self.env.scene.terrain.cfg.prim_path, lambda p: p.GetTypeName() == "Mesh"
            )
            if prim is None:
                raise RuntimeError("PGMT requires a generated terrain mesh")
            geometry = UsdGeom.Mesh(prim)
            points = np.asarray(geometry.GetPointsAttr().Get())
            transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            points = points @ transform[:3, :3].T + transform[:3, 3]
            indices = np.asarray(geometry.GetFaceVertexIndicesAttr().Get())
            self._mesh = convert_to_warp_mesh(points, indices, device=self.command.device)
        mesh = self._mesh
        starts = xy.new_zeros(xy.numel() // 2, 3)
        starts[:, :2] = xy.reshape(-1, 2)
        starts[:, 2] = 100.0
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1
        hits, _, normals, _ = raycast_mesh(starts, directions, mesh, return_normal=True)
        valid = torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1)
        normals = normals * torch.where(normals[:, 2:3] < 0, -1, 1)
        return (
            hits[:, 2].reshape(shape),
            normals.reshape(*shape, 3),
            valid.reshape(shape),
        )

    def update(self, reset_ids=None):
        c, env = self.command, self.env
        xy = c.robot_anchor_pos_w[:, :2]
        terrain = env.scene.terrain
        if terrain.cfg.terrain_type == "generator":
            origins = terrain.terrain_origins
            size = xy.new_tensor(terrain.cfg.terrain_generator.size)
            first = origins[0, 0, :2] - size / 2
            ij = ((xy - first) / size).floor().long()
            self.actual_tile.copy_(ij)
            inside = ((ij >= 0) & (ij < ij.new_tensor(origins.shape[:2]))).all(-1)
            # The generator has ten fixed columns: flat 0/1, slope 2/3, stairs 4/5, boxes 6/7, rough 8/9.
            self.family.copy_(ij[:, 1].clamp(0, 9) // 2)
            self.level.copy_(ij[:, 0].clamp(0, 9))
            assigned = torch.stack((terrain.terrain_levels, terrain.terrain_types), -1)
            self.outside.copy_(~inside | (ij != assigned).any(-1))
            away = (xy - env.scene.env_origins[:, :2]).abs().amax(-1) > 1.1
            self.off_platform_steps += away
        feet = c.robot_body_pos_w[:, c.foot_ids]
        if reset_ids is not None:
            # PhysX articulation link caches may still describe the old episode
            # until the next forward pass. The placed reset pose is already known.
            feet = feet.clone()
            feet[reset_ids] = c.body_pos_w[reset_ids][:, c.foot_ids]
        height, normal, valid = self.query(feet[..., :2])
        root_h, _, root_valid = self.query(xy)
        if not (valid.all() and root_valid.all()):
            raise FloatingPointError("PGMT terrain ray missed: refuse invalid height targets")
        self.foot_height, self.foot_normal, self.foot_valid = height, normal, valid
        # Reference contact selects support; flight uses terrain under root (keeps reference flight clearance).
        labels = c.reference_contacts[:, c.foot_ids] & c.reference_contact_valid[:, c.foot_ids]
        target = torch.where(
            labels.any(-1),
            (height * labels).sum(-1) / labels.sum(-1).clamp_min(1),
            root_h,
        )
        blend = 1 - torch.exp(xy.new_tensor(-env.step_dt / c.cfg.support_filter_seconds))
        self.support_z.lerp_(target, blend)
        self.support_z[~self.initialized] = target[~self.initialized]
        self.initialized[:] = True
        if reset_ids is not None:
            self.support_z[reset_ids] = target[reset_ids]
            self.off_platform_steps[reset_ids] = 0

    def tolerance(self, error, maximum):
        from .terrain_math import tolerance_error

        return tolerance_error(
            error,
            self.level,
            self.family,
            maximum,
            (self.command.cfg.tolerance_alpha if self.command.cfg.enable_tolerance else 0.0),
        )
