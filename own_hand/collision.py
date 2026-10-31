"""Custom-hand preshape approach sweep against the unsegmented scene.

Inspired by 1030 CustomHandCollisionDetector, but accepts the resolved hand
transform directly: depth is already consumed by anchoring and is never added
again. Uses per-link mesh distance (and interiors for watertight links), avoiding
random surface-sampling holes. This does not check arm links or finger closure.
"""
import numpy as np
import open3d as o3d


class HandCollisionDetector:
    def __init__(self, scene_points, margin=0.003, step=0.003, mesh_provider=None):
        points = np.asarray(scene_points, dtype=float).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            raise ValueError("Collision scene must contain finite points")
        if not np.isfinite([margin, step]).all() or margin < 0 or step <= 0:
            raise ValueError("Invalid collision margin/step")
        # Keep raw scene points: table removal and SOR can hide obstacles.
        self.points, self.margin, self.step = points, float(margin), float(step)
        self.mesh_provider = mesh_provider
        self._cache = {}

    def _links(self, grasp_type):
        key = int(grasp_type)
        if key not in self._cache:
            if self.mesh_provider is None:
                import own_hand_grasp as ohg
                meshes = ohg.link_meshes_for_motors(ohg.motor_positions(key))
            else:
                meshes = self.mesh_provider(key)
            links = []
            for mesh in meshes:
                if mesh.is_empty():
                    raise ValueError("Empty collision link mesh")
                scene = o3d.t.geometry.RaycastingScene(nthreads=1)
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
                links.append((scene, mesh.get_min_bound(), mesh.get_max_bound(),
                              mesh.is_watertight()))
            if not links:
                raise ValueError("Missing hand collision geometry")
            self._cache[key] = links
        return self._cache[key]

    def check(self, grasp_type, plan, threshold=0):
        if threshold < 0:
            raise ValueError("Negative collision threshold")
        T = np.asarray(plan["T_camera_hand"], dtype=float).reshape(4, 4)
        R, translation = T[:3, :3], T[:3, 3]
        if (not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or
                not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
            raise ValueError("Invalid hand transform")
        distance = float(plan["approach_distance"])
        direction = np.asarray(plan["pregrasp_direction_camera"], dtype=float).reshape(3)
        if (not np.isfinite(distance) or distance < 0 or
                not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1)):
            raise ValueError("Invalid approach sweep")
        n = max(1, int(np.ceil(distance / self.step)))
        # Half-step padding covers gaps between constant-orientation samples.
        padding = self.margin + distance / n / 2
        links = self._links(grasp_type)
        surface_only = sum(not link[3] for link in links)
        local_scene = (self.points - translation) @ R
        direction_local = direction @ R
        # Cull only points outside the entire swept hand AABB; retain every
        # measured point inside it (including the table and thin obstacles).
        low = np.min([link[1] for link in links], axis=0)
        high = np.max([link[2] for link in links], axis=0)
        sweep_low = np.minimum(low, low + direction_local * distance) - padding
        sweep_high = np.maximum(high, high + direction_local * distance) + padding
        local_scene = local_scene[np.all(local_scene >= sweep_low, axis=1) &
                                  np.all(local_scene <= sweep_high, axis=1)]
        hits = np.zeros(len(local_scene), dtype=bool)
        for offset in np.linspace(0, distance, n + 1):
            points = local_scene - direction_local * offset
            for scene, low, high, watertight in links:
                indices = np.flatnonzero(~hits & np.all(points >= low - padding, axis=1)
                                         & np.all(points <= high + padding, axis=1))
                if not len(indices):
                    continue
                query = o3d.core.Tensor(points[indices].astype(np.float32))
                values = (scene.compute_signed_distance(query, nthreads=1, nsamples=3)
                          if watertight else scene.compute_distance(query, nthreads=1))
                hits[indices[values.numpy() <= padding]] = True
        count = int(hits.sum())
        return {"passed": count <= threshold, "collision_points": count,
                "threshold": int(threshold), "margin_m": self.margin,
                "sweep_step_m": self.step, "surface_only_links": surface_only,
                "model": "own_urdf_preshape", "scope": "preshape_descent_only"}
