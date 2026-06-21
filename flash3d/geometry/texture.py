"""Texture mapping: UV mapping, atlas generation, mesh texturing from views."""

from __future__ import annotations

import math

import numpy as np


class UVMapper:
    """UV coordinate generation for 3D meshes.

    Supports spherical, cylindrical, planar, and box projection methods.
    """

    @staticmethod
    def spherical_projection(vertices: np.ndarray) -> np.ndarray:
        """Project vertices to UV using spherical mapping.

        Args:
            vertices: (V, 3) mesh vertex positions.

        Returns:
            (V, 2) UV coordinates in [0, 1].
        """
        centered = vertices - vertices.mean(axis=0)
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = centered / norms

        u = 0.5 + np.arctan2(normalized[:, 2], normalized[:, 0]) / (2 * math.pi)
        v = 0.5 + np.arcsin(np.clip(normalized[:, 1], -1, 1)) / math.pi

        return np.stack([u, v], axis=-1)

    @staticmethod
    def cylindrical_projection(
        vertices: np.ndarray, axis: int = 1,
    ) -> np.ndarray:
        """Project vertices using cylindrical mapping around a given axis.

        Args:
            vertices: (V, 3) vertex positions.
            axis: Which axis is the cylinder axis (0=x, 1=y, 2=z).

        Returns:
            (V, 2) UV coordinates.
        """
        centered = vertices - vertices.mean(axis=0)
        axes = [0, 1, 2]
        axes.remove(axis)
        a1, a2 = axes

        u = 0.5 + np.arctan2(centered[:, a2], centered[:, a1]) / (2 * math.pi)
        v_range = centered[:, axis]
        v_min, v_max = v_range.min(), v_range.max()
        v = (v_range - v_min) / max(v_max - v_min, 1e-8)

        return np.stack([u, v], axis=-1)

    @staticmethod
    def planar_projection(
        vertices: np.ndarray, plane: str = "xy",
    ) -> np.ndarray:
        """Project vertices onto a plane.

        Args:
            vertices: (V, 3) vertex positions.
            plane: Projection plane ('xy', 'xz', 'yz').

        Returns:
            (V, 2) UV coordinates.
        """
        axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        a1, a2 = axis_map.get(plane, (0, 1))

        coords = vertices[:, [a1, a2]]
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins
        ranges = np.maximum(ranges, 1e-8)
        uvs = (coords - mins) / ranges
        return uvs

    @staticmethod
    def box_projection(
        vertices: np.ndarray, normals: np.ndarray,
    ) -> np.ndarray:
        """Box/cube projection: choose projection plane per vertex based on normal.

        Args:
            vertices: (V, 3) positions.
            normals: (V, 3) vertex normals.

        Returns:
            (V, 2) UV coordinates.
        """
        abs_normals = np.abs(normals)
        dominant = abs_normals.argmax(axis=1)

        uvs = np.zeros((len(vertices), 2))
        for axis in range(3):
            mask = dominant == axis
            if not mask.any():
                continue
            axes = [0, 1, 2]
            axes.remove(axis)
            a1, a2 = axes
            coords = vertices[mask][:, [a1, a2]]
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            ranges = np.maximum(maxs - mins, 1e-8)
            uvs[mask] = (coords - mins) / ranges

        return uvs


class TextureAtlas:
    """Texture atlas generator that packs UV charts into a single texture.

    Performs simple bin-packing of face charts into a square texture atlas.
    """

    def __init__(self, atlas_size: int = 2048) -> None:
        self.atlas_size = atlas_size
        self.atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
        self._next_row = 0
        self._next_col = 0
        self._row_height = 0

    def pack_chart(
        self,
        chart_image: np.ndarray,
        padding: int = 2,
    ) -> tuple[int, int]:
        """Pack a UV chart into the atlas.

        Args:
            chart_image: (H, W, 3) texture patch.
            padding: Pixels of padding between charts.

        Returns:
            (offset_x, offset_y) in the atlas.
        """
        h, w = chart_image.shape[:2]

        if self._next_col + w + padding > self.atlas_size:
            self._next_row += self._row_height + padding
            self._next_col = 0
            self._row_height = 0

        if self._next_row + h > self.atlas_size:
            raise RuntimeError("Atlas is full, increase atlas_size")

        ox, oy = self._next_col, self._next_row
        self.atlas[oy:oy+h, ox:ox+w] = chart_image
        self._next_col += w + padding
        self._row_height = max(self._row_height, h)

        return ox, oy

    def get_atlas(self) -> np.ndarray:
        return self.atlas

    def save(self, path: str) -> None:
        from PIL import Image
        img = Image.fromarray(self.atlas)
        img.save(path)


class MeshTexturer:
    """Texture a mesh from multi-view images by projecting and blending.

    Projects each face into visible camera views, samples color from the
    best-scoring view, and writes the result into a texture atlas.
    """

    def __init__(self, atlas_size: int = 2048) -> None:
        self.atlas_size = atlas_size

    def texture_from_views(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        images: list[np.ndarray],
        intrinsics_list: list[np.ndarray],
        extrinsics_list: list[np.ndarray],
        vertex_normals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a texture atlas from multi-view images.

        Args:
            vertices: (V, 3) mesh vertices.
            faces: (F, 3) triangle faces.
            images: List of (H, W, 3) images as uint8.
            intrinsics_list: List of (3, 3) camera intrinsics.
            extrinsics_list: List of (4, 4) world-to-camera matrices.
            vertex_normals: (V, 3) optional vertex normals for visibility.

        Returns:
            atlas: (atlas_size, atlas_size, 3) texture atlas.
            uvs: (V, 2) UV coordinates.
        """
        if vertex_normals is not None:
            uvs = UVMapper.box_projection(vertices, vertex_normals)
        else:
            uvs = UVMapper.spherical_projection(vertices)

        atlas = np.zeros((self.atlas_size, self.atlas_size, 3), dtype=np.uint8)

        for face_idx in range(len(faces)):
            face = faces[face_idx]
            face_verts = vertices[face]
            face_center = face_verts.mean(axis=0)

            best_view = -1
            best_score = -1.0

            if vertex_normals is not None:
                face_normal = vertex_normals[face].mean(axis=0)
                face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)

            for view_idx, (K, E) in enumerate(zip(intrinsics_list, extrinsics_list)):
                R, t = E[:3, :3], E[:3, 3]
                cam_pos_world = -R.T @ t
                view_dir = cam_pos_world - face_center
                view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

                if vertex_normals is not None:
                    cos_angle = np.dot(face_normal, view_dir)
                    if cos_angle < 0.1:
                        continue
                    score = cos_angle
                else:
                    score = 1.0

                pts_cam = (R @ face_center.reshape(3, 1) + t.reshape(3, 1)).flatten()
                if pts_cam[2] <= 0:
                    continue

                px = K @ pts_cam
                px = px[:2] / px[2]
                H, W = images[view_idx].shape[:2]
                if 0 <= px[0] < W and 0 <= px[1] < H:
                    if score > best_score:
                        best_score = score
                        best_view = view_idx

            if best_view >= 0:
                self._sample_face_texture(
                    atlas, uvs, face, face_verts,
                    images[best_view],
                    intrinsics_list[best_view],
                    extrinsics_list[best_view],
                )

        return atlas, uvs

    def _sample_face_texture(
        self,
        atlas: np.ndarray,
        uvs: np.ndarray,
        face: np.ndarray,
        face_verts: np.ndarray,
        image: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
    ) -> None:
        """Sample texture for a single face from a view and write to atlas."""
        face_uvs = uvs[face]
        atlas_size = atlas.shape[0]

        u_min = int(face_uvs[:, 0].min() * atlas_size)
        u_max = int(face_uvs[:, 0].max() * atlas_size) + 1
        v_min = int(face_uvs[:, 1].min() * atlas_size)
        v_max = int(face_uvs[:, 1].max() * atlas_size) + 1

        u_min = max(0, min(u_min, atlas_size - 1))
        u_max = max(0, min(u_max, atlas_size))
        v_min = max(0, min(v_min, atlas_size - 1))
        v_max = max(0, min(v_max, atlas_size))

        R, t = E[:3, :3], E[:3, 3]
        H_img, W_img = image.shape[:2]

        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                uv = np.array([u / atlas_size, v / atlas_size])
                bary = self._compute_barycentric(uv, face_uvs)
                if bary is None or (bary < -0.01).any():
                    continue

                pt_3d = bary @ face_verts
                pt_cam = R @ pt_3d + t
                if pt_cam[2] <= 0:
                    continue

                px = K @ pt_cam
                px_2d = px[:2] / px[2]
                ix, iy = int(px_2d[0]), int(px_2d[1])

                if 0 <= ix < W_img and 0 <= iy < H_img:
                    atlas[v, u] = image[iy, ix]

    @staticmethod
    def _compute_barycentric(
        point: np.ndarray, triangle: np.ndarray,
    ) -> np.ndarray | None:
        """Compute barycentric coordinates of point in triangle (2D)."""
        v0 = triangle[1] - triangle[0]
        v1 = triangle[2] - triangle[0]
        v2 = point - triangle[0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return None

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return np.array([u, v, w])


def save_textured_mesh_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_path: str,
    output_path: str,
) -> None:
    """Save a textured mesh as OBJ with material (MTL) file.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) face indices.
        uvs: (V, 2) UV coordinates.
        texture_path: Path to the texture image file.
        output_path: Output OBJ file path.
    """
    mtl_path = output_path.replace(".obj", ".mtl")
    mtl_name = "textured_material"

    with open(mtl_path, "w") as f:
        f.write(f"newmtl {mtl_name}\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("d 1.0\n")
        f.write(f"map_Kd {texture_path}\n")

    with open(output_path, "w") as f:
        f.write(f"mtllib {mtl_path}\n")
        f.write(f"usemtl {mtl_name}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        for face in faces:
            f.write(
                f"f {face[0]+1}/{face[0]+1} "
                f"{face[1]+1}/{face[1]+1} "
                f"{face[2]+1}/{face[2]+1}\n"
            )
