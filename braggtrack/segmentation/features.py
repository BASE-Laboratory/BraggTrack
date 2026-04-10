"""Feature extraction for segmented 3D instances."""

from __future__ import annotations


def _jacobi_eigenvalues_symmetric_3x3(a: list[list[float]], iters: int = 25) -> tuple[float, float, float]:
    m = [[a[i][j] for j in range(3)] for i in range(3)]

    def max_offdiag() -> tuple[int, int]:
        pairs = [(0, 1), (0, 2), (1, 2)]
        p, q = max(pairs, key=lambda t: abs(m[t[0]][t[1]]))
        return p, q

    for _ in range(iters):
        p, q = max_offdiag()
        if abs(m[p][q]) < 1e-12:
            break
        theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
        t = (1.0 / (abs(theta) + (theta * theta + 1.0) ** 0.5))
        if theta < 0:
            t = -t
        c = 1.0 / (1.0 + t * t) ** 0.5
        s = t * c

        app = m[p][p]
        aqq = m[q][q]
        apq = m[p][q]

        m[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        m[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        m[p][q] = 0.0
        m[q][p] = 0.0

        for r in range(3):
            if r not in (p, q):
                mrp = m[r][p]
                mrq = m[r][q]
                m[r][p] = c * mrp - s * mrq
                m[p][r] = m[r][p]
                m[r][q] = s * mrp + c * mrq
                m[q][r] = m[r][q]

    vals = [m[0][0], m[1][1], m[2][2]]
    vals.sort(reverse=True)
    return vals[0], vals[1], vals[2]


def extract_instance_table(
    labels: list[list[list[int]]],
    intensity: list[list[list[float]]],
) -> list[dict[str, float | int]]:
    """Compute centroid, bbox, integrated intensity, covariance and eigenvalues."""

    stats: dict[int, dict[str, float | int | list[float] | list[int]]] = {}

    for z, plane in enumerate(labels):
        for y, row in enumerate(plane):
            for x, label in enumerate(row):
                if label <= 0:
                    continue
                val = float(intensity[z][y][x])
                st = stats.setdefault(
                    label,
                    {
                        "count": 0,
                        "sum_i": 0.0,
                        "sum_z": 0.0,
                        "sum_y": 0.0,
                        "sum_x": 0.0,
                        "min_z": z,
                        "max_z": z,
                        "min_y": y,
                        "max_y": y,
                        "min_x": x,
                        "max_x": x,
                        "coords": [],
                    },
                )
                st["count"] = int(st["count"]) + 1
                st["sum_i"] = float(st["sum_i"]) + val
                st["sum_z"] = float(st["sum_z"]) + z * val
                st["sum_y"] = float(st["sum_y"]) + y * val
                st["sum_x"] = float(st["sum_x"]) + x * val
                st["min_z"] = min(int(st["min_z"]), z)
                st["max_z"] = max(int(st["max_z"]), z)
                st["min_y"] = min(int(st["min_y"]), y)
                st["max_y"] = max(int(st["max_y"]), y)
                st["min_x"] = min(int(st["min_x"]), x)
                st["max_x"] = max(int(st["max_x"]), x)
                st["coords"].append((z, y, x, val))

    rows: list[dict[str, float | int]] = []
    for label in sorted(stats):
        st = stats[label]
        sum_i = float(st["sum_i"]) if float(st["sum_i"]) > 0 else 1e-12
        cz = float(st["sum_z"]) / sum_i
        cy = float(st["sum_y"]) / sum_i
        cx = float(st["sum_x"]) / sum_i

        czz = cyy = cxx = czy = czx = cyx = 0.0
        for z, y, x, w in st["coords"]:  # type: ignore[index]
            dz = z - cz
            dy = y - cy
            dx = x - cx
            czz += w * dz * dz
            cyy += w * dy * dy
            cxx += w * dx * dx
            czy += w * dz * dy
            czx += w * dz * dx
            cyx += w * dy * dx

        cov = [
            [czz / sum_i, czy / sum_i, czx / sum_i],
            [czy / sum_i, cyy / sum_i, cyx / sum_i],
            [czx / sum_i, cyx / sum_i, cxx / sum_i],
        ]
        l1, l2, l3 = _jacobi_eigenvalues_symmetric_3x3(cov)

        rows.append(
            {
                "label": label,
                "voxel_count": int(st["count"]),
                "integrated_intensity": float(st["sum_i"]),
                "centroid_mu": cx,
                "centroid_chi": cy,
                "centroid_d": cz,
                "bbox_min_z": int(st["min_z"]),
                "bbox_max_z": int(st["max_z"]),
                "bbox_min_y": int(st["min_y"]),
                "bbox_max_y": int(st["max_y"]),
                "bbox_min_x": int(st["min_x"]),
                "bbox_max_x": int(st["max_x"]),
                "cov_zz": cov[0][0],
                "cov_yy": cov[1][1],
                "cov_xx": cov[2][2],
                "cov_zy": cov[0][1],
                "cov_zx": cov[0][2],
                "cov_yx": cov[1][2],
                "eig_1": l1,
                "eig_2": l2,
                "eig_3": l3,
            }
        )

    return rows
