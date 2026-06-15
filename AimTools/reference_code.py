from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy.spatial.distance import cdist as _scipy_cdist
except ImportError:  # pragma: no cover - optional acceleration
    _scipy_cdist = None

try:
    import igraph as _igraph
except ImportError:  # pragma: no cover - optional acceleration
    _igraph = None

try:
    from ortools.sat.python import cp_model as _cp_model
except ImportError:  # pragma: no cover - optional acceleration
    _cp_model = None


@dataclass
class RigidMatch:
    """One selected rigid template match."""

    template_index: int
    template_indices: tuple[int, ...]
    observed_indices: tuple[int, ...]
    rotation: np.ndarray
    translation: np.ndarray
    max_error: float
    rms_error: float
    score: float

    @property
    def matched_count(self) -> int:
        return len(self.observed_indices)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix R for p_target = R @ p_source + t."""
        return self.rotation

    @property
    def translation_vector(self) -> np.ndarray:
        """3D translation vector t for p_target = R @ p_source + t."""
        return self.translation

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform matrix [[R, t], [0, 0, 0, 1]]."""
        return make_transform_matrix(self.rotation, self.translation)

    @property
    def rotate_matrix(self) -> np.ndarray:
        """4x4 homogeneous rotation-only matrix with zero translation."""
        return make_rotation_matrix(self.rotation)

    def as_transform_dict(self) -> dict[str, object]:
        """Return a compact serializable view of this match and its matrices."""
        return {
            "template_index": self.template_index,
            "template_indices": self.template_indices,
            "observed_indices": self.observed_indices,
            "rotation_matrix": self.rotation_matrix,
            "translation_vector": self.translation_vector,
            "transform_matrix": self.transform_matrix,
            "rotate_matrix": self.rotate_matrix,
            "max_error": self.max_error,
            "rms_error": self.rms_error,
        }


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply p' = R p + t to row-vector points with shape (n, 3)."""
    return np.asarray(points, dtype=float) @ rotation.T + translation


def make_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform matrix from 3x3 R and 3D t."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if translation.shape != (3,):
        raise ValueError("translation must have shape (3,)")

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def make_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous rotation-only matrix from a 3x3 rotation."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    rotate = np.eye(4)
    rotate[:3, :3] = rotation
    return rotate


def matches_to_transform_dicts(matches: Sequence[RigidMatch]) -> list[dict[str, object]]:
    """Export selected matches with Transform and Rotate matrices."""
    return [match.as_transform_dict() for match in matches]


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return R, t minimizing ||R source_i + t - target_i|| in least squares.

    The returned rotation is forced to be a proper rotation, det(R) = +1.
    With fewer than 3 non-collinear points, the rotation is not uniquely
    determined, but the returned transform is still a least-squares solution.
    """
    source = _as_points(source, "source")
    target = _as_points(target, "target")
    if len(source) != len(target):
        raise ValueError("source and target must have the same number of points")
    if len(source) == 0:
        raise ValueError("at least one point is required")
    if len(source) == 1:
        rotation = np.eye(3)
        translation = target[0] - source[0]
        return rotation, translation

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    x = source - source_center
    y = target - target_center

    covariance = x.T @ y
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    translation = target_center - rotation @ source_center
    return rotation, translation


def match_rigid_point_sets(
    observed_points: Sequence[Sequence[float]],
    templates: Sequence[Sequence[Sequence[float]]],
    max_error: float,
    *,
    distance_tolerance: float | None = None,
    max_candidates_per_template: int | None = 100,
    one_instance_per_template: bool = True,
    return_all_candidates: bool = False,
    clique_backend: str = "auto",
    selection_backend: str = "auto",
    cp_sat_time_limit_seconds: float = 2.0,
) -> list[RigidMatch] | tuple[list[RigidMatch], list[RigidMatch]]:
    """
    Match several rigid 3D point-set templates inside one observed point set.

    This implements the precise small-point-count approach:

    1. For each template, search correspondence cliques with pairwise distance
       consistency:

           | ||q_i - q_j|| - ||p_a - p_b|| | <= distance_tolerance

       By default, distance_tolerance = 2 * max_error.

    2. For every complete correspondence, solve the rigid transform with
       Kabsch/SVD.

    3. Keep only candidates whose matched point distances are all <= max_error.

    4. Select a globally compatible set of candidates so that each observed
       point is used at most once. By default each template can be selected at
       most once.

    Optional acceleration:
    - Install scipy to speed up distance matrices.
    - Install python-igraph to use its C-backed clique enumeration for larger
      association graphs.
    - Install ortools to solve the final mutually exclusive candidate selection
      with CP-SAT.

    Assumptions:
    - Each template is expected to appear completely if it appears at all.
    - For a unique 3D pose, a template should contain at least 3 non-collinear
      points. Smaller or degenerate templates are accepted, but their rotation
      may be underdetermined.

    Parameters
    ----------
    observed_points:
        Array-like shape (n, 3). This may contain outliers/noise points.
    templates:
        List of template point arrays, each shape (m_k, 3).
    max_error:
        Hard maximum allowed Euclidean error for every matched point.
    distance_tolerance:
        Pairwise distance consistency threshold. If None, uses 2 * max_error.
    max_candidates_per_template:
        Keep only this many verified candidates per template, sorted by score.
        Use None to keep all candidates.
    one_instance_per_template:
        If True, at most one pose is selected for each template.
    return_all_candidates:
        If True, returns (selected_matches, verified_candidates).
    clique_backend:
        "auto", "python", or "igraph". "auto" uses igraph when it is available
        and the association graph is large enough to benefit.
    selection_backend:
        "auto", "python", or "ortools". "auto" uses ortools when available.
    cp_sat_time_limit_seconds:
        Time limit for the optional OR-Tools CP-SAT global selection backend.

    Returns
    -------
    selected_matches:
        List of RigidMatch objects.
    """
    observed = _as_points(observed_points, "observed_points")
    template_arrays = [_as_points(t, f"templates[{i}]") for i, t in enumerate(templates)]

    if max_error < 0:
        raise ValueError("max_error must be non-negative")
    if distance_tolerance is None:
        distance_tolerance = 2.0 * max_error + 1e-12
    if distance_tolerance < 0:
        raise ValueError("distance_tolerance must be non-negative")
    if clique_backend not in {"auto", "python", "igraph"}:
        raise ValueError('clique_backend must be "auto", "python", or "igraph"')
    if selection_backend not in {"auto", "python", "ortools"}:
        raise ValueError('selection_backend must be "auto", "python", or "ortools"')

    all_candidates: list[RigidMatch] = []
    for template_index, template in enumerate(template_arrays):
        candidates = _find_template_candidates(
            observed=observed,
            template=template,
            template_index=template_index,
            max_error=max_error,
            distance_tolerance=distance_tolerance,
            backend=clique_backend,
            max_verified_candidates=max_candidates_per_template,
        )
        candidates.sort(key=lambda c: (-c.score, c.rms_error, c.max_error))
        if max_candidates_per_template is not None:
            candidates = candidates[:max_candidates_per_template]
        all_candidates.extend(candidates)

    selected = _select_non_overlapping_candidates(
        all_candidates,
        observed_count=len(observed),
        one_instance_per_template=one_instance_per_template,
        backend=selection_backend,
        cp_sat_time_limit_seconds=cp_sat_time_limit_seconds,
    )
    selected.sort(key=lambda c: c.template_index)

    if return_all_candidates:
        return selected, all_candidates
    return selected


def _find_template_candidates(
    *,
    observed: np.ndarray,
    template: np.ndarray,
    template_index: int,
    max_error: float,
    distance_tolerance: float,
    backend: str,
    max_verified_candidates: int | None,
) -> list[RigidMatch]:
    m = len(template)
    n = len(observed)
    if m == 0 or n == 0 or m > n:
        return []

    if m == 1:
        return [
            _make_candidate(
                observed=observed,
                template=template,
                template_index=template_index,
                template_indices=(0,),
                observed_indices=(j,),
                max_error=max_error,
            )
            for j in range(n)
        ]

    if backend == "igraph" and _igraph is None:
        raise RuntimeError('clique_backend="igraph" requires python-igraph')
    if _should_use_igraph_backend(backend, m, n):
        return _find_template_candidates_igraph(
            observed=observed,
            template=template,
            template_index=template_index,
            max_error=max_error,
            distance_tolerance=distance_tolerance,
            max_verified_candidates=max_verified_candidates,
        )

    return _find_template_candidates_python(
        observed=observed,
        template=template,
        template_index=template_index,
        max_error=max_error,
        distance_tolerance=distance_tolerance,
    )


def _find_template_candidates_python(
    *,
    observed: np.ndarray,
    template: np.ndarray,
    template_index: int,
    max_error: float,
    distance_tolerance: float,
) -> list[RigidMatch]:
    m = len(template)
    n = len(observed)
    template_order = _template_search_order(template)
    template_dist = _pairwise_distances(template)
    observed_dist = _pairwise_distances(observed)
    all_observed_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def compatible_observed_mask(template_a: int, template_b: int, observed_a: int) -> int:
        target_distance = template_dist[template_a, template_b]
        ok = np.abs(observed_dist[observed_a] - target_distance) <= distance_tolerance
        ok[observed_a] = False
        return _bool_array_to_mask(ok)

    found_by_observed_set: dict[int, RigidMatch] = {}
    assignment = [-1] * m

    def dfs(depth: int, used_observed_mask: int) -> None:
        if depth == m:
            observed_indices_by_template = [-1] * m
            for ordered_pos, observed_index in enumerate(assignment):
                observed_indices_by_template[template_order[ordered_pos]] = observed_index

            candidate = _make_candidate(
                observed=observed,
                template=template,
                template_index=template_index,
                template_indices=tuple(range(m)),
                observed_indices=tuple(observed_indices_by_template),
                max_error=max_error,
            )
            if candidate is None:
                return

            observed_set_mask = _indices_to_mask(candidate.observed_indices)
            previous = found_by_observed_set.get(observed_set_mask)
            if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
                found_by_observed_set[observed_set_mask] = candidate
            return

        template_next = template_order[depth]
        allowed = all_observed_mask & ~used_observed_mask
        for previous_depth in range(depth):
            template_prev = template_order[previous_depth]
            observed_prev = assignment[previous_depth]
            allowed &= compatible_observed_mask(template_prev, template_next, observed_prev)
            if allowed == 0:
                return

        for observed_next in _iter_mask_indices(allowed):
            assignment[depth] = observed_next
            dfs(depth + 1, used_observed_mask | (1 << observed_next))
            assignment[depth] = -1

    dfs(0, 0)
    return list(found_by_observed_set.values())


def _find_template_candidates_igraph(
    *,
    observed: np.ndarray,
    template: np.ndarray,
    template_index: int,
    max_error: float,
    distance_tolerance: float,
    max_verified_candidates: int | None,
) -> list[RigidMatch]:
    if _igraph is None:
        raise RuntimeError("python-igraph is not installed")

    m = len(template)
    n = len(observed)
    template_dist = _pairwise_distances(template)
    observed_dist = _pairwise_distances(observed)

    vertex_to_pair = [(template_i, observed_j) for template_i in range(m) for observed_j in range(n)]
    edges: list[tuple[int, int]] = []
    not_same_observed = ~np.eye(n, dtype=bool)

    for template_i in range(m - 1):
        for template_k in range(template_i + 1, m):
            ok = (np.abs(observed_dist - template_dist[template_i, template_k]) <= distance_tolerance) & not_same_observed
            observed_left, observed_right = np.nonzero(ok)
            if len(observed_left) == 0:
                continue
            left_vertices = template_i * n + observed_left
            right_vertices = template_k * n + observed_right
            edges.extend(zip(left_vertices.tolist(), right_vertices.tolist()))

    graph = _igraph.Graph(n=len(vertex_to_pair), edges=edges, directed=False)
    cliques = graph.cliques(min=m, max=m)

    found_by_observed_set: dict[int, RigidMatch] = {}
    for clique in cliques:
        observed_indices_by_template = [-1] * m
        valid = True
        for vertex in clique:
            template_i, observed_j = vertex_to_pair[int(vertex)]
            if observed_indices_by_template[template_i] != -1:
                valid = False
                break
            observed_indices_by_template[template_i] = observed_j
        if not valid or any(index == -1 for index in observed_indices_by_template):
            continue

        candidate = _make_candidate(
            observed=observed,
            template=template,
            template_index=template_index,
            template_indices=tuple(range(m)),
            observed_indices=tuple(observed_indices_by_template),
            max_error=max_error,
        )
        if candidate is None:
            continue

        observed_set_mask = _indices_to_mask(candidate.observed_indices)
        previous = found_by_observed_set.get(observed_set_mask)
        if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
            found_by_observed_set[observed_set_mask] = candidate

    _ = max_verified_candidates
    return list(found_by_observed_set.values())


def _make_candidate(
    *,
    observed: np.ndarray,
    template: np.ndarray,
    template_index: int,
    template_indices: tuple[int, ...],
    observed_indices: tuple[int, ...],
    max_error: float,
) -> RigidMatch | None:
    source = template[list(template_indices)]
    target = observed[list(observed_indices)]
    rotation, translation = estimate_rigid_transform(source, target)
    transformed = apply_transform(source, rotation, translation)
    errors = np.linalg.norm(transformed - target, axis=1)
    max_observed_error = float(errors.max(initial=0.0))
    if max_observed_error > max_error + 1e-9:
        return None

    rms_error = float(np.sqrt(np.mean(errors * errors))) if len(errors) else 0.0
    score = 1_000_000.0 * len(observed_indices) - 1000.0 * rms_error - max_observed_error
    return RigidMatch(
        template_index=template_index,
        template_indices=template_indices,
        observed_indices=observed_indices,
        rotation=rotation,
        translation=translation,
        max_error=max_observed_error,
        rms_error=rms_error,
        score=score,
    )


def _select_non_overlapping_candidates(
    candidates: Sequence[RigidMatch],
    *,
    observed_count: int,
    one_instance_per_template: bool,
    backend: str,
    cp_sat_time_limit_seconds: float,
) -> list[RigidMatch]:
    if not candidates:
        return []
    if backend == "ortools" and _cp_model is None:
        raise RuntimeError('selection_backend="ortools" requires ortools')
    if _should_use_ortools_backend(backend):
        return _select_non_overlapping_candidates_ortools(
            candidates,
            observed_count=observed_count,
            one_instance_per_template=one_instance_per_template,
            time_limit_seconds=cp_sat_time_limit_seconds,
        )

    return _select_non_overlapping_candidates_python(
        candidates,
        observed_count=observed_count,
        one_instance_per_template=one_instance_per_template,
    )


def _select_non_overlapping_candidates_python(
    candidates: Sequence[RigidMatch],
    *,
    observed_count: int,
    one_instance_per_template: bool,
) -> list[RigidMatch]:
    if not candidates:
        return []

    candidate_items = [
        (candidate, _indices_to_mask(candidate.observed_indices))
        for candidate in candidates
    ]

    if one_instance_per_template:
        grouped: dict[int, list[tuple[RigidMatch, int]]] = {}
        for item in candidate_items:
            grouped.setdefault(item[0].template_index, []).append(item)
        groups = list(grouped.values())
        for group in groups:
            group.sort(key=lambda item: (-item[0].score, item[0].rms_error, item[0].max_error))
        groups.sort(key=lambda group: (-group[0][0].score, len(group)))

        best_score = float("-inf")
        best_selection: list[RigidMatch] = []
        current: list[RigidMatch] = []
        suffix_upper = [0.0] * (len(groups) + 1)
        for i in range(len(groups) - 1, -1, -1):
            suffix_upper[i] = suffix_upper[i + 1] + max(0.0, groups[i][0][0].score)

        def dfs_group(group_index: int, used_mask: int, score: float) -> None:
            nonlocal best_score, best_selection
            if score + suffix_upper[group_index] < best_score - 1e-9:
                return
            if group_index == len(groups):
                if _selection_is_better(current, score, best_selection, best_score):
                    best_score = score
                    best_selection = list(current)
                return

            # Skip this template if none of its candidates is compatible.
            dfs_group(group_index + 1, used_mask, score)

            for candidate, observed_mask in groups[group_index]:
                if used_mask & observed_mask:
                    continue
                current.append(candidate)
                dfs_group(group_index + 1, used_mask | observed_mask, score + candidate.score)
                current.pop()

        dfs_group(0, 0, 0.0)
        return best_selection

    sorted_items = sorted(
        candidate_items,
        key=lambda item: (-item[0].score, item[0].rms_error, item[0].max_error),
    )
    suffix_upper = [0.0] * (len(sorted_items) + 1)
    for i in range(len(sorted_items) - 1, -1, -1):
        suffix_upper[i] = suffix_upper[i + 1] + max(0.0, sorted_items[i][0].score)

    best_score = float("-inf")
    best_selection: list[RigidMatch] = []
    current: list[RigidMatch] = []

    def dfs_item(index: int, used_mask: int, score: float) -> None:
        nonlocal best_score, best_selection
        if score + suffix_upper[index] < best_score - 1e-9:
            return
        if index == len(sorted_items):
            if _selection_is_better(current, score, best_selection, best_score):
                best_score = score
                best_selection = list(current)
            return

        candidate, observed_mask = sorted_items[index]
        if not (used_mask & observed_mask):
            current.append(candidate)
            dfs_item(index + 1, used_mask | observed_mask, score + candidate.score)
            current.pop()
        dfs_item(index + 1, used_mask, score)

    _ = observed_count
    dfs_item(0, 0, 0.0)
    return best_selection


def _select_non_overlapping_candidates_ortools(
    candidates: Sequence[RigidMatch],
    *,
    observed_count: int,
    one_instance_per_template: bool,
    time_limit_seconds: float,
) -> list[RigidMatch]:
    if _cp_model is None:
        raise RuntimeError("ortools is not installed")

    model = _cp_model.CpModel()
    variables = [model.NewBoolVar(f"c_{i}") for i in range(len(candidates))]

    by_observed: list[list[int]] = [[] for _ in range(observed_count)]
    by_template: dict[int, list[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        for observed_index in candidate.observed_indices:
            by_observed[observed_index].append(candidate_index)
        by_template.setdefault(candidate.template_index, []).append(candidate_index)

    for candidate_indices in by_observed:
        if candidate_indices:
            model.Add(sum(variables[i] for i in candidate_indices) <= 1)

    if one_instance_per_template:
        for candidate_indices in by_template.values():
            model.Add(sum(variables[i] for i in candidate_indices) <= 1)

    # CP-SAT requires integer coefficients. The score already has large point
    # count separation; this scaling keeps sub-millimeter tie-breaking useful.
    objective_terms = [
        int(round(candidate.score * 1000.0)) * variables[i]
        for i, candidate in enumerate(candidates)
    ]
    model.Maximize(sum(objective_terms))

    solver = _cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(0.01, float(time_limit_seconds))
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (_cp_model.OPTIMAL, _cp_model.FEASIBLE):
        return _select_non_overlapping_candidates_python(
            candidates,
            observed_count=observed_count,
            one_instance_per_template=one_instance_per_template,
        )

    return [
        candidate
        for variable, candidate in zip(variables, candidates)
        if solver.BooleanValue(variable)
    ]


def _selection_is_better(
    current: Sequence[RigidMatch],
    current_score: float,
    best: Sequence[RigidMatch],
    best_score: float,
) -> bool:
    if current_score > best_score + 1e-9:
        return True
    if abs(current_score - best_score) > 1e-9:
        return False

    current_errors = (sum(c.rms_error for c in current), sum(c.max_error for c in current), -len(current))
    best_errors = (sum(c.rms_error for c in best), sum(c.max_error for c in best), -len(best))
    return current_errors < best_errors


def _template_search_order(points: np.ndarray) -> list[int]:
    m = len(points)
    if m <= 2:
        return list(range(m))

    distances = _pairwise_distances(points)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    first = int(first)
    second = int(second)
    if first == second:
        return list(range(m))

    selected = [first, second]
    remaining = set(range(m)) - set(selected)

    base = points[second] - points[first]
    base_norm = np.linalg.norm(base)
    if base_norm > 0.0 and remaining:
        third = max(
            remaining,
            key=lambda idx: np.linalg.norm(np.cross(base, points[idx] - points[first])) / base_norm,
        )
        selected.append(int(third))
        remaining.remove(third)

    while remaining:
        next_index = max(
            remaining,
            key=lambda idx: min(distances[idx, selected_index] for selected_index in selected),
        )
        selected.append(int(next_index))
        remaining.remove(next_index)

    return selected


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    if _scipy_cdist is not None:
        return _scipy_cdist(points, points)
    delta = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(delta, axis=2)


def _should_use_igraph_backend(backend: str, template_count: int, observed_count: int) -> bool:
    if backend == "python":
        return False
    if backend == "igraph":
        return True
    if _igraph is None:
        return False
    # Building the full association graph costs O((m*n)^2). For very small
    # cases the bitset DFS is usually faster and allocates less memory.
    return template_count * observed_count >= 250


def _should_use_ortools_backend(backend: str) -> bool:
    if backend == "python":
        return False
    if backend == "ortools":
        return True
    return _cp_model is not None


def _as_points(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _bool_array_to_mask(values: np.ndarray) -> int:
    mask = 0
    for index in np.flatnonzero(values):
        mask |= 1 << int(index)
    return mask


def _indices_to_mask(indices: Iterable[int]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def _iter_mask_indices(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _candidate_sort_key(candidate: RigidMatch) -> tuple[float, float, float]:
    return (-candidate.score, candidate.rms_error, candidate.max_error)


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    template_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    template_b = rng.normal(size=(6, 3))

    def random_rotation() -> np.ndarray:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    r_a = random_rotation()
    r_b = random_rotation()
    transformed_a = apply_transform(template_a, r_a, np.array([4.0, -1.0, 2.0]))
    transformed_b = apply_transform(template_b, r_b, np.array([-2.0, 3.0, 1.0]))
    observed_demo = np.vstack(
        [
            transformed_a + rng.normal(scale=0.002, size=transformed_a.shape),
            transformed_b + rng.normal(scale=0.002, size=transformed_b.shape),
            rng.uniform(-6, 6, size=(15, 3)),
        ]
    )
    rng.shuffle(observed_demo)

    matches = match_rigid_point_sets(
        observed_demo,
        [template_a, template_b],
        max_error=0.02,
        clique_backend="auto",
        selection_backend="auto",
    )
    for match in matches:
        print(
            f"template={match.template_index}, points={match.observed_indices}, "
            f"max_error={match.max_error:.6f}, rms={match.rms_error:.6f}"
        )
        print("rotation_matrix:")
        print(match.rotation_matrix)
        print("translation_vector:")
        print(match.translation_vector)
        print("transform_matrix:")
        print(match.transform_matrix)
