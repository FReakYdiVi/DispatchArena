"""Standalone coarse-zone simulator for Andheri, Mumbai.

This module is deliberately independent from the current DispatchArena episode
logic. It builds a deterministic graph snapshot with shortest-path ETAs that can
later serve as the geography layer for courier, restaurant, and customer
simulation.
"""

from __future__ import annotations

import heapq
from typing import Dict, Optional

from pydantic import Field

from dispatch_arena.models import DispatchArenaModel, Node
from dispatch_arena.server.andheri_graph import AndheriGraphData, AndheriPreset, AndheriZoneMetadata, build_andheri_zone_graph


class AndheriSimulatorSnapshot(DispatchArenaModel):
    """Serializable snapshot of the Andheri graph and its ETA expansion."""

    area_name: str = "andheri_mumbai_coarse"
    preset: AndheriPreset
    seed: int | None = None
    nodes: list[Node] = Field(default_factory=list)
    node_metadata: dict[str, AndheriZoneMetadata] = Field(default_factory=dict)
    adjacency_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    travel_time_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    shortest_paths: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    traffic_events: list[str] = Field(default_factory=list)

    def as_dispatch_primitives(self) -> dict[str, object]:
        """Return the exact geography primitives that DispatchArena already uses.

        This keeps future integration simple: restaurants, drivers, and
        customers can later be layered on top of the same nodes and
        travel-time matrix without rebuilding the graph logic.
        """

        return {
            "nodes": [node.model_copy(deep=True) for node in self.nodes],
            "travel_time_matrix": {
                src: dict(dst_map) for src, dst_map in self.travel_time_matrix.items()
            },
        }

    def summary_text(self) -> str:
        return (
            f"area={self.area_name}; preset={self.preset.value}; "
            f"nodes={len(self.nodes)}; traffic_events={len(self.traffic_events)}"
        )


class AndheriSimulator:
    """Resettable simulator for coarse Andheri geography and ETA lookups."""

    def __init__(self, default_preset: AndheriPreset | str = AndheriPreset.EASY) -> None:
        self.default_preset = default_preset if isinstance(default_preset, AndheriPreset) else AndheriPreset(default_preset)
        self._snapshot: Optional[AndheriSimulatorSnapshot] = None

    @property
    def snapshot(self) -> AndheriSimulatorSnapshot:
        if self._snapshot is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")
        return self._snapshot

    def reset(self, seed: int | None = None, preset: AndheriPreset | str | None = None) -> AndheriSimulatorSnapshot:
        resolved_preset = self.default_preset if preset is None else preset
        graph = build_andheri_zone_graph(resolved_preset, seed=seed)
        travel_time_matrix, shortest_paths = _compute_all_pairs_shortest_paths(graph)
        self._snapshot = AndheriSimulatorSnapshot(
            area_name=graph.area_name,
            preset=graph.preset,
            seed=seed,
            nodes=graph.nodes,
            node_metadata=graph.node_metadata,
            adjacency_matrix=graph.adjacency_matrix,
            travel_time_matrix=travel_time_matrix,
            shortest_paths=shortest_paths,
            traffic_events=graph.traffic_events,
        )
        return self._snapshot

    def estimate_eta(self, src: str, dst: str) -> int:
        return self.snapshot.travel_time_matrix[src][dst]

    def shortest_path(self, src: str, dst: str) -> list[str]:
        return list(self.snapshot.shortest_paths[src][dst])


def _compute_all_pairs_shortest_paths(
    graph: AndheriGraphData,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, list[str]]]]:
    travel_time_matrix: dict[str, dict[str, int]] = {}
    shortest_paths: dict[str, dict[str, list[str]]] = {}
    for source in graph.adjacency_matrix:
        distances, previous = _dijkstra(graph.adjacency_matrix, source)
        travel_time_matrix[source] = {}
        shortest_paths[source] = {}
        for target in graph.adjacency_matrix:
            travel_time_matrix[source][target] = distances[target]
            shortest_paths[source][target] = _reconstruct_path(previous, source, target)
    return travel_time_matrix, shortest_paths


def _dijkstra(
    adjacency: dict[str, dict[str, int]],
    source: str,
) -> tuple[dict[str, int], dict[str, str | None]]:
    distances = {node_id: float("inf") for node_id in adjacency}
    previous: dict[str, str | None] = {node_id: None for node_id in adjacency}
    distances[source] = 0
    frontier: list[tuple[int, str]] = [(0, source)]

    while frontier:
        current_distance, node_id = heapq.heappop(frontier)
        if current_distance > distances[node_id]:
            continue
        for neighbor, weight in adjacency[node_id].items():
            candidate = current_distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node_id
                heapq.heappush(frontier, (candidate, neighbor))

    if any(value == float("inf") for value in distances.values()):
        missing = [node_id for node_id, value in distances.items() if value == float("inf")]
        raise ValueError(f"Graph is not fully reachable from {source}: {missing}")

    return {node_id: int(value) for node_id, value in distances.items()}, previous


def _reconstruct_path(previous: dict[str, str | None], source: str, target: str) -> list[str]:
    if source == target:
        return [source]

    path = [target]
    cursor = target
    while cursor != source:
        cursor = previous[cursor]
        if cursor is None:
            raise ValueError(f"No path found from {source} to {target}")
        path.append(cursor)
    path.reverse()
    return path
