"""Coarse Andheri zone graph and traffic preset overlays.

This module intentionally avoids raw street-level simulation. It defines a
hand-curated graph for a compact Andheri area that can later feed the existing
DispatchArena simulator via ``nodes`` and ``travel_time_matrix`` primitives.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, Iterable, Tuple

from pydantic import Field

from dispatch_arena.models import DispatchArenaModel, Node


class AndheriPreset(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AndheriZoneMetadata(DispatchArenaModel):
    """Metadata for a coarse Andheri node."""

    node_id: str
    label: str
    lat: float
    lon: float
    x_km: float
    y_km: float
    tags: list[str] = Field(default_factory=list)

    def as_dispatch_node(self) -> Node:
        return Node(id=self.node_id, kind="zone", label=self.label)


class AndheriEdge(DispatchArenaModel):
    """Directed or bidirectional corridor between coarse nodes."""

    src: str
    dst: str
    base_minutes: int = Field(ge=1)
    tags: list[str] = Field(default_factory=list)
    bidirectional: bool = True


class AndheriGraphData(DispatchArenaModel):
    """Direct graph state before shortest-path expansion."""

    area_name: str = "andheri_mumbai_coarse"
    preset: AndheriPreset
    seed: int | None = None
    nodes: list[Node] = Field(default_factory=list)
    node_metadata: dict[str, AndheriZoneMetadata] = Field(default_factory=dict)
    adjacency_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    traffic_events: list[str] = Field(default_factory=list)


class _PresetOverlay(DispatchArenaModel):
    global_multiplier: float = 1.0
    tag_multipliers: dict[str, float] = Field(default_factory=dict)
    directional_multipliers: dict[str, float] = Field(default_factory=dict)
    seeded_slowdown_multiplier: float = 1.0
    seeded_slowdown_count: int = 0


_BASE_NODES: tuple[AndheriZoneMetadata, ...] = (
    AndheriZoneMetadata(
        node_id="lokhandwala",
        label="Lokhandwala",
        lat=19.1427,
        lon=72.8258,
        x_km=0.1,
        y_km=1.5,
        tags=["residential", "west"],
    ),
    AndheriZoneMetadata(
        node_id="four_bungalows",
        label="Four Bungalows",
        lat=19.1368,
        lon=72.8278,
        x_km=0.5,
        y_km=1.2,
        tags=["residential", "connector", "west"],
    ),
    AndheriZoneMetadata(
        node_id="oshiwara",
        label="Oshiwara",
        lat=19.1481,
        lon=72.8341,
        x_km=0.9,
        y_km=1.8,
        tags=["arterial", "market", "northwest"],
    ),
    AndheriZoneMetadata(
        node_id="veera_desi",
        label="Veera Desai",
        lat=19.1298,
        lon=72.8388,
        x_km=1.2,
        y_km=1.0,
        tags=["arterial", "market", "central_west"],
    ),
    AndheriZoneMetadata(
        node_id="juhu_circle",
        label="Juhu Circle",
        lat=19.1136,
        lon=72.8260,
        x_km=0.4,
        y_km=0.2,
        tags=["coastal", "connector", "southwest"],
    ),
    AndheriZoneMetadata(
        node_id="andheri_station_w",
        label="Andheri Station West",
        lat=19.1197,
        lon=72.8464,
        x_km=1.8,
        y_km=0.7,
        tags=["station", "bottleneck", "market"],
    ),
    AndheriZoneMetadata(
        node_id="andheri_station_e",
        label="Andheri Station East",
        lat=19.1195,
        lon=72.8519,
        x_km=2.1,
        y_km=0.7,
        tags=["station", "bottleneck", "east"],
    ),
    AndheriZoneMetadata(
        node_id="chakala",
        label="Chakala",
        lat=19.1118,
        lon=72.8602,
        x_km=2.6,
        y_km=0.4,
        tags=["arterial", "east", "airport_corridor"],
    ),
    AndheriZoneMetadata(
        node_id="marol_naka",
        label="Marol Naka",
        lat=19.1158,
        lon=72.8778,
        x_km=3.7,
        y_km=0.6,
        tags=["arterial", "airport_corridor", "bottleneck"],
    ),
    AndheriZoneMetadata(
        node_id="seepz_gate",
        label="SEEPZ Gate",
        lat=19.1304,
        lon=72.8731,
        x_km=3.4,
        y_km=1.3,
        tags=["industrial", "business", "east"],
    ),
)

_BASE_EDGES: tuple[AndheriEdge, ...] = (
    AndheriEdge(src="lokhandwala", dst="four_bungalows", base_minutes=4, tags=["west_residential"]),
    AndheriEdge(src="lokhandwala", dst="oshiwara", base_minutes=5, tags=["west_residential"]),
    AndheriEdge(src="four_bungalows", dst="juhu_circle", base_minutes=5, tags=["coastal_connector"]),
    AndheriEdge(src="four_bungalows", dst="veera_desi", base_minutes=4, tags=["market", "connector"]),
    AndheriEdge(src="oshiwara", dst="veera_desi", base_minutes=4, tags=["arterial"]),
    AndheriEdge(src="juhu_circle", dst="veera_desi", base_minutes=6, tags=["connector"]),
    AndheriEdge(src="juhu_circle", dst="andheri_station_w", base_minutes=6, tags=["station_approach"]),
    AndheriEdge(src="veera_desi", dst="andheri_station_w", base_minutes=4, tags=["station_approach", "market"]),
    AndheriEdge(src="andheri_station_w", dst="andheri_station_e", base_minutes=6, tags=["station_crossing", "bottleneck"]),
    AndheriEdge(src="andheri_station_w", dst="chakala", base_minutes=12, tags=["east_west_connector", "bottleneck"]),
    AndheriEdge(src="andheri_station_e", dst="chakala", base_minutes=4, tags=["east_arterial"]),
    AndheriEdge(src="andheri_station_e", dst="seepz_gate", base_minutes=7, tags=["east_arterial"]),
    AndheriEdge(src="chakala", dst="marol_naka", base_minutes=4, tags=["airport_corridor", "bottleneck"]),
    AndheriEdge(src="chakala", dst="seepz_gate", base_minutes=6, tags=["industrial_corridor"]),
    AndheriEdge(src="marol_naka", dst="seepz_gate", base_minutes=5, tags=["industrial_corridor", "bottleneck"]),
)

_PRESET_OVERLAYS: dict[AndheriPreset, _PresetOverlay] = {
    AndheriPreset.EASY: _PresetOverlay(),
    AndheriPreset.MEDIUM: _PresetOverlay(
        global_multiplier=1.05,
        tag_multipliers={
            "market": 1.10,
            "station_approach": 1.15,
            "station_crossing": 1.20,
            "airport_corridor": 1.15,
            "industrial_corridor": 1.10,
            "bottleneck": 1.10,
        },
        directional_multipliers={
            "andheri_station_w->andheri_station_e": 1.10,
            "chakala->marol_naka": 1.10,
        },
        seeded_slowdown_multiplier=1.15,
        seeded_slowdown_count=1,
    ),
    AndheriPreset.HARD: _PresetOverlay(
        global_multiplier=1.10,
        tag_multipliers={
            "market": 1.20,
            "station_approach": 1.25,
            "station_crossing": 1.40,
            "airport_corridor": 1.30,
            "industrial_corridor": 1.20,
            "bottleneck": 1.25,
            "east_west_connector": 1.20,
        },
        directional_multipliers={
            "andheri_station_w->andheri_station_e": 1.25,
            "andheri_station_e->andheri_station_w": 1.10,
            "chakala->marol_naka": 1.20,
            "marol_naka->chakala": 1.10,
            "veera_desi->andheri_station_w": 1.15,
        },
        seeded_slowdown_multiplier=1.30,
        seeded_slowdown_count=3,
    ),
}

_SEEDED_SLOWDOWN_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("veera_desi", "andheri_station_w"),
    ("andheri_station_w", "andheri_station_e"),
    ("andheri_station_e", "chakala"),
    ("chakala", "marol_naka"),
    ("marol_naka", "seepz_gate"),
    ("andheri_station_e", "seepz_gate"),
)


def build_andheri_zone_graph(preset: AndheriPreset | str = AndheriPreset.EASY, seed: int | None = None) -> AndheriGraphData:
    """Build the direct weighted graph for the Andheri simulator."""

    resolved_preset = preset if isinstance(preset, AndheriPreset) else AndheriPreset(preset)
    overlay = _PRESET_OVERLAYS[resolved_preset]
    node_metadata = {node.node_id: node for node in _BASE_NODES}
    nodes = [node.as_dispatch_node() for node in _BASE_NODES]
    adjacency = {node.node_id: {node.node_id: 0} for node in _BASE_NODES}
    seeded_overrides, traffic_events = _build_seeded_slowdowns(resolved_preset, seed)

    for edge in _BASE_EDGES:
        _store_weight(adjacency, edge.src, edge.dst, _edge_weight(edge, edge.src, edge.dst, overlay, seeded_overrides))
        if edge.bidirectional:
            _store_weight(adjacency, edge.dst, edge.src, _edge_weight(edge, edge.dst, edge.src, overlay, seeded_overrides))

    return AndheriGraphData(
        preset=resolved_preset,
        seed=seed,
        nodes=nodes,
        node_metadata=node_metadata,
        adjacency_matrix=adjacency,
        traffic_events=traffic_events,
    )


def node_ids() -> list[str]:
    return [node.node_id for node in _BASE_NODES]


def _build_seeded_slowdowns(preset: AndheriPreset, seed: int | None) -> tuple[dict[str, float], list[str]]:
    overlay = _PRESET_OVERLAYS[preset]
    if overlay.seeded_slowdown_count <= 0:
        return {}, []

    rng = random.Random(seed)
    selected = rng.sample(list(_SEEDED_SLOWDOWN_CANDIDATES), k=overlay.seeded_slowdown_count)
    overrides = {f"{src}->{dst}": overlay.seeded_slowdown_multiplier for src, dst in selected}
    events = [f"seeded slowdown on {src}->{dst} x{overlay.seeded_slowdown_multiplier:.2f}" for src, dst in selected]
    return overrides, events


def _edge_weight(
    edge: AndheriEdge,
    src: str,
    dst: str,
    overlay: _PresetOverlay,
    seeded_overrides: dict[str, float],
) -> int:
    multiplier = overlay.global_multiplier
    for tag in edge.tags:
        multiplier *= overlay.tag_multipliers.get(tag, 1.0)
    key = f"{src}->{dst}"
    multiplier *= overlay.directional_multipliers.get(key, 1.0)
    multiplier *= seeded_overrides.get(key, 1.0)
    return max(1, round(edge.base_minutes * multiplier))


def _store_weight(adjacency: dict[str, dict[str, int]], src: str, dst: str, weight: int) -> None:
    if dst not in adjacency[src] or weight < adjacency[src][dst]:
        adjacency[src][dst] = weight


def iter_edges(graph: AndheriGraphData) -> Iterable[Tuple[str, str, int]]:
    for src, neighbors in graph.adjacency_matrix.items():
        for dst, weight in neighbors.items():
            if src == dst:
                continue
            yield src, dst, weight
