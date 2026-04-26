"""Read-only API surface for the standalone Andheri simulator."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from dispatch_arena.server.andheri_graph import AndheriPreset
from dispatch_arena.server.andheri_simulator import AndheriSimulator


def build_andheri_router() -> APIRouter:
    router = APIRouter(prefix="/api/andheri", tags=["andheri"])

    @router.get("/presets")
    def presets() -> dict:
        return {"presets": [preset.value for preset in AndheriPreset]}

    @router.get("/snapshot")
    def snapshot(
        preset: AndheriPreset = Query(default=AndheriPreset.EASY),
        seed: int | None = Query(default=None),
    ) -> dict:
        simulator = AndheriSimulator()
        data = simulator.reset(seed=seed, preset=preset)
        return {"snapshot": data.to_dict()}

    @router.get("/nodes")
    def nodes(
        preset: AndheriPreset = Query(default=AndheriPreset.EASY),
        seed: int | None = Query(default=None),
    ) -> dict:
        simulator = AndheriSimulator()
        data = simulator.reset(seed=seed, preset=preset)
        return {
            "area_name": data.area_name,
            "preset": data.preset.value,
            "seed": data.seed,
            "nodes": [node.to_dict() for node in data.nodes],
            "node_metadata": {
                node_id: metadata.to_dict() for node_id, metadata in data.node_metadata.items()
            },
        }

    @router.get("/path")
    def path(
        src: str = Query(...),
        dst: str = Query(...),
        preset: AndheriPreset = Query(default=AndheriPreset.EASY),
        seed: int | None = Query(default=None),
    ) -> dict:
        simulator = AndheriSimulator()
        data = simulator.reset(seed=seed, preset=preset)

        if src not in data.node_metadata:
            raise HTTPException(status_code=404, detail=f"Unknown src node: {src}")
        if dst not in data.node_metadata:
            raise HTTPException(status_code=404, detail=f"Unknown dst node: {dst}")

        return {
            "area_name": data.area_name,
            "preset": data.preset.value,
            "seed": data.seed,
            "src": src,
            "dst": dst,
            "eta_minutes": simulator.estimate_eta(src, dst),
            "path": simulator.shortest_path(src, dst),
            "traffic_events": list(data.traffic_events),
        }

    return router
