import unittest

from dispatch_arena.server.andheri_graph import AndheriPreset
from dispatch_arena.server.andheri_simulator import AndheriSimulator


class AndheriSimulatorTests(unittest.TestCase):
    def test_reset_returns_dispatch_compatible_primitives(self):
        simulator = AndheriSimulator()
        snapshot = simulator.reset(seed=7, preset=AndheriPreset.EASY)
        primitives = snapshot.as_dispatch_primitives()

        self.assertEqual(len(primitives["nodes"]), 10)
        self.assertEqual(set(primitives["travel_time_matrix"]), {node.id for node in snapshot.nodes})
        for node in snapshot.nodes:
            self.assertEqual(primitives["travel_time_matrix"][node.id][node.id], 0)

    def test_graph_is_fully_reachable(self):
        simulator = AndheriSimulator()
        snapshot = simulator.reset(seed=7, preset=AndheriPreset.EASY)
        expected_nodes = {node.id for node in snapshot.nodes}

        for source, distances in snapshot.travel_time_matrix.items():
            self.assertEqual(set(distances), expected_nodes)
            for target, minutes in distances.items():
                self.assertGreaterEqual(minutes, 0, msg=f"{source}->{target} should have non-negative ETA")

    def test_shortest_path_prefers_cheaper_multi_hop_route(self):
        simulator = AndheriSimulator()
        simulator.reset(seed=7, preset=AndheriPreset.EASY)

        self.assertEqual(simulator.estimate_eta("andheri_station_w", "chakala"), 10)
        self.assertEqual(
            simulator.shortest_path("andheri_station_w", "chakala"),
            ["andheri_station_w", "andheri_station_e", "chakala"],
        )

    def test_same_seed_is_deterministic(self):
        simulator_a = AndheriSimulator()
        simulator_b = AndheriSimulator()

        snapshot_a = simulator_a.reset(seed=11, preset=AndheriPreset.HARD)
        snapshot_b = simulator_b.reset(seed=11, preset=AndheriPreset.HARD)

        self.assertEqual(snapshot_a.to_dict(), snapshot_b.to_dict())

    def test_medium_and_hard_presets_increase_station_crossing_eta(self):
        simulator = AndheriSimulator()
        easy = simulator.reset(seed=3, preset=AndheriPreset.EASY)
        medium = simulator.reset(seed=3, preset=AndheriPreset.MEDIUM)
        hard = simulator.reset(seed=3, preset=AndheriPreset.HARD)

        route = ("andheri_station_w", "andheri_station_e")
        self.assertLess(easy.travel_time_matrix[route[0]][route[1]], medium.travel_time_matrix[route[0]][route[1]])
        self.assertLess(medium.travel_time_matrix[route[0]][route[1]], hard.travel_time_matrix[route[0]][route[1]])

    def test_seeded_traffic_events_exist_in_non_easy_presets(self):
        simulator = AndheriSimulator()
        medium = simulator.reset(seed=1, preset=AndheriPreset.MEDIUM)
        hard = simulator.reset(seed=1, preset=AndheriPreset.HARD)

        self.assertEqual(len(medium.traffic_events), 1)
        self.assertEqual(len(hard.traffic_events), 3)


if __name__ == "__main__":
    unittest.main()
