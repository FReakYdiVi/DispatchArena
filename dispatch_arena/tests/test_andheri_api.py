import unittest

from fastapi.testclient import TestClient

from dispatch_arena.server.app import create_app


class AndheriApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_presets_endpoint_lists_all_presets(self):
        response = self.client.get("/api/andheri/presets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["presets"], ["easy", "medium", "hard"])

    def test_snapshot_endpoint_is_deterministic_for_same_seed(self):
        params = {"preset": "hard", "seed": 7}
        first = self.client.get("/api/andheri/snapshot", params=params)
        second = self.client.get("/api/andheri/snapshot", params=params)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())

    def test_path_endpoint_returns_expected_eta_and_path(self):
        response = self.client.get(
            "/api/andheri/path",
            params={
                "preset": "easy",
                "seed": 7,
                "src": "andheri_station_w",
                "dst": "chakala",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["eta_minutes"], 10)
        self.assertEqual(payload["path"], ["andheri_station_w", "andheri_station_e", "chakala"])

    def test_path_endpoint_rejects_unknown_nodes(self):
        response = self.client.get(
            "/api/andheri/path",
            params={
                "preset": "easy",
                "seed": 7,
                "src": "unknown",
                "dst": "chakala",
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("Unknown src node", response.json()["detail"])

    def test_nodes_endpoint_returns_lightweight_graph_metadata(self):
        response = self.client.get("/api/andheri/nodes", params={"preset": "medium", "seed": 3})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["preset"], "medium")
        self.assertEqual(len(payload["nodes"]), 10)
        self.assertIn("lokhandwala", payload["node_metadata"])


if __name__ == "__main__":
    unittest.main()
