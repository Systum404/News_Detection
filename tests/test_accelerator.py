import unittest
from unittest.mock import patch
from News_Detection.accelerator import (
    get_runtime_environment,
    get_hardware_status,
    get_optimal_device,
    benchmark_acceleration,
)


class TestRuntimeEnvironment(unittest.TestCase):
    """Tests for dynamic platform, architecture, and runtime telemetry discovery."""

    def test_runtime_environment_native(self):
        """Verifies native runtime string contains Python version and architecture."""
        runtime = get_runtime_environment()
        self.assertIsInstance(runtime, str)
        self.assertTrue(runtime.startswith("Python 3."))
        self.assertIn("(", runtime)
        self.assertIn(")", runtime)

    @patch("platform.python_version", return_value="3.12.8")
    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_runtime_environment_linux_x86_64(self, mock_machine, mock_sys, mock_py):
        """Ensures x86_64 Linux container environments are accurately represented."""
        runtime = get_runtime_environment()
        self.assertEqual(runtime, "Python 3.12.8 (Linux x86_64)")

    @patch("platform.python_version", return_value="3.12.5")
    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="aarch64")
    def test_runtime_environment_linux_aarch64(self, mock_machine, mock_sys, mock_py):
        """Ensures ARM64 Linux container environments (e.g. AWS Graviton) are accurately represented."""
        runtime = get_runtime_environment()
        self.assertEqual(runtime, "Python 3.12.5 (Linux aarch64)")

    @patch("platform.python_version", return_value="3.12.15")
    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_runtime_environment_macos_arm64(self, mock_machine, mock_sys, mock_py):
        """Ensures macOS Darwin is presented as friendly 'macOS arm64'."""
        runtime = get_runtime_environment()
        self.assertEqual(runtime, "Python 3.12.15 (macOS arm64)")

    @patch("platform.python_version", return_value="3.11.9")
    @patch("platform.system", return_value="Windows")
    @patch("platform.machine", return_value="AMD64")
    def test_runtime_environment_windows_amd64(self, mock_machine, mock_sys, mock_py):
        """Ensures Windows AMD64 environments are accurately represented."""
        runtime = get_runtime_environment()
        self.assertEqual(runtime, "Python 3.11.9 (Windows AMD64)")


class TestHardwareAccelerator(unittest.TestCase):
    """Tests for hardware discovery, quantization modes, and benchmark routines."""

    def test_get_hardware_status_keys(self):
        """Verifies dictionary structure of hardware diagnostic metadata."""
        status = get_hardware_status()
        self.assertIn("active_device", status)
        self.assertIn("device_name", status)
        self.assertIn("half_precision_fp16", status)
        self.assertIn("mps_available", status)
        self.assertIn("cuda_available", status)
        self.assertIn("quantization_modes", status)
        self.assertIn("runtime", status)
        self.assertIsInstance(status["quantization_modes"], list)
        self.assertEqual(status["runtime"], get_runtime_environment())

    @patch("platform.machine", return_value="x86_64")
    def test_quantization_modes_x86_64(self, mock_machine):
        """Verifies x86_64 targets report AVX2/AVX-512 instead of ARM64 NEON."""
        status = get_hardware_status()
        self.assertIn("x86_64 AVX2/AVX-512 Vectorized", status["quantization_modes"])

    @patch("platform.machine", return_value="arm64")
    def test_quantization_modes_arm64(self, mock_machine):
        """Verifies ARM64 targets report ARM64 NEON Vectorized."""
        status = get_hardware_status()
        self.assertIn("ARM64 NEON Vectorized", status["quantization_modes"])

    @patch.dict("os.environ", {"DEVICE_OVERRIDE": "cpu"})
    def test_optimal_device_cpu_override(self):
        """Verifies manual DEVICE_OVERRIDE=cpu forces CPU device."""
        device, dtype, name = get_optimal_device()
        self.assertIn("cpu", str(device).lower())
        self.assertIn("CPU", name)

    def test_benchmark_acceleration_execution(self):
        """Verifies synthetic tensor benchmark returns valid latency metrics."""
        result = benchmark_acceleration(iterations=2)
        self.assertIn("cpu_latency_ms", result)
        self.assertIn("accelerated_latency_ms", result)
        self.assertIn("speedup_factor", result)
        self.assertGreater(result["cpu_latency_ms"], 0)


class TestHealthApiRuntimeIntegration(unittest.TestCase):
    """Verifies that /health accurately exposes the dynamic runtime string."""

    def test_health_endpoint_exposes_dynamic_runtime(self):
        from starlette.testclient import TestClient
        from News_Detection import health_api

        client = TestClient(health_api.app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["runtime"], get_runtime_environment())
        self.assertNotIn("ARM64 Native", data["runtime"] if "Darwin" not in data["runtime"] and "arm64" not in data["runtime"] else "")


if __name__ == "__main__":
    unittest.main()

