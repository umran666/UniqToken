"""Regression tests for the single-distribution v1 packaging contract."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleasePackagingTests(unittest.TestCase):
    def test_root_metadata_declares_one_mixed_distribution(self) -> None:
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('build-backend = "maturin"', metadata)
        self.assertIn('name = "uniqtoken-core"', metadata)
        self.assertIn('version = "1.0.0"', metadata)
        self.assertIn('manifest-path = "crates/uniqtoken_core/Cargo.toml"', metadata)
        self.assertIn('module-name = "uniqtoken_core.uniqtoken_core"', metadata)
        self.assertIn('python-packages = ["uniqtoken", "uniqtoken_core", "benchmarks"]', metadata)

    def test_ci_exercises_install_uninstall_and_sdist_contracts(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        self.assertIn("tools/verify_release_wheel.py", workflow)
        self.assertIn("Verify clean install, uninstall, and reinstall", workflow)
        self.assertIn('python -I -c "import importlib.metadata, uniqtoken, uniqtoken_core', workflow)
        self.assertIn("python -m maturin sdist", workflow)
        self.assertIn("dist/uniqtoken_core-1.0.0.tar.gz", workflow)
        self.assertIn("publish-dist/*", workflow)

    def test_release_verifier_checks_both_import_names(self) -> None:
        verifier = (ROOT / "tools" / "verify_release_wheel.py").read_text(encoding="utf-8")

        self.assertIn("import uniqtoken\n", verifier)
        self.assertIn("import uniqtoken_core", verifier)
        self.assertIn('pip", "uninstall"', verifier)
        self.assertIn("REQUIRED_PYTHON_FILES", verifier)
        self.assertIn("native_extension_sha256", verifier)
        self.assertIn("legacy uniqtoken distribution", verifier)


if __name__ == "__main__":
    unittest.main()
