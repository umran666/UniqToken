"""Regression tests for the supported and security-patched CI dependency contract."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyPolicyTests(unittest.TestCase):
    def test_declared_test_dependencies_are_security_patched(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.10"', pyproject)
        self.assertIn('"torch==2.13.0"', pyproject)
        self.assertIn('"pyarrow==23.0.1"', pyproject)
        self.assertIn('"accelerate==1.15.0"', pyproject)
        self.assertIn('"transformers==5.10.4"', pyproject)

    def test_ci_installs_the_declared_test_extra(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        self.assertIn('python-version: ["3.10", "3.11", "3.12"]', workflow)
        self.assertIn('python -m pip install -e ".[test]"', workflow)
        self.assertNotIn("torch==", workflow)
        self.assertNotIn("pyarrow==", workflow)
        self.assertNotIn("accelerate==", workflow)


if __name__ == "__main__":
    unittest.main()
