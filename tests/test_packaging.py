import unittest
from importlib.resources import files


class PackagingTests(unittest.TestCase):
    def test_standalone_build_support_is_installed_as_package_data(self):
        resources = files("workspace_browser._packaging")
        self.assertTrue(resources.joinpath("workspace-browser.spec").is_file())
        self.assertTrue(resources.joinpath("runtime_hook_kaleido.py").is_file())


if __name__ == "__main__":
    unittest.main()
