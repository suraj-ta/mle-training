import importlib


def test_package_import():
    """
    Test to verify that the nonstandardcode package and its modules can be installed.
    """
    try:
        nonstandardcode = importlib.import_module("nonstandardcode")
        assert nonstandardcode is not None, "Failed to import nonstandardcode package"

        ingest_data = importlib.import_module("nonstandardcode.ingest_data")
        train = importlib.import_module("nonstandardcode.train")
        score = importlib.import_module("nonstandardcode.score")

        assert ingest_data is not None, "Failed to import ingest_data module"
        assert train is not None, "Failed to import train module"
        assert score is not None, "Failed to import score module"

        print("All modules imported successfully!")
    except Exception as e:
        raise AssertionError(f"Installation test failed: {e}")


if __name__ == "__main__":
    test_package_import()
