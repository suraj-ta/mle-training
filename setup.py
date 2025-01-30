from setuptools import find_packages, setup

setup(
    name="nonstandardcode",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "numpy",
        "joblib",
    ],
)
