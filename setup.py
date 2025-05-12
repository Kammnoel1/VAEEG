from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "mne", "numpy", "pandas", "joblib", "tqdm", "interval3",
        # …any other deps…
    ],
    # …other setup args…
)
