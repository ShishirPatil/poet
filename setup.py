from setuptools import setup

setup(
    name="poet-ai",
    version="0.1.0",
    description="",  # TODO
    packages=["poet"],  # find_packages()
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "loguru",
        "gurobipy",
        "graphviz",
        "toposort",
        "pulp",
    ],
    extras_require={
        "test": [
            "pytest",
            "matplotlib",
            "graphviz",
            "ray",
            "tqdm",
            "pandas",
            "plotly",
            "psutil",
            "wandb",
        ],
    },
)
