import os
from setuptools import setup, find_packages

# package metadata
NAME = 'lf2i'
DESCRIPTION = 'Likelihood-Free Frequentist Inference'
KEYWORDS = "likelihood-free inference simulator likelihood posterior parameter"
URL = "https://github.com/lee-group-cmu/lf2i"
EMAIL = "lee.group.cmu@gmail.com"
AUTHOR = "Luca Masserano, Niccolò Dalmasso, David Zhao, Rafael Izbicki, Ann B. Lee"
REQUIRES_PYTHON = ">=3.9.0, <3.12"

REQUIRED = [
    "alphashape",
    "catboost>=1.2.8",
    "click",
    "descartes",
    "joblib",
    "matplotlib>=3.6.0",
    "numpy>=1.26.4",
    "scikit-learn>=1.5.2",
    "scipy>=1.11.4",
    "seaborn>=0.12.1",
    "torch>=1.12.1",
    "tqdm",
    "xgboost>=1.6.2"
]

EXTRAS = {
    "examples": ["sbi>=0.19.2"],
    "dev": [
        "aquirdturtle-collapsible-headings",
        "ipykernel",
        "jupyterlab",
        "furo",  # sphinx theme
        "myst-parser",
        "nbsphinx",
        "pytest",
        "sphinx",
        "sphinx-mdinclude",
        "alphashape"
    ]
}

ENTRY_POINTS = {
    "console_scripts": ["lf2i=lf2i.cli:lf2i"]
}

pwd = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version="0.3.1",
    description=DESCRIPTION,
    keywords=KEYWORDS,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    license="MIT",
    entry_points=ENTRY_POINTS,
    include_package_data=True
)
