import io
import os
from setuptools import setup, find_packages

# package metadata
NAME = 'lf2i'
DESCRIPTION = 'Likelihood-Free Frequentist Inference'
KEYWORDS = "likelihood-free inference simulator likelihood posterior parameter"
URL = "https://github.com/lee-group-cmu/lf2i"
EMAIL = "lee.group.cmu@gmail.com"
AUTHOR = "Luca Masserano, Niccolo Dalmasso, Ann B. Lee"
REQUIRES_PYTHON = ">=3.9.0"

REQUIRED = [

]

EXTRAS = [

]

pwd = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(pwd, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version="",
    description=DESCRIPTION,
    keywords=KEYWORDS,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages="",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="",
    classifiers=[],
    # $ setup.py publish support.
    cmdclass=None, # dict(upload=UploadCommand),
)
