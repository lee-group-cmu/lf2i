# LF2I: Likelihood-Free Frequentist Inference

[![GitHub](https://img.shields.io/github/license/lee-group-cmu/lf2i?style=flat-square)](./LICENSE.txt)

`lf2i` is a Python package for likelihood-free inference, also known as simulation based inference. It allows to do inference on parameters of interest by constructing confidence regions with strong statistical guarantees.

The constructed confidence regions are guaranteed to cover the true value with the desired frequentist coverage probability, *regardless of*
- the size of the observed sample: the coverage guarantee holds even for finite sample sizes;
- the true value of the parameter of interest: the coverage guarantee holds point-wise over the parameter space (i.e., not only on average)

## Installation

The package is still under development.

```
git clone git@github.com:lee-group-cmu/lf2i.git
cd lf2i
pip install -e ".[dev]"
```

## Usage

TBD

## References

TBD
