# LF2I: Likelihood-Free Frequentist Inference

[![GitHub](https://img.shields.io/github/license/lee-group-cmu/lf2i?style=flat-square)](./LICENSE.txt)

`lf2i` is a Python package for likelihood-free inference. It allows to do inference on parameters of interest by constructing confidence regions with strong statistical guarantees.

The constructed confidence regions are guaranteed to cover the true value with the desired frequentist coverage probability, *regardless of*
1. the prior distribution over the parameters of interest;
2. the true value of the parameters of interest: the coverage guarantee holds point-wise over the parameter space (i.e., not only on average); and
3. the size of the observed sample: the coverage guarantee holds even for finite sample sizes.

## Install

The package is still under development, but will soon be released on PyPI. For now, it can be installed in `dev` mode with the following commands: 

```
git clone git@github.com:lee-group-cmu/lf2i.git
cd lf2i
pip install -e ".[dev]"
```

## Usage

See [tutorials](https://github.com/lee-group-cmu/lf2i/tree/main/tutorials) for a few basic examples.

## Feedback and Contributions

We strongly encourage users to leave feedback and report bugs either by using the *Issues* tab, or by contacting us directly. The current maintainer can be reached [here](mailto:lmassera@andrew.cmu.edu).

If you want to contribute, feel free to open an issue and/or a pull request.

## References

LF2I is based on the following research articles:\
    - [Confidence sets and hypothesis testing in a likelihood-free inference setting (2020), PMLR, 119:2323-2334](http://proceedings.mlr.press/v119/dalmasso20a/dalmasso20a.pdf)\
    - [Likelihood-Free Frequentist Inference: Confidence Sets with Correct Conditional Coverage (2021)](https://arxiv.org/pdf/2107.03920.pdf)\
    - [Simulation-Based Inference with Waldo: Confidence Regions by Leveraging Prediction Algorithms or Posterior Estimators for Inverse Problems (2022)](https://arxiv.org/pdf/2205.15680.pdf)
