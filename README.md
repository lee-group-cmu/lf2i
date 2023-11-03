# LF2I: Likelihood-Free Frequentist Inference

[![PyPI](https://img.shields.io/pypi/v/lf2i)](https://pypi.org/project/lf2i/)
[![GitHub](https://img.shields.io/github/license/lee-group-cmu/lf2i)](./LICENSE.txt)

<!--- - [LF2I: Likelihood-Free Frequentist Inference](#lf2i-likelihood-free-frequentist-inference)---> 
- [LF2I: Likelihood-Free Frequentist Inference](#lf2i-likelihood-free-frequentist-inference)
  - [Getting Started](#getting-started)
    - [What is LFI?](#what-is-lfi)
    - [What does LF2I do?](#what-does-lf2i-do)
    - [Structure of LF2I](#structure-of-lf2i)
    - [Usage](#usage)
  - [Install](#install)
  - [Feedback and Contributions](#feedback-and-contributions)
  - [References](#references)

## Getting Started

### What is LFI?
`lf2i` is a Python package for likelihood-free inference; that is, inference on the parameters $\boldsymbol{\theta}$ of a statistical model $F_{\boldsymbol{\theta}}$ in a setting where the likelihood $\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}):=p(\mathcal{D}|\boldsymbol{\theta})$ cannot be evaluated but is *implicitly* encoded by a high-fidelity simulator for $F_{\boldsymbol{\theta}}$. In other words, one can simulate data in batches of size $n$, $\mathcal{D}=(X_1, \dots, X_n)$, for any given $\boldsymbol{\theta}$ in the parameter space.

### What does LF2I do?
`lf2i` constructs confidence regions for internal parameters with correct *coverage* across the whole parameter space, that is, sets $\mathcal{R}(\mathcal{D})$ satisfying $\mathbb{P}({\boldsymbol{\theta}} \in \mathcal{R}(\mathcal{D})) = 1 - \alpha\text{ } \text{ }\forall\theta\in\Theta$, where $(1 − \alpha) \in (0, 1)$ is a prespecified confidence level.\
Coverage is guaranteed regardless of
1. the prior distribution over the parameters of interest;
2. the true value of the parameters of interest: the coverage guarantee holds point-wise over the parameter space (i.e., not only on average); and
3. the size of the observed sample: the coverage guarantee holds even for finite sample sizes, including for the case of one observation, i.e. $n=1$.

### Structure of LF2I
`lf2i` is based on the equivalence of confidence sets and hypothesis tests. It leverages supervised machine learning methods to efficiently execute the Neyman construction of confidence sets. The framework has three separate modules for estimating 
1. test statistics (such as [ACORE](https://arxiv.org/pdf/2002.10399.pdf), [BFF](https://arxiv.org/abs/2107.03920), [Waldo](https://arxiv.org/abs/2205.15680), etc...);
2. critical values for a level $\alpha$ test; and
3. empirical coverage

across the entire parameter space. See the figure below for a schematic diagram.\
<br>
<p align="center">
<img class="hide-on-website" src="https://lee-group-cmu.github.io/lf2i/_images/lf2i_framework.png" alt="drawing" width="50%"/>
</p>

While *1.* and *2.* are used to construct the confidence sets, *3.* is an independent diagnostic tool that can be used to check whether a given parameter region (such as, `lf2i` confidence sets, posterior credible regions, prediction sets, etc …) has the right conditional coverage. Because `lf2i` is modular, users can construct valid confidence sets using any test statistic of their choice. 

### Usage
`lf2i` offers a simple interface that allows you to get started quickly. The entry point is the `LF2I` class in the `lf2i.inference` module, which contains classes to wrap the different functionalities. The method `inference` merges steps *1.* and *2.* to return confidence sets with correct coverage. The method `diagnostics` performs step *3.* as an independent check of empirical coverage of the constructed parameter regions.

Check the [website](https://lee-group-cmu.github.io/lf2i/) for the full documentation, complete of tutorials on the [Waldo](https://arxiv.org/pdf/2205.15680.pdf) test statistics. Tutorials on likelihood-based test statistics like ACORE and BFF are coming soon!


## Install

The package is under active development, and is available on PyPI at this [link](https://pypi.org/project/lf2i/). It can be installed using `pip`:

```python
pip install lf2i
```

The diagnostics module leverages smoothing splines implemented in `R`, which is assumed to be installed along with the `mgcv` package.

## Feedback and Contributions

We strongly encourage users to leave feedback and report bugs either by using the *Issues* tab, or by contacting us directly. The current maintainer can be reached [here](mailto:lmassera@andrew.cmu.edu).

If you want to contribute, feel free to open an issue and/or a pull request.

## References

LF2I is based on the following research articles:\
    - [Confidence sets and hypothesis testing in a likelihood-free inference setting (ICML 2020)](http://proceedings.mlr.press/v119/dalmasso20a/dalmasso20a.pdf)\
    - [Likelihood-Free Frequentist Inference: Bridging Classical Statistics and Machine Learning for Reliable Simulator-Based Inference (2021)](https://arxiv.org/pdf/2107.03920.pdf)\
    - [Simulation-Based Inference with Waldo: Confidence Regions by Leveraging Prediction Algorithms and Posterior Estimators for Inverse Problems (AISTATS 2023)](https://arxiv.org/pdf/2205.15680.pdf)
