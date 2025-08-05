---
layout: page
title: "Adjoint-accelerated Programmable Inference for Large PDEs"
description: "Enabling Adjoint in Probabilistic Programming Language Turing.jl"
img: assets/img/12.jpg
importance: 1
category: ML
---

Many traditional scientific methods codify physical principles, such as conservation of mass and momentum, into partial differential equations (PDEs), which are then solved numerically. Although these numerical solutions can be extremely detailed, they are not necessarily accurate. Their accuracy requires good models and accurate model parameters. This is why experimental testing remains a crucial component of the engineering design process. CFD simulations and experimental results are usually discarded once a design has been chosen, however, despite producing gigabytes of information that could be useful in the future. This project is partly motivated by a desire to use this data effectively and sustainably. 

The aim of this project is to generate and popularise methods that exploit (i) the vast amounts of data generated from experiments and PDE solvers (ii) the high-quality physics-based priors that are expressed through PDEs, and (iii) recent developments in probabilistic programming. For this aim, a Bayesian framework is ideal: The PDEs is treated as a model in which some parameters are fixed while others are expressed as probability distributions. The prior distributions are asserted by the user and then, as data arrives, the posterior probability distributions of the parameters are updated deterministically. 

The challenge with implementing PDEs within Probabilistic Programs is that function evaluations are usually prohibitively expensive. This means that traditional sampling methods such as Markov Chain Monte Carlo (MCMC) become intractable. PDEs present an opportunity, however, because the outputs of many PDEs vary smoothly with their parameters. For these PDEs, Gaussian prior distributions lead to nearly Gaussian posterior distributions, meaning that Laplace’s approximation becomes sufficiently accurate to be useful. This can be combined with adjoint methods such that tens of thousands of parameters of large PDE problems can feasibly be inferred cheaply, even when data is sparse. The Gaussian assumption can be tested a posteriori with MCMC if desired, albeit at considerable expense. 