---
layout: page
title: Symbolic Model Discovery
description: Discovering governing equation from temporal observation of the states
img: assets/img/ODR-BINDy-Lorenz-10NSR.svg
importance: 2
category: ML
related_publications: true
---

The Bayesian Identification of Nonlinear Dynamics (BINDy) framework is the next iteration of the popular SINDy framework for learning equations from data. By proper treatment of the noise in the data through the reformulation of the model discovery problem in the Bayesian framework, BINDy has shown major improvement in noise robustness and data efficiency in the learning problem. It also invited more mathematically rigorous treatment of the SINDy framework and found the field on a firmer foundation.

Our first work {% cite Fung2025 %} on BINDy has illustrated the importance and rigour of interpreting the model selection problem in the Bayesian sense. By exploiting linear regression the same way SINDy has and using Laplace approximation, our algorithm is almost as fast as the original SINDy, while being more noise robust. 

Our second go at the framework {% cite Fung2025a %} focus more in cases when measurement noise is large enough to distort noise propagation through the nonlinear library. We showed that in those cases, the noise model should include both noise in measurments and noise in the model arising from numerical trucation or stohcasticity. By explioting information from neighbouring time-points and through nonlinear optimisation, our [ODR-BINDy algorithm](https://github.com/llfung/ODR-BINDy/) can denoise the data, perform parameter estimation and select the most parsimonious model at the same time. In our benchmark, ODR-BINDy outperforms all existing methods in noise robustness and data requirements.

{% include figure.liquid loading="eager" path="assets/img/Lorenz_Success.svg" title="Lorenz63 Success Rate" class="img-fluid rounded z-depth-1" %}
Success Rate of recovering Lorenz63 from data using different SINDy derivatives

Want to give our algorithm a try? Why not check out [our MATLAB applet designed for 2D/3D time-series data](https://github.com/llfung/ODR-BINDy-MATLABApp)?
