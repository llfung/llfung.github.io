---
layout: page
title: Learning & Forecasting Satellite Orbits
description: Applying ODR-BINDy to learn the control laws that keeps satelites in orbit
img: assets/img/Satellite.webp
importance: 4
category: ML
---

Without correct and active manoeuvring, the 10,000+ satellites circling the Earth can quickly fall out of orbit due to various perturbatory effects such as drag, solar radiation, and gravity from third bodies (such as the sun and moon). While we understand these mechanics quite well and with great precision, the fact that they are constantly manoeuvred makes it extremely difficult to predict their long-term trajectory. However, in order to steer them and avoid collisions, we need accurate predictions of their trajectories. Satellite control laws are rarely shared for strategic reasons, making collision avoidance even more difficult in an already congested sky. Instead, we may learn the control law utilised from data, allowing us to better our forecast of their trajectory and hence increase overall safety. 

To recover an interpretable control rule and improve forecasting, a symbolic-based ML approach like SINDy appeared appropriate. However, the parameter regime of satellite orbit proves to be particularly difficult for derivative-based ML methods such as SINDy, because the control rules governing these manoeuvres are orders of magnitude smaller than the dominant physics from gravity. Meanwhile, traditional parameter estimation using a data-assimilation-style adjoint solver, while better suited for the task, cannot accommodate a long measurement history in a single iteration. This reduces the capability of the subsequent data-intensive model identification process.

The ODR-BINDy algorithm combines the advantages of SINDy and classical data assimilation. Not only is it noise-resistant in model creation and capable of de-noising data during learning, but it also supports assimilation of arbitrarily long temporal data, making it ideal for the vast historical dataset of satellite orbit data. Furthermore, the system takes advantage of the fact that tiny manoeuvres amplify in time in orbital motion, allowing for very precise reconstruction of the control law and thus delivering actionable manoeuvre information and forecasts.