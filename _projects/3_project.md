---
layout: page
title: NN vs FEM
description: "Which method is better at solving PDEs? To understand that, We need a deep dive into representation theory."
img: assets/img/NNvsFEM.png
importance: 3
category: ML
---

Physics-Informed Neural Networks (PINNs) are a neural technique to solving Partial Differential Equations (PDEs) that has been controversial since its inception. Some see it as the future, beating out the older numerical methods. While others think that PINNs are overrated. Whatever future has in store for PINNs, one cannot ignore them.

Significant focus has been placed on comparing PINNs performance with classic numerical methods like the Finite Element Method (FEM), but barely any have tried to understand where their differences come from. We want to understand exactly that. By drawing out the differences (or similarities) on the way established methods and PINNs represent the solution of a PDE, we can better highlight each method's pros and cons, perhaps even improving each through inspiration of their counterpart.

The scope of this project is to bridge the gap between FEM and PINN because we believe that at their core, these two methods are actually more similar than not.

Want to learn more? Visit out [blog page](https://llfung.github.io/blog/)