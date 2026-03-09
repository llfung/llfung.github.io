---
layout: post
title: Are FEM and PINNs Really That Different?
date: 2026-03-09 09:00:00
description: Numerical methods and PINNs have been regarded as two separate modelling paradigms, with a lot of research focusing on trying to prove which method is better. However, we believe that more focus should be placed on understanding their differences and how each method can inform each other. In this blog we show that FEM and PINNs are more alike than people choose to believe.
tags: ideas
categories: misc
---

Before starting this blog, I would like to thank Dr. Fung for giving me the opportunity to write this blog post. I would also welcome any critique or comments on the work. Best way to contact me is via email on sean.demarco.02@gmail.com

# Introduction

The following article belongs to a multipart blog series on Physics-Informed Neural Networks (PINNs). The scope of this series is to explain and discuss all-things PINNs, starting from the basic concepts, to a more mathematical look at PINNs, to exploring their use cases in solving Partial Differential Equations (PDEs), and to aspects that explore more abstract ideas such as causality and information travel. The goal of this blog series is to provide a platform for open discussion on a small fragment of Scientific Machine Learning (SciML) (and because it is impossible to personally do in depth studies on every single aspect, so hopefully it inspires someone to go and build upon this research).

## What are PINNs and What is PINN's Purpose

PINNs were first introduced by Raissi, Perdikaris and Karniadakis in 2017 and formalised into a publication published in 2019 [1]. PINNs quickly took the academic world by storm by integrating physical laws expressed as differential equations in the loss function of neural networks. Structurally, the introduced PINN has the same architecture as vanilla neural networks however, do not require data to be trained. Rather, PINNs are trained by setting the governing equations as the loss function. The model parameters are then optimised such that the governing PDE is satisfied at chosen collocation points, corresponding to the global minima of the optimisation space. Automatic differentiation plays a large part in the accuracy of PINNs due to its capability of finding the derivatives across the neural architecture to numerical precision. Despite PINNs having been proven to be accurate solvers, many users have found them to be a struggle in general use [2]. PINNs have a complex loss landscape, making convergence to the global minima difficult [3]. A lot of work has been done to find ways of improving the convergence properties of PINNs, ranging from architectural changes [4], experimentation with optimisation techniques [5] and even data grounding [6] which showed to simplify the loss landscape. PINNs's reputation for being notoriously hard to optimise has been well-documented and is perhaps the biggest pushback against this method in becoming a standard in solving PDEs [2]. 

Therefore, the scope of a PINN is to fit network parameters such that a PDE is satisfied. Currently, most approaches of solving PDEs are numerically done using finite difference, element or volume methods. These methods are all mesh-based, with the nature of a mesh-based solver, the generation of a grid is required. For complex domains, generating the grid such that FEM can be applied becomes a complex problem in end of itself, becoming unfeasible [5]. The drawback of numerical methods is that solution accuracy relies strictly on the spatial resolution of the problem space and on the complexity of reconstruction, which can be refined through h-type and p-type refinement respectively [7]. Due to their dependence on spatial resolution, numerical methods suffer from the curse of dimensionality [8-9]. Moreover, numerical methods requires the solver to be tailored to the problem being solved by introducing stabilisation techniques and phenomenon-specific constructions. On the other hand, NNs do not strictly require a grid, but rather just need collocation points at which the loss is determined to move onto the next optimisation step. Inherently, this enables NNs to overcome the curse of dimensionality [10], since convergence to a solution is not strictly dependant on spatial resolution. In addition, NNs can be classified as ”plug-and-play”, in which minimal adjustments to the solver are needed to adapt to different equation types. These two avenues to NNs show PINNs to be an exciting candidate for the future of solving PDEs. However, the main drawbacks from NNs come from training. As mentioned previously, no matter what (very optimistically), PINNs converge to a solution. This does not guarantee that the converged solution is representative of the governing PDE. In fact, due to the complex nature of the optimisation space of PINNs, it is unlikely that the PINN solution is accurate to numeric precision. Studies such as Rathore et al. [11] have shown that PINNs require a near zero training loss to be highly accurate, as even small loss values are susceptible to ill-conditions and under-optimisation. Achieving such small loss values represent a difficult task.

## Literature's Comparison of FEM and PINNs

As is expected with exploring new methods, the first question always revolves around how this new method compares to more traditional methods. To explore this avenue, the FEM was selected for direct comparison. The reason for this will be discussed and explored further in this work.

From the literature I have reviewed up to this date (Late 2025), only two studies have directly compared PINNs and the FEM, the first study was done by Grossmann et al.[12] who used the Poisson equation in 1D, 2D and 3D, the Schrödinger equation in 1D and 2D and the Allen-Cahn equation in 1D to compare the methods. The scope of using such varied equations is to expose both methods to varying conditions and solution spaces. Grossmann et al.[12] ultimately showed that although PINNs are capable of overcoming the curse of dimensionality, the solution cost and time were not more favourable than FEM in all cases. The study showed inconclusive results where no method was the clear favourite in solving PDEs. The other literature which directly compared the FEM and PINNs was a study by Sikora et al.[13] who compared PINNs and Variational-PINNs with highorder and continuity FEMs in solving the Eriksson-Johnson problem. The paper showed that in order for the FEM to solve the Eriksson-Johnson problem, two stabilisation methods were required, namely the streamline upwind Petrov-Galerkin method and the residual minimisation method. The drawback of both is that adaptive meshing is required, further increasing the curse of dimensionality of the FEM problem. On the other hand, the PINN implementation was simpler, the only adjustments needed were to adapt the loss function to include the governing PDE and some terms for the boundary and initial conditions, however, due to the nature of the solution, a non-uniform discretisation was required. The need of a non-uniform discretisation came from pre-existing understanding of the boundary layer requiring greater refinement to successfully capture the characteristics in the boundary, mimicking what is done in FEM in the presence of large parameter gradients. This problem highlights a significant challenge with PINNs; where convergence to a correct solution with PINNs could require a priori knowledge.

The aim of this work is to explore this comparison further through a mathematical analysis of FEM and PINN, with the goal of highlighting the mathematical similarities and methods between the two methods and how these numeric differences manifest themselves computationally.

# Theory

The mathematics behind neural networks is often seen as a black-box, with PINNs being similar. This was definitely the case for me when I first interacted with PINNs. Only by doing a deep diver into the mathematics behind PINNs and reducing the structure to a single-layer was I able to grasp what PINNs actually do.

The goal of this section of the blog is to highlight that by reducing PINNs, to a single-layer, fixing the input layer weights and analytically optimising the output-layer in a linear least squares fashion - bringing the PINNs to a form known as an Extreme Learning Machine (ELM), that a mathematical equivalence between FEM and a single-layer perceptron PINN can be drawn.

## What Are Extreme Learning Machines

{% include figure.liquid 
   loading="eager" 
   path="assets/img/femvspinn/Screenshot.png" 
   class="img-fluid rounded z-depth-1"
   caption="Figure 1: Structure of an Extreme Learning Machine (ELM) showing the random hidden layer and analytically determined output weights."
%}

ELMs were first presented by Huang et. al[4]. An ELM has the same structure as a standard feedforward NN, however, it is trained differently. Consider a single-layer feedforward NN as seen in the figure 1; to convert this to an ELM, the input-hidden layers weights, $w_1$ and biases, $b_1$, are fixed, it is a general rule-of-thumb to assign and fix these randomly[4], [14], the hidden-output weights, $w_2$ and biases, $b_2$, are then analytically trained using a linear-least squares method[4]. ELMs can be SLPs or even MLPs[15], [16]. The significant advantage of using ELMs is that the complexity of the problem is reduced compared to standard PINNs due to the constraint on the training space. Furthermore, Huang et. al presents the derivation for the training of a single-layer frozen network as a linear least squares optimisation problem, where the global minima has a definite result by solving a linear set of equations. This is much simpler compared to using gradient descent as the optimisation solver.

Implementing an ELM algorithm involves the following steps[4]:

1. Select a shallow one layer NN
2. Fix the weights and biases of the input layer
3. Forward pass the input layer
4. Setup a system of a linear set of equations
5. Train the model using a linear least squares approach 

Following the work of Huang et. al[4], ELMs have been developed, ranging from kernel-based ELMs[17], to finite basis ELMs[5], to even Physics Informed Extreme Learning Machines (PIELMs). PIELMs were first presented by Dwivedi and Srinivasan[14] and attempted to combine ELMs[4] and PINNs[1], by creating a framework that enabled an ELM to be trained on a physics equation. The goal was to overcome the computational cost and complexity of PINNs by exploiting the lightweight analytical solution method of ELMs. Dwivedi and Srinivasan[14] present the mathematics behind PIELMs and show that for 1D and 2D cases of fluid dynamic related problems, the PIELMs were capable of accurately solving numerous PDEs in complex domains.

### Problem Definition

Consider a general problem governed by the 1D Poisson equation such that the problem is defined as

Solve 

$$
\frac{d^2 u(x)}{dx^2} = f(x) \tag{1}
$$

in the domain, $x \in [0,1]$ given $u(0) = u(1) =0$.

Analytically, this problem can be easily solved by integrating both sides of the equation with respect to x and substituting the boundary conditions. Although the analytical solution is simple, the goal of the study is to benchmark the performance of ELMs compared to the FEM to solve PDEs, therefore the analytical solution enables for a strong target to which both methods could be scrutinised. Furthermore this 1D problem can be easily scaled to higher dimensions.

It is important to note that by defining problem (1) as the main problem to be tested, the study is constrained to boundary value problems and not initial value problems. Defining an initial value problem might have an influence on the performance of either method. The reason why an initial value problem was not tested was due to time constraint. However, this presents possibilities should a study wish to build on the work done in this project.

### FEM Formulation

Consider the domain defined in problem space (1) and discretise the domain into $N$ sections.

Let $u(x)$ be the solution to equation (1) at an arbitrary $x$. Then $u(x)$ can be approximated by a linear sum of the weighted basis functions belonging to each element such that

$$
\hat{u}(x) = \sum_{j=1}^{N}U_j \varphi_j \tag{2}
$$

Where, $\hat{u}(x)$ is the approximate solution at a given $x$, $U_j$ and $\varphi_j$ are the weights and basis functions at node $j$ respectively.

It is possible to select any basis function, however it was decided to select a piecewise-linear base function as it is simple and cost-effective.

Consider Poisson's equation as defined in equation (1) and substitute the approximation, $\hat{u}(x)$

$$
\frac{d^2 \hat{u}(x)}{dx^2} - f(x) = 0 \tag{3}
$$

By the method of weighted residuals

$$
\int_{\Omega} \left(\frac{d^2 \hat{u}(x)}{dx^2} - f(x) \right)v_i dx= 0 \quad; i = 1,2,...,N \tag{4}
$$

Where $v_i$ is some test function at the $i^{th}$ element.

Using the Galerkin method, the basis function and the test function are selected to be the same. Noting that the second order term can be integrated by parts and the boundary conditions substituted into the equation such that

$$
\sum_{j=2}^{N-1} U_j \int_{0}^{1} \frac{d\varphi_j}{dx} \frac{d\varphi_i}{dx} \, dx = \int_{0}^{1} f(x) \varphi_i \, dx \quad; i = 1, 2, \ldots, N \tag{5}
$$

From equation (5) it can be seen that, the limits of the summation were changed due to the boundary conditions. Computationally, the Dirichlet boundary conditions can be enforced by directly inserting rows in the stiffness matrix. Moreover, equation (5) shows that if the basis function and forcing function $f(x)$ are known then the system becomes a set of linear simultaneous equations with $U_j$ as the unknown. In the galerkin method with a linear piecewise basis function, the weight $U_j$ corresponds directly with the approximate solution at the $j^{th}$ node.

The piecewise linear function can be defined as

$$
\varphi_j(x) =
\begin{cases}
\frac{x - x_{j-1}}{x_j - x_{j-1}}, & x_{j-1} < x < x_j \\
\frac{x_{j+1} - x}{x_{j+1} - x_j}, & x_j < x < x_{j+1}
\end{cases} \tag{6}
$$

Let $A_{ij} = \int_{0}^{1} \frac{d\varphi_j}{dx} \frac{d\varphi_i}{dx}$. Then, by definition of the test function, $A_{ij}$ is non-zero when $x \in [x_{i-1}, x_i, x_{i+1}]$. $A_{ij}$ can then be defined as

$$
A_{ij} = \int_{x_{i-1}}^{x_{i+1}} \frac{d\varphi_j}{dx} \frac{d\varphi_i}{dx} \, dx \tag{7}
$$

$$
A_{ij} = \int_{x_{i-1}}^{x_i} \frac{d\varphi_j}{dx} \frac{d\varphi_i}{dx} \, dx + \int_{x_i}^{x_{i+1}} \frac{d\varphi_j}{dx} \frac{d\varphi_i}{dx} \, dx \tag{8}
$$

$$
A_{ij} = \frac{d\varphi_j^{(i)}}{dx} \cdot \frac{d\varphi_i^{(i)}}{dx} + \frac{d\varphi_j^{(i+1)}}{dx} \cdot \frac{d\varphi_i^{(i+1)}}{dx} \tag{9}
$$

But, at one selection of $i$, $\varphi_j$ exists only when $j = i-1, i, i+1$. Therefore, assuming an equispaced mesh.

$$
A_{i,i-1} = -\frac{1}{h} \tag{10}
$$

$$
A_{i,i} = \frac{2}{h} \tag{11}
$$

$$
A_{i,i+1} = -\frac{1}{h} \tag{12}
$$

Moreover, the RHS of equation (5) can be found by the Riemann sum definition of an integral

$$
b_i = f(x_i)h \tag{13}
$$

Where, $h$ is the mesh spacing

The problem is now setup in the form of a linear set of equations and can therefore be solved.

### PIELM Formulation

Let the loss, $E$ be defined by a mean squared physics-informed loss function based on the 1D Poisson equation as

$$
E = \sum_{i=1}^{N} \left( \frac{d^2 \tilde{u}}{dx^2}(x_i) - f(x_i) \right)^2 \tag{14}
$$

To solve the optimisation problem, the equation must first be setup. Following the method presented by Huang et. al, a single layer NN was used. Then, the output of the NN is found by equation (15).

$$
\tilde{u} = W_i^{(2)} \, \sigma^{(1)} \left( W_i^{(1)} x_j + b_i^{(1)} \right) + b^{(2)} \tag{15}
$$

such that

$$
W^{(1)} \in \mathbb{R}^{H \times 1}, \quad 
x \in \mathbb{R}^{1 \times N}, \quad 
b^{(1)} \in \mathbb{R}^{H \times 1}, \quad 
W^{(2)} \in \mathbb{R}^{1 \times H}, \quad 
b^{(2)} \in \mathbb{R}^{1 \times 1} \tag{16}
$$

Where, $H$, $N$ and $\sigma$ are the number of hidden dimensions, the number of sample points and the activation function respectively. Note that the superscript number refers to the layer in which the parameters belong.

In an SLP-ELM, the parameters of the first layer are frozen therefore the non-trained section of the ELM can be defined as

$$
Z_{ij} = \sigma^{(1)} \left( W_i^{(1)} x_j + b_i^{(1)} \right) \tag{17}
$$

then

$$
\tilde{u} = W_i^{(2)} Z_{ij} + b^{(2)} \tag{18}
$$

The bias of the output layer can either be trained or fixed to some arbitrary value. For this derivation, the output bias, $b^{(2)}$ was set to 0.

Taking the laplacian and noting that the biases and weights are not functions of the domain

$$
\frac{d^2 \tilde{u}}{dx^2} = W^{(2)} \,\frac{d^2 {Z}_{ij}}{dx^2} \tag{19}
$$

Substituting equation (19) in equation (14)

$$
E = \sum_{j=1}^{m} \left[ \sum_{i=1}^{N} 
\left( w_i^{(2)} \frac{d^2 Z_{ij}}{dx^2}(x_j) - f(x_j) \right) \right]^2 \equiv \left\| M \mathbf{a} - \mathbf{f} \right\|^2 \tag{20}
$$

The global minima of the optimisation can be solved analytically such that

$$
\mathbf{a} = \left( M^T M \right)^{-1} M^T \mathbf{f} \tag{21}
$$

Enforcing boundary conditions can be done through many different methods. The two common methods are: $(1)$ including the boundary term in the loss term — creating a soft boundary constraint or $(2)$ using the Lagrange multiplier — hard constraining the boundary conditions. Applying the Lagrange multiplier can be done by setting up the optimisation space as follows

$$
\min_{\mathbf{a}} \|M \mathbf{a} - \mathbf{f}\|^2 \quad \text{subject to} \quad C \mathbf{a} = \mathbf{d} \tag{22}
$$

Where, $C$ represents the nodal outputs at boundaries and $\mathbf{d}$ represents the imposed boundary conditions.

The minimisation problem can be solved by taking the derivative of equation (22) with respect to $\mathbf{a}$ and to the Lagrange multiplier and equating them both to zero. Then, the matrix system of equations become.

$$
\begin{bmatrix}
M^T M & C^T \\
C & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\mathbf{a} \\
\mathbf{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
M^T \mathbf{f} \\
\mathbf{d}
\end{bmatrix} \tag{23}
$$

### Mathematical Comparison of FEM and ELM

The following section presents the mathematical similarity between FEM and PI-ELMs. The goal of this section is to show how in essence, FEM and PI-ELMs are similar if not identical.

Consider equation (4) derived using the method of weighted residuals

$$
\int_{\Omega} \left(\frac{d^2 \hat{u}(x)}{dx^2} - f(x) \right)v(x) \, dx= 0 \tag{24}
$$

Where, $v(x)$ is a test function which projects the residuals onto the domain.

Integrating the left-hand side term by parts

$$
\left[ \frac{d \hat{u}}{dx} v(x) \right]_{\partial\Omega} 
- \int_\Omega \frac{d\hat{u}}{dx} \frac{dv}{dx} \, dx
= \int_\Omega f(x) v(x) \, dx \tag{25}
$$

As done in equation (2), let the approximate solution be the linear sum of weighted basis functions such that

$$
\hat{u}(x) = \sum_{j=1}^{N}U_j \varphi_j(x) \tag{26}
$$

Where, $U_j$ is a scalar weight — this is the parameter to be solved for and $\varphi_j(x)$ is a basis function which models the solution space.

Substituting this definition in equation (25) and writing in index notation

$$
\underbrace{\left[ \frac{d u}{dx} v(x) \right]_{\partial\Omega}}_{\text{Boundary Conditions}}
- 
U_j \underbrace{\int_0^1 \varphi'_j v'_i \, dx}_{\text{Stiffness Matrix}}
=
\underbrace{\int_0^1 f(x_i) v_i \, dx}_{\text{Forcing Terms}}
\quad ; i=1,2,\ldots,M \tag{27}
$$

Where, $M$ represents the number of points in the test space. To map the weighted residuals into the test space $v(x)$ must be selected.

In ELM we attempt to minimise a residual as defined by the governing equation using a least square method and applying it to each collocation point. Where the loss to be optimised is defined as

$$
E = \sum_{j=1}^{m} \left[ \sum_{i=1}^{N} 
\left( w_i^{(2)} \frac{d^2 Z_{ij}}{dx^2}(x_j) - f(x_j) \right) \right]^2 \tag{28}
$$

Where, all parameters have the same meaning as before.

By minimising $L$, the global minima to the optimisation problem can be found using equation (21).

Then from (21) it follows that

$$
\sum_{i=1}^{N} \left( w_i^{(2)} \frac{d^2 Z_{ij}}{dx^2}(x_j) - f(x_j) \right) = 0 \tag{29}
$$

$$
\Rightarrow \int_\Omega \left( w_i^{(2)} \frac{d^2 Z_{ij}}{dx^2} - f(x_j) \right) \delta(x - x_j) \, dx = 0
\quad ; j = 1, 2, \ldots, m \tag{30}
$$

Where, $\delta(x - x_j)$ is the shifted dirac delta function.

Let $\left( \delta(x - x_j) = \tilde{V}_j(x) \right)$, then:

$$
w_i^{(2)} \int_\Omega \left( \frac{d^2 Z_{ij}}{dx^2} \right) \tilde{V}_j(x) \, dx = 
\int_\Omega f(x_j) \tilde{V}_j(x) \, dx \tag{31}
$$

Integrating the second order term by parts

$$
\underbrace{\left[w_i^{(2)} \frac{d Z_{ij}}{dx} \tilde{V}_j(x) \right]_{\partial\Omega}}_{\text{Boundary Conditions}}
- \underbrace{w_i^{(2)}\int_\Omega Z_{ij}' \tilde{V}_j' \, dx}_{\text{Stiffness Matrix}}
= \underbrace{\int_\Omega f(x_j) \tilde{V}_j(x) \, dx}_{\text{Forcing Terms}} \tag{32}
$$

Comparing equations (27) and (32) the similarities between the methods are apparent. In equation (27) the selection of the test function was left abstract on purpose due to the range of choice and in equation (32), the dirac delta function was represented in a more general manner. This was done so that the similarity between the methods is more visually evident. Starting from different points, both the FEM and ELM were brought into the weak form of the governing equation. From equation (32) it is evident that in the case of ELM, the activation function of the input layer is the equivalent to the basis function in the FEM and the dirac delta function is the equivalent to the test function in the FEM formulation. This shows that mathematically, FEM and ELM solve similar equations through stiffness matrix inversion and both solve for the weights of their respective trial functions. The differences between the methods appear to be: (1) the method through which the discretised space is reconstructed, that is for FEM the choice of basis and trial function and for ELM, the choice of activation function (synonymous to basis function) and the output layer activation, typically an identity activation (synonymous to the trial function) and (2) the way through which the residuals is enforced to be zero that is, for the FEM this is done by solving the weak form of the PDE directly and for the ELM, this is done using a linear least squares method as shown in equation (21).

Numerically, the difference in reconstructing the solution is evident in the matrix to be inverted. In the ELM, the input activation is globally supported meaning that each activation node influences the entire domain. On the other hand, in the FEM the basis function is locally supported, meaning that it only influences the section of the domain where it is located. The global and local support manifests itself in either a dense or sparse stiffness matrix. In the ELM case, the stiffness matrix is globally supported and hence dense, whereas for the FEM case, the stiffness matrix is locally supported and hence sparse. The difference in stiffness matrices puts into question the validity of using an ELM approach for higher-dimensional problems as inverting large dense matrices is computationally costly as well as not allowing for smart storage such as using Skyline storage for sparse matrices. A summarised comparison of the methods can be seen in the table below.

| **Characteristic** | **FEM** | **ELM** |
|--------------------|--------|--------|
| Mathematical foundation | Weighted residual method (weak form) | Weighted residual method (collocation / least squares, strong form) |
| Trial functions | Locally supported trial function | Globally supported hidden-layer activation functions |
| Test functions | Up to user | Dirac delta functions at collocation points |
| Residual enforcement | Weak form solution | Minimisation of strong form via linear-least squares method |
| Unknowns | Coefficients of FEM trial functions | Output layer weights |
| System matrix | Sparse stiffness matrix (local influence) | Dense system matrix (global influence) |

### Physical Representation of Mathematical Analysis

From the mathematical analysis, it was shown that ELMs can be brought into the FEM mathematical framework. This shows that ELMs can be viewed in a FEM lens, having globally supported activation functions as trial functions and dirac delta functions as test functions. Mathematically, this manifestes in a dense stiffness matrix for ELMs, but what are the physical implications of these mathematical differeces? That is, how does the reconstruction of the PDE change when using FEM or ELM?

FEM's reconstruction of a solution space is well known, by splitting a domain into finite elements (hence the name) the solution can be locally reconstructed using properly weighted functions. An example of a classic visual interpretation is presented as reconstructing a circle using straight lines, the more lines, the more accurate the circle can be reconstructed.

ELM's reconstruction of the solution space is similar, however in ELM's case, the domain is not split. Rather the solution space is reconstructed by the sum of the PINN's nodal activations.  Figure 2 shows the true solution of an arbitrary 1D function and it's reconstruction using FEM (having 10 linear piecewise functions) and ELM (having 10 nodes in the network, but only 4 are visible).

{% include figure.liquid 
   loading="eager" 
   path="assets/img/femvspinn/FEMvsELMreconst2.svg" 
   class="img-fluid rounded z-depth-1"
   caption="Figure 2: The solution reconstruction differences between FEM and ELM, where FEM locally reconstructs the solution and ELM reconstructs the solution by summing the nodal contributions"
%}

First, consider FEM's reconstruction. Each linear piecewise function acts in a local range of the solution, only influencing its neighbouring elements. The sum of these can be expressed in a stiffness matrix, which has non-zero entries only at which each element is weighed. This creates a sparse stiffness matrix.

Now consider ELM's reconstruction. In this scenario, the figure visualises 4 nodal activations out of 10 total nodes, since the activation is globally supported, that is it influences the whole domain, the activation is non-zero for all points in the domain. In the figure the activations look like straight lines, but rather they are highly stretched sigmoid shapes. It takes a bit of abstract perspective, but it is possible to see that by summing these activations, the PDE is reconstructed. The summed reconstruction is visible, matching closely to the analytical solution. Therefore, due to the global influence of each node, it would be expected that the stiffness matrix is dense.

# Performance Testing Results

Following the problem definition, to better understand the practical implications of the mathematical differences between the methods, a simple 1D boundary-value Poisson problem was used. The forcing function of the equation was used to expose the methods to varying conditions. They were tested using 3 different forcings.

1. Polynomial Forcing, used to study convergence properties
2. Variable Frequency Forcing, used to evaluate the ability of each method to capture rapidly changing properties
3. Multi-scale forcing, used to asses performance on problems exhibiting multi-scale phenomena


## Testing on a Polynomial Forcing Function

{% include figure.liquid 
   loading="eager" 
   path="assets/img/femvspinn/ploy_htype_conv_fixed.svg" 
   class="img-fluid rounded z-depth-1"
   caption="Figure 3: The convergence solutions of FEM and ELM"
%}

Figure 3 shows the scores of; the FEM model, the ELM model where the number of domain points are equal to the number of hidden dimensions mimicking a Galerkin style, and a series of ELMs with fixed hidden dimensions. Both the FEM and ELM converge to highly accurate results for the polynomial forcing function; however, ELM has a sharper convergence profile compared to FEM's smoother convergence. The ELM convergence appears to be highly oscillatory at approximately float32 machine precision, which could be attributed to numerical errors which may be of the same order of magnitude as the error itself. In this 1D case, both methods have similair computational costs, however with an increase in dimension it would be expected that the stiffness matrices rapidly increase in size. FEM's construction creates a sparse stiffness matrix, the sparseness of the matrix can be exploited for efficient matrix storage and inversion whereas, ELM's construction creates a dense stiffness matrix, requiring higher memory and computational demands. Therefore, The scalability of the ELM case potentially requires further work to alleviate the negatives of inverting and storing dense matrices.


## Testing the High Frequency Response

{% include figure.liquid 
    loading="eager" 
    path="assets/img/femvspinn/highfreq3.svg" 
    class="img-fluid rounded z-depth-1"
    caption="Figure 4: The high frequency response of FEM, standard ELM and an ELM having the initialised weights appropriately rescaled" 
%}

Figure 4 shows the response of FEM and ELM to high-frequency forcing functions. The figures show that for high frequencies, FEM was highly accurate, however, the ELM decreased in accuracy with an increase in frequency. Consequently, failing to converge to the high frequency soluition. The ELM solution seems to converge to a lower frequency solution, this apparent convergence to a lower frequency solution hints at the inherent spectral bias of PINNs, where it is well known that PINNs converge quicker to lower frequency solutions[18]. To overcome the convergence of the ELM to the lower frequency solution, the first layer weights can be adjusted to contract or extend the sigmoid activation as needed. The test was therefore repeated with the scalar weighting of the first layer initialisation adjusted for each test. The trace "ELM - scaled" in figure 4, shows that manually influencing the sigmoid activation enabled the ELM to converge to the high frequency solution. This example shows the simplest form of spectral bias that a PINN could experience being solved through manual adjustments of the initialised weights. Overcoming the spectral bias through weight adjustment puts into question whether PINNs inherently have spectral bias, or rather it is a result of the optimisers used, highlighting an important research question which could potentially be considered for future works. 

Although manually adjusting the weights was successful in the case of hard-constrained ELM, the scalar multiplier is abstract and sensitive, and small variations of the scalar were observed to highly influence the accuracy of the ELM. Therefore, for unseen cases where the analytical solution is not necessarily known, there exists a strong possibility of converging to an incorrect solution, especially when high frequencies or rapid gradient changes exist in the solution. Therefore, the high frequency tests display that without a priori knowledge of the exact solution, the use of ELMs for unknown solutions is difficult. This observation complements what was observed in Sikora et al.'s[13] study; where for the PINN to accurately solve the advection-diffusion problem, a non-uniform collocation density refined around the boundary layer was required. Both situations confirming that knowledge of the solution itself is required for PINNs to be accurate. Contrasting the ELM, FEM worked in the high frequency tests due to the locally supported nature of its formulation, where FEM "learns" in the reduced domain at which the element is constrained to act. Moseley et. al[5] describe FEM's locally supported nature and its improved performance in more detail, and use it to inspire an attempt to overcome PINNs inaccuracy when dealing with high frequencies through the use of finite basis PINNs. Finite basis PINNs may be described as a PINN which is trained through; breaking down the domain into subdomains - synonymous to finite elements, training PINNs on each subdomain, and then assembling the smaller PINNs.

## Testing on a Composite Frequency Response

{% include figure.liquid 
    loading="eager" 
    path="assets/img/femvspinn/compfreq.svg" 
    class="img-fluid rounded z-depth-1" 
    caption="Figure 5: The multi-scale response of FEM, a standard ELM and an ELM with appropriate scaling of the initialised parameters"
%}

Figure 5 shows the results of FEM and ELM in the composite frequency test. Similarly to what was observed in the high frequency test; FEM was able to capture the solution profile, whereas ELM appeared to converge to the lower-frequency solution. As a result, rescaling the weightings to enable the ELM to better reconstruct high frequency features was done. The result of the rescaling is visible in the "ELM - scaled" trace in figure 5. The trace shows that with suitable scaling of the initialised weights, the ELM can be adapted to capture higher frequency profiles even in a multi-scale task. This test displays another instance of how the model's apparent spectral bias can be shifted through intelligent weighting. 


# Conclusions and Recommendations for Future Works

The research showed that with an analysis of the mathematics behind the methods, the FEM, PIELMs and by consequence, PINNs, are not too dissimilar to one another. In fact, PIELM mathematically can be viewed from a FEM lens, where the trial functions are a globally supported activation, and the test functions are a shifted Dirac delta function. Therefore, PIELM can be interpreted as a form of globally supported FEM. This observation highlights that although numerically FEM and PINN have differences, more emphasis should be placed in understanding the similarities between the methods rather than their differences.

## Future Work Recommendations

Rather than focusing on the differences of FEM and PINNs and viewing scientific machine learning and numerical modelling as two distinct fields, more research should focus on highlighting their similarities and learning from each other. Therefore, based on the above research I have two main recommendations: (1) Further work developing the mathematical equivalences between FEM and PINNs, this includes broadening problem classes, increasing dimensionality, convergence analyses and comparison to SOTA FEM amongst others, and (2) Improvements to SciML models based on FEM and vice versa.

# References

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal ofComputational Physics, vol. 378, pp. 686–707, Feb. 2019. doi: 10.1016/j.jcp.2018.10.045.

[2] P.-Y. Chuang and L. A. Barba, “Experience report of physics-informed neural networks in fluid simulations:Pitfalls and frustration,” in Proceedings of the 21st Python in Science Conference, SciPy, May 2022, pp

[3] P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell, Challenges in training pinns: A loss landscapeperspective, Jun. 2024. [Online]. Available: http://arxiv.org/abs/2402.01868.

[4] G. B. Huang, Q. Y. Zhu, and C. K. Siew, “Extreme learning machine: Theory and applications,” Neurocomputing,
vol. 70, no. 1–3, pp. 489–501, Dec. 2006, issn: 0925-2312. doi: 10.1016/J.NEUCOM.2005.12.126.

[5] B. Moseley, A. Markham, and T. Nissen-Meyer, “Finite basis physics-informed neural networks (fbpinns):
A scalable domain decomposition approach for solving differential equations,” Advances in Computational
Mathematics, vol. 49, no. 4, p. 62, 2023, issn: 1572-9044. doi: 10.1007/s10444-023-10065-9.

[6] V. Gopakumar, S. Pamela, and D. Samaddar, “Loss landscape engineering via data regulation on pinns,” Machine Learning with Applications, vol. 12, p. 100 464, Jun. 2023, issn: 2666-8270. doi: 10.1016/J.MLWA.2023.100464. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2666827023000178.

[7] B. A. Szabo and A. K. Mehta, “P-convergent finite element approximations in fracture mechanics,” International Journal for Numerical Methods in Engineering, vol. 12, no. 3, pp. 551–560, 1978. doi: https://doi.org/10.1002/nme.1620120313. eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/nme.1.[Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620120313.2. 

[8] I. Babuvška and W. C. Rheinboldt, “Error estimates for adaptive finite element computations,” SIAM Journal on Numerical Analysis, vol. 15, no. 4, pp. 736–754, 1978. doi: 10.1137/0715049. eprint: https: //doi.org/10.1137/0715049. [Online]. Available: https://doi.org/10.1137/0715049.

[9] G. Karniadakis and S. Sherwin, “Spectral/hp element methods for computational fluid dynamics,” in Jun. 2005. doi: 10.1093/acprof:oso/9780198528692.001.0001.

[10] Z. Hu, K. Shukla, G. E. Karniadakis, and K. Kawaguchi, “Tackling the curse of dimensionality with physicsinformed neural networks,” Neural Networks, vol. 176, Aug. 2024, issn: 18792782. doi: 10.1016/j.neunet.
2024.106369. [Online]. Available: http://arxiv.org/abs/2307.12306%20http://dx.doi.org/10.1016/j.
neunet.2024.106369.

[11] P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell, Challenges in training pinns: A loss landscape perspective, Jun. 2024. [Online]. Available: http://arxiv.org/abs/2402.01868.     

[12] T. G. Grossmann, U. J. Komorowska, J. Latz, and C.-B. Schönlieb, Can physics-informed neural networks
beat the finite element method? preprint, 2023.

[13] M. Sikora, P. Krukowski, A. Paszyńska, and M. Paszyński, “Comparison of physics informed neural networks and finite element method solvers for advection-dominated diffusion problems,” Journal of Computational Science, vol. 81, p. 102 340, Sep. 2024, issn: 1877-7503. doi: 10.1016/J.JOCS.2024.102340. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1877750324001339

[14] V. Dwivedi and B. Srinivasan, Physics informed extreme learning machine (pielm) – a rapid method for the numerical solution of partial differential equations, 2019. [Online]. Available: http://arxiv.org/abs/1907.03507.

[15] J. Tang, C. Deng, and G.-B. Huang, “Extreme learning machine for multilayer perceptron,” IEEE Transactions
on Neural Networks and Learning Systems, vol. 27, no. 4, pp. 809–821, 2016. doi: 10.1109/TNNLS.
2015.2424995.

[16] G. A. Kale and C. Karakuzu, “Multilayer extreme learning machines and their modeling performance on dynamical systems,” Applied Soft Computing, vol. 122, p. 108 861, 2022, issn: 1568-4946. doi: https:// doi.org/10.1016/j.asoc.2022.108861.

[17] C. M. Wong, C. M. Vong, P. K. Wong, and J. Cao, “Kernel-based multilayer extreme learning machines for representation learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 3, pp. 757–762, 2018. doi: 10.1109/TNNLS.2016.2636834.

[18] N. Rahaman, A. Baratin, D. Arpit, et al., “On the spectral bias of neural networks,” 36th International Conference on Machine Learning, ICML 2019, vol. 2019-June, pp. 9230–9239, Jun. 2018. [Online]. Available:https://arxiv.org/pdf/1806.08734.
