---
layout: post
title: Diffusion models
date: 2022-12-14 11:59:00-0400
description:
# tags: DL
categories: generative-model deep-learning diffusion-model
giscus_comments: true
---
*TOC*

* TOC
{:toc}

*Reference:*

Official links:

1. DDPM (NeurIPS 2020)

   Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

   Github: [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

   Website: [https://hojonathanho.github.io/diffusion/](https://hojonathanho.github.io/diffusion/)

2. Improved DDPM (ICML 2021)

   Paper: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

   Github: [https://github.com/openai/improved-diffusion](https://github.com/openai/improved-diffusion)

3. Guided Diffusion Models (NeurIPS 2021)

   Paper: [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

   Github: [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)

Blog:

   1. [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
   2. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## Markov chain

Quote from [Wikipedia](https://en.wikipedia.org/wiki/Markov_chain#:~:text=A%20Markov%20chain%20or%20Markov,the%20state%20of%20affairs%20now.%22):


A **Markov chain** or **Markov process** is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. Informally, this may be thought of as, "What happens next depends only on the state of affairs now".


## Main idea of Diffusion Model

Diffusion models works by **destroying training data** through the successive addition if Gaussian noise, and then **learning to recover** the data by reversing this noising process. After training, we can use the Diffusion Model to generate data by simply **passing randomly sampled noise through the learned denoising process**.

### *Forward process* (or *diffusion process*)

Specifically, a Diffusion Model is a latent variable model which maps to the latent space using a fixed Markov chain. This chain gradually adds noise to the data in order to obtain the **approximate posterior** $$q(X_{1:T}\vert X_0)$$, where $$X_1,\ldots,X_T$$ are the latent variables with the same dimensionality as $$X_0$$. See figure below.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image.png" alt="The Markov chain manifested for image data." width="600pt"/>

A parameterization of the forward process (combing Markov assumption):

\begin{equation}
\label{eq:forward-process}
q(X_{1:T} \vert X_0) := \prod^T_{t=1} q(X_t \vert X_{t-1}) := \prod^T_{t=1} \mathcal{N}(X_t;\sqrt{1-\beta_t}X_{t-1},\beta_t\mathbf{I})
\end{equation}

where $$\beta_1,\dots,\beta_T$$ is a variance schedule (either learned or fixed) which, if well-behaved, **ensures that $$X_T$$ is nearly an isotropic Gaussian for sufficiently large $$T$$**.

*Let's skip the proof for now.*

### *Reverse process* (or *reverse diffusion process*)

Ultimately, the image is asymptotically transformed to pure Gaussian noise. The goal of training a diffusion model is to **learn the reverse process**, i.e. training $$p_\theta (X_{t-1} \vert X_t)$$. See figure below, by traversing backwards along this chain, we can generate new data.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-1.png" alt="The reverse process of the Markov chain." width="600pt"/>

Starting with the pure Gaussian noise $$p(X_T)=\mathcal{N}(X_T;\mathbf{0},\mathbf{I})$$, the model learns the joint distribution $$p_\theta(X_0;T)$$ as:

\begin{equation}
p_\theta(X_0;T) := p(X_T) \prod^T_{t=1} p_\theta(X_{t-1}\vert X_t) := p(X_T) \prod^T_{t=1} \mathcal{N}(X_{t-1};\mu_\theta(x_t,t),\varSigma_\theta(X_t,t))
\end{equation}

where the time-dependent parameters of the Gaussian transitions are learned.

## Benefits of Diffusion Models

1. Diffusion Models currently produce State-of-the-Art image quality.
2. Not requiring adversarial training.
3. Scalability and parallelizability.

## Training

A Diffusion Model is trained by **finding the reverse Markov transitions that maximize the likelihood of the training data**. In practice, training equivalently consists of minimizing the variational upper bound on the negative log likelihood.

\begin{equation}
\mathbb{E} [= \log p_\theta (X_0)] \leqslant \mathbb{E}_ {q} [-\log \frac{p_\theta (X_{0:T})}{q(X_{1:T} \vert X_0)}] =: L_{vlb}
\end{equation}

Variational lower bound $$L_{vlb}$$ is technically an upper bound (the negative of the ELBO) which we are trying to minimize. We will try to rewrite the $$L_{vlb}$$ in terms of Kullback-Leibler (KL) Divergences because the transition distributions in the Markov chain are Gaussians, and **the KL divergence between Gaussians has a closed form**.

\begin{equation}
D_{KL}(P\parallel Q) = \int_{-\infty}^{\infty} p(x)\log(\frac{p(x)}{q(x)}) dx
\end{equation}

Casting $$L_{vlb}$$ in terms of KL Divergences

\begin{equation}
L_{vlb} := L_0 + L_1 + \ldots + L_{T-1} + L_T
\end{equation}

where

\begin{equation}
L_0 := -\log p_\theta(X_0 \vert X_1)
\end{equation}

\begin{equation}
L_{t-1} := D_{KL}(q(X_{t-1} \vert X_t,X_0)\parallel p_\theta(X_{t-1} \vert X_t))
\end{equation}

\begin{equation}
L_T := D_{KL}(q(X_T \vert X_0)\parallel p(X_T))
\end{equation}

*Let's skip the proof for now.*

Conditioning the forward process posterior on $$X_0$$ in $$L_{t-1}$$ results in a tractable form that leads to **all KL divergences being comparisons between Gaussians**. This means that the divergences can be exactly calculated with closed-form expressions rather than with Monte Carlo estimates.

## Summary

1. Diffusion Models are **highly flexible** and allow for any architecture whose input and output dimensionality are the same to be used. Many implementations use **U-Net-like** architectures.
2. The **training objective** is to maximize the likelihood of t he training data. This is manifested as tuning the model parameters to **minimize the variational upper bound of the negative log likelihood of the data**.
3. Almost all terms in the objective function can be cast as **KL Divergences** as a result of the Markov assumption. This values **become tenable to calculated** given that we are using Gaussians, therefore omitting the need to perform MonteCarlo approximation.
4. A discrete decoder is used to obatin log likelihoods across pixel values as the last step in the reverse diffusion process.
