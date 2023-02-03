---
layout: post
title: Statistical Distributions
date: 2023-02-01 11:59:00-0400
description:
# tags: DL
categories: Bayesian, Markov
giscus_comments: true
---
*TOC*

* TOC
{:toc}

*Reference:*

Mainly collected from Wikipedia.

# Basic knowledge

1. Cumulative distribution function (**CDF**)

    The cumulative distribution function of a real-valued random variable $$ X $$ is the function given by

    $$
    F_X (x) = P(X \leq x)
        $$

2. Probability density function (**PDF**)

    A function that defines the relationship between continuous random variables and their probabilities. If the random variables are discrete, we call it **Probability mass function (PMF)**.

3. Law of total probability

   $$
   P(A) = \sum_n P(A \cap B_n) \text{ or } P(A) = \sum_n P(A \mid B_n) P(B_n)
   $$

4. Gamma function

    For every positive integer $$n$$

    $$
    \Gamma (n) = (n-1) !
    $$

5. Bayesian statistics

    $$
    \text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
    $$

    Given a prior belief that a PDF is $$p(\theta)$$ and that the observations $$x$$ have a likelihood $$p(x \mid \theta )$$, then the posterior probability is defined as

    $$
    p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)}
    $$

    where $$p(\theta)$$ is the normalizing constant and is calculated as

    $$
    p(x) = \int p(x\mid\theta) p(\theta) d(\theta)
    $$

6. Conjugate prior

    In Bayesian probability, if the posterior distribution $$ p ( \theta \mid x ) $$ is in the same probability distribution family as the prior probability distribution $$ p(\theta ) $$, the **prior** and **posterior** are then called **conjugate distributions**, and the prior is called a **conjugate prior** for the likelihood function $$ p(x\mid \theta ) $$.

# Common distributions

## Symmetric probability distribution

A probability distribution is said to be symmetric if and only if there exists a value $$ x_{0} $$ such that

$$
    f(x_{0}-\delta )=f(x_{0}+\delta )
$$

for all real numbers $$\delta$$ , where $$f$$ is the PDF if the distribution is continuous or the probability mass function if the distribution is discrete.

## Uniform distribution

Notation: $$U[a,b]$$

PDF:

$$
f(x)={\begin{cases}{\frac {1}{b-a}}&\mathrm {for} \ a\leq x\leq b,\\[8pt]0&\mathrm {for} \ x<a\ \mathrm {or} \ x>b\end{cases}}
$$

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Uniform_Distribution_PDF_SVG.svg" alt="PDF of continuous uniform distribution" width="300pt"/>
<figcaption>PDF of continuous uniform distribution</figcaption>
</figure>

CDF:

$$
f(x)={\begin{cases}0&{\text{for }}x<a\\[8pt]{\frac {x-a}{b-a}}&{\text{for }}a\leq x\leq b\\[8pt]1&{\text{for }}x>b\end{cases}}
$$

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/6/63/Uniform_cdf.svg" alt="CDF of continuous uniform distribution" width="300pt"/>
<figcaption>CDF of continuous uniform distribution</figcaption>
</figure>

## Binomial distribution

Notation: $$B(n,p)$$

The binomial distribution with parameters $$n$$ and $$p$$ is the discrete probability distribution of the number of successes in a sequence of $$n$$ independent experiments, each asking a yes–no question, and each with its own Boolean-valued outcome: success (with probability $$p$$) or failure (with probability $$q = 1 − p$$).

PMF:

$$
{\displaystyle f(k,n,p)=\Pr(k;n,p)=\Pr(X=k)={\binom {n}{k}}p^{k}(1-p)^{n-k}} \text{}
$$

for k = 0, 1, 2, ..., n, where

$$
{\displaystyle {\binom {n}{k}}={\frac {n!}{k!(n-k)!}}}
$$

is the binomial coefficient. The formula can be understood as follows: $$k$$ successes occur with probability $$p^k$$ and $$n − k$$ failures occur with probability $$(1-p)^{n-k}$$. However, the $$k$$ successes can occur anywhere among the $$n$$ trials, and there are $${\tbinom {n}{k}}$$ different ways of distributing $$k$$ successes in a sequence of $$n$$ trials.

CDF:

A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment, and a sequence of outcomes is called a Bernoulli process; for a single trial, i.e., $$n = 1$$, the binomial distribution is a [Bernoulli distribution](#bernoulli-distribution).

## Bernoulli distribution

Bernoulli distribution is the **discrete** probability distribution of a random variable which takes the value $$1$$ with probability $$p$$ and the value $$0$$ with probability $$q=1-p$$.

If $$X$$ is a random variable with this distribution, then:

$$
{\displaystyle \Pr(X=1)=p=1-\Pr(X=0)=1-q.}
$$

The PMF $$f$$ of this distribution, over possible outcomes $$k$$, is

$$
{\displaystyle f(k;p)={\begin{cases}p&{\text{if }}k=1,\\q=1-p&{\text{if }}k=0.\end{cases}}}
$$

or

$$
{\displaystyle f(k;p)=p^{k}(1-p)^{1-k}\quad {\text{for }}k\in \{0,1\}}
$$

or

$$
{\displaystyle f(k;p)=pk+(1-p)(1-k)\quad {\text{for }}k\in \{0,1\}.}
$$

The Bernoulli distribution is a special case of the [binomial distribution](#binomial-distribution) with $$n = 1$$.

## Beta distribution

Notation: $$B(\alpha, \beta)$$

A family of continuous probability distributions defined on the interval $$[ 0 , 1 ]$$ in terms of two positive parameters, denoted by alpha ($$\alpha$$) and beta ($$\beta$$), that appear as exponents of the variable and its complement to $$1$$, respectively, and control the shape of the distribution.

PDF:

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg" alt="PDF of Beta distribution" width="300pt"/>
<figcaption>PDF of Beta distribution</figcaption>
</figure>

CDF:

<figure>
<img src="https://upload.wikimedia.org/wikipedia/commons/1/11/Beta_distribution_cdf.svg" alt="CDF of Beta distribution" width="300pt"/>
<figcaption>CDF of Beta distribution</figcaption>
</figure>

In Bayesian inference, the beta distribution is the [conjugate prior](#basic-knowledge) probability distribution for the Bernoulli, binomial, negative binomial and geometric distributions.
