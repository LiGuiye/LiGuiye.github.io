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

Mainly collected from Wikipedia and class notes of the class STAT 5100 in CU Boulder.

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

4. Bayesian statistics

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

5. Conjugate prior

    In Bayesian probability, if the posterior distribution $$ p ( \theta \mid x ) $$ is in the same probability distribution family as the prior probability distribution $$ p(\theta ) $$, the **prior** and **posterior** are then called **conjugate distributions**, and the prior is called a **conjugate prior** for the likelihood function $$ p(x\mid \theta ) $$.

6. Manually calculate a p-value

    The p-value for:

    a lower-tailed test is specified by: $$\text{p-value} = P(TS \leq ts \mid H_0 \text{ is true}) = \text{cdf}(ts)$$

    an upper-tailed test is specified by: $$\text{p-value} = P(TS \geq ts \mid H_0 \text{ is true}) = 1 - \text{cdf}(ts)$$

    $$\text{TS}$$ is "Test statistic", $$\text{ts}$$ is the observed value of the test statistic calculated from your sample.

7. Distributions of Certain Sums

   - A Sum of Bernoullis is Binomial
   - A Sum of Binomials is Binomial
   - A Sum of Poissons is Poisson
   - A Sum of Geometrics is Negative Binomial

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

## Bernoulli distribution

Notation: $$Bernoulli(p)$$ or $$Bern(p)$$

Let $$X$$ be a random variable that takes on the values $$1$$ and $$0$$ with respective probabilities $$p$$ and $$1 − p$$. Then $$X$$ is said to have a *Bernoulli distribution with parameter $$p$$*.

$$
{\displaystyle \Pr(X=1)=p=1-\Pr(X=0)=1-q.}
$$

The PMF of this distribution, over possible outcomes $$x$$, is

$$
P(X=x)={
    \begin{cases}
    p &{\text{if }}x=1,\\
    q=1-p &{\text{if }}x=0.\end{cases}}
$$

or

$$
{\displaystyle P(X=x)=p^{x}(1-p)^{1-x}\quad {\text{for }}x\in \{0,1\}}
$$

or

$$
{\displaystyle P(X=x)=px+(1-p)(1-x)\quad {\text{for }}x\in \{0,1\}.}
$$

The mean or expected value of the Bernoulli distribution is

$$
E[X] = \sum_x x P (X = x) = 0 (1 − p) + 1 p = p
$$

$$
E[X^2] = \sum_x x P (X = x) = 0^2 (1 − p) + 1^2 p = p
$$

Then the variance of this distribution is

$$
\text{Var}[X] = E[X^2] − (E[X])^2 = p − p^2 = p(1 − p)
$$

The Bernoulli distribution is a special case of the [binomial distribution](#binomial-distribution) with $$n = 1$$.

## Binomial distribution

Notation: $$B(n,p)$$ or $$bin(n,p)$$

The binomial distribution with parameters $$n$$ and $$p$$ is the discrete probability distribution of *the number of successes in a sequence of $$n$$ independent experiments*, each asking a yes–no question, and each with its own Boolean-valued outcome: success (with probability $$p$$) or failure (with probability $$q = 1 − p$$).

PMF:

$$
P(X=x)={\binom {n}{x}}p^{x}(1-p)^{n-x}
$$

for x = 0, 1, 2, ..., n, where

$$
{\displaystyle {\binom {n}{x}}={\frac {n!}{x!(n-x)!}}}
$$

is the binomial coefficient. The formula can be understood as follows: $$x$$ successes occur with probability $$p^x$$ and $$n − x$$ failures occur with probability $$(1-p)^{n-x}$$. However, the $$x$$ successes can occur anywhere among the $$n$$ trials, and there are $${\tbinom {n}{x}}$$ different ways of distributing $$x$$ successes in a sequence of $$n$$ trials.

$$
E[X] = np
$$

$$
Var[X] = np(1 − p)
$$

A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment, and a sequence of outcomes is called a Bernoulli process; for a single trial, i.e., $$n = 1$$, the binomial distribution is a [Bernoulli distribution](#bernoulli-distribution).

## Negative binomial distribution

Notation: $$\mathrm {NB} (r,\,p)$$ or $$negbin(r,p)$$

$$r > 0$$ — number of successes until the experiment is stopped (integer, but the definition can also be extended to reals)

$$p \in [0,1]$$ — success probability in each experiment (real)

## Poisson distribution

Notation $$Poisson(\lambda)$$

A random variable $$X$$ has a Poisson distribution with parameter $$\lambda > 0$$ if $$X$$ has PDF

$$
\begin{aligned}
P (X = x)
&= {\begin{cases}
\frac{e^{−\lambda}\lambda^x}{x!} &, \quad x = 0, 1, 2, \ldots\\
0  &, \quad \text{otherwise}
\end{cases}} \\
&= \frac{e^{−\lambda}\lambda^x}{x!} I_{\{0,1,2,\dots\}}(x)
\end{aligned}
$$

## Geometric distribution

Notation: $$geom(p)$$

Consider a sequence of independent trials of an experiment where each trial can result in either “Success” ($$S$$) or “Failure” ($$F$$). Let $$0 \leq p \leq 1$$ be the probability of *success* on any one trial.

Definition 1 ("number of trials" model):

Let $$x$$ be the number of trials until the first success. There will be $$x-1$$ failures, each with probability $$1-p$$

PMF:

$$
P (X = x) = (1 − p)^{x−1} p \text{ for } x \text{ in } \{1, 2, 3, \ldots \}
$$

Definition 2 ("number of failures" model):

Let $$x$$ be the number of failures before the first success. There will be $$x$$ failures, each with probability $$1-p$$.

PMF:

$$
P (X = x) = (1 − p)^{x} p \text{ for } x \text{ in } \{0, 1, 2, \ldots \},
$$

## Exponential distribution

Notation: $$exp(\lambda)$$ or $$rate=\lambda$$

Let $$\lambda > 0$$ be a fixed parameter and consider the continuous random variable with PDF

$$
\begin{aligned}
P (X = x)
&= {\begin{cases}
\lambda e^{-\lambda x} &, \quad x > 0\\
0  &, \quad \text{otherwise}
\end{cases}} \\
&= \lambda e^{-\lambda x} I_{(0,\infty)}(x)
\end{aligned}
$$

CDF:

$$
F (x) = P (X \leq x) = \int_o^x \lambda e^{−\lambda u} du = 1 − e^{−λx}
$$

Tail probability:

$$
P (X > x) = 1 − P (X \leq x) = 1 − F (x) = e^{−λx}.
$$

## Gamma distribution

Notation:

- Gamma distribution $$\Gamma(\alpha, \beta)$$
- Gamma function $$\Gamma(\alpha)$$

Let $$\alpha > 0$$ and $$ \beta > 0$$ be fixed parameters and consider the continuous random variable with PDF

$$
\begin{aligned}
f(x)
&= {\begin{cases}
\frac{1}{\Gamma(\alpha)}\beta^{\alpha}x^{\alpha-1}e^{-\beta x} &, \quad x > 0\\
0  &, \quad \text{otherwise}
\end{cases}} \\
&= \frac{1}{\Gamma(\alpha)}\beta^{\alpha}x^{\alpha-1}e^{-\beta x} I_{(0,\infty)}(x)
\end{aligned}
$$

$$\alpha$$ is known as a *shape parameter* and $$\beta$$ is known as an *inverse scale parameter*. The pdf is given in terms of the *gamma function* which is defined, for $$\alpha > 0$$ as

$$
\Gamma(\alpha) = \int_0^{\infty} x^{\alpha -1} e^{-x}dx
$$

Properties of Gamma function:
- For $$\alpha >1$$, $$\Gamma(\alpha) = (\alpha-1)\Gamma(\alpha-1)$$
- If $$n\geq 1$$ is an integer, $$\Gamma(n) = (n-1)!$$
- (Since $$0!=1$$), $$\Gamma(1)=1$$

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

In Bayesian inference, the beta distribution is the [conjugate prior](#basic-knowledge) probability distribution for the [Bernoulli](#bernoulli-distribution), [binomial](#binomial-distribution), [negative binomial](#negative-binomial-distribution) and [geometric distributions](#geometric-distribution).
