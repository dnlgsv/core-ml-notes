---
layout: default
title: Maximum A Posteriori Estimation
parent: Introduction
nav_order: 3
---

# Maximum A Posteriori Estimation

Maximum A Posteriori (MAP) estimation is the Bayesian extension of Maximum Likelihood Estimation (MLE). MLE chooses the parameters that make the observed data most likely. MAP chooses the parameters that are most plausible after combining the data with prior beliefs about the parameters.

The idea is simple:

> MLE asks: which parameter best explains the data?
>
> MAP asks: which parameter best explains the data while also agreeing with the prior?

This is exactly why regularization appears so naturally in machine learning. A penalty term in an optimization problem can often be interpreted as a prior belief over the parameters.

## Intuition: likelihood plus prior

Suppose a coin lands heads with probability $\theta$. You toss it $n$ times and observe $k$ heads.

MLE estimates

$$
\hat{\theta}_{\text{MLE}} = \frac{k}{n}
$$

If $n = 10$ and $k = 7$, MLE gives $0.7$. But what if the sample is very small? If you toss a coin once and get heads, MLE says $\hat{\theta}=1$. That is usually too extreme.

MAP lets us express a prior belief before seeing the data. For example, we may believe that coins are usually not extremely biased. The prior then pulls the estimate away from unstable extremes.

<iframe src="MAP/map_coin_prior.html" width="100%" height="590" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

The posterior combines two forces:

- The **likelihood** says which values of $\theta$ explain the observed data.
- The **prior** says which values of $\theta$ were plausible before seeing the data.

With more data, the likelihood becomes sharper and usually dominates the prior. With little data, the prior can matter a lot.

## Formal definition

Let the dataset be $\mathcal{D} = \{x_1, \dots, x_n\}$ and let the model depend on parameters $\theta$.

Bayes' rule gives the posterior distribution:

$$
P(\theta \mid \mathcal{D}) =
\frac{P(\mathcal{D} \mid \theta)P(\theta)}{P(\mathcal{D})}
$$

The denominator $P(\mathcal{D})$ does not depend on $\theta$. Therefore, when we optimize over $\theta$, we can write

$$
P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)P(\theta)
$$

The **maximum a posteriori estimate** is

$$
\hat{\theta}_{\text{MAP}}
= \arg\max_{\theta} P(\theta \mid \mathcal{D})
= \arg\max_{\theta} P(\mathcal{D} \mid \theta)P(\theta)
$$

In log form:

$$
\hat{\theta}_{\text{MAP}}
= \arg\max_{\theta}
\left[
\log P(\mathcal{D} \mid \theta) + \log P(\theta)
\right]
$$

This equation is the key bridge between Bayesian estimation and regularized optimization.

## MAP for a coin with a Beta prior

For a Bernoulli coin model, the likelihood is

$$
P(\mathcal{D} \mid \theta)
= \theta^k(1-\theta)^{n-k}
$$

A common prior for $\theta$ is the Beta distribution:

$$
P(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

Here $\alpha$ and $\beta$ control the prior belief:

- Larger $\alpha$ favors heads.
- Larger $\beta$ favors tails.
- $\alpha=\beta=1$ is uniform, so MAP becomes very close to MLE.
- $\alpha=\beta>1$ favors values near $0.5$.

The unnormalized posterior is

$$
P(\theta \mid \mathcal{D})
\propto
\theta^k(1-\theta)^{n-k}
\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

Combine powers:

$$
P(\theta \mid \mathcal{D})
\propto
\theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1}
$$

So the posterior is also a Beta distribution:

$$
\theta \mid \mathcal{D}
\sim
\operatorname{Beta}(k+\alpha,\ n-k+\beta)
$$

When $k+\alpha>1$ and $n-k+\beta>1$, the MAP estimate is the mode:

$$
\boxed{
\hat{\theta}_{\text{MAP}}
=
\frac{k+\alpha-1}{n+\alpha+\beta-2}
}
$$

This looks like MLE with extra pseudo-observations from the prior. For example, a symmetric prior with $\alpha=\beta=2$ behaves like one extra head and one extra tail.

## MAP and regularization

Many machine learning objectives can be read as MAP estimation. Suppose the model parameters are $\mathbf{w}$. MAP maximizes

$$
\log P(\mathcal{D} \mid \mathbf{w}) + \log P(\mathbf{w})
$$

Equivalently, minimizing the negative log-posterior gives

$$
-\log P(\mathcal{D} \mid \mathbf{w}) - \log P(\mathbf{w})
$$

The first term is the data loss. The second term becomes a regularization penalty.

<iframe src="MAP/map_regularization_bridge.html" width="100%" height="520" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

### Gaussian prior leads to L2 regularization

Assume a zero-mean Gaussian prior on the weights:

$$
P(\mathbf{w}) \propto
\exp\!\left(-\frac{\lambda}{2}\|\mathbf{w}\|_2^2\right)
$$

Then

$$
-\log P(\mathbf{w})
=
\frac{\lambda}{2}\|\mathbf{w}\|_2^2 + C
$$

So MAP estimation becomes likelihood-based training plus an L2 penalty.

### Laplace prior leads to L1 regularization

Assume a zero-mean Laplace prior:

$$
P(\mathbf{w}) \propto
\exp(-\lambda\|\mathbf{w}\|_1)
$$

Then

$$
-\log P(\mathbf{w})
=
\lambda\|\mathbf{w}\|_1 + C
$$

So MAP estimation becomes likelihood-based training plus an L1 penalty. This prior puts more mass near zero and encourages sparse parameters.

## MAP for a Gaussian mean

Suppose the data are generated as

$$
x_i \sim \mathcal{N}(\mu, \sigma^2)
$$

and assume $\sigma^2$ is known. MLE estimates $\mu$ using the sample mean:

$$
\hat{\mu}_{\text{MLE}} = \bar{x}
$$

Now place a Gaussian prior on $\mu$:

$$
\mu \sim \mathcal{N}(\mu_0, \tau^2)
$$

The MAP estimate becomes a weighted average of the data mean and the prior mean:

$$
\hat{\mu}_{\text{MAP}}
=
\frac{\frac{n}{\sigma^2}\bar{x} + \frac{1}{\tau^2}\mu_0}
{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}
$$

This formula is worth reading slowly:

- If $n$ is large, the data term dominates.
- If the prior variance $\tau^2$ is small, the prior is very confident and pulls the estimate toward $\mu_0$.
- If the prior is weak, meaning $\tau^2$ is large, MAP approaches MLE.

## MAP vs MLE vs full Bayesian inference

MAP is Bayesian in the sense that it uses a prior and a posterior. But it still returns a single point estimate.

Full Bayesian inference keeps the whole posterior distribution:

$$
P(\theta \mid \mathcal{D})
$$

That distribution represents uncertainty over parameter values. MAP only takes its highest point:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid \mathcal{D})
$$

This makes MAP computationally convenient, but it can hide uncertainty. If the posterior is wide, skewed, or multimodal, the single MAP estimate may be a poor summary.

## Key properties of MAP

MAP is useful because it keeps the optimization flavor of MLE while adding a way to encode prior information.

- **Prior-sensitive with small data**: when the dataset is small, the prior can strongly affect the estimate.
- **MLE as a special case**: with a uniform or very weak prior, MAP often reduces to MLE.
- **Regularization interpretation**: common penalties such as L1 and L2 correspond to log-priors.
- **Not invariant like MLE**: the MAP estimate can change under reparameterization because probability densities transform with Jacobian factors.
- **Point estimate only**: MAP does not preserve the full uncertainty contained in the posterior.

## Summary table

| Question | Answer |
| --- | --- |
| What does MAP maximize? | $P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta)P(\theta)$ |
| How is MAP related to MLE? | MAP is MLE plus a prior term |
| What happens with a uniform prior? | MAP usually becomes MLE |
| Why use the log-posterior? | Products become sums and optimization is easier |
| Where does L2 regularization come from? | A Gaussian prior on parameters |
| Where does L1 regularization come from? | A Laplace prior on parameters |
| What is the main limitation? | MAP gives one point estimate and can hide posterior uncertainty |

## Self-check questions

Use these questions to check whether the main ideas are clear before moving on.

<iframe src="MAP/map_self_check_questions.html" width="100%" height="430" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

## Common interview questions

These are the versions of MAP questions that often appear in machine learning and statistics interviews.

<iframe src="MAP/map_interview_questions.html" width="100%" height="540" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

## Where these ideas reappear

- **Regularized regression**: ridge regression is MAP with a Gaussian prior; lasso is MAP with a Laplace prior.
- **Bayesian machine learning**: MAP is often used when the full posterior is too expensive to compute.
- **Naive Bayes and smoothing**: additive smoothing can be interpreted through Dirichlet priors.
- **Neural network weight decay**: L2 weight decay is closely related to a Gaussian prior on weights.
- **Small-data modeling**: priors help stabilize estimates when likelihood alone is too noisy.
- **Probabilistic graphical models**: MAP inference often means finding the most likely configuration of hidden variables.
