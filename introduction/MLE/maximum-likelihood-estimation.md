---
layout: default
title: Maximum Likelihood Estimation
parent: Introduction
nav_order: 2
---

# Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is one of the central ideas in statistics and machine learning. The idea is simple: MLE chooses the parameters that make the data we actually observed most likely under the model.

If you toss a coin 10 times and observe 7 heads, a reasonable guess is that the probability of heads is close to 0.7. MLE turns that intuition into a precise optimization problem.

## Intuition: fit the parameter to the data

Suppose a coin lands heads with probability $\theta$. After $n$ tosses, you observe $k$ heads. MLE asks:

> Which value of $\theta$ makes this exact outcome most likely?

The interactive chart below shows that for $n = 10$ and $k = 7$, the likelihood peaks at $\theta = 0.7$. If you keep the same ratio but increase the sample size, for example $n = 20$ and $k = 14$, the peak stays in the same place while the curve becomes narrower. More data means more certainty.

<iframe src="mle_coin_intuition.html" width="100%" height="560" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

There is one subtle but important naming convention here. When $\theta$ is fixed, $P(\mathcal{D} \mid \theta)$ is the probability of the data. When the observed data $\mathcal{D}$ are fixed and we vary $\theta$, the same expression is viewed as a function of the parameter. That function is called the **likelihood**.

## Formal definition

Let the dataset be $\mathcal{D} = \{x_1, \dots, x_n\}$ and let the model depend on parameters $\theta$.

The **likelihood function** is the probability, or probability density for continuous data, of observing the data under those parameters:

$$
\mathcal{L}(\theta) = P(\mathcal{D} \mid \theta) = \prod_{i=1}^{n} P(x_i \mid \theta)
$$

The product appears because we usually assume the observations are independent.

The **maximum likelihood estimate** is the parameter value that maximizes this function:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \mathcal{L}(\theta)
$$

## Why we use the log-likelihood

In practice, multiplying many small probabilities quickly produces numbers that underflow to zero in floating-point arithmetic. That is why we maximize the log-likelihood instead:

$$
\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log P(x_i \mid \theta)
$$

This helps for two reasons:

- The logarithm is monotonic, so it does not change the location of the maximum.
- Products become sums, which are easier to analyze, differentiate, and optimize.

<iframe src="likelihood_vs_loglikelihood.html" width="100%" height="560" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

## MLE for a coin: analytic derivation

For a coin with $k$ heads in $n$ tosses, the Bernoulli log-likelihood is

$$
\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)
$$

This is the log-likelihood of one particular sequence with $k$ heads and $n-k$ tails. If we instead ask for the probability of seeing exactly $k$ heads in any order, the likelihood includes the binomial coefficient $\binom{n}{k}$. That coefficient does not depend on $\theta$, so it does not affect which value of $\theta$ maximizes the likelihood.

Differentiate with respect to $\theta$ and set the derivative to zero:

$$
\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0
$$

Solve for $\theta$:

$$
\frac{k}{\theta} = \frac{n-k}{1-\theta}
$$

$$
k(1-\theta) = (n-k)\theta
$$

$$
k - k\theta = n\theta - k\theta
$$

$$
k = n\theta
$$

$$
\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}}
$$

So the familiar sample proportion is not just intuition. It is the exact maximum likelihood solution.

## MLE for a Gaussian distribution

Now consider continuous data such as heights, weights, or measurement errors. Assume the data are independently sampled from a Gaussian distribution:

$$
x_i \sim \mathcal{N}(\mu, \sigma^2)
$$

The density of one observation is

$$
P(x_i \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

For continuous variables, this value is a probability density, not the probability of observing exactly one point. Exact points have probability zero in continuous distributions. MLE still works the same way: it chooses the parameters that give the observed sample the highest joint density.

For the full dataset, the log-likelihood is

$$
\ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$

Maximizing with respect to $\mu$ gives

$$
\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

Maximizing with respect to $\sigma^2$ gives

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
$$

That is why the sample mean naturally appears as the MLE of the Gaussian mean. Notice that the MLE variance uses $1/n$, not $1/(n-1)$. The latter is the unbiased estimator, which is a different objective.

<iframe src="mle_gaussian_fitting.html" width="100%" height="620" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

With small samples, the fitted curve can differ noticeably from the true distribution. As the sample size grows, the estimate stabilizes and approaches the true parameter values. This is one expression of **consistency**.

## From MLE to common loss functions

MLE is not an isolated topic. It directly explains why several standard machine learning losses look the way they do.

### Logistic regression and cross-entropy

For binary classification, each label $y_i \in \{0, 1\}$ is modeled as a Bernoulli random variable with predicted probability $\hat{p}_i$:

$$
P(y_i \mid x_i, \mathbf{w}) = \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}
$$

The log-likelihood over the dataset is

$$
\ell(\mathbf{w}) = \sum_{i=1}^{n} \left[y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i)\right]
$$

If we negate this expression, maximizing likelihood becomes minimizing a loss. That loss is exactly **binary cross-entropy**.

### Linear regression and mean squared error

If a regression model assumes Gaussian noise around the prediction,

$$
y_i = f(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

then maximizing the Gaussian likelihood is equivalent to minimizing the sum of squared errors. This is the probabilistic origin of **mean squared error (MSE)**.

![From maximum likelihood estimation to cross-entropy](mle_to_cross_entropy_chain.svg)

## Key properties of MLE

MLE is popular because, under standard regularity assumptions, it has several strong asymptotic properties:

These are long-run guarantees. They explain why MLE is so widely used, but they do not mean every small dataset gives an unbiased or highly accurate estimate.

- **Consistency**: as $n \to \infty$, the estimate converges to the true parameter.
- **Asymptotic normality**: for large samples, the estimate is approximately Gaussian around the true value.
- **Asymptotic efficiency**: it achieves the lowest possible variance among a broad class of estimators.
- **Invariance**: if $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ for a transformation $g$.

<iframe src="mle_properties_stepper.html" width="100%" height="520" style="border: 1px solid #d0d7de; border-radius: 12px; background: #fff;" loading="lazy"></iframe>

## Summary table

| Question | Answer |
| --- | --- |
| What do we maximize? | $\mathcal{L}(\theta) = \prod_i P(x_i \mid \theta)$ |
| Why use a logarithm? | Numerical stability and easier optimization |
| How do we solve it? | Analytically when possible, otherwise with gradient-based optimization |
| Where does cross-entropy come from? | MLE for a Bernoulli model |
| Where does mean squared error come from? | MLE for Gaussian noise |
| How does regularization fit in? | It appears naturally in Maximum A Posteriori (MAP) estimation through a prior |

## Self-check questions

Use these questions to check whether the main ideas are clear before moving on.

<details>
<summary>1. What is the difference between probability and likelihood?</summary>

Probability treats the parameters as fixed and asks how likely different data outcomes are. Likelihood treats the observed data as fixed and asks which parameter values make those data most plausible.
</details>

<details>
<summary>2. Why does MLE for a coin give $\hat{\theta} = k/n$?</summary>

The likelihood is maximized when the model's predicted probability of heads matches the observed fraction of heads. Differentiating the Bernoulli log-likelihood gives $k/\theta - (n-k)/(1-\theta) = 0$, which solves to $\hat{\theta} = k/n$.
</details>

<details>
<summary>3. Why can we maximize log-likelihood instead of likelihood?</summary>

The logarithm is monotonic, so it does not change the location of the maximum. It also turns products into sums, which are easier to compute, differentiate, and optimize.
</details>

<details>
<summary>4. Why is negative log-likelihood used as a loss?</summary>

Optimization libraries usually minimize objectives. MLE is a maximization problem, so we multiply the log-likelihood by $-1$ and minimize the negative log-likelihood instead.
</details>

<details>
<summary>5. Why does Gaussian MLE lead to mean squared error?</summary>

Under Gaussian noise with fixed variance, the log-likelihood contains a negative squared-error term. Maximizing that log-likelihood is therefore equivalent to minimizing the sum of squared errors.
</details>

<details>
<summary>6. Why does the MLE variance use $1/n$ instead of $1/(n-1)$?</summary>

MLE chooses the parameter that maximizes likelihood, not the estimator with zero finite-sample bias. The $1/(n-1)$ version is the unbiased sample variance, which optimizes a different criterion.
</details>

## Common interview questions

These are the versions of MLE questions that often appear in machine learning and statistics interviews.

<details>
<summary>1. What assumptions are usually made when writing the likelihood as a product?</summary>

The standard assumption is that observations are independent, often independent and identically distributed (i.i.d.). Independence lets us write the joint likelihood as $\prod_i P(x_i \mid \theta)$.
</details>

<details>
<summary>2. Is likelihood a probability distribution over parameters?</summary>

No. In frequentist MLE, the likelihood is a function of the parameter, but it is not a probability distribution over the parameter. To put a probability distribution over parameters, we need a Bayesian prior and posterior.
</details>

<details>
<summary>3. What is the difference between MLE and MAP?</summary>

MLE maximizes $P(\mathcal{D} \mid \theta)$. MAP maximizes $P(\theta \mid \mathcal{D})$, which is proportional to $P(\mathcal{D} \mid \theta)P(\theta)$. MAP adds a prior, and that prior often acts like regularization.
</details>

<details>
<summary>4. Why is cross-entropy the loss for logistic regression?</summary>

Logistic regression models labels as Bernoulli random variables. The negative Bernoulli log-likelihood is exactly binary cross-entropy.
</details>

<details>
<summary>5. Why is MSE connected to Gaussian noise?</summary>

If the target is modeled as $y_i = f(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0,\sigma^2)$, the Gaussian negative log-likelihood is proportional to the sum of squared residuals, plus constants.
</details>

<details>
<summary>6. What can go wrong with MLE?</summary>

MLE can be biased in small samples, sensitive to outliers, and unstable when the model is misspecified. In some models, the likelihood may also be unbounded or have multiple local maxima.
</details>

<details>
<summary>7. Why do we often minimize average negative log-likelihood instead of total negative log-likelihood?</summary>

Averaging by $n$ does not change the minimizer, but it makes the scale of the loss less dependent on dataset size. This is useful for reporting, comparing runs, and choosing learning rates.
</details>

<details>
<summary>8. What does it mean if two parameter values have similar likelihood?</summary>

It means the observed data do not strongly distinguish between those parameter values. With more informative data, the likelihood curve usually becomes sharper around the best estimate.
</details>

## Where these ideas reappear

- **Regression losses**: linear regression with Gaussian noise leads to mean squared error.
- **Classification losses**: logistic regression and softmax classification lead to binary and categorical cross-entropy.
- **Probabilistic models**: Naive Bayes, Gaussian Mixture Models, Hidden Markov Models, probabilistic PCA, and factor analysis all use likelihood-based fitting.
- **Latent-variable models**: Expectation-Maximization (EM) maximizes likelihood when some variables are not directly observed.
- **Regularized estimation**: Maximum A Posteriori (MAP) extends MLE by adding a prior, which links probabilistic estimation to L1 and L2 regularization.
- **Deep learning and generative modeling**: many standard losses are negative log-likelihoods under a chosen output distribution, while models such as autoregressive models and normalizing flows optimize likelihood directly.
