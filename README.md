# MiniGPT
Implementation of benchmark autoregressive text generation algorithms at a mini scale.

This repo uses `python 3.11`, `pytorch 2.0+` and my model zoo repo `github/pfrwilson/torchzero`. 

***warning - this repo is not maintained***

## Quickstart
`train.py` is the main script to run. 


## Theory

Autoregressive generative modeling is the driving force behind the current state of the art in generative language modeling. Given a multivariate random variable $$X = (X_1, X_2, ..., X_n), $$ the multivariate probability distribution $P(X)$ can be factored as follows: 
$$P(X) = P(X_1) \prod_{i=2}^{n} P(X_i|X_1, ..., X_{i-1}),$$
Given this factorization, we aim to approximate the distribution by another distribution $Q$ where the individual terms $Q(X_i|X_1, ..., X_{i-1})$ are simple distributions that can be easily sampled. For example, we could have $Q(X_i=x_i|X_1=x_1, ..., X_{i-1}=x_{i-1}) = \mathcal{N}\big(x_i; \mu(x_1, ..., x_{i-1}), \sigma(x_1, ..., x_{i-1})\big)$ (normal distribution) or $Q(X_i=x_i|X_1=x_1, ..., X_{i-1}=x_{i-1}) = \text{Categorical}\big(x_i; p(x_1, ..., x_{i-1}))$ (categorical distribution). As usual, the aim is to define $\mu(x_1, ..., x_{i-1}), \sigma(x_1, ..., x_{i-1})$ or $p(x_1, ..., x_{i-1})$ to be a neural network parameterized by theta, and define a suitable loss function. 

We want to optimize a loss function that will result in the best fit between the data-generating distribution $P$ and its approximation $Q$. We can do so by minimizing the cross-entropy loss $$H(P, Q) = \mathbb{E}_{X\sim P}[ -\log Q(X)].$$
As usual, we approximate this expectation by collecting samples $\{x^{(1)}, x^{(2)}, ..., x^{(N)}\}$ of data from $P(X)$, and using the monte carlo estimate: 
$$H(P, Q)\approx \frac{1}{N} \sum_{i=1}^{N} -\log Q(X=x^{(i)})$$ 
***Note that the amount of data determines how well this approximation works, and, consequently, how well the model will generalize. This is true everywhere in deep learning***. 

If we define a suitable loss function for each term in the sum, we can optimize the whole loss by stochastic gradient descent. 
Let's see how the individual terms break down:
$$ -\log Q(X^{(i)}) = -\log \Big( Q(X_1=x_1^{(i)}) \prod_{j=2}^{n} Q(X_j=x_j^{(i)}|X_1=x_1^{(i)}, ..., X_{j-1}=x_{j-1}^{(i)})\Big)$$
$$= -\log Q(X_1=x_1^{(i)}) + \sum_{j=2}^{n} -\log Q(X_j=x_j^{(i)}|X_1=x_1^{(i)}, ..., X_{j-1}=x_{j-1}^{(i)}) $$
Note that each term in the sum is just the cross entropy on the simplified probability distribution $Q(X_j=x_j^{(i)}|X_1=x_1^{(i)}, ..., X_{j-1}=x_{j-1}^{(i)})$. For example, supposing the categorical case, if our neural network $f_{\theta}$ maps $x_1^{(i)}, ..., x_{j-1}^{(i)}$ to parameters $\mathbf{p}$ of a categorical distribution, each term is the simple categorical cross-entropy loss. Then the gradient for each term is easy to compute with respect to the network parameters $\theta$ through backpropagation, and $\theta$ can be optimized for stochastic gradient descent. If we were considering the normal distribution, we would typically assume that $\sigma$ is fixed to the unit diagonal matrix and let the network $f_{\theta}$ maps $x_1^{(i)}, ..., x_{j-1}^{(i)}$ to $\mu$, and the resulting cross-entropy loss would actually be the mean-squared error. 

Concretely, pseudocode of computing loss on a single training example in the categorical case is: 
```python
loss = 0 
X = train_example_i
criterion = CategoricalCrossEntropy()

p = net(None) # first entry - no inputs
loss += criterion(p, X[0])
for j in range(1, len(X)):
    p = net(X[:j])
    loss += criterion(p, X[j])
```

The generative process now admits a very easy implementation: 
```python
outputs = zeros(n)

# first step - no inputs 
p = net(None)
outputs[0] = sample_categorical(p)

for j in range(1, n): 
    p = net(outputs[:j])
    outputs[j] = sample_categorical(p)
```

## Practical Implementation

To convert the above theory into an implementation of AI text generation, we need a few things: 

1. A way to convert our data (text) to a categorical vocabulary
2. A concrete neural network implementation for $f_\theta$

(1.) is fairly easy - we use the byte pair encoding tokenization which "fits" to a body of text. First, every character in the text has its own token. Next, we iteratively go through the text and find the most common pair of tokens, assign that its own token, and replace instances of that pair in the text with the token. This process is repeated until we have a vocabulary of the desired lengh. The vocabulary depends on the text. For example in the bible, the word "David" is so common that it gets its own token. ChatGPT uses an ENORMOUS vocabulary, but we keep our vocabulary small as our data and nets are small.

(2.) We offer a few implementations here - first, the recurrent neural network which was the de-facto sequence processing model before transformer, and second, the transformer (Vaswani et. al 2017) model which now reigns supreme.

## Results 

After 90+ epochs of training on the bible, the large LSTM model produces such gems as these: 

"27:11 Now the LORD was execute judgment, and delivered me out of all thine enemies."

"17:9 He smote them, and cast them down, some man, and the vail, above upon all them that shout."

Obviously, the bible is not enough data to adequately learn english grammar. But, it learns the style and pacing of biblical text. Interesting...

The transformer model gave such outputs as these: 

"2:23 And I have built up in the field, and took it the locust, and set the baldy with the gate; and bass through the and: and, behold, I will verbring the s water, and safely.

2:24 For amites vexed upon our head; thou hast heard, and said unto them, Son of man, looked thine also own land.

2:25 At not thou fear more than all the people, and tell me, and hast done to them are lifted up in the gates of the coverer.

2:26 And the king’s spear was straightway: and the abomination was toward him, saying, Enos us, and take thee away and cause thee to dry."

It seems to have learned grammar slightly better, but still has many mistakes. The biggest difference I noted here was that the verse numbering actually was in ascending order. The LSTM model completely failed to learn this, and just gave random verse numbers. This highlights the benefits of the self-attention mechanism in the transformer, which can look backwards across many tokens and identify the correct verse number to step it by one. 

## Next directions

To improve these models, we already know what is needed. More data, more compute, bigger model.