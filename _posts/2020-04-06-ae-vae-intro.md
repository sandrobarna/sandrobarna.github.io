---
layout: post
title: Intuitive Review of Autoencoders and Variational Autoencoders
tags: [deep learning, generative modelling, autoencoders, variational autoencoders]
---

This is my first post where I talk about Autoencoder and Variational Autoencoder neural networks as well as some important facts about these models. Emphasis will be put on intuitive aspects rather than theory and math. The motivation came from my recent interest and involvement with these models which includes two of my workshops on the topic: The first one at the [DataFest Tbilisi 2020](http://www.datafest.ge) and the second at [AMLD 2020](https://appliedmldays.org/) conference in Lausanne, Switzerland (material available [here](https://github.com/MaxinAI/amld2020-workshop)).

# Autoencoders

In its simplest form, an Autoencoder (shortly, AE) is a fully-connected (feed-forward) neural network depicted below:

<center>
<img src="/img/ae_intro/ae.svg" />
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 1. A typical autoencoder neural network architecture.</div></center>
</center>
<br />

The whole neural network can be broken down in three logical pieces: 1) An encoder - first part with gradually shrinking width denoted as $f(x)$ on the picture; 2) middle layer $h$, a.k.a hidden/latent/code/bottleneck layer; 3) A decoder - last part with a gradually increasing width $g(x)$. It's important to note that first (the input) and the last layers must have exact same shape.

Given some dataset $x_i\sim\mathcal{D}$, an objective is to learn to copy data points as accurately as possible, i.e. learning an identity map from input to the output. More concretely, given an input data point $x$, the goal is to output some $\hat{x}$ which is as close to the original input $x$ as possible. Mathematically, it is the following optimization problem: 


<center>$$\arg\min_{\theta}\sum_{i=1}^{n}\lVert{x_i-g(f(x_i))}\rVert_2^2$$</center>
<br />
Where $\theta$ represents weights of an AE and $x_i$ - data points of the dataset. $\lVert.\rVert_2^2$ denotes squared L2 norm. In terms of the loss function, it's a MSE (mean squared error) between input and output vectors. The network can be trained using SGD. 

Basically, that's all! But let's slowly digest what we've just described. First of all, it's an unsupervised learning problem, as we haven't mentioned any explicit labels, just data points $x_i$. Secondly, why the heck do we need to learn copying inputs? Isn't it a useless task? Technically yes, but learning copying produces a very interesting side-effect that is a whole point of using this AE thing! 

In order to understand this side-effect, let's first note that AE could've just passed input from layer to layer, all the way to the last one and that would've been a perfect copy! That's not interesting at all. But wait, recall that as we go from the encoder to the decoder, layer width gradually shrinks and then increases again to its original size, making a bottleneck in the middle (layer $h$, in the above picture). This specific architectural nuance renders mentioned layer-by-layer copy infeasible. In fact, the encoder is forced to throw away part of the information to make enough room for the rest to squeeze in the bottleneck, whereas, the decoder has to do it's best to reconstruct/recover original input from partially corrupted/lost data (taken from bottleneck layer). During the training, the encoder learns to recognize parts of the input with the least impact on the reconstruction quality and drops such parts. Meanwhile, the decoder learns to recover as much information as possible to make the copy accurate. This whole process resembles a lossy compression algorithm, such as JPEG, where an encoder encodes large [RAW](https://en.wikipedia.org/wiki/Raw_image_format) image into compressed representation (layer $h$ in AE terminology) and a decoder decodes it back and displays an image to the user. Reconstruction quality doesn't match the original, of course, but it's still very good, usually even unnoticeable by a random user eye. Note that, JPEG uses deterministic compression/decompression algorithm, whereas, in AEs, these two are learnt.

The fact that middle/bottleneck layer $h$ learns a "compressed code/representation" of an original input (like raw and jpg images in above example) is the very side-effect that naturally occurs during AE training on copying task. Why do we care? Because what data is left after the encoder phase, is the collection of the most critical and discriminative characteristic features of the original input data point, required for doing the best possible reconstruction. In other words, a bottleneck layer $h$ is an automatically learned feature vector for our original data point, in a sense that it summarizes the most important aspects of it.

Besides, semantical similarities/dissimilarities between data points in input space are reflected in latent space (set of all latent vectors) as well. If we have three images: A, B - white cats, C - black dog, in latent space A and B will be close to each other (in terms of some distance measure) and relatively far from C. This is amazing!

Alright, I hope I have you some insight and intuition about internals of vanilla autoencoders. I say vanilla, because there are many other types of autoencoders out there, but most of them are built on top of the core ideas of vanilla autoencoders we've discussed so far.
 
### Technical Details

In this section, I am gonna point out some technical details (including implementation considerations) which I believe you should know.

#### Encoder and Decoder Network Architectures

In practice, various sorts of architectures are being used for Encoder and Decoder, not necessarily same or symmetrical as illustrated in above picture. For example, in Computer Vision you can use popular CNN-based architectures like [ResNet](https://arxiv.org/abs/1512.03385), in NLP or Time-Series you can use [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) or [Transformer](https://arxiv.org/abs/1706.03762), or other crazy architecture you come up with...

In fully-connected example above, we used decreasing/increasing layer width in Encoder/Decoder, respectively, to control information flow capacity and achieve bottleneck effect. In other architectures, you might use other constraining techniques. For example, in CNN-based architectures you can use max-pooling or strided convolutional layers for downsampling (i.e reducing spatial dimensions of the input) in the Encoder. As for decoding, popular choice is to pick up nearest-neighbor upscaling followed by a convolution or using [transposed convolutions](https://github.com/vdumoulin/conv_arithmetic). The latter one is notorious for causing [checkboard artifacts](https://distill.pub/2016/deconv-checkerboard/) in some cases, so one can also consider using [subpixcel convolutions](https://arxiv.org/pdf/1609.05158.pdf).

#### Right size for the middle layer $h$

Unfortunately, there is no magic formula for picking a right size for the bottleneck layer $h$. Practitioners usually consider several factors when estimating the right size (note that, these are just rough guidlines and might or might not work depending on the task):

- Dataset Complexity - The more complex dataset becomes, the "wider" bottleneck is required to squeeze in information. Toy datasets like [MNIST](http://yann.lecun.com/exdb/mnist) can have $h$ as small as 2-10 and big datasets might have 100-1000.
- Representational Power of Encoder/Decoder - With increasing encoder/decoder model complexity, the bottleneck layer can be made smaller. Theoretically, an encoder can be so complex that it can just learn to map every individual data point from training set to an integer index, and the decoder map that integer index back to the data point. In this case, the bottleneck will be just a single neuron! Well, practically such things usually don't happen, but we need to make sure encoder/decoders aren't overcomplicated and don't overfit.
- Training Time - By keeping encoder/decoder architecture fixed, the narrower the bottleneck layer becomes, the more training time we need to better compress the data and fit inside the bottleneck. 
- Performance on the Downstream Task - as we briefly mentioned above and as we will get back to it later below, automatic feature extraction is one of the applications of an AE (layer $h$ is a feature vector). That being said, you treat $h$'s size as a hyperparameter and select best one using a cross-validation on the downstream task.   

<br />
### Applications

<br />
#### Dimensionality Reduction & Representation Learning

Dimensionality reduction is a classical use-case of an autoencoder. Once trained, we take only an encoder part and feed the whole dataset to it. The outputs (i.e layer $h$ activations) will represent original data points in a lower-dimensional space. Afterwards, these vectors can be used as an input to another (downstream) supervised/unsupervised Machine Learning algorithm. Can't we just use original data points as inputs? Well, there are several reasons why using dimensionality reduction can help:

- Semi-Supervised Learning - Suppose you need to do a classification, however, only 10% of your data is labelled which is not enough to achieve a desired score. In this scenario, you may want to first train an autoencoder on the full data (labelled+unlabelled) and then use learned feature vectors of the labelled portion as an input to the classifier. Even though, you still use labelled portion of data for classification, feature vectors now contain encoded patterns from the whole dataset. In many cases, this can boost the result.
- Sparse Input - Sometimes our original data vectors are sparse, i.e. large vectors with the majority of zero elements. Such vectors occur naturally when we use one-hot encodings for nominal categorical variables. Apart from excessive memory consumption and computational cost, it's empirically shown that densifying sparse input using autoencoder-like techniques is beneficial in wide range of tasks, for instance, in NLP (check out [word embeddings](https://nlp.stanford.edu/projects/glove/), for more insights on this last one).

Before we move on, it's worth mentioning that a feed-forward autoencoder with only a single hidden layer and no activation function (i.e $input\mapsto h\mapsto output$) produces same effect as applying a [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) on the data. Details on this can be seen [here](https://arxiv.org/pdf/1804.10253.pdf). 

#### Anomaly Detection

Antoencoders are also frequently used in anomaly/outlier detection. Usually, these tasks can be formulated as an extremely-imbalanced binary classification problems where the vast majority of the data points are labelled negative (no anomaly). Class imbalance and scarcity of positive samples can make traditional classification algorithms hard to train, so unsupervised methods are usually in favour.

Our approach is to train an autoencoder only on negative class. During evaluation, we look at the reconstruction error (value of the loss function) on the test set samples and the data points with reconstruction error higher than some threshold are considered as anomalies.

The logical assumption here is that negative samples are similar to each other and dissimilar to anomalies, i.e. they are outliers. An autoencoder is optimized to produce low reconstruction error on negatives and, therefore, a positive sample should cause relatively higher error.

For more details on this topic, check out [this](https://www.kaggle.com/mlg-ulb/creditcardfraud) Kaggle challenge and the [post](https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders) talking about the autoencoders in fraud detection.

I'd also recommend a nice Python library containing various methods on anomaly detection: [pyod](http://pyod.readthedocs.io/)

# Variational Autoencoders (VAE)

Let's imagine we have a dataset $D=\{x_0, x_1, ... x_n\}$ that comes from some unknown distribution $P(X)$, sometimes also referred as the data generation process.

Variational Autoencoder, shortly VAE, is a probabilistic generative model that allows us to perform two this: 
1. Learn the unknown distribution of the data, i.e approximate $P(X)$ using given $D$. 
2. Draw samples from the learned distribution, i.e. generate new data points.

In this section I'll try to explain how does VAE achieve these two objectives and how can we leverage them to do pretty amazing stuff! Before we proceed, it should be pointed out that VAE concept has emerged from the underlying mathematical framework known as Variational Bayesian Inference (where the name "Variational" comes from). Here I am not gonna dive deep into theory and proofs (I've included some references for this) but rather keep it as simple and intuitive as possible. Still, some basic math notation will be used. 

A simple VAE is illustrated in the figure below. Even though a reader can immediately draw some parallels between AE and VAE, there are some fundamental differences (like the weird green box on the picture) that make VAE rather different animal. 

<br /><br />
<img src="/img/ae_intro/vae.svg" />
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 2. A typical variational autoencoder neural network architecture.</div></center>
<br /><br />

Recall, that AE's encoder maps the input to some point in the latent space and the decoder takes it back to the original space. As for the VAE, the encoder learns to map the input not to a single point, but to the set of infinitely-many potential points described as a probability distribution, namely, a Gaussian with diagonal covariance matrix! More concretely, instead of outputting a single latent vector, VAE's encoder outputs two vectors: The mean and the covariance - parameters (sufficient statistics) of such Gaussian (diagonal covariance is used for simplicity, more details given later). Meanwhile, the decoder is trained to map samples drawn from this Gaussian back to the vicinity of the original input. This process is depicted in the above picture, where the green box denotes sampling procedure.

VAE uses a loss function that is a sum of two components: 1) **MSE reconstruction error** between input and output, similar to AE. It kinda makes sense, because we want decoder output to be close to original input. 2) **Regularization term** enforcing that all Gaussians keep close to the standard normal $\mathcal{N}(0, 1)$ during the training (see figure 2). Mathematically, this term represents a [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between our Gaussian and standard normal.

**Q**: How do we generate new data samples using trained VAE?<br />
**A**: First, we sample a standard normal random vector and pass it to the decoder, which will transform it into the data sample vector. Why standard normal? Well, because we know that regularizer term in loss function forces all Gaussians to be close to the standard one. Without it, Gaussians produced by encoder would highly differ from each other, making it hard to "guess" which one we need to sample from.

Generally, this kinda "two-step" sampling is very common in programming: We start by sampling from a simple distribution such as standard uniform (`rand()` function is available in every modern programming language) and then we pass it to some mathematical function that transforms it into any desired distribution sample. See [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling), for one possible way of doing this.

Alright, up to this point I hope you've developed at least some basic feelings about how does the described process learn distribution and allow generating new data points out of it. Now, let's talk about training.

#### Stochastic Back-Propagation

Similar to AE, VAE is also trained using SGD, however, there is a subtle issue: How do we back-propagate gradient in a sampling layer (green box in the illustration)? Well, here comes so-called "reparametrization trick", which 
involves modifying sampling layer in a way that deterministic paths appear from decoder to encoder, allowing gradient information to flow backwards (see figure 3.). In particular, we take advantage of the fact that any Gaussian can be written as mean shifted and scaled version of standard normal, i.e $X\sim\mathcal{N}(\mu, \sigma^2) \iff X\sim\mu + Z * \sigma, Z\sim\mathcal{N}(0, 1)$. This way, we make parameters of Gaussian learnable!

<br /><br />
<img src="/img/ae_intro/vae-reparametrization.svg" />
<br />
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 3. VAE diagram before (on the left) and after (on the right) reparametrization. Deterministic nodes are shown in blue, stochastic - in green. Two-sided thick arrows indicate forward and backward flow, whereas, one-sided arrows allow only forward pass.</div></center>
<br /><br />

#### Continuity of Latent Space 

Let's take a single data point and feed it to the vanilla AE's encoder. Now, what happens if we slightly perturb the resulting vector $h$, i.e. add some small epsilon to it? How is it gonna influence the reconstruction? Well, one reasonable answer would be that slight change should result in slight change in reconstruction as well. However, this is not usually the case with Autoencoders, because their latent space usually isn't continuous! By introducing a change, we might get out of the latent space and get inadequate reconstruction (something not in the data space). In the case of VAE, this isn't a problem, because VAE's design produces continuous latent space! Refer to the below diagram (figure 4) for visualizations.  

<br /><br />
<center><img src="/img/ae_intro/ae_latent.png" /></center>
<center><img src="/img/ae_intro/vae_latent.png" /></center>
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 4. Visualization of 2D latent space of AE (top) and VAE (bottom) trained on MNIST dataset. On the top-left section of AE's image, a region outside the data space is observed - pictures definitely don't correspond to valid digits. This is caused by regions of discontinuity in AE's latent space.</div></center>
<br /><br />

We can do the so-called "visual arithmetics" with VAE. In case of facial image dataset, we can subtract two encodings - "person with eyeglasses" and "person without eyeglasses" - and get an encoding of "eyeglass information". Further, we can add this resulting vector to arbitrary images to put sunglasses on them (figure 4). Continuity makes sure there are no "gaps" in the latent space and we accidentally don't get outside the valid space by performing arithmetics.

<br /><br />
<img src="/img/ae_intro/vae_arithmetics.png" />
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 5. Example of adding "eyeglass information vector" to the arbitrary image encodings in the latent space. Reconstructions of the resulting sums are shown.</div></center>
<br /><br />


#### Effect of the KL Regularization

As we mentioned above, KL (Kullback–Leibler) divergence between encoder's produced Gaussian and standard normal keeps the whole latent space tight, kinda bounded within a spherical region of latent space, centered at the origin. This, first of all, it highly reduces overfitting because Gaussian centers can't grow arbitrarily large and, secondly, plays an important role in getting rid of any possible "large gaps" within the latent space making it continuous in any region (figure 6). Besides, if the latent space didn't have this particular tight circular shape, it would've been hard to do "two-step" sampling described above, since not every initial standard normal sample would be in valid region of latent space (due to gaps).

One can also control the contribution of the regularization term in the final loss by using weighting. The new loss would be $Loss=MSE + \beta * KL$, where $\beta$ is a real number weighting the regularization term.


There is some work indicating that increasing $\beta$ can help learning more "disentangled" latent representations (see [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl) for details and [this](https://arxiv.org/pdf/1804.03599v1.pdf) explanation paper). Disentanglement is a characteristic of latent space where single latent vector element is responsible for a single semantic aspect of the original data. In case of our face dataset example, one dimension, for instance, can only affect the degree of "smile" on the face, second - "hair color", third - "lip size", etc... Single element can't affect multiple factors! Well, of course disentanglement is something you definitely wanna see in your model, however, it's really hard to achieve and even impossible in real datasets (see ICML 2019 [best paper](https://arxiv.org/pdf/1811.12359.pdf) award work on a related topic).

<br />
<center>
<video width="500" controls loop>
  <source src="/img/ae_intro/vae.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
<video width="500" controls loop>
  <source src="/img/ae_intro/vae_without_kl.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
</center>
<center><div style='width: 60%; font-size: 11pt; color: gray;'>Figure 6. Dynamics of VAE's 2D latent space on MNIST. Each circle represents a latent vector (center of Gaussian), size of a circle - total variance (trace of the covariance matrix), color - digit label (0-9). Top animation is VAE with KL weight $\beta=1$ and bottom with no regularization, i.e $\beta=0$. We can see how does KL enforce circular shape. Clustering takes place regardless of regularization presence.</div></center>
<br /><br />

#### Limitations

In the real world VAEs usually lag behind other generative models in terms of generated data quality. Indeed, GANs and/or Normalizing Flows (e.g. [OpenAI Glow](https://openai.com/blog/glow/))  show superior performance when it comes to realistic, high-resolution image generation. Note that, this applies not only to image but other data domains as well.

There is a wide-spread opinion that using Gaussian for latent vectors isn't powerful enough to deal with highly complex datasets found in the real world, mainly, due to its unimodal nature (e.g. check out conclusion section https://github.com/dojoteef/dvae). Some research suggests this problem can be mitigated by using more complex distributions, like [Mixture of Gaussians](https://openreview.net/forum?id=BJJLHbb0-) or, going even further, by employing [adversarial training](https://arxiv.org/pdf/1511.05644.pdf). 

# References

 - https://arxiv.org/abs/1312.6114 - original VAE paper.
 - https://www.deeplearningbook.org/ - very nice Deep Learning book. See section on Autoencoders.
 - http://bjlkeng.github.io/posts/variational-autoencoders/ - good explanation of mathematical details.
 - https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html
 - https://arxiv.org/pdf/1401.4082.pdf - Stochastic Backpropagation and Approximate Inference
in Deep Generative Models.
 - https://www.aclweb.org/anthology/D19-1370.pdf
 
 
 <br />
 