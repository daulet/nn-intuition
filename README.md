# Neural Net Intuition

> If I've seen further it is by standing on the shoulders of giants. -_Gandalf (probably)_

Like many things, engineers solve problems by braking it down into smaller, known problems. Arguably neural nets, aka [software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35), are the next level of abstraction for solving increasingly complex problems. Quality of a model is driven not just by quality and size of training data, but also its architecture. As these model architectures grow in size and variety, it is important to maintain intuition behind their building blocks and how various architectures compress information.

# Building Blocks

## Attention
Input is fed into three parallel flows: _query_, _key_ and _value_. Each flow applies distinct linear transformation. _Query_ and _key_ are combined (MatMul) into an attention filter. Attention filter is then combined with _value_ to produce filtered value, which could learn to extract particular aspect of the input, e.g. in case of image input attention layer could learn to extract image background, or conversely highlight the subject, or extract style like photography vs painting vs animation.
### Intuition
Attention is a similarity search, i.e. for a given _query_ what _keys_ match it the best, that mapping can be then used to extract _values_ for corresponding matching keys. We calculate cosine similarity of the query and corresponding key to identify subset of positions (in tensor) to pay attention to (filter). We then apply the filter to the value to finally extract the filtered value.

## Batch Norm
Although the reason for its usefulness is [debatable](https://en.wikipedia.org/wiki/Batch_normalization#Theoretical_Understanding), intuitively it normalizes the input to account for distribution shift between training and evaluation data. 

## Convolution
A sliding window (kernel) transformation across input, to extract localized features, e.g. lines, curves. Hierarchy of convolution layers can learn to extract more complex features, e.g. curve -> semicircle -> a letter.

## Embedding
A representation of input data in multidimensional space, could be used as intermediate representation across different modalities, e.g. between image and its text description.

## Residual
Residual connections help with knowledge preservation (information lost while passing through layers of a neural net) and vanishing gradient problem.

# Architecture

## Convolutional Neural Network (CNN)

Architecture exploits the fact that neighbor pixels of an image are related and could describe a meaningful pattern. First convolution layer could extract features like a horizontal line, a vertical line, a curve etc. The second layer could learn to combine these features into shapes, e.g. longer lines, a grid, circles etc. Recursively each layer combines features from the previous layer to learn more and more complex visual representations relevant to the problem. Since the output of each layer is also a stack (channels) of images, intermediate layers could be human-inspectable.

## Fully Connected Network

Since each neuron is connected to each neuron of the next layer, this network can discover arbitrary relationships between input variables and the output. While that can be said about any neural network, this is the most generic architecture typically used in absence of any inherent structure in the input (e.g. image of real world, words from english dictionary etc). Larger nets can learn easier as more parameters can describe more complex functions, but might not generalize well, since it might memorize training dataset instead of extracting useful patterns. For training, start with the smallest network that learns, since fewer parameters forces the model to extract useful variables when it does not have capacity to store all of the training data.

## Transformer

### Architecture

At high level transformer consists of Encoder and Decoder components. Encoder parses the Transformer input, while Decoder combines output of Encoder and previously generated Transformer output to generate next output token. If there are no previous tokens (first inference) use a special token for “beginning”of the output. The process repeats until Decoder generates a special “end” token.

Input is transformed into an “embedding”, representing the input along with position of each token (word, in case of text input). Position embedding allows parallel processing of input sequence (compared to sequential processing of RNNs). Each attention layer is complemented with feed forward layer, presumably, to build connections that attention layer couldn't. The last part of Decoder is a classifier where each word of the corpus is a separate category.

The key idea is to use attention in different ways (it's all you need). First multi-headed attention of Encoder extracts multiple attention relationships in the input. Second, masked attention enables self-supervisied training by masking target sequence at any position. Third, it applies attention filter of the input sequence (output of Encoder) to previously generated output to predict what should be the next output token.

### Intuition

Encoder extracts multiple key "focus" relationships in the input (via multiheaded attention). Then that focus is appliead on each run of Decoder to keep the output faithful to original sentiment.  

## Autoencoders

Model consists of a concatenated encoder, low-dimensional latent space and decoder. It is trained by reproducing the same output from decoder.
Low-dimensional latent space is used as a choke point to compress information in a smaller dimension space.
Autoencoders cannot be used for data generation, since the model outputs faithful reconstruction. Use cases: image reconstruction, noise removal.

## Diffusion model

Model learns by reversing a gradual noising process. The Noising process adds noise step by step, eventually resulting in pure noise.
At inference one can sample from random noise and walk back to a generated output. Since the process is stochastic, it could be used to generate variations of the same input by repeatedly running inference.

# Model Transferability

The following are unproven ways one can transfer knowledge of one model into another:

## Presence -> Detection
If you learned to recognize a presence of an object in a scene, you could inspect layer activations to detect the object in the scene, e.g. draw a bounding box around it.

## Classifier -> Example Generator
If you have a classifier for a given task, you could combine it with GAN to generate more examples of each class.
