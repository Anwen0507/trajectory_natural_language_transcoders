# Natural Language Transcoders (NLT) with Trajectory Objective
The goal of my project is to understand what computation is performed in a stack of layers.

The natural language transcoder has a verbalizer that converts a trajectory, or series, of activations at certain checkpoints in a stack of layers into an explanation of the computation over that stack, and a reconstructor that reconstructs the trajectory of activations given the explanation and the first activation in the trajectory. For example, if we want to explain $L$ layers and we have a checkpoint every $n$ layers, then the trajectory of activations will be $(\mathbf{a}_0, \mathbf{a}_1, \dots, \mathbf{a}_k)$ where $nk = L$.

This project is inspired by Anthropic's [recent natural language autoencoders (NLAs) work](https://transformer-circuits.pub/2026/nla/index.html), and this code is adapted from their corresponding [codebase](https://github.com/kitft/natural_language_autoencoders). The distinction of this project from an NLA is in the objective: an NLA reconstructs the activation it was given, whereas my method would attempt to construct a trajectory of activations through a textual bottleneck.
