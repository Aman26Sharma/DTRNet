# DTRNet: Dynamic Token Routing Network to Reduce Quadratic Costs in Transformers

<a target="_blank" href="">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/Aman26Sharma/DTRNet">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<!-- <a target="_blank" href="https://twitter.com/DongfuJiang/status/1805438506137010326">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a> -->
<br>


## Abstract

Transformers achieve state-of-the-art results across many tasks, but their uniform application of quadratic self-attention to every token at every layer makes them computationally expensive. %However, many tokens do not require such heavy computation: layer-wise cosine similarity analysis of dense Transformers reveals that inner-layer token embeddings change only marginally across adjacent layers, indicating substantial computational redundancy. We introduce DTRNet (Dynamic Token Routing Network), an improved Transformer architecture that allows tokens to dynamically skip the quadratic cost of cross-token mixing while still receiving lightweight linear updates. By preserving the MLP module and reducing the attention cost for most tokens to linear, DTRNet ensures that every token is explicitly updated while significantly lowering overall computation. This design offers an efficient and effective alternative to standard dense attention. Once trained, DTRNet blocks routes only ~10\% of tokens through attention at each layer while maintaining performance comparable to a full Transformer. Its efficiency gains, scales with sequence length, offering significant reduction in FLOPs for long-context inputs. By decoupling token updates from attention mixing, DTRNet substantially reduces the quadratic share of computation, providing a simple, efficient, and scalable alternative to Transformers.

## Architecture

<div align="center">
<img src="assets/DTRNet_arch.jpg" width="700" alt="DTRNet Architecture"/>
<p><em>Figure 1: DTRNet Layer. Left: tokens routed to the self-attention path undergo full cross-token mixing. Right: tokens routed to the projection-only (bypass) path skip mixing and receive a token-local update via the value projection (W_V) and output projection (W_O), followed by the shared feed-forward network (FFN). Both paths share parameters.</em></p>
</div>

