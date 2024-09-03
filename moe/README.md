# Mixture Of Experts (MoE)

![MoE](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)

Enhances performance by utilizing a set of specialized models, or "experts," for different tasks. This framework allows only a subset of experts to be activated for any given input, optimizing computational resources and improving efficiency.

> Note: my code is a simplified version, as it does not add/handle noise,randomness, etc.

Resources:
- https://huggingface.co/blog/moe
- https://cameronrwolfe.substack.com/p/conditional-computation-the-birth
- https://github.com/lucidrains/mixture-of-experts
- https://www.youtube.com/watch?v=0U_65fLoTq0

