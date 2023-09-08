# GAU - TensorFlow
Gated Attention Unit (TensorFlow implementation) from the paper [Transformer Quality in Linear Time](https://arxiv.org/pdf/2202.10447.pdf).

## Roadmap
- [x] GAU module, Transformer Model
- [x] AutoregressiveWrapper (top_p, top_k)
- [x] Non-Causal GAU functionality
- [x] Rotary Embeddings
- [ ] ScaleNorm + FixNorm experiment from the [paper](https://arxiv.org/pdf/1910.05895.pdf)
- [ ] Gradient Checkpointing
- [x] Add RMSNorm

> [!WARNING]
> This repository is under developemnt, but please feel free to explore and provide any feedback or suggestions you may have. :construction:

## Usage

```python
import tensorflow as tf
from gau_tensorflow import Transformer

model = Transformer(
    emb_dim = 128,        # embedding dimension
    n_tokens = 50256,     # number of tokens used in the vocabulary
    depth = 4,            # number of blocks stacked in the model
    causal = True         # autoregressive functionality
)

x = tf.random.uniform([1, 512], 0, 50256, tf.int64)
logits = model(x, training = False)
```

### Interpolate Sequence Positions

```python
for i in range(model.depth):
    model.get_layer('blocks').get_layer(f'gau{i}').rotary_pos_embs.interpolate_factor = 2.0
```


## Citations

```bibtex
@article{Hua2022TransformerQI,
    title   = {Transformer Quality in Linear Time},
    author  = {Weizhe Hua and Zihang Dai and Hanxiao Liu and Quoc V. Le},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.10447}
}
```

```bibtex
@article{Toan2019TransformerwT,
    title   = {Transformers without Tears: Improving the Normalization of Self-Attention},
    author  = {Toan Q. Nguyen and Julian Salazar},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/2019.05895}
}
```
