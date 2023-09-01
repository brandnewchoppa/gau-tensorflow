import tensorflow as tf

from tensorflow import einsum, reshape, cast
from tensorflow import math

from keras import Model, Sequential
from keras.layers import Layer
from keras.initializers import ones, zeros

from keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Embedding
)

class ScaleNorm(Layer):
    """
    Scale Normalization (ScaleNorm)
    https://arxiv.org/pdf/2202.10447.pdf

    Normalizes activation vectors to a single learned length 'g', and achieving
    a much simpler scaled l2 normalization. 

    Projects the d-dimensional vectors onto a (d-1)-dimensional hypershpere with
    learned length radius 'g'. Replaces the 2d scale and shift parameters of 
    LayerNorm with a single learned scalar.

    References:
    https://arxiv.org/pdf/1910.05895.pdf
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self,
                 *,
                 eps : float = 1e-5,
                 scale_by_dim : bool = False,
                 **kwargs):
        super(ScaleNorm, self).__init__(**kwargs)
        self.eps = eps
        self.scale_by_dim = scale_by_dim

    def build(self, _):

        self.scale = self.add_weight(
            name = 'scale',
            shape = (),
            initializer = ones())

        self.built = True

    def call(self, x):
        norm = tf.norm(x, axis = -1, keepdims = True)

        if self.scale_by_dim:
            norm = norm * (x.shape[-1] ** -.5)

        norm = tf.clip_by_value(x, self.eps, norm.dtype.max)
        norm = self.scale / norm
        return x * norm
    
class OffsetScale(Layer):
    """
    Offset Scale (OffsetScale)
    https://arxiv.org/pdf/2202.10447.pdf

    Apply per-dim scalars (gamma) and offsets (beta) to 'x' (similar to the
    learnable variables in LayerNorm).

    Used in Gated Attention Unit to create Q, K as two cheap transformations
    to Z.

    References:
    https://arxiv.org/pdf/1706.03762.pdf (based on query, key)
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self,
                 splits : int = 1,
                 **kwargs):
        super(OffsetScale, self).__init__(**kwargs)
        self.splits = splits

    def build(self, x_shape):
        e = x_shape[-1]

        self.gamma = self.add_weight(
            name = 'gamma',
            shape = (self.splits, e),
            initializer = ones())

        self.beta = self.add_weight(
            name = 'beta',
            shape = (self.splits, e),
            initializer = zeros())

        self.built = True

    def call(self, x):
        out = einsum('...e, se -> ...se', x, self.gamma) + self.beta
        return tf.unstack(out, axis = -2)
    
class RelativePositionBias(Layer):
    """
    Relative Position Bias (RelativePositionBias)
    https://arxiv.org/pdf/2202.10447.pdf

    Clipping the maximum distance also enables the model to generalize to
    sequence lengths not seen during training.

    References:
    https://arxiv.org/pdf/1803.02155.pdf
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self,
                 *,
                 scale : float,
                 n_buckets : int = 32,
                 max_distance : int = 128,
                 **kwargs):
        super(RelativePositionBias, self).__init__(**kwargs)
        self.scale = scale
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = Embedding(n_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        n_buckets = 32,
        max_distance = 128):

        n = -relative_position
        n = math.maximum(n, tf.zeros_like(n))

        max_exact = n_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + cast(
            math.log(cast(n, 'float32') / max_exact) / math.log(max_distance / max_exact) * (n_buckets - max_exact),
            'int64')
        val_if_large = math.minimum(val_if_large, cast(tf.fill(val_if_large.shape, n_buckets - 1), 'int64'))

        return tf.where(is_small, cast(n, 'int64'), val_if_large)

    def forward(self, x):
        i, j = x.shape[-2:]
        q_pos = tf.range(i, dtype = tf.dtypes.int64)
        k_pos = tf.range(j, dtype = tf.dtypes.int64)
        rel_pos = reshape(k_pos, [1] + k_pos.shape) - reshape(q_pos, q_pos.shape + [1])
        rp_bucket = self._relative_position_bucket(rel_pos, self.n_buckets, self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = reshape(values, values.shape[:-1])
        return bias * self.scale
    
class GAU(Layer):
    """
    Gated Attention Unit (GAU)
    https://arxiv.org/pdf/2202.10447.pdf

    Formulates the attention and Gated Linear Unit (GLU) as a unified layer and
    to share their computation as much as possible.

    First apply a ScaleNorm on the input 'x' and formulates the two gates
    'u' and 'v'. On the other hand computes the attention 'A = attn(x, v)',
    and apply an element-wise multiplication (u * A). Finally the dense layer
    which closes the inverse bottleneck and transforms the embeddings back to
    original size along feature axis.

    References:
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self,
                 *,
                 qk_dim : int = 64,
                 expansion_factor : int = 2,
                 dropout_rate : float = .2,
                 norm_type : str = 'scale_norm',
                 shift_tokens : bool = False,
                 **kwargs):
        super(GAU, self).__init__(**kwargs)
        self.qk_dim = qk_dim
        self.expansion_factor = expansion_factor
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.shift_tokens = shift_tokens

    def build(self, x_shape):
        e = x_shape[-1]

        if self.norm_type == 'scale_norm':
            self.norm = ScaleNorm()
        else:
            self.norm = LayerNormalization()

        self.to_uv = Dense(
            (e * self.expansion_factor) * 2,
            activation = 'silu')

        self.to_qk = Dense(
            self.qk_dim,
            activation = 'silu')

        self.scale_offset = OffsetScale(
            splits = 2)

        self.rel_pos_bias = RelativePositionBias(
            scale = e ** .5)

        self.dropout = Dropout(
            rate = self.dropout_rate)

        self.to_out = Sequential([
            Dense(e),
            Dropout(rate = self.dropout_rate)
        ])

        self.built = True

    def _attn(self, x, v):
        n = cast(x.shape[-2], 'float32')
        z = self.to_qk(x)
        q, k = self.scale_offset(z)
        qk = einsum('bns, bms -> bnm', q, k)
        
        a = tf.nn.relu(qk / n + self.rel_pos_bias(qk)) ** 2
        mask = tf.cast(tf.linalg.band_part(tf.ones([n, n]), -1, 0), 'bool')
        a = tf.where(mask, a, -1e10)
        
        return einsum('bnm, bme -> bne', a, v)

    def _shift_tokens(self, x):
        x_shift, x_pass = tf.split(x, 2, axis = -1)
        x_shift = tf.pad(x_shift, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
        return tf.concat([ x_shift, x_pass ], axis = -1)

    def call(self, x):
        shortcut, x = x, self.norm(x)

        if self.shift_tokens:
            x = self._shift_tokens(x)

        u, v = tf.split(self.to_uv(x), 2, axis = -1)
        x = u * self.dropout(self._attn(x, v))
        return self.to_out(x) + shortcut
    
class ScaledSin(Layer):
    """
    Sinusoidal Position Embedding with scaling factor. (ScaledSin)
    https://arxiv.org/pdf/2202.10447.pdf

    References:
    https://arxiv.org/pdf/1706.03762.pdf
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self):
        super(ScaledSin, self).__init__()

    def build(self, x_shape):
        d = x_shape[-1]

        self.scale = self.add_weight(
            name = 'scale',
            shape = (),
            initializer = tf.constant_initializer(1 / d ** .5))

        self.inv_freq = 1. / (10000 ** (tf.range(0, d, 2, 'float32') / d))

    def call(self, x):
        n = x.shape[-2]

        pos = tf.range(n, dtype = self.inv_freq.dtype)
        pos = einsum('s, d -> sd', pos, self.inv_freq)
        scaled_emb = tf.concat([ math.sin(pos), math.cos(pos) ], axis = -1)
        return scaled_emb * self.scale
    
class GAUTransformer(Model):
    def __init__(self,
                 *,
                 emb_dim : int,
                 n_tokens : int,
                 depth : int = 4,
                 qk_dim : int = 64,
                 expansion_factor : int = 2,
                 dropout_rate : float = .2,
                 norm_type : str = 'scale_norm',
                 shift_tokens : bool = False,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.token_emb = Embedding(n_tokens, emb_dim)
        self.abs_pos_emb = ScaledSin()

        self.blocks = Sequential([
            GAU(
                qk_dim = qk_dim,
                expansion_factor = expansion_factor,
                dropout_rate = dropout_rate,
                norm_type = norm_type,
                shift_tokens = shift_tokens
            ) for _ in range(depth)])

        self.to_logits = Sequential([
            LayerNormalization(),
            Dense(n_tokens)
        ])

    def call(self, x):
        x = self.token_emb(x)
        x = self.abs_pos_emb(x) + x

        x = self.blocks(x)
        return self.to_logits(x)
