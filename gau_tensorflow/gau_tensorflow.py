import tensorflow as tf

from rope_tensorflow import RoPE
from tensorflow import einsum, reshape, cast
from tensorflow import math

from keras import Model, Sequential
from keras.layers import Layer
from keras.saving import register_keras_serializable

from keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Embedding
)

@register_keras_serializable(package = 'GAUTensorFlow')
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
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, _):

        self.scale = self.add_weight(
            name = 'scale',
            shape = (),
            initializer = tf.ones_initializer())

        self.built = True

    def call(self, x):
        norm = tf.norm(x, axis = -1, keepdims = True)
        norm = norm * (x.shape[-1] ** -.5)
        norm = tf.clip_by_value(x, self.eps, norm.dtype.max)
        norm = self.scale / norm
        return x * norm

    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
class RMSNorm(Layer):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    https://arxiv.org/pdf/1910.07467.pdf

    A well-known explanation of the success of LayerNorm is its re-centering
    and re-scaling invariance property. However RMSNorm only focuses on
    re-scaling invariance and regularizes the summed inputs simply according
    to the root mean square statistic.

    Intuitively, RMSNorm simplifies LayerNorm by totally removing the
    mean statistic at the cost of sacrificing the invariance that mean
    normalization affords.
    """

    def __init__(self,
                 *,
                 eps : float = 1e-8,
                 use_bias : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.use_bias = use_bias

    def build(self, x_shape):
        d = x_shape[-1]

        self.scale = self.add_weight(
            name = 'scale',
            shape = (1, d),
            initializer = tf.ones_initializer())

        self.offset = self.add_weight(
            name = 'offset',
            shape = (1, d) if self.use_bias else (1,),
            initializer = tf.zeros_initializer())

        self.built = True

    def call(self, x):
        ms = tf.reduce_mean(tf.math.square(x), axis = -1, keepdims = True)
        return self.scale * x * tf.math.rsqrt(ms + self.eps) + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'use_bias': self.use_bias
        })
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
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
                 *,
                 splits : int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.splits = splits

    def build(self, x_shape):
        d = x_shape[-1]

        self.gamma = self.add_weight(
            name = 'gamma',
            shape = (self.splits, d),
            initializer = tf.keras.initializers.ones())

        self.beta = self.add_weight(
            name = 'beta',
            shape = (self.splits, d),
            initializer = tf.keras.initializers.zeros())

        self.built = True

    def call(self, x):
        out = einsum('...e, se -> ...se', x, self.gamma) + self.beta
        return tf.unstack(out, axis = -2)

    def get_config(self):
        config = super().get_config()
        config.update({'splits': self.splits})
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
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
        super().__init__(**kwargs)
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
            math.log(cast(n, tf.float32) / max_exact) / math.log(max_distance / max_exact) * (n_buckets - max_exact),
            tf.int32)
        val_if_large = math.minimum(val_if_large, cast(tf.fill(val_if_large.shape, n_buckets - 1), tf.int32))

        return tf.where(is_small, cast(n, tf.int32), val_if_large)

    def call(self, x):
        i, j = x.shape[-2:]
        q_pos = tf.range(i, dtype = tf.int32)
        k_pos = tf.range(j, dtype = tf.int32)
        rel_pos = reshape(k_pos, [1] + k_pos.shape) - reshape(q_pos, q_pos.shape + [1])
        rp_bucket = self._relative_position_bucket(rel_pos, self.n_buckets, self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = reshape(values, values.shape[:-1])
        return bias * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'n_buckets': self.n_buckets,
            'max_distance': self.max_distance,
            'relative_attention_bias': self.relative_attention_bias
        })
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
class ReLUSquared(Layer):
    """
    ReLU Squared (ReLUSquared)
    https://arxiv.org/pdf/2104.07012.pdf

    They introduce a novel, simple method for achieving sparsity in attention, by
    replacing the softmax activation with ReLU, and show that sparsity naturally
    emerges from such a formulation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return math.square(tf.nn.relu(x))

@register_keras_serializable(package = 'GAUTensorFlow')
class LaplacianAttnFn(Layer):
    """
    Laplacian Attention Function (LaplacianAttnFn)
    https://arxiv.org/abs/2209.10655

    Replacement for Squared ReLU via architecture search techniques which has
    shown faster convergence speed and competitive generalization performance
    on language tasks.
    """

    def __init__(self,
                 *,
                 use_n : bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_n = use_n
        self.PI = 3.14159265358

    def call(self, x):
    
        ## With the increasing length of the vectors the results are starting
        ## to become 1.0, to prevent this I introduced a new variable into
        ## the function to squeeze the domain a little bit. With this
        ## modification the values won't become 1.0 at the near end of the vecs.
        n = tf.saturate_cast(x.shape[-2], x.dtype) if self.use_n else 2.0

        mu = tf.saturate_cast(math.sqrt(0.5), x.dtype)
        std = tf.saturate_cast(math.sqrt(0.25 * self.PI), x.dtype)
        inner = (x - mu) / (std * tf.cast(math.sqrt(n), x.dtype))
        return 0.5 * (1 + math.erf(inner))

    def get_config(self):
        config = super().get_config()
        config.update({'use_n': self.use_n})
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
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
                 causal : bool = False,
                 dropout_rate : float = .2,
                 norm_type : str = 'layer_norm',
                 shift_tokens : bool = False,
                 use_rope : bool = False,
                 laplace_attn_fn : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.qk_dim = qk_dim
        self.expansion_factor = expansion_factor
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.shift_tokens = shift_tokens
        self.use_rope = use_rope
        self.laplace_attn_fn = laplace_attn_fn

    def build(self, x_shape):
        d = x_shape[-1]

        if self.norm_type == 'scale_norm':
            self.norm = ScaleNorm()
        elif self.norm_type == 'layer_norm':
            self.norm = LayerNormalization()
        elif self.norm_type == 'rms_norm':
            self.norm = RMSNorm()

        self.to_uv = Dense(
            (d * self.expansion_factor) * 2,
            activation = 'silu')

        self.to_qk = Dense(
            self.qk_dim,
            activation = 'silu')

        self.scale_offset = OffsetScale(
            splits = 2)

        self.rotary_pos_embs = RoPE(
            dim = self.qk_dim // 2)

        self.rel_pos_bias = RelativePositionBias(
            scale = d ** .5)

        self.dropout = Dropout(
            rate = self.dropout_rate)

        self.to_out = Sequential([
            Dense(d),
            Dropout(rate = self.dropout_rate)
        ])

        if self.laplace_attn_fn:
            self.attn_fn = LaplacianAttnFn()
        else:
            self.attn_fn = ReLUSquared()

        self.built = True

    def _attn(self, x, v):
        n = cast(x.shape[-2], x.dtype)
        z = self.to_qk(x)
        q, k = self.scale_offset(z)

        if self.use_rope:
            q, k = self.rotary_pos_embs.rotate([q, k])
        
        qk = einsum('bns, bms -> bnm', q, k)
        a = self.attn_fn(qk / n + self.rel_pos_bias(qk))

        if self.causal:
            mask = tf.cast(tf.linalg.band_part(tf.ones([n, n]), -1, 0), tf.bool)
            a = tf.where(mask, a, 0.0)
        
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'qk_dim': self.qk_dim,
            'expansion_factor': self.expansion_factor,
            'causal': self.causal,
            'dropout_rate': self.dropout_rate,
            'norm_type': self.norm_type,
            'shift_tokens': self.shift_tokens,
            'use_rope': self.use_rope,
            'laplace_attn_fn': self.laplace_attn_fn
        })
        return config

@register_keras_serializable(package = 'GAUTensorFlow')
class ScaledSin(Layer):
    """
    Sinusoidal Position Embedding with scaling factor. (ScaledSin)
    https://arxiv.org/pdf/2202.10447.pdf

    References:
    https://arxiv.org/pdf/1706.03762.pdf
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, x_shape):
        d = x_shape[-1]

        self.scale = self.add_weight(
            name = 'scale',
            shape = (),
            initializer = tf.constant_initializer(1 / d ** .5))

        self.inv_freq = 1. / (10000 ** (tf.range(0, d, 2, tf.float32) / d))

    def call(self, x):
        n = x.shape[-2]

        pos = tf.range(n, dtype = self.inv_freq.dtype)
        pos = einsum('s, d -> sd', pos, self.inv_freq)
        scaled_emb = tf.concat([ math.sin(pos), math.cos(pos) ], axis = -1)
        return tf.cast(scaled_emb, x.dtype) * self.scale

@register_keras_serializable(package = 'GAUTensorFlow')
class GAUTransformer(Model):
    def __init__(self,
                 *,
                 emb_dim : int,
                 n_tokens : int,
                 depth : int = 4,
                 qk_dim : int = 64,
                 expansion_factor : int = 2,
                 causal : bool = False,
                 dropout_rate : float = .2,
                 norm_type : str = 'layer_norm',
                 shift_tokens : bool = False,
                 use_rope : bool = False,
                 laplace_attn_fn : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_tokens = n_tokens
        self.depth = depth
        self.qk_dim = qk_dim
        self.expansion_factor = expansion_factor
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.shift_tokens = shift_tokens
        self.use_rope = use_rope
        self.laplace_attn_fn = laplace_attn_fn
                     
        self.token_emb = Embedding(n_tokens, emb_dim, name = 'embeddings')
        self.abs_pos_emb = ScaledSin(name = 'scaled_sin')

        self.blocks = Sequential([
            GAU(
                qk_dim = qk_dim,
                expansion_factor = expansion_factor,
                causal = causal,
                dropout_rate = dropout_rate,
                norm_type = norm_type,
                shift_tokens = shift_tokens,
                use_rope = use_rope,
                laplace_attn_fn = laplace_attn_fn,
                name = f'gau{i}'
            ) for i in range(depth)], name = 'blocks')

        self.to_logits = Sequential([
            LayerNormalization(),
            Dense(n_tokens)
        ], name = 'logits')

    def call(self, x):
        x = self.token_emb(x)
        x = self.abs_pos_emb(x) + x
        x = self.blocks(x)
        return self.to_logits(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'n_tokens': self.n_tokens,
            'depth': self.depth,
            'qk_dim': self.qk_dim,
            'expansion_factor': self.expansion_factor,
            'causal': self.causal,
            'dropout_rate': self.dropout_rate,
            'norm_type': self.norm_type,
            'shift_tokens': self.shift_tokens,
            'use_rope': self.use_rope,
            'laplace_attn_fn': self.laplace_attn_fn
        })
