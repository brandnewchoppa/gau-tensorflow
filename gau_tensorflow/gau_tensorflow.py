import tensorflow as tf

from tensorflow import einsum, reshape, cast
from tensorflow import math

from keras import Model, Sequential
from keras.layers import Layer

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
        super().__init__(**kwargs)
        self.eps = eps
        self.scale_by_dim = scale_by_dim

    def build(self, _):

        self.scale = self.add_weight(
            name = 'scale',
            shape = (),
            initializer = tf.ones_initializer())

        self.built = True

    def call(self, x):
        norm = tf.norm(x, axis = -1, keepdims = True)

        if self.scale_by_dim:
            norm = norm * (x.shape[-1] ** -.5)

        norm = tf.clip_by_value(x, self.eps, norm.dtype.max)
        norm = self.scale / norm
        return x * norm

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
                 p : float = -1.0,
                 use_bias : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.p = p
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
            'int64')
        val_if_large = math.minimum(val_if_large, cast(tf.fill(val_if_large.shape, n_buckets - 1), tf.int32))

        return tf.where(is_small, cast(n, 'int64'), val_if_large)

    def forward(self, x):
        i, j = x.shape[-2:]
        q_pos = tf.range(i, dtype = tf.int32)
        k_pos = tf.range(j, dtype = tf.int32)
        rel_pos = reshape(k_pos, [1] + k_pos.shape) - reshape(q_pos, q_pos.shape + [1])
        rp_bucket = self._relative_position_bucket(rel_pos, self.n_buckets, self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = reshape(values, values.shape[:-1])
        return bias * self.scale

def rotate_half(x):
    x_shape = x.shape
    x = tf.reshape(x, x.shape[:-1] + [ x.shape[-1] // 2, 2 ])
    x1, x2 = tf.split(x, 2, axis = -1)
    x1, x2 = tf.squeeze([x1, x2], axis = -1)
    x = tf.stack([ -x2, x1 ], axis = -1)
    return tf.reshape(x, x_shape)

def apply_rotary_emb(freqs, x, scale = 1.0):
    freqs = tf.cast(freqs, x.dtype)
    scale = tf.cast(scale, x.dtype)
    rotary_dim = freqs.shape[-1]
    x_left, x, x_right = x[..., :0], x[..., 0:rotary_dim], x[..., rotary_dim:]
    x = (x * tf.math.cos(freqs) * scale) + (rotate_half(x) * tf.math.sin(freqs) * scale)
    return tf.concat([ x_left, x, x_right ], axis = -1)

class RoPE(Layer):
    '''
    Rotary Position Emebddings (RoPE)
    https://arxiv.org/pdf/2104.09864.pdf

    Combines the concept of absolute and relative position embeddings.
    RoPE naturally incorporates relative position information through rotation
    matrix product instead of altering terms in the expanded formulation of
    additive position encoding when applied with self-attention.

    ## Length Extrapolatable Rotary Embeddings
    https://arxiv.org/pdf/2212.10554.pdf

    References:
    https://github.com/lucidrains/rotary-embedding-torch (implementation)
    https://blog.eleuther.ai/rotary-embeddings/ (explanation)
    '''
    
    def __init__(self,
                 dim : int,
                 theta : int = 10000,
                 xpos_scale_base : int = 512,
                 interpolate_factor : float = 1.0,
                 theta_rescale_factor : float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs = 1.0 / (theta ** (tf.range(0, dim, 2)[:(dim // 2)] / dim))
        self.interpolate_factor = interpolate_factor

        self.scale = (tf.range(0, dim, 2, self.freqs.dtype) + .4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

    def _calc_freqs(self, x):
        n = x.shape[-2]
        seq_pos = tf.range(n, dtype = self.freqs.dtype) / self.interpolate_factor
        freqs = tf.einsum('..., f-> ...f', seq_pos, self.freqs)
        freqs = tf.repeat(freqs, 2, axis = -1)
        return freqs, seq_pos

    def rotate(self, x):
        if isinstance(x, list):
            q, k = x
            d = q.shape[-1]
            freqs, seq_pos = self._calc_freqs(q)

            power = (seq_pos - d // 2) / self.scale_base
            scale = self.scale ** tf.transpose(power[tf.newaxis])
            scale =  tf.concat([ scale, scale ], axis = -1)

            q = apply_rotary_emb(freqs, q, scale = scale)
            k = apply_rotary_emb(freqs, k, scale = scale ** -1)
            return q, k

        freqs, _ = self._calc_freqs(x)
        return apply_rotary_emb(freqs, x)

class LaplacianAttnFn(Layer):
    """
    Laplacian Attention Function (LaplacianAttnFn)
    https://arxiv.org/abs/2209.10655

    Replacement for Squared ReLU via architecture search techniques which has
    shown faster convergence speed and competitive generalization performance
    on language tasks.
    """

    def __init__(self,
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
                 norm_type : str = 'scale_norm',
                 shift_tokens : bool = False,
                 use_rotary_embs : bool = False,
                 laplace_attn_fn : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.qk_dim = qk_dim
        self.expansion_factor = expansion_factor
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.norm_type = norm_type
        self.shift_tokens = shift_tokens
        self.use_rotary_embs = use_rotary_embs
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
            self.attn_fn = lambda x : tf.math.square(tf.nn.relu(x))

        self.built = True

    def _attn(self, x, v):
        n = cast(x.shape[-2], x.dtype)
        z = self.to_qk(x)
        q, k = self.scale_offset(z)

        if self.use_rotary_embs:
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
    
class ScaledSin(Layer):
    """
    Sinusoidal Position Embedding with scaling factor. (ScaledSin)
    https://arxiv.org/pdf/2202.10447.pdf

    References:
    https://arxiv.org/pdf/1706.03762.pdf
    https://github.com/lucidrains/FLASH-pytorch (implementation logic)
    """

    def __init__(self):
        super().__init__()

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
                 use_rotary_embs : bool = False,
                 laplace_attn_fn : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
                     
        self.token_emb = Embedding(n_tokens, emb_dim)
        self.abs_pos_emb = ScaledSin()

        self.blocks = Sequential([
            GAU(
                qk_dim = qk_dim,
                expansion_factor = expansion_factor,
                causal = causal,
                dropout_rate = dropout_rate,
                norm_type = norm_type,
                shift_tokens = shift_tokens,
                use_rotary_embs = use_rotary_embs,
                laplace_attn_fn = laplace_attn_fn,
                name = f'gau{i}'
            ) for i in range(depth)], name = 'blocks')

        self.to_logits = Sequential([
            LayerNormalization(),
            Dense(n_tokens)
        ], name = 'logits')

    @tf.function(jit_compile = True)
    def call(self, x):
        x = self.token_emb(x)
        x = self.abs_pos_emb(x) + x
        x = self.blocks(x)
        return self.to_logits(x)
