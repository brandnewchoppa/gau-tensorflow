import tensorflow as tf

from keras.layers import Layer

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
            n = q.shape[-1]
            freqs, seq_pos = self._calc_freqs(q)

            power = (seq_pos - n // 2) / self.scale_base
            scale = self.scale ** tf.transpose(power[tf.newaxis])
            scale =  tf.concat([ scale, scale ], axis = -1)

            q = apply_rotary_emb(freqs, q, scale = scale)
            k = apply_rotary_emb(freqs, k, scale = scale ** -1)
            return q, k

        freqs, _ = self._calc_freqs(x)
        return apply_rotary_emb(freqs, x)