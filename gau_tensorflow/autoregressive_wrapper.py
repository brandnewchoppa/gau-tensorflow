import tensorflow as tf

from keras import Model

def top_k_fn(logits, k = 0):
    '''
    Top-K Sampling (top_k)
    https://arxiv.org/pdf/1805.04833.pdf

    The K most likely next words are filtered and the probability mass is
    redistributed among only those K next words.
    '''

    top_k = tf.clip_by_value(k, clip_value_min = 1, clip_value_max = logits.shape[-1])
    values, _ = tf.math.top_k(logits, k = top_k)
    indices_to_remove = logits < values[..., -1][tf.newaxis]
    return tf.where(indices_to_remove, tf.fill(logits.shape, float('-inf')), tf.cast(logits, 'float32'))

def top_p_fn(logits, p = .9):
    '''
    Top-P Sampling (top_p)
    https://arxiv.org/pdf/1904.09751.pdf

    Chooses from the smallest possible set of words whose cumulative probability
    exceeds the probability p.
    '''

    sorted_logits = tf.cast(tf.sort(logits, direction = 'DESCENDING', axis = -1), 'float32')
    cum_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis = -1), axis = -1)
    sorted_indices_to_remove = cum_probs > (1 - p)
    sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis = -1)
    sorted_indices_to_remove = tf.concat([
        tf.zeros_like(sorted_indices_to_remove[:, :1]),
        sorted_indices_to_remove[:, 1:]
    ], axis = -1)
    return tf.where(sorted_indices_to_remove, tf.fill(logits.shape, float('-inf')), sorted_logits)

class AutoregressiveWrapper(Model):
    def __init__(self,
                 model,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def generate(self,
                 input_tokens : tf.Tensor,
                 max_new_tokens : int,
                 eos_token = None,
                 temperature : float = 1.0,
                 top_k : int = 50,
                 top_p : float = 1.0,
                 max_tokens : int = 20):
        out = tf.TensorArray(
            dtype = input_tokens.dtype,
            size = 0,
            dynamic_size = True)

        for idx, token in enumerate(input_tokens[-1]):
            out = out.write(idx, token[tf.newaxis])

        size = tf.size(input_tokens)
        for i in tf.range(size, size + max_new_tokens):

            if i + 1 == max_tokens:
                break

            x = tf.transpose(out.stack())
            logits = self.model(x, training = False)[:, -1, :]

            logits = top_k_fn(x, top_k)
            logits = top_p_fn(x, top_p)
            probs = tf.nn.softmax(logits / temperature, axis = -1)
            sample = tf.random.categorical(probs, 1, dtype = input_tokens.dtype)

            out = out.write(i + 1, sample[0])

            if sample == eos_token:
                break

        out = tf.transpose(out.stack())
        return out

    def call(self, x : tf.Tensor, training : bool = False):
        return self.model(x, training = training)
