from gau_tensorflow import GAUTransformer

import tensorflow as tf
from keras import mixed_precision

import time, os, psutil
from tqdm import tqdm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    strategy = tf.distribute.OneDeviceStrategy(device = gpus[0])
    mixed_precision.set_global_policy('mixed_float16')

process = psutil.Process(os.getpid())

## Load the tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

## Model
CONTEXT_SIZE = 512
EMBEDDING_DIM = 128
VOCAB_SIZE = len(tokenizer)
DEPTH = 6

## Training
LR = 1e-04
EPOCHS = 30
BATCH_SIZE = 64

with strategy.scope():
    model = GAUTransformer(
        emb_dim = EMBEDDING_DIM,
        n_tokens = VOCAB_SIZE,
        depth = DEPTH,
        causal = True,
        use_rotary_embs = True,
        laplace_attn_fn = True
    )

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate = LR,
        decay_steps = EPOCHS,
        end_learning_rate = 1e-1
    )

    optimizer = tf.keras.optimizers.Lion(
        learning_rate = lr_schedule,
        weight_decay = 1e-02,
        clipvalue = 0.75,
        jit_compile = False
    )

    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction = tf.keras.losses.Reduction.NONE
    )

print(f'Global Policy: {mixed_precision.global_policy().name}')
print(f'Compute dtype: {model.compute_dtype}')
print(f'Variable dtype: {model.variable_dtype}')

## Initialize the model weights by a starting shape
model.build(tf.TensorShape([None, CONTEXT_SIZE]))

## Preprocess the dataset
from datasets import load_dataset

wikitext = load_dataset('wikitext', 'wikitext-2-v1', split = 'train')

## Context size + extra shift token
block_size = CONTEXT_SIZE + 1

def group_examples(examples):
    examples = tokenizer(examples['text'])
    concatenated_examples = { key : sum(examples[key], []) for key in examples.keys() }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        key : [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for key, t in concatenated_examples.items()
    }

    ## Minor optimization possibility by leaving behind the labels
    #result['labels'] = result['input_ids'].copy()

    return result

## Tokenize text
tokenized_set = wikitext.map(
    group_examples,
    batched = True,
    drop_last_batch = True,
    remove_columns = ['text'],
    num_proc = 4
)

print(f'Number of tokens: {len(tokenized_set) * block_size : ,}\n\n')
print(tokenized_set)

## Create the Prefetch dataset
train_set = tokenized_set.to_tf_dataset(
    columns = 'input_ids',
    label_cols = [],
    batch_size = BATCH_SIZE,
    shuffle = True,
    prefetch = tf.data.AUTOTUNE,
    num_workers = 4
)

## Cast to tf.uint16 to save memory
train_set = train_set.map(lambda x : tf.cast(x, tf.uint16))

print(train_set, end = '\n\n')
print(train_set.take(1).get_single_element())

input_signature = [
    tf.TensorSpec(shape = (None, CONTEXT_SIZE), dtype = tf.uint16, name = 'tokens'),
    tf.TensorSpec(shape = (None, CONTEXT_SIZE), dtype = tf.uint16, name = 'labels')
]

@tf.function(input_signature = input_signature, jit_compile = True)
def train_step(tokens, labels):
    with tf.GradientTape() as tape:
        logits = model(tokens, training = True)
        loss = tf.reduce_mean(model.loss(labels, logits))
        scaled_loss = model.optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(scaled_grads)
    return loss, grads

for epoch in range(EPOCHS):

    ## Track time and loss
    total_loss = 0
    start_time = time.time()

    ## Iterate over the batches of the dataset
    pbar = tqdm(train_set, desc = 'Epoch : ' + str(epoch+1) + '/' + str(EPOCHS))
    for step, batch in enumerate(pbar):

        ## Shift tokens
        tokens, labels = batch[:, :-1], batch[:, 1:]

        ## Forward pass
        loss, grads = train_step(tokens, labels)

        ## Update the model parameters
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        ## Track epoch
        total_loss += loss
        epoch_time = time.time() - start_time
        pbar.set_postfix_str('Loss: {0:.2f}, Time: {1:.2f}s, Mem: {2:.2f}%'.format(
            total_loss.numpy() / len(train_set), epoch_time, process.memory_percent()),
            refresh = True)