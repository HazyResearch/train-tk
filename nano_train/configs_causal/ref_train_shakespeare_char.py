# character-level shakespeare model

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 100 # don't print too too often
always_save_checkpoint = False # we expect to overfit on this small dataset, so only save when val improves

# wandb
wandb_log = True 
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-ref'

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 768 # context of up to 256 previous characters

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
TK_kernel = False

# optimizer and scheduler
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially