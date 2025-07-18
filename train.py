import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    flash_attn = False
)
diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,
    sampling_timesteps=50,
    objective='pred_x0',
    min_snr_loss_weight=True,
    min_snr_gamma=5,
    offset_noise_strength=0.0,
    adaptive=True, # True for spatially adaptive noise
)

trainer = Trainer(
    diffusion,
    "/home/hice1/avarma49/scratch/ffhq-128/",
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder="/home/hice1/avarma49/scratch/results/results128_semantic_mask",
    # results_folder="/home/hice1/avarma49/scratch/results/results128_edgeaware_mask",
    calculate_fid = True,              # whether to calculate fid during training
    num_fid_samples=5000,
    spatial_mask_type = "semantic",
    mask_folder = "/home/hice1/avarma49/scratch/ffhq-128-masks" # spatial_mask_type should be semantic
)

# resume from checkpoint
# trainer.load(131)
trainer.train()