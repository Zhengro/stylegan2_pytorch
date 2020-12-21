import torch
import numpy as np
from networks_stylegan2 import D_stylegan2, G_main

g = G_main(is_validation=True, randomize_noise=False)
g.load_state_dict(torch.load("stylegan2-generator.pt"))
d = D_stylegan2()
d.load_state_dict(torch.load("stylegan2-discriminator.pt"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} ...")
g.to(device)
d.to(device)

# generate random faces and print scores
seeds = range(20)
for seed in seeds:
    rnd = np.random.RandomState(seed)
    z = torch.as_tensor(rnd.randn(1, 512), dtype=torch.float32, device=device)
    image_out = g(z)
    score_out = d(image_out, None)
    print(score_out)
