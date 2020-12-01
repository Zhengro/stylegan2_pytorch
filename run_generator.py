import torch
import numpy as np
import PIL.Image
from networks_stylegan2 import G_main


def convert_to_unit8(image):
    drange = [image.min(), image.max()]
    scale = 255 / (drange[1] - drange[0])
    image = image * scale + (0.5 - drange[0] * scale)
    return image.astype(np.uint8)


def generate_images_from_dlatents():
    g = G_main(is_validation=True, randomize_noise=False)
    g.load_state_dict(torch.load("stylegan2-generator.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} ...")
    g.to(device)
    state_dict = g.state_dict()

    # generate average face
    synthesis = g.synthesis
    w_avg = state_dict["dlatent_avg"]
    w_avg = w_avg[np.newaxis, np.newaxis, :]
    w_avg = w_avg.repeat(1, 18, 1).to(device)
    image_out = synthesis(w_avg)
    image_out = image_out[0].permute(1, 2, 0).detach().cpu().numpy()
    image_out = convert_to_unit8(image_out)
    PIL.Image.fromarray(image_out, "RGB").save("avg.png")


def generate_images_from_seeds():
    g = G_main(is_validation=True, randomize_noise=False)
    g.load_state_dict(torch.load("stylegan2-generator.pt"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} ...")
    g.to(device)

    # generate random faces
    seeds = range(20)
    for seed in seeds:
        rnd = np.random.RandomState(seed)
        z = torch.as_tensor(rnd.randn(1, 512), dtype=torch.float32, device=device)
        image_out = g(z)
        image_out = image_out[0].permute(1, 2, 0).detach().cpu().numpy()
        image_out = convert_to_unit8(image_out)
        PIL.Image.fromarray(image_out, "RGB").save(f"seed{seed}.png")


if __name__ == "__main__":

    generate_images_from_seeds()
