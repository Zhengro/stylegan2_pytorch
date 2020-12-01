import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from ops.upfirdn_2d import upsample_2d, upsample_conv_2d, conv_downsample_2d


# ----------------------------------------------------------------------------
# Main generator network.
class G_main(nn.Module):
    """Composed of two sub-networks (mapping and synthesis) that are defined below."""
    def __init__(self,
                 truncation_psi=0.5,            # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=None,        # Number of layers for which to apply the truncation trick. None = disable.
                 truncation_psi_val=None,       # Value for truncation_psi to use during validation.
                 truncation_cutoff_val=None,    # Value for truncation_cutoff to use during validation.
                 dlatent_avg_beta=0.995,        # Decay for tracking the moving average of W during training. None = disable.
                 style_mixing_prob=0.9,         # Probability of mixing styles during training. None = disable.
                 is_training=False,             # Network is under training? Enables and disables specific features.
                 is_validation=False,           # Network is under validation? Chooses which value to use for truncation_psi.
                 return_dlatents=False,         # Return dlatents in addition to the images?
                 **kwargs):                     # Arguments for sub-networks (mapping and synthesis).
        super(G_main, self).__init__()

        # Validate arguments.
        assert not is_training or not is_validation
        if is_validation:
            truncation_psi = truncation_psi_val
            truncation_cutoff = truncation_cutoff_val
        if is_training or (truncation_psi is not None and not isinstance(truncation_psi, torch.Tensor) and truncation_psi == 1):
            truncation_psi = None
        if is_training:
            truncation_cutoff = None
        if not is_training or (dlatent_avg_beta is not None and not isinstance(dlatent_avg_beta, torch.Tensor) and dlatent_avg_beta == 1):
            dlatent_avg_beta = None
        if not is_training or (style_mixing_prob is not None and not isinstance(style_mixing_prob, torch.Tensor) and style_mixing_prob <= 0):
            style_mixing_prob = None

        self.dlatent_avg_beta = dlatent_avg_beta
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.return_dlatents = return_dlatents

        # Setup components.
        self.synthesis = G_synthesis_stylegan2(**kwargs)
        self.num_layers = self.synthesis.num_layers
        dlatent_size = self.synthesis.dlatent_size
        self.mapping = G_mapping(dlatent_broadcast=self.num_layers, **kwargs)

        # Setup variables.
        self.register_buffer("lod_in", torch.zeros([]))
        self.register_buffer("dlatent_avg", torch.zeros([dlatent_size]))

    def forward(self, latents_in, labels_in=None):
        """
        :param latents_in: Latent vectors (Z) [minibatch, latent_size].
        :param labels_in: Conditioning labels [minibatch, label_size].
        :return: images_out: Generated images [minibatch, num_channels, resolution, resolution] (see G_synthesis_stylegan2)
        """
        assert latents_in.dtype == torch.float32, "Data type of latents_in is wrong."
        # Evaluate mapping network.
        dlatents = self.mapping(latents_in, labels_in)

        # Update moving average of W.
        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = torch.mean(dlatents[:, 0], dim=0)
                self.dlatent_avg = batch_avg + (self.dlatent_avg - batch_avg) * self.dlatent_avg_beta

        # Perform style mixing regularization.
        if self.style_mixing_prob is not None:
            latents2 = torch.randn_like(latents_in)
            dlatents2 = self.mapping(latents2, labels_in)
            layer_idx = torch.arange(self.num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = self.num_layers - self.lod_in.to(dtype=torch.int32) * 2
            if torch.rand([]) < self.style_mixing_prob:
                mixing_cutoff = torch.randint(low=1, high=cur_layers, size=[], dtype=torch.int32)
            else:
                mixing_cutoff = cur_layers
            dlatents = torch.where((layer_idx < mixing_cutoff).expand(dlatents.shape), dlatents, dlatents2)

        # Apply truncation trick.
        if self.truncation_psi is not None:
            layer_idx = np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
            layer_psi = np.ones(layer_idx.shape, dtype=np.float32)
            if self.truncation_cutoff is None:
                layer_psi *= self.truncation_psi
            else:
                layer_psi = torch.where(layer_idx < self.truncation_cutoff, layer_psi * self.truncation_psi, layer_psi)
            dlatents = self.dlatent_avg + (dlatents - self.dlatent_avg) * layer_psi

        # Evaluate synthesis network.
        images_out = self.synthesis(dlatents)

        # Return requested outputs.
        if self.return_dlatents:
            return images_out, dlatents
        return images_out


# ----------------------------------------------------------------------------
# Mapping network.
class G_mapping(nn.Module):
    """Transforms the input latent code (z) to the disentangled latent code (w)."""
    def __init__(self,
                 latent_size=512,              # Latent vector (Z) dimensionality.
                 label_size=0,                 # Label dimensionality, 0 if no labels.
                 dlatent_size=512,             # Disentangled latent (W) dimensionality.
                 dlatent_broadcast=None,       # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
                 mapping_layers=8,             # Number of mapping layers.
                 mapping_fmaps=512,            # Number of activations in the mapping layers.
                 mapping_lrmul=0.01,           # Learning rate multiplier for the mapping layers.
                 normalize_latents=True,       # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 dtype=torch.float32,          # Data type to use for activations and outputs.
                 **_kwargs):                   # Ignore unrecognized keyword args.
        super(G_mapping, self).__init__()

        self.dlatent_size = dlatent_size
        self.normalize_latents = normalize_latents
        self.mapping_lrmul = mapping_lrmul
        self.dlatent_broadcast = dlatent_broadcast
        self.dtype = dtype

        if label_size:
            self.embedding_layer = nn.Linear(label_size, latent_size, bias=False)
            nn.init.normal_(self.embedding_layer.weight)
            in_features = latent_size * 2
        else:
            in_features = latent_size

        self.dense_weight = nn.ParameterList()
        self.runtime_coef = []
        self.dense_bias = nn.ParameterList()
        hidden_dims = [dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps for layer_idx in range(mapping_layers)]
        for i, h_dim in enumerate(hidden_dims):

            dense_weight = nn.Parameter(torch.normal(mean=0, std=1.0 / self.mapping_lrmul, size=[h_dim, in_features]))
            self.dense_weight.append(dense_weight)
            self.runtime_coef.append(1 / np.sqrt(in_features) * self.mapping_lrmul)

            dense_bias = nn.Parameter(torch.zeros([h_dim]))
            self.dense_bias.append(dense_bias)
            in_features = h_dim

    def forward(self, latents_in, labels_in=None):
        """
        :param latents_in: Latent vectors (Z) [minibatch, latent_size].
        :param labels_in: Conditioning labels [minibatch, label_size].
        :return: dlatents: Disentangled latent (W) [minibatch, dlatent_broadcast, latent_size].
        """
        # Inputs.
        if labels_in is not None:
            # Embed labels and concatenate them with latents.
            x = torch.cat((latents_in, self.embedding_layer(labels_in)), dim=1)
        else:
            x = latents_in
        if self.normalize_latents:
            # Normalize latents.
            x = x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-8)

        # Mapping layers.
        for i, (dense_weight, dense_bias) in enumerate(zip(self.dense_weight, self.dense_bias)):
            x = F.linear(x, dense_weight * self.runtime_coef[i], bias=dense_bias * self.mapping_lrmul)
            x = F.leaky_relu(x, negative_slope=0.2) * np.sqrt(2)  # Multiply np.sqrt(2) (See original implementation of fused_bias_act.py)

        if self.dlatent_broadcast is not None:
            # Broadcast.
            x = x[:, np.newaxis].repeat(1, self.dlatent_broadcast, 1)
            assert list(x.shape) == [latents_in.shape[0], self.dlatent_broadcast, self.dlatent_size]

        assert x.dtype == self.dtype, "Data type of dlatents is wrong."
        return x


# ----------------------------------------------------------------------------
# StyleGAN2 synthesis network.
class G_synthesis_stylegan2(nn.Module):
    """Implements skip connections."""
    def __init__(self,
                 dlatent_size=512,              # Disentangled latent (W) dimensionality.
                 num_channels=3,                # Number of output color channels.
                 resolution=1024,               # Output resolution.
                 fmap_base=16 << 10,            # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,                # log2 feature map reduction when doubling the resolution.
                 fmap_min=1,                    # Minimum number of feature maps in any layer.
                 fmap_max=512,                  # Maximum number of feature maps in any layer.
                 randomize_noise=True,          # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
                 architecture='skip',           # Architecture: 'orig', 'skip', 'resnet'.
                 dtype=torch.float32,           # Data type to use for activations and outputs.
                 resample_kernel=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
                 fused_modconv=True,            # Implement modulated_conv2d_layer() as a single fused op?
                 **_kwargs):                    # Ignore unrecognized keyword args.
        super(G_synthesis_stylegan2, self).__init__()

        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        assert architecture in ['orig', 'skip', 'resnet']
        self.num_layers = self.resolution_log2 * 2 - 2
        self.dlatent_size = dlatent_size
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_min = fmap_min
        self.fmap_max = fmap_max
        self.randomize_noise = randomize_noise
        self.architecture = architecture
        self.dtype = dtype
        self.resample_kernel = resample_kernel
        self.kernel = 3

        # Noise inputs.
        for layer_idx in range(self.num_layers - 1):
            res = (layer_idx + 5) // 2
            self.register_buffer(f"noise_input{layer_idx}", torch.randn(size=[1, 1, 2 ** res, 2 ** res]))

        # Constant input.
        self.constant_input = nn.Parameter(torch.randn(size=[1, self.nf(1), 4, 4]))

        # Early layers.
        fmaps_in, fmaps_out = self.nf(1), self.nf(1)
        self.conv = ModulatedConv2dLayer(fmaps_in, fmaps_out, self.kernel, up=False, resample_kernel=self.resample_kernel)
        self.noise_bias = NoiseBiasBlock(self.randomize_noise, getattr(self, f"noise_input0"), fmaps_out=fmaps_out)
        if self.architecture == 'skip':
            self.torgb = TorgbBlock(fmaps_out, num_channels, resample_kernel=self.resample_kernel)

        # Main layers.
        self.convs = nn.ModuleList()
        self.noise_biass = nn.ModuleList()
        if self.architecture == 'skip':
            self.torgbs = nn.ModuleList()
        for res in range(3, self.resolution_log2 + 1):
            fmaps_out = self.nf(res-1)
            self.convs.append(ModulatedConv2dLayer(fmaps_in, fmaps_out, self.kernel, up=True, resample_kernel=self.resample_kernel))
            self.noise_biass.append(NoiseBiasBlock(self.randomize_noise, getattr(self, f"noise_input{res*2-5}"), fmaps_out=fmaps_out))
            self.convs.append(ModulatedConv2dLayer(fmaps_out, fmaps_out, self.kernel, up=False, resample_kernel=self.resample_kernel))
            self.noise_biass.append(NoiseBiasBlock(self.randomize_noise, getattr(self, f"noise_input{res*2-4}"), fmaps_out=fmaps_out))
            if self.architecture == 'skip' or res == self.resolution_log2:
                self.torgbs.append(TorgbBlock(fmaps_out, num_channels, resample_kernel=self.resample_kernel))
            fmaps_in = fmaps_out

    def nf(self, stage):
        return np.clip(int(self.fmap_base / (2.0 ** (stage * self.fmap_decay))), self.fmap_min, self.fmap_max)

    def forward(self, dlatents_in, resolution_log2=None):
        """
        :param dlatents_in: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        :return: images_out: Generated images [minibatch, num_channels, resolution, resolution]
        """
        if resolution_log2 is not None:
            self.resolution_log2 = resolution_log2
        # Early layers.
        y = None
        constant_input = self.constant_input.repeat(dlatents_in.shape[0], 1, 1, 1)
        out = self.conv(constant_input, dlatents_in[:, 0])
        out = self.noise_bias(out)
        if self.architecture == 'skip':
            y = self.torgb(out, dlatents_in[:, 1], y)

        # Main layers.
        for res in range(3, self.resolution_log2 + 1):
            out = self.convs[res*2-6](out, dlatents_in[:, res*2-5])
            out = self.noise_biass[res*2-6](out)
            out = self.convs[res*2-5](out, dlatents_in[:, res*2-4])
            out = self.noise_biass[res*2-5](out)
            if self.architecture == 'skip' or res == self.resolution_log2:
                y = self.torgbs[res-3](out, dlatents_in[:, res*2-3], y)

        images_out = y
        assert images_out.dtype == self.dtype
        return images_out


# ----------------------------------------------------------------------------
class TorgbBlock(nn.Module):
    def __init__(self,
                 fmaps_out,
                 num_channels,
                 resample_kernel=None):
        super(TorgbBlock, self).__init__()

        self.conv = ModulatedConv2dLayer(fmaps_in=fmaps_out, fmaps_out=num_channels, kernel=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros([1, num_channels, 1, 1]))
        self.resample_kernel = resample_kernel

    def forward(self, x, dlatent, y):
        out = self.conv(x, dlatent)
        out = out + self.bias
        if y is not None:
            y = upsample_2d(y, self.resample_kernel)
            out = out + y
        return out


# ----------------------------------------------------------------------------
class NoiseBiasBlock(nn.Module):
    def __init__(self,
                 randomize_noise,
                 noise_input,
                 fmaps_out):
        super(NoiseBiasBlock, self).__init__()

        self.randomize_noise = randomize_noise
        self.noise_input = noise_input
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros([1, fmaps_out, 1, 1]))

    def forward(self, x):
        if self.randomize_noise:
            noise = torch.randn(size=[x.shape[0], 1, x.shape[2], x.shape[3]])
        else:
            noise = self.noise_input
        x = x + noise.to(x.device) * self.noise_strength
        x = x + self.bias
        x = F.leaky_relu(x, negative_slope=0.2) * np.sqrt(2)
        return x


# ----------------------------------------------------------------------------
class ModulatedConv2dLayer(nn.Module):
    def __init__(self,
                 fmaps_in,
                 fmaps_out,
                 kernel,
                 dlatent_size=512,
                 up=False,
                 down=False,
                 demodulate=True,
                 resample_kernel=None):
        super(ModulatedConv2dLayer, self).__init__()

        self.fmaps_in = fmaps_in
        self.fmaps_out = fmaps_out
        self.kernel = kernel
        self.dlatent_size = dlatent_size
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.resample_kernel = resample_kernel
        self.padding = self.kernel // 2

        self.conv_weight = nn.Parameter(torch.randn([1, self.fmaps_out, self.fmaps_in, self.kernel, self.kernel]))
        self.convw_runtime_coef = 1 / np.sqrt(self.kernel * self.kernel * self.fmaps_in)
        self.dense_weight = nn.Parameter(torch.randn([self.fmaps_in, self.dlatent_size]))
        self.densew_runtime_coef = 1 / np.sqrt(self.dlatent_size)
        self.dense_bias = nn.Parameter(torch.ones([self.fmaps_in]))

    def forward(self, x, dlatent):
        """
        :param x: Convolution weights [minibatch, fmaps_in, height, width]
        :param dlatent: Disentangled latent of an layer [minibatch, dlatent_size].
        :return:
        """
        batch_size, fmaps_in, height, width = x.shape

        # Modulate.
        style = F.linear(dlatent, self.dense_weight * self.densew_runtime_coef, bias=self.dense_bias)  # [BI] Transform incoming W to style.
        conv_weight = self.conv_weight * self.convw_runtime_coef * style.view(batch_size, 1, fmaps_in, 1, 1)  # [BOIkk] Scale input feature maps.

        # Demodulate.
        if self.demodulate:
            d = torch.rsqrt(torch.sum(torch.square(conv_weight), dim=[2, 3, 4]) + 1e-8)  # [BO] Scaling factor.
            conv_weight = conv_weight * d.view(batch_size, self.fmaps_out, 1, 1, 1)  # [BOIkk] Scale output feature maps.

        # Reshape input.
        x = x.reshape(1, batch_size*fmaps_in, height, width)  # [1(BI)hw] Fused => reshape minibatch to convolution groups.
        conv_weight = conv_weight.view(batch_size * self.fmaps_out, fmaps_in, self.kernel, self.kernel)  # [(BO)Ikk]

        # Convolution with optional up/downsampling.
        if self.up:
            x = upsample_conv_2d(x, conv_weight, k=self.resample_kernel)
        elif self.down:
            x = conv_downsample_2d(x, conv_weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, conv_weight, padding=self.padding, groups=batch_size)
            x = x.view(batch_size, self.fmaps_out, x.shape[2], x.shape[3])
        return x
