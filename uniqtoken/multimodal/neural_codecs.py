from __future__ import annotations

import math
from typing import Any, cast, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =========================================================================
# 1. 2D VISUAL NEURAL CODEC (VQ-VAE Architecture)
# =========================================================================

if HAS_TORCH:

    class ResidualBlock2D(nn.Module):
        """2D Residual Convolutional Block with GELU activations."""

        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.act(self.conv1(x))
            out = self.conv2(out)
            return self.act(out + residual)

    class VisualEncoder(nn.Module):
        """
        2D Convolutional Encoder downsampling images by 4x into latent feature maps.
        Input: [B, C, H, W] -> Output: [B, D, H/4, W/4]
        """

        def __init__(self, in_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 128):
            super().__init__()
            self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)  # 2x down
            self.res1 = ResidualBlock2D(hidden_dim)
            self.conv_down = nn.Conv2d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1)  # 2x down
            self.res2 = ResidualBlock2D(latent_dim)
            self.act = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.conv_in(x))
            h = self.res1(h)
            h = self.act(self.conv_down(h))
            h = self.res2(h)
            return h

    class VisualDecoder(nn.Module):
        """
        2D Transposed Convolutional Decoder upsampling latent feature maps by 4x back to pixels.
        Input: [B, D, H/4, W/4] -> Output: [B, C, H, W]
        """

        def __init__(self, out_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 128):
            super().__init__()
            self.res1 = ResidualBlock2D(latent_dim)
            self.conv_up1 = nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1)  # 2x up
            self.res2 = ResidualBlock2D(hidden_dim)
            self.conv_up2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)  # 2x up
            self.act = nn.GELU()

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = self.res1(z)
            h = self.act(self.conv_up1(h))
            h = self.res2(h)
            h = self.conv_up2(h)
            return h

    class NeuralVisualCodec(nn.Module):
        """
        End-to-End 2D Neural Visual Codec (VQ-VAE).

        Bridges continuous 2D images to discrete codebook tokens using
        learned Convolutional Encoders, Straight-Through Estimator (STE) Quantization,
        and Transposed Convolutional Decoders.
        """

        def __init__(
            self,
            in_channels: int = 3,
            hidden_dim: int = 64,
            latent_dim: int = 128,
            num_tokens: int = 512,
            beta_commit: float = 0.25,
        ):
            if num_tokens < 1:
                raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
            if latent_dim < 1 or hidden_dim < 1 or in_channels < 1:
                raise ValueError("in_channels, hidden_dim, latent_dim must be >= 1")
            super().__init__()
            self.encoder = VisualEncoder(in_channels, hidden_dim, latent_dim)
            self.decoder = VisualDecoder(in_channels, hidden_dim, latent_dim)
            self.embedding = nn.Embedding(num_tokens, latent_dim)
            self.embedding.weight.data.uniform_(-1.0 / math.sqrt(latent_dim), 1.0 / math.sqrt(latent_dim))
            self.num_tokens = num_tokens
            self.latent_dim = latent_dim
            self.beta_commit = beta_commit

        @staticmethod
        def _pad_image(x: torch.Tensor) -> torch.Tensor:
            """Pad right and bottom edges so strided layers preserve all pixels."""
            if x.dim() != 4:
                raise ValueError("image tensor must have shape [batch, channels, height, width]")
            pad_h = (-x.shape[-2]) % 4
            pad_w = (-x.shape[-1]) % 4
            return F.pad(x, (0, pad_w, 0, pad_h))

        def encode_to_tokens(self, x: torch.Tensor) -> Tuple[List[str], torch.Tensor, Tuple[int, int]]:
            """
            Encodes an image tensor [B, C, H, W] into discrete visual token strings.
            Returns (token_strings, code_indices, (grid_h, grid_w)).
            """
            z_e = self.encoder(self._pad_image(x))  # [B, D, H', W']
            b, d, gh, gw = z_e.shape

            # Flatten spatial dimensions: [B*H'*W', D]
            z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)

            # Compute squared Euclidean distance to codebook vectors: ||z - e||^2
            dist = (
                torch.sum(z_flat**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            )
            indices = torch.argmin(dist, dim=1)  # [B*H'*W']

            token_strings = [f"<|vis_{idx.item():04d}|>" for idx in indices]
            return token_strings, indices.view(b, gh, gw), (gh, gw)

        def decode_from_indices(
            self,
            indices: torch.Tensor,
            grid_h: int,
            grid_w: int,
            output_size: Optional[Tuple[int, int]] = None,
        ) -> torch.Tensor:
            """
            Reconstructs the continuous image tensor from discrete code indices.
            """
            device = self.embedding.weight.device
            if indices.dim() == 1 and indices.numel() == grid_h * grid_w:
                b = 1
            elif indices.dim() == 3:
                b = indices.shape[0]
            elif indices.dim() == 2 and indices.shape[1] == grid_h * grid_w:
                b = indices.shape[0]  # [B, H*W] per-batch flat indices
            elif indices.dim() == 2 and indices.shape[0] == grid_h * grid_w:
                b = 1  # single flat image
            else:
                raise ValueError(f"indices shape {tuple(indices.shape)} is not compatible with grid {grid_h}x{grid_w}")
            if indices.device != device:
                indices = indices.to(device)
            flat_indices = indices.view(-1)
            if flat_indices.numel() != b * grid_h * grid_w:
                raise ValueError(
                    f"indices contain {flat_indices.numel()} entries, expected "
                    f"{b * grid_h * grid_w} for grid {grid_h}x{grid_w}"
                )
            z_q = self.embedding(flat_indices).view(b, grid_h, grid_w, self.latent_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, D, H', W']
            reconstructed = self.decoder(z_q)
            if output_size is not None:
                height, width = output_size
                if height < 1 or width < 1:
                    raise ValueError("output_size values must be positive")
                reconstructed = reconstructed[..., :height, :width]
            return reconstructed

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward training step with Straight-Through Estimator and Commitment Loss.
            """
            original_height, original_width = x.shape[-2:]
            z_e = self.encoder(self._pad_image(x))
            b, d, gh, gw = z_e.shape
            z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, d)

            dist = (
                torch.sum(z_flat**2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            )
            indices = torch.argmin(dist, dim=1)
            z_q_flat = self.embedding(indices)
            z_q = z_q_flat.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()

            # Straight-Through Estimator (STE)
            z_q_ste = z_e + (z_q - z_e).detach()
            x_recon = self.decoder(z_q_ste)[..., :original_height, :original_width]

            # VQ-VAE Loss Terms
            recon_loss = F.mse_loss(x_recon, x)
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_e, z_q.detach())
            total_loss = recon_loss + codebook_loss + self.beta_commit * commit_loss

            return {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "commit_loss": commit_loss,
                "x_recon": x_recon,
                "indices": indices.view(b, gh, gw),
            }


# =========================================================================
# 2. 1D ACOUSTIC NEURAL CODEC (EnCodec / SoundStream Architecture)
# =========================================================================

if HAS_TORCH:

    class ResidualBlock1D(nn.Module):
        """1D Dilated Residual Convolutional Block for Audio."""

        def __init__(self, channels: int, dilation: int = 1):
            super().__init__()
            self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.act(self.conv1(x))
            out = self.conv2(out)
            return self.act(out + residual)

    class AudioEncoder(nn.Module):
        """
        1D Convolutional Encoder downsampling 1D audio waveforms into temporal latents.
        Input: [B, 1, T] -> Output: [B, D, T / 320]
        """

        def __init__(self, in_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 128):
            super().__init__()
            # Strided downsampling: 4x, 4x, 4x, 5x = 320x total temporal downsampling
            self.conv_in = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, stride=1, padding=3)
            self.down1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=4, padding=2)
            self.res1 = ResidualBlock1D(hidden_dim, dilation=1)
            self.down2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=8, stride=4, padding=2)
            self.res2 = ResidualBlock1D(hidden_dim * 2, dilation=2)
            self.down3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=8, stride=4, padding=2)
            self.res3 = ResidualBlock1D(hidden_dim * 2, dilation=1)
            self.down4 = nn.Conv1d(hidden_dim * 2, latent_dim, kernel_size=10, stride=5, padding=3)
            self.res4 = ResidualBlock1D(latent_dim, dilation=1)
            self.act = nn.ELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.act(self.conv_in(x))
            h = self.res1(self.act(self.down1(h)))
            h = self.res2(self.act(self.down2(h)))
            h = self.res3(self.act(self.down3(h)))
            h = self.res4(self.act(self.down4(h)))
            return h

    class AudioDecoder(nn.Module):
        """
        1D Transposed Convolutional Decoder upsampling temporal latents 320x back to waveform.
        Input: [B, D, T / 320] -> Output: [B, 1, T]
        """

        def __init__(self, out_channels: int = 1, hidden_dim: int = 64, latent_dim: int = 128):
            super().__init__()
            # Upsampling: 5x, 4x, 4x, 4x = 320x total temporal upsampling
            self.res1 = ResidualBlock1D(latent_dim, dilation=1)
            self.up1 = nn.ConvTranspose1d(
                latent_dim,
                hidden_dim * 2,
                kernel_size=10,
                stride=5,
                padding=3,
                output_padding=1,
            )
            self.res2 = ResidualBlock1D(hidden_dim * 2, dilation=1)
            self.up2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, kernel_size=8, stride=4, padding=2)
            self.res3 = ResidualBlock1D(hidden_dim * 2, dilation=2)
            self.up3 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=8, stride=4, padding=2)
            self.res4 = ResidualBlock1D(hidden_dim, dilation=1)
            self.up4 = nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=8, stride=4, padding=2)
            self.conv_out = nn.Conv1d(hidden_dim, out_channels, kernel_size=7, stride=1, padding=3)
            self.act = nn.ELU()

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h = self.res1(z)
            h = self.res2(self.act(self.up1(h)))
            h = self.res3(self.act(self.up2(h)))
            h = self.res4(self.act(self.up3(h)))
            h = self.act(self.up4(h))
            # tanh confines output to [-1, 1]: the L1 loss in NeuralAudioCodec
            # assumes waveforms are normalized to that range.
            out = torch.tanh(self.conv_out(h))
            return out

    class NeuralAudioCodec(nn.Module):
        """
        End-to-End 1D Neural Audio Codec with Learned Residual Vector Quantization (RVQ).

        Downsamples 16kHz continuous audio waveforms into 50Hz latent frames, quantizes each
        frame hierarchically through N_q learned codebooks, and synthesizes waveforms back.
        """

        def __init__(
            self,
            in_channels: int = 1,
            hidden_dim: int = 64,
            latent_dim: int = 128,
            num_quantizers: int = 4,
            codebook_size: int = 256,
        ):
            if num_quantizers < 1:
                raise ValueError(f"num_quantizers must be >= 1, got {num_quantizers}")
            if codebook_size < 1:
                raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
            if latent_dim < 1 or hidden_dim < 1 or in_channels < 1:
                raise ValueError("in_channels, hidden_dim, latent_dim must be >= 1")
            super().__init__()
            self.encoder = AudioEncoder(in_channels, hidden_dim, latent_dim)
            self.decoder = AudioDecoder(in_channels, hidden_dim, latent_dim)
            self.num_quantizers = num_quantizers
            self.codebook_size = codebook_size
            self.latent_dim = latent_dim

            # N_q codebook embedding layers
            self.quantizers = nn.ModuleList([nn.Embedding(codebook_size, latent_dim) for _ in range(num_quantizers)])
            for module in self.quantizers:
                quantizer = cast(nn.Embedding, module)
                quantizer.weight.data.uniform_(-1.0 / math.sqrt(latent_dim), 1.0 / math.sqrt(latent_dim))

        @staticmethod
        def _pad_audio(audio: torch.Tensor) -> torch.Tensor:
            """Pad waveforms to complete 320-sample encoder frames."""
            if audio.dim() != 3:
                raise ValueError("audio tensor must have shape [batch, channels, samples]")
            if audio.shape[-1] < 1:
                raise ValueError("audio tensor must contain at least one sample")
            target_length = max(320, math.ceil(audio.shape[-1] / 320) * 320)
            return F.pad(audio, (0, target_length - audio.shape[-1]))

        def encode_to_tokens(self, audio: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
            """
            Encodes audio waveform [B, 1, T] into hierarchical RVQ discrete tokens.
            """
            h = self.encoder(self._pad_audio(audio))  # [B, D, T']
            b, d, t_prime = h.shape

            residual = h.permute(0, 2, 1).contiguous().view(-1, d)  # [B*T', D]
            all_indices: List[torch.Tensor] = []
            for q_idx, module in enumerate(self.quantizers):
                quantizer = cast(nn.Embedding, module)
                dist = (
                    torch.sum(residual**2, dim=1, keepdim=True)
                    + torch.sum(quantizer.weight**2, dim=1)
                    - 2 * torch.matmul(residual, quantizer.weight.t())
                )
                indices = torch.argmin(dist, dim=1)  # [B*T']
                all_indices.append(indices)

                q_vec = quantizer(indices)
                residual = residual - q_vec

            stacked_indices = torch.stack(all_indices, dim=1)  # [B*T', N_q]

            indices_3d = stacked_indices.view(b, t_prime, self.num_quantizers)
            tokens: List[str] = []
            for batch_idx in range(b):
                tokens.extend(["<|audio_start|>", f"<|aud_len_{audio.shape[-1]}|>"])
                for frame_idx in range(t_prime):
                    for q_idx in range(self.num_quantizers):
                        code_id = indices_3d[batch_idx, frame_idx, q_idx].item()
                        tokens.append(f"<|aud_q{q_idx}_{code_id:04d}|>")
                tokens.append("<|audio_end|>")
            return tokens, indices_3d

        def decode_from_indices(self, indices: torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
            """
            Synthesizes waveform from hierarchical RVQ code indices [B, T', N_q].
            """
            b, t_prime, n_q = indices.shape
            if n_q != self.num_quantizers:
                raise ValueError(f"expected {self.num_quantizers} quantizer stages in indices, got {n_q}")
            param = next(self.parameters())
            device = param.device
            dtype = param.dtype
            if indices.device != device:
                indices = indices.to(device)
            z_q_total = torch.zeros(b * t_prime, self.latent_dim, device=device, dtype=dtype)

            flat_indices = indices.view(b * t_prime, n_q)
            for q_idx in range(n_q):
                q_ind = flat_indices[:, q_idx]
                z_q_total = z_q_total + self.quantizers[q_idx](q_ind)

            z_q = z_q_total.view(b, t_prime, self.latent_dim).permute(0, 2, 1).contiguous()
            reconstructed = self.decoder(z_q)
            if output_length is not None:
                if output_length < 1:
                    raise ValueError("output_length must be positive")
                reconstructed = reconstructed[..., :output_length]
            return reconstructed

        def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Forward training pass with Multi-Stage RVQ loss and Straight-Through Estimators.
            """
            original_audio = audio
            original_length = audio.shape[-1]
            h = self.encoder(self._pad_audio(audio))
            b, d, t_prime = h.shape
            z_e = h.permute(0, 2, 1).contiguous().view(-1, d)
            residual = z_e

            z_q_sum = torch.zeros_like(residual)
            loss_commit = torch.tensor(0.0, device=audio.device)

            all_indices: List[torch.Tensor] = []
            for q_idx, module in enumerate(self.quantizers):
                quantizer = cast(nn.Embedding, module)
                dist = (
                    torch.sum(residual**2, dim=1, keepdim=True)
                    + torch.sum(quantizer.weight**2, dim=1)
                    - 2 * torch.matmul(residual, quantizer.weight.t())
                )
                indices = torch.argmin(dist, dim=1)
                all_indices.append(indices)
                z_q = quantizer(indices)

                loss_commit = (
                    loss_commit + F.mse_loss(z_q, residual.detach()) + 0.25 * F.mse_loss(residual, z_q.detach())
                )
                residual = residual - z_q
                z_q_sum = z_q_sum + z_q

            # Straight-Through Estimator
            z_q_ste = z_e + (z_q_sum - z_e).detach()
            z_q_ste = z_q_ste.view(b, t_prime, d).permute(0, 2, 1).contiguous()
            audio_recon = self.decoder(z_q_ste)[..., :original_length]

            loss_recon = F.l1_loss(audio_recon, original_audio)
            total_loss = loss_recon + loss_commit

            return {
                "loss": total_loss,
                "recon_loss": loss_recon,
                "commit_loss": loss_commit,
                "audio_recon": audio_recon,
                "indices": torch.stack(all_indices, dim=1).view(b, t_prime, self.num_quantizers),
            }
else:
    NeuralVisualCodec: Any = None  # type: ignore[no-redef]
    NeuralAudioCodec: Any = None  # type: ignore[no-redef]


# =========================================================================
# 3. FALLBACK / REFERENCE WRAPPER
# =========================================================================


class NeuralCodecFacade:
    """
    Public Facade providing unified access to Neural Visual & Audio Codecs.
    """

    @staticmethod
    def is_available() -> bool:
        return HAS_TORCH

    @staticmethod
    def get_visual_codec(num_tokens: int = 512) -> Optional[Any]:
        if not HAS_TORCH:
            return None
        return NeuralVisualCodec(num_tokens=num_tokens)

    @staticmethod
    def get_audio_codec(num_quantizers: int = 4, codebook_size: int = 256) -> Optional[Any]:
        if not HAS_TORCH:
            return None
        return NeuralAudioCodec(num_quantizers=num_quantizers, codebook_size=codebook_size)
