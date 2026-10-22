import inspect
from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange


# --- Minimal self-contained ConvTasNet & BaseEncoderMaskerDecoder components from Asteroid (MIT License) ---

def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def jitable_shape(tensor: torch.Tensor) -> torch.Tensor:
    return torch.tensor(tensor.shape)


def _unsqueeze_to_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    return x


def _shape_reconstructed(reconstructed: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


class _LayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        dims: List[int] = torch.arange(1, len(x.shape)).tolist()
        mean = x.mean(dim=dims, keepdim=True)
        var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        value = (x - mean) / torch.sqrt(var2 + EPS)
        return self.apply_gain_and_bias(value)


class ChanLN(_LayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        return self.apply_gain_and_bias((x - mean) / (var + EPS).sqrt())


class CumLN(_LayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        batch, chan, spec_len = x.size()
        cum_sum = torch.cumsum(x.sum(1, keepdim=True), dim=-1)
        cum_pow_sum = torch.cumsum(x.pow(2).sum(1, keepdim=True), dim=-1)
        cnt = torch.arange(
            start=chan, end=chan * (spec_len + 1), step=chan, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        cum_mean = cum_sum / cnt
        cum_var = cum_pow_sum / cnt - cum_mean.pow(2)
        return self.apply_gain_and_bias((x - cum_mean) / (cum_var + EPS).sqrt())


class FeatsGlobLN(_LayerNorm):
    def forward(self, x, EPS: float = 1e-8):
        dims: List[int] = torch.arange(2, len(x.shape)).tolist()
        mean = x.mean(dim=dims, keepdim=True)
        var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        value = (x - mean) / torch.sqrt(var2 + EPS)
        return self.apply_gain_and_bias(value)


class BatchNorm(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() < 2 or input.dim() > 4:
            raise ValueError(f"expected 4D or 3D input (got {input.dim()}D input)")


_NORMS = {
    "GlobLN": GlobLN, "gLN": GlobLN,
    "ChanLN": ChanLN, "cLN": ChanLN,
    "CumLN": CumLN, "cgLN": CumLN,
    "FeatsGlobLN": FeatsGlobLN, "fgLN": FeatsGlobLN,
    "BatchNorm": BatchNorm, "bN": BatchNorm, "BN": BatchNorm,
}


def _get_norm(identifier):
    if identifier is None:
        return None
    if callable(identifier):
        return identifier
    if isinstance(identifier, str) and identifier in _NORMS:
        return _NORMS[identifier]
    raise ValueError(f"Could not interpret normalization identifier: {identifier}")


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


_ACTIVATIONS = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": lambda dim=None: nn.Softmax(dim=dim),
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "swish": Swish,
}


def _get_activation(identifier):
    if identifier is None:
        return None
    if callable(identifier):
        return identifier
    if isinstance(identifier, str) and identifier in _ACTIVATIONS:
        return _ACTIVATIONS[identifier]
    raise ValueError(f"Could not interpret activation identifier: {identifier}")


class _Chop1d(nn.Module):
    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = _get_norm(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = _get_norm(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2**x // 2 if not causal else (conv_kernel_size - 1) * 2**x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = _get_activation(mask_act)
        sig = inspect.signature(mask_nl_class)
        if "dim" in sig.parameters:
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        return self.output_act(score)


class BaseEncoderMaskerDecoder(nn.Module):
    def __init__(self, encoder, masker, decoder, encoder_activation=None):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.encoder_activation = encoder_activation
        self.enc_activation = _get_activation(encoder_activation or "linear")()

    def forward_encoder(self, wav: torch.Tensor) -> torch.Tensor:
        tf_rep = self.encoder(wav)
        return self.enc_activation(tf_rep)

    def forward_masker(self, tf_rep: torch.Tensor) -> torch.Tensor:
        return self.masker(tf_rep)

    def apply_masks(self, tf_rep: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
        return est_masks * tf_rep.unsqueeze(1)

    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        return self.decoder(masked_tf_rep)


# --- De-limiter Model Wrappers ---

class BaseEncoderMaskerDecoderWithConfigs(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav) if self.use_encoder else wav
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks) if self.apply_mask else est_masks

        if self.use_decoder:
            decoded = self.forward_decoder(masked_tf_rep)
            reconstructed = pad_x_to_y(decoded, wav)
            return masked_tf_rep, _shape_reconstructed(reconstructed, shape)
        return masked_tf_rep


class BaseEncoderMaskerDecoderWithConfigsMaskOnOutput(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)
        self.nb_channels = kwargs.get("nb_channels", 2)
        self.decoder_activation = kwargs.get("decoder_activation", "sigmoid")
        if self.decoder_activation == "sigmoid":
            self.act_after_dec = nn.Sigmoid()
        elif self.decoder_activation == "relu":
            self.act_after_dec = nn.ReLU()
        elif self.decoder_activation == "relu6":
            self.act_after_dec = nn.ReLU6()
        elif self.decoder_activation == "tanh":
            self.act_after_dec = nn.Tanh()
        elif self.decoder_activation == "none":
            self.act_after_dec = nn.Identity()
        else:
            self.act_after_dec = nn.Sigmoid()

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav) if self.use_encoder else wav
        if self.nb_channels == 2:
            tf_rep = rearrange(tf_rep, "b c f t -> b (c f) t")
        est_masks = self.forward_masker(tf_rep)

        if self.use_decoder:
            if self.nb_channels == 2:
                est_masks = rearrange(est_masks, "b 1 f t -> b f t")
            est_masks_decoded = self.forward_decoder(est_masks)
            est_masks_decoded = pad_x_to_y(est_masks_decoded, wav)
            est_masks_decoded = self.act_after_dec(est_masks_decoded)
            decoded = wav * est_masks_decoded
            return est_masks_decoded, decoded
        return (est_masks,)


class BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid(BaseEncoderMaskerDecoder):
    def __init__(self, encoder, masker, decoder, encoder_activation=None, **kwargs):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.use_encoder = kwargs.get("use_encoder", True)
        self.apply_mask = kwargs.get("apply_mask", True)
        self.use_decoder = kwargs.get("use_decoder", True)
        self.nb_channels = kwargs.get("nb_channels", 2)
        self.decoder_activation = kwargs.get("decoder_activation", "none")
        if self.decoder_activation == "sigmoid":
            self.act_after_dec = nn.Sigmoid()
        elif self.decoder_activation == "relu":
            self.act_after_dec = nn.ReLU()
        elif self.decoder_activation == "relu6":
            self.act_after_dec = nn.ReLU6()
        elif self.decoder_activation == "tanh":
            self.act_after_dec = nn.Tanh()
        elif self.decoder_activation == "none":
            self.act_after_dec = nn.Identity()
        else:
            self.act_after_dec = nn.Sigmoid()

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav) if self.use_encoder else wav
        if self.nb_channels == 2:
            tf_rep = rearrange(tf_rep, "b c f t -> b (c f) t")
        est_masks = self.forward_masker(tf_rep)
        if self.nb_channels == 2:
            tf_rep = rearrange(tf_rep, "b (c f) t -> b c f t", c=self.nb_channels)
        masked_tf_rep = est_masks * tf_rep if self.apply_mask else est_masks

        if self.use_decoder:
            decoded = self.forward_decoder(masked_tf_rep)
            reconstructed = pad_x_to_y(decoded, wav)
            reconstructed = self.act_after_dec(reconstructed)
            return masked_tf_rep, _shape_reconstructed(reconstructed, shape)
        return masked_tf_rep
