import torch

from comfy import latent_formats

from .taehv import TAEHV

LATENT_FORMATS_CLASS = {
    "taehv": latent_formats.HunyuanVideo,
    "taew2_1": latent_formats.Wan21,
}

class TAEVDWrapper(TAEHV):
    def __init__(self, model_name, decoder_time_upscale=(True, True), decoder_space_upscale=(True, True, True)):
        super().__init__(checkpoint_path=None, decoder_time_upscale=decoder_time_upscale, decoder_space_upscale=decoder_space_upscale)
        assert model_name in ["taehv", "taew2_1"]
        self.model_name = model_name

    def encode(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.shape[2] == 1: # has only one frame:
            x = torch.expand_copy(x, (-1, -1, 4, -1, -1))

        x = x.mul_(0.5).add_(0.5)
        x = x.transpose(1, 2)
        x = self.encode_video(x, parallel=False, show_progress_bar=False)
        x = x.transpose(1, 2)
        x = LATENT_FORMATS_CLASS[self.model_name]().process_out(x)

        return x

    def decode(self, x):
        x = LATENT_FORMATS_CLASS[self.model_name]().process_in(x)

        x = x.transpose(1, 2)
        x = self.decode_video(x, parallel=False, show_progress_bar=False)
        x = x.transpose(2, 1)
        x = x.mul_(2).sub_(1)

        return x
