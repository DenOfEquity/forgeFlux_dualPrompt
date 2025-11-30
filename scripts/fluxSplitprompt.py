from backend import memory_management
from backend.modules.k_prediction import PredictionFlux, PredictionDiscreteFlow

from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.sdxl import StableDiffusionXL
try:
    from backend.diffusion_engine.sd35 import StableDiffusion3
except:
    StableDiffusion3 = None

import gradio
from gradio_rangeslider import RangeSlider
import torch
import math
import numpy
import torchvision.transforms.functional as TF

from modules import scripts, shared, images
from modules.ui_components import InputAccordion#, ToolButton
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes
from modules_forge.forge_canvas.canvas import ForgeCanvas
from PIL import Image, ImageFilter
from modules.api.api import decode_base64_to_image


##  Flux transparent VAE - from https://github.com/RedAIGC/Flux-version-LayerDiffuse
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, 16, kernel_size=3, padding=1, stride=1)),
        )
    def forward(self, x):
        return self.blocks(x)


class UNet1024(torch.nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 4,
        down_block_types: tuple = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: tuple = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: tuple = (32, 32, 64, 128, 256, 512, 512), layers_per_block: int = 2,
        mid_block_scale_factor: float = 1, downsample_padding: int = 1, downsample_type: str = "conv",
        upsample_type: str = "conv", dropout: float = 0.0, act_fn: str = "silu",
        attention_head_dim: int = 8, norm_num_groups: int = 4,
        norm_eps: float = 1e-5, latent_c: int = 16,
    ):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(torch.nn.Conv2d(latent_c, block_out_channels[2], kernel_size=1))
        self.down_blocks = torch.nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = torch.nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block( down_block_type, num_layers=layers_per_block, in_channels=input_channel,
                out_channels=output_channel, temb_channels=None, add_downsample=not is_final_block, resnet_eps=norm_eps,
                resnet_act_fn=act_fn, resnet_groups=norm_num_groups, attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding, resnet_time_scale_shift="default", downsample_type=downsample_type, dropout=dropout,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1], temb_channels=None, dropout=dropout, resnet_eps=norm_eps, resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor, resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups, attn_groups=None, add_attention=True,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type, num_layers=layers_per_block + 1, in_channels=input_channel, out_channels=output_channel,
                prev_output_channel=prev_output_channel, temb_channels=None, add_upsample=not is_final_block,
                resnet_eps=norm_eps, resnet_act_fn=act_fn, resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default", upsample_type=upsample_type, dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        sample = self.mid_block(sample, emb)
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class FluxTransparentVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, alpha=300.0):
        super().__init__()
        self.dtype = dtype

        self.encoder = LatentTransparencyOffsetEncoder()
        # self.encoder.to(dtype=self.dtype)
        self.alpha = alpha
        self.decoder = UNet1024()
        # self.decoder.to(dtype=self.dtype)


    def decode(self, origin_pixel, latent):
        latent_for_decoder = latent.to('cuda', dtype=self.dtype)
        origin_pixel_for_decoder = origin_pixel.to('cuda', dtype=self.dtype)
        y = self.decoder(origin_pixel_for_decoder, latent_for_decoder)
        return y.clamp(0,1)
##  End: Flux transparent VAE


class forgeMultiPrompt(scripts.Script):
    sorting_priority = 0

    glc_backup_flux = None
    glc_backup_sdxl = None
    glc_backup_sd3 = None
    clearConds = False
    sigmasBackup = None
    prediction_typeBackup = None
    text_encoder_device_backup = None
    text_encoder_offload_device_backup = None

    flux_use_T5 = True
    flux_use_CL = True
    SDXL_use_CL = True
    SDXL_use_CG = True
    SD3_use_CL = True
    SD3_use_CG = True
    SD3_use_T5 = True

    transparentVAE = None

    def __init__(self):
        if forgeMultiPrompt.glc_backup_flux is None:
            forgeMultiPrompt.glc_backup_flux = Flux.get_learned_conditioning
        if forgeMultiPrompt.glc_backup_sdxl is None:
            forgeMultiPrompt.glc_backup_sdxl = StableDiffusionXL.get_learned_conditioning
        if forgeMultiPrompt.glc_backup_sd3 is None and StableDiffusion3 is not None:
            forgeMultiPrompt.glc_backup_sd3 = StableDiffusion3.get_learned_conditioning
        if forgeMultiPrompt.text_encoder_device_backup is None:
            forgeMultiPrompt.text_encoder_device_backup = memory_management.text_encoder_device
            forgeMultiPrompt.text_encoder_offload_device_backup = memory_management.text_encoder_offload_device

    def splitPrompt (prompt, countTextEncoders):
        promptTE1 = []
        promptTE2 = []
        promptTE3 = []

        for p in prompt:
            splitPrompt = p.split('SPLIT')

            countSplits = min (countTextEncoders, len(splitPrompt))
            match countSplits:
                case 3:         #   sd3
                    promptTE1.append(splitPrompt[0].strip())
                    promptTE2.append(splitPrompt[1].strip())
                    promptTE3.append(splitPrompt[2].strip())
                case 2:         #   sdxl, flux, hunyuan future proofing or SD3 with incomplete SPLITs
                    promptTE1.append(splitPrompt[0].strip())
                    promptTE2.append(splitPrompt[1].strip())
                    promptTE3.append(p)
                case 1:         #   sd1,    or Any if SPLIT not used
                    promptTE1.append(p)
                    promptTE2.append(p)
                    promptTE3.append(p)
                case _:
                    promptTE1.append(p)
                    promptTE2.append(p)
                    promptTE3.append(p)

        return promptTE1, promptTE2, promptTE3

    def patched_text_encoder_offload():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        else:
            return forgeMultiPrompt.text_encoder_offload_device_backup()
    def patched_text_encoder_gpu2():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        else:
            return forgeMultiPrompt.text_encoder_device_backup()
    def patched_text_encoder_gpu():
        return torch.cuda.current_device()#torch.device("cuda")
    def patched_text_encoder_cpu():
        return memory_management.cpu#torch.device("cpu")


    @torch.inference_mode()
    def patched_glc_sd3(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        np = len(prompt)

        CLIPLprompt, CLIPGprompt, T5prompt = forgeMultiPrompt.splitPrompt (prompt, 3)

        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)
        if force_zero_negative_prompt:
            l_pooled = torch.zeros([np, 768])
            g_pooled = torch.zeros([np, 1280])
            cond_l = torch.zeros([np, 77, 768])
            cond_g = torch.zeros([np, 77, 1280])
            cond_t5 = torch.zeros([np, 256, 4096])
        else:
            if forgeMultiPrompt.SD3_use_CG:
                cond_g, g_pooled = self.text_processing_engine_g(CLIPGprompt)
            else:
                cond_g = torch.zeros([np, 77, 1280])
                g_pooled = torch.zeros([np, 1280])

            if forgeMultiPrompt.SD3_use_CL:
                cond_l, l_pooled = self.text_processing_engine_l(CLIPLprompt)
            else:
                cond_l = torch.zeros([np, 77, 768])
                l_pooled = torch.zeros([np, 768])

            if forgeMultiPrompt.SD3_use_T5 and getattr(shared.opts, 'sd3_enable_t5', True):
                cond_t5 = self.text_processing_engine_t5(T5prompt)
            else:
                cond_t5 = torch.zeros([np, 256, 4096])

        #   conds get concatenated later, in dimension 2, so sizes of dimension 1 must match
        #   padding with zero
        pad = cond_g.size(1) - cond_l.size(1)
        if pad > 0:
            padding = (0,0, 0, pad, 0,0)
            cond_l = torch.nn.functional.pad (cond_l, padding, mode='constant', value=0)
        elif pad < 0:
            padding = (0,0, 0, -pad, 0,0)
            cond_g = torch.nn.functional.pad (cond_g, padding, mode='constant', value=0)

        cond_lg = torch.cat([cond_l, cond_g.to(cond_l.device)], dim=-1)
        cond_lg = torch.nn.functional.pad(cond_lg, (0, 4096 - cond_lg.shape[-1]))

        if type(cond_t5) is list:
            crossattn = []
            for i in range(len(cond_t5)):
                ca = torch.cat([cond_lg[i], cond_t5[i]], dim=-2)
                crossattn.append(ca)
            cond = dict(
                crossattn=crossattn,
                vector=torch.cat([l_pooled, g_pooled.to(cond_l.device)], dim=-1),
            )
        else:
            cond = dict(
                crossattn=torch.cat([cond_lg, cond_t5.to(cond_l.device)], dim=-2),
                vector=torch.cat([l_pooled, g_pooled.to(cond_l.device)], dim=-1),
            )

        return cond


    @torch.inference_mode()
    def patched_glc_flux(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        np = len(prompt)

        #   make 2 prompt lists, split each prompt in original list based on 'SPLIT'
        CLIPprompt, T5prompt, _ = forgeMultiPrompt.splitPrompt (prompt, 2)

        if forgeMultiPrompt.flux_use_CL:
            cond_l, pooled_l = self.text_processing_engine_l(CLIPprompt)
        else:
            pooled_l = torch.zeros([np, 768])

        if forgeMultiPrompt.flux_use_T5:
            cond_t5 = self.text_processing_engine_t5(prompt)
        else:
            cond_t5 = torch.zeros([np, 256, 4096])

        cond = dict(crossattn=cond_t5, vector=pooled_l)

        if self.use_distilled_cfg_scale:
            distilled_cfg_scale = getattr(prompt, 'distilled_cfg_scale', 3.5) or 3.5
            cond['guidance'] = torch.FloatTensor([distilled_cfg_scale] * len(prompt))
            print(f'Distilled CFG Scale: {distilled_cfg_scale}')
        else:
            print('Distilled CFG Scale will be ignored for Schnell')

        return cond


    @torch.inference_mode()
    def patched_glc_sdxl(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        np = len(prompt)

        #   make 2 prompt lists, split each prompt in original list based on 'SPLIT'
        CLIPLprompt, CLIPGprompt, _ = forgeMultiPrompt.splitPrompt (prompt, 2)

        if forgeMultiPrompt.SDXL_use_CL:
            cond_l = self.text_processing_engine_l(CLIPLprompt)
        else:
            cond_l = torch.zeros([np, 77, 768])

        if forgeMultiPrompt.SDXL_use_CG:
            cond_g, clip_pooled = self.text_processing_engine_g(CLIPGprompt)
        else:
            cond_g = torch.zeros([np, 77, 1280])
            clip_pooled = torch.zeros([np, 1280])

        #   conds get concatenated later, in dimension 2, so sizes of dimension 1 must match
        #   padding with zero
        pad = cond_g.size(1) - cond_l.size(1)
        if pad > 0:
            padding = (0,0, 0, pad, 0,0)
            cond_l = torch.nn.functional.pad (cond_l, padding, mode='constant', value=0)
        elif pad < 0:
            padding = (0,0, 0, -pad, 0,0)
            cond_g = torch.nn.functional.pad (cond_g, padding, mode='constant', value=0)

        width = getattr(prompt, 'width', 1024) or 1024
        height = getattr(prompt, 'height', 1024) or 1024
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        crop_w = shared.opts.sdxl_crop_left
        crop_h = shared.opts.sdxl_crop_top
        target_width = width
        target_height = height

        out = [
            self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
            self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
            self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))
        ]

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1).to(clip_pooled)

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)

        if force_zero_negative_prompt:
            clip_pooled = torch.zeros_like(clip_pooled)
            cond_l = torch.zeros_like(cond_l)
            cond_g = torch.zeros_like(cond_g)

        cond = dict(
            crossattn=torch.cat([cond_l, cond_g.to(cond_l.device)], dim=2),
            vector=torch.cat([clip_pooled, flat.to(clip_pooled.device)], dim=1),
        )

        return cond

    def title(self):
        return "Forge2 extras"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:

            with gradio.Row():
                _ = gradio.Markdown("""
                    ### multi-prompt separator keyword: **SPLIT**
                """)
                prediction_type = gradio.Dropdown(label='Set model prediction type', choices=['default', 'epsilon', 'const', 'v_prediction', 'edm'], value='default', type='value')

            with gradio.Accordion(label="FluxTools", open=False):
                with gradio.Tab("Canny / Depth", id="F2E_FT"):
                    gradio.Markdown("Select Flux Canny or Depth model in **Checkpoint** menu.")
                    gradio.Markdown("Use an appropriately *preprocessed* control image.")
                    with gradio.Row():
                        with gradio.Column():
                            control_image = gradio.Image(label="Control image", type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            control_strength = gradio.Slider(label="Strength", minimum = 0.0, maximum = 2.0, step = 0.01, value=1.0)
                            control_time = RangeSlider(label="Start / End", minimum = 0.0, maximum = 1.0, step = 0.01, value=(0.0, 0.8))
                            image_info = gradio.Markdown("Control image aspect ratio: *no image*")

                with gradio.Tab("Fill", id="F2E_FT_f"):
                    gradio.Markdown("Select Flux Fill model in **Checkpoint** menu")
                    gradio.Markdown("If this tab is used, it takes priority over Canny / Depth.")
                    with gradio.Row():
                        fill_image = ForgeCanvas(height=300, contrast_scribbles=shared.opts.img2img_inpaint_mask_high_contrast, scribble_color=shared.opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=True, scribble_alpha=75, scribble_alpha_fixed=True, scribble_softness_fixed=True)

                with gradio.Tab("Redux", id="F2E_FT_r1"):
                    gradio.Markdown("Redux can be combined with another tool, or used alone.")
                    gradio.Markdown("Select an image to use for Redux.")
                    with gradio.Row():
                        with gradio.Column():
                            redux_image1 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            redux_str1 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                            redux_time1 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                            swap12 = gradio.Button("swap redux 1 and 2")
                            swap13 = gradio.Button("swap redux 1 and 3")
                            swap14 = gradio.Button("swap redux 1 and 4")

                with gradio.Tab("Redux-2", id="F2E_FT_r2"):
                    gradio.Markdown("Multiple images can be used for Redux.")
                    gradio.Markdown("Select an image to use for Redux.")
                    with gradio.Row():
                        with gradio.Column():
                            redux_image2 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            redux_str2 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                            redux_time2 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                            swap21 = gradio.Button("swap redux 2 and 1")
                            swap23 = gradio.Button("swap redux 2 and 3")
                            swap24 = gradio.Button("swap redux 2 and 4")

                with gradio.Tab("Redux-3", id="F2E_FT_r3"):
                    gradio.Markdown("Multiple images can be used for Redux.")
                    gradio.Markdown("Select an image to use for Redux.")
                    with gradio.Row():
                        with gradio.Column():
                            redux_image3 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            redux_str3 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                            redux_time3 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                            swap31 = gradio.Button("swap redux 3 and 1")
                            swap32 = gradio.Button("swap redux 3 and 2")
                            swap34 = gradio.Button("swap redux 3 and 4")

                with gradio.Tab("Redux-4", id="F2E_FT_r4"):
                    gradio.Markdown("Multiple images can be used for Redux.")
                    gradio.Markdown("Select an image to use for Redux.")
                    with gradio.Row():
                        with gradio.Column():
                            redux_image4 = gradio.Image(show_label=False, type="pil", height=300, sources=["upload", "clipboard"])
                        with gradio.Column():
                            redux_str4 = gradio.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                            redux_time4 = RangeSlider(label="Start / End", minimum=0.0, maximum=1.0, step=0.01, value=(0.0, 0.8))
                            swap41 = gradio.Button("swap redux 4 and 1")
                            swap42 = gradio.Button("swap redux 4 and 2")
                            swap43 = gradio.Button("swap redux 4 and 3")


                def redux_swap(imageA, strA, timeA, imageB, strB, timeB):
                    return imageB, strB, timeB, imageA, strA, timeA

                swap_1 = [redux_image1, redux_str1, redux_time1]
                swap_2 = [redux_image2, redux_str2, redux_time2]
                swap_3 = [redux_image3, redux_str3, redux_time3]
                swap_4 = [redux_image4, redux_str4, redux_time4]

                swap12.click(fn=redux_swap, inputs=swap_1+swap_2, outputs=swap_1+swap_2)
                swap13.click(fn=redux_swap, inputs=swap_1+swap_3, outputs=swap_1+swap_3)
                swap14.click(fn=redux_swap, inputs=swap_1+swap_4, outputs=swap_1+swap_4)

                swap21.click(fn=redux_swap, inputs=swap_2+swap_1, outputs=swap_2+swap_1)
                swap23.click(fn=redux_swap, inputs=swap_2+swap_3, outputs=swap_2+swap_3)
                swap24.click(fn=redux_swap, inputs=swap_2+swap_4, outputs=swap_2+swap_4)

                swap31.click(fn=redux_swap, inputs=swap_3+swap_1, outputs=swap_3+swap_1)
                swap32.click(fn=redux_swap, inputs=swap_3+swap_2, outputs=swap_3+swap_2)
                swap34.click(fn=redux_swap, inputs=swap_3+swap_4, outputs=swap_3+swap_4)

                swap41.click(fn=redux_swap, inputs=swap_4+swap_1, outputs=swap_4+swap_1)
                swap42.click(fn=redux_swap, inputs=swap_4+swap_2, outputs=swap_4+swap_2)
                swap43.click(fn=redux_swap, inputs=swap_4+swap_3, outputs=swap_4+swap_3)

            with InputAccordion(False, label="Flux Transparent VAE") as transparent_vae:
                _ = gradio.Markdown("""
                    * include the Flux LayerDiffuse LoRA in the prompt
                    * transparent VAE must be located in the models directory as `models/TransparentVAE.pth`
                    * download from https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/
                """)

            with gradio.Accordion('Shift for Flow models', open=False):
                with gradio.Row():
                    shift = gradio.Slider(label='Shift - 0: use default.', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                    max = gradio.Slider(label='Max Shift - 0: non-dynamic', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                with gradio.Row():
                    shiftHR = gradio.Slider(label='HighRes Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                    maxHR = gradio.Slider(label='HighRes Max Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)

            with gradio.Accordion('Text encoders control', open=False):
                te_device = gradio.Radio(label="device for text encoders", choices=["default", "cpu", "gpu", "gpu-2"], value="default", info="note: gpu-2 uses default behaviour if there isn't a second CUDA device. gpu-2 also uses same device for offload.")
                with gradio.Row(visible=(StableDiffusion3 is not None)):
                    SD3_use_T5 = gradio.Checkbox(value=forgeMultiPrompt.SD3_use_T5, label="SD3: use T5")
                    SD3_use_CL = gradio.Checkbox(value=forgeMultiPrompt.SD3_use_CL, label="SD3: use CLIP-L")
                    SD3_use_CG = gradio.Checkbox(value=forgeMultiPrompt.SD3_use_CG, label="SD3: use CLIP-G")
                with gradio.Row():
                    flux_use_T5 = gradio.Checkbox(value=forgeMultiPrompt.flux_use_T5, label="Flux: use T5")
                    flux_use_CL = gradio.Checkbox(value=forgeMultiPrompt.flux_use_CL, label="Flux: use CLIP (pooled)")
                with gradio.Row():
                    SDXL_use_CL = gradio.Checkbox(value=forgeMultiPrompt.SDXL_use_CL, label="SDXL: use CLIP-L")
                    SDXL_use_CG = gradio.Checkbox(value=forgeMultiPrompt.SDXL_use_CG, label="SDXL: use CLIP-G")

                def update_info (image):
                    if image is None:
                        return "Control image aspect ratio: *no image*"
                    else:
                        return f"Control image aspect ratio: {round(image.size[0] / image.size[1], 3)} ({image.size[0]} \u00D7 {image.size[1]})"

                control_image.change(fn=update_info, inputs=[control_image], outputs=[image_info], show_progress=False)

            with InputAccordion(False, label="Flex.2") as use_flex2:
                gradio.Markdown("Select Flex.2 model in **Checkpoint** menu.")
                gradio.Markdown("Inputs are optional. Control image must be appropriately *preprocessed* (depth / line / pose).")
                with gradio.Row():
                    with gradio.Column():
                        flex2_image = ForgeCanvas(height=388, contrast_scribbles=shared.opts.img2img_inpaint_mask_high_contrast, scribble_color=shared.opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=True, scribble_alpha=75, scribble_alpha_fixed=True, scribble_softness_fixed=True)
                    with gradio.Column():
                        flex2_control = gradio.Image(label="Control image", type="pil", height=300, sources=["upload", "clipboard"])
                        flex2_strength = gradio.Slider(label="Strength", minimum = 0.0, maximum = 2.0, step = 0.01, value=1.0)
                        flex2_time = RangeSlider(label="Start / End", minimum = 0.0, maximum = 1.0, step = 0.01, value=(0.0, 0.8))


        self.infotext_fields = [
            (enabled, lambda d: d.get("fmp_enabled", False)),
            (shift,           "fmp_shift"),
            (max,             "fmp_max"),
            (shiftHR,         "fmp_shiftHR"),
            (maxHR,           "fmp_maxHR"),
            (te_device,       "fmp_te_device"),
            (prediction_type, "fmp_prediction"),
            (flux_use_T5,     "fmp_fluxT5"),
            (flux_use_CL,     "fmp_fluxCL"),
            (SDXL_use_CL,     "fmp_sdxlCL"),
            (SDXL_use_CG,     "fmp_sdxlCG"),
            (SD3_use_CL,      "fmp_sd3CL"),
            (SD3_use_CG,      "fmp_sd3CG"),
            (SD3_use_T5,      "fmp_sd3T5"),
        ]

        def clearCondCache ():
            forgeMultiPrompt.clearConds = True

        enabled.input     (fn=clearCondCache, inputs=None, outputs=None)
        flux_use_T5.input (fn=clearCondCache, inputs=None, outputs=None)
        flux_use_CL.input (fn=clearCondCache, inputs=None, outputs=None)
        SDXL_use_CL.input (fn=clearCondCache, inputs=None, outputs=None)
        SDXL_use_CG.input (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_CL.input  (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_CG.input  (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_T5.input  (fn=clearCondCache, inputs=None, outputs=None)

        return enabled, transparent_vae, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, control_image, control_strength, control_time, redux_image1, redux_image2, redux_image3, redux_image4, redux_str1, redux_str2, redux_str3, redux_str4, redux_time1, redux_time2, redux_time3, redux_time4, fill_image.background, fill_image.foreground, use_flex2, flex2_image.background, flex2_image.foreground, flex2_control, flex2_strength, flex2_time

    def after_extra_networks_activate(self, p, *script_args, **kwargs):
        enabled = script_args[0]
        if enabled:
            te_device = script_args[6]
            match te_device:
                case "gpu-2":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_gpu2
                    memory_management.text_encoder_offload_device = forgeMultiPrompt.patched_text_encoder_offload
                case "gpu":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_gpu
                case "cpu":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_cpu
                case _:
                    pass

    def process(self, params, *script_args, **kwargs):
        enabled, transparent_vae, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, control_image, control_strength, control_time, redux_image1, redux_image2, redux_image3, redux_image4, redux_str1, redux_str2, redux_str3, redux_str4, redux_time1, redux_time2, redux_time3, redux_time4, fill_image, fill_mask, use_flex2, flex2_image, flex2_mask, flex2_control, flex2_strength, flex2_time = script_args

        #   clear conds if usage has changed - must do this even if extension has been disabled
        if forgeMultiPrompt.clearConds:
            params.clear_prompt_cache()
            forgeMultiPrompt.clearConds = False

        if enabled:
            forgeMultiPrompt.flux_use_T5 = flux_use_T5
            forgeMultiPrompt.flux_use_CL = flux_use_CL
            forgeMultiPrompt.SDXL_use_CL = SDXL_use_CL
            forgeMultiPrompt.SDXL_use_CG = SDXL_use_CG
            forgeMultiPrompt.SD3_use_CL  = SD3_use_CL
            forgeMultiPrompt.SD3_use_CG  = SD3_use_CG
            forgeMultiPrompt.SD3_use_T5  = SD3_use_T5

            params.extra_generation_params.update({
                "fmp_enabled"   :   enabled,
                "fmp_te_device" :   te_device,
            })

            if isinstance(shared.sd_model.forge_objects.unet.model.predictor, PredictionFlux) or isinstance(shared.sd_model.forge_objects.unet.model.predictor, PredictionDiscreteFlow):
                if shift > 0.0:
                    params.extra_generation_params.update({
                        "fmp_shift"     :   shift,
                        "fmp_max"       :   max,
                    })
                if shiftHR > 0.0:
                    params.extra_generation_params.update({
                        "fmp_shiftHR"   :   shiftHR,
                        "fmp_maxHR"     :   maxHR,
                    })

            isMPModel = not (params.sd_model.is_sd1 or params.sd_model.is_sd2)
            if isMPModel:
                if params.sd_model.is_sdxl:
                    StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.patched_glc_sdxl
                    params.extra_generation_params.update({
                        "fmp_sdxlCL"    :   SDXL_use_CL,
                        "fmp_sdxlCG"    :   SDXL_use_CG,
                    })
                elif getattr(params.sd_model, 'is_sd3', False):
                    StableDiffusion3.get_learned_conditioning = forgeMultiPrompt.patched_glc_sd3
                    params.extra_generation_params.update({
                        "fmp_sd3CL"    :   SD3_use_CL,
                        "fmp_sd3CG"    :   SD3_use_CG,
                        "fmp_sd3T5"    :   SD3_use_T5,
                    })
                else:
                    Flux.get_learned_conditioning = forgeMultiPrompt.patched_glc_flux
                    params.extra_generation_params.update({
                        "fmp_fluxT5"    :   flux_use_T5,
                        "fmp_fluxCL"    :   flux_use_CL,
                    })

            if prediction_type != 'default':
                forgeMultiPrompt.prediction_typeBackup = params.sd_model.forge_objects.unet.model.predictor.prediction_type
                params.sd_model.forge_objects.unet.model.predictor.prediction_type = prediction_type

                params.extra_generation_params.update({
                    "fmp_prediction"     :   prediction_type,
                })

        return

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, transparent_vae, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, control_image, control_strength, control_time, redux_image1, redux_image2, redux_image3, redux_image4, redux_str1, redux_str2, redux_str3, redux_str4, redux_time1, redux_time2, redux_time3, redux_time4, fill_image, fill_mask, use_flex2, flex2_image, flex2_mask, flex2_control, flex2_strength, flex2_time = script_args
        if enabled:
            # print (shared.sd_model.model_config.unet_config)

            if not hasattr(shared.sd_model.model_config.unet_config, 'depth') or shared.sd_model.model_config.unet_config['depth'] != 8:
                use_flex2 = False

            if isinstance(shared.sd_model.forge_objects.unet.model.predictor, PredictionFlux) or isinstance(shared.sd_model.forge_objects.unet.model.predictor, PredictionDiscreteFlow):

                def sigma (timestep, s, d):
                    if d > 0.0:
                        m = (d - shift) / (4096 - 256)
                        b = shift - m * 256
                        mu = 16 * m + b

                        return math.exp(mu) / (math.exp(mu) + (1 / timestep - 1) ** 1.0)
                    else:
                        return s * timestep / (1 + (s - 1) * timestep)

                if params.is_hr_pass:
                    thisShift = shiftHR if shiftHR > 0.0 else shift
                    dynamic = maxHR if maxHR > 0.0 else max
                else:
                    thisShift = shift
                    dynamic = max

                if thisShift > 0.0:
                    if forgeMultiPrompt.sigmasBackup is None:
                        forgeMultiPrompt.sigmasBackup = shared.sd_model.forge_objects.unet.model.predictor.sigmas
                    ts = sigma((torch.arange(1, 10000 + 1, 1) / 10000), thisShift, dynamic)
                    shared.sd_model.forge_objects.unet.model.predictor.sigmas = ts

            if params.iteration > 0:    # batch count
                # FluxTools setup done on iteration 0
                return

            if not params.sd_model.is_webui_legacy_model():
                x = kwargs['x']
                n, c, h, w = x.size()

                if use_flex2:
                    if flex2_image is None:
                        flex_latent = torch.zeros([1, 16, h, w])
                        flex_mask = torch.ones([1, 1, h, w])
                    else:
                        if isinstance (flex2_image, str):
                            flex2_image = decode_base64_to_image(flex2_image)
                        image = flex2_image.convert('RGB').resize((w*8, h*8))
                        image = numpy.array(image) / 255.0
                        image = numpy.transpose(image, (2, 0, 1))
                        image = torch.tensor(image).unsqueeze(0)

                        mask_A = flex2_mask.getchannel('A').convert('L')
                        mask_A = mask_A.point(lambda v: 255 if v > 128 else 0)
                        mask_A = mask_A.resize((w, h))
                        mask_A = numpy.array(mask_A) / 255.0
                        flex_mask = torch.tensor(mask_A).unsqueeze(0).unsqueeze(0)

                        flex_latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                        flex_latent *= (1.0 - flex_mask.to(flex_latent.device))

                    if flex2_control is None:
                        flex_control = torch.zeros([1, 16, h, w])
                        forgeMultiPrompt.start = 0.0
                        forgeMultiPrompt.end = 1.0
                    else:
                        if isinstance (flex2_control, str):
                            flex2_control = decode_base64_to_image(flex2_control)
                        control_image = flex2_control.resize((w*8, h*8))
                        control_image = numpy.array(control_image) / 255.0
                        control_image = numpy.transpose(control_image, (2, 0, 1))
                        control_image = torch.tensor(control_image).unsqueeze(0)
                        flex_control = images_tensor_to_samples(control_image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                        flex_control *= flex2_strength
                        forgeMultiPrompt.start = flex2_time[0]
                        forgeMultiPrompt.end = flex2_time[1]

                    forgeMultiPrompt.strength = 1.0
                    forgeMultiPrompt.latent = torch.cat([flex_latent, flex_mask.to(flex_latent.device), flex_control.to(flex_latent.device)], dim=1)

                else:   # FluxTools
                    if (fill_image is not None and fill_mask is not None):
                        if isinstance (fill_image, str):
                            fill_image = decode_base64_to_image(fill_image)
                        if isinstance (fill_mask, str):
                            fill_mask = decode_base64_to_image(fill_mask)

                        mask_A = fill_mask.getchannel('A').convert('L')
                        mask_A_I = mask_A.point(lambda v: 0 if v > 128 else 255)
                        mask_A = mask_A.point(lambda v: 255 if v > 128 else 0)

                        # mask_A_I = mask_A.point(lambda v: 255-v)
                        # mask_A = mask_A.point(lambda v: 255 if v > 0 else 0)

                        mask = Image.merge('RGBA', (mask_A_I, mask_A_I, mask_A_I, mask_A))

                        image = Image.alpha_composite(fill_image, mask).convert('RGB')
                        image = image.resize((w*8, h*8))
                        image = numpy.array(image) / 255.0
                        image = numpy.transpose(image, (2, 0, 1))
                        image = torch.tensor(image).unsqueeze(0)

                        mask = mask_A.resize((w*8, h*8))
                        mask = numpy.array(mask) / 255.0
                        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
                        #mask = mask[:, 0, :, :]
                        mask = mask.view(1, h, 8, w, 8)
                        mask = mask.permute(0, 2, 4, 1, 3)
                        mask = mask.reshape(1, 64, h, w)

                        latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)

                        forgeMultiPrompt.latent = torch.cat([latent, mask.to(latent.device)], dim=1)

                        del latent, image, mask

                        forgeMultiPrompt.start = 0.0
                        forgeMultiPrompt.end = 1.0
                        forgeMultiPrompt.strength = 1.0
                    elif control_image and control_strength > 0:
                        if isinstance (control_image, str):
                            control_image = decode_base64_to_image(control_image)
                        image = control_image.resize((w*8, h*8))
                        image = numpy.array(image) / 255.0
                        image = numpy.transpose(image, (2, 0, 1))
                        image = torch.tensor(image).unsqueeze(0)

                        latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
                        forgeMultiPrompt.latent = latent
                        del image

                        forgeMultiPrompt.start = control_time[0]
                        forgeMultiPrompt.end = control_time[1]
                        forgeMultiPrompt.strength = control_strength
                    else:
                        forgeMultiPrompt.latent = None

                redux_images = [redux_image1, redux_image2, redux_image3, redux_image4]
                redux_strengths = [redux_str1, redux_str2, redux_str3, redux_str4]
                redux_times = [redux_time1, redux_time2, redux_time3, redux_time4]

                if redux_images != [None, None, None, None] and redux_strengths != [0, 0, 0, 0]:
                    from transformers import SiglipImageProcessor, SiglipVisionModel
                    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder

                    feature = SiglipImageProcessor.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="feature_extractor")
                    encoder = SiglipVisionModel.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="image_encoder")
                    embedder = ReduxImageEncoder.from_pretrained("Runware/FLUX.1-Redux-dev", subfolder="image_embedder")

                    embeds = []
                    for i in range(len(redux_images)):
                        if redux_images[i] is None or redux_strengths[i] == 0:
                            continue

                        if isinstance (redux_images[i], str):
                            redux_images[i] = decode_base64_to_image(redux_images[i])
                        image = feature.preprocess(
                            images=redux_images[i], do_resize=True, return_tensors="pt", do_convert_rgb=True
                        )

                        image_enc_hidden_states = encoder(**image).last_hidden_state

                        embeds.append((redux_strengths[i] * embedder(image_enc_hidden_states).image_embeds, redux_times[i][0], redux_times[i][1]))
                        del image_enc_hidden_states

                    del feature, encoder, embedder

                    forgeMultiPrompt.image_embeds = embeds
                else:
                    forgeMultiPrompt.image_embeds = None

                def apply_control(self):
                    lastStep = self.total_sampling_steps - 1
                    thisStep = self.sampling_step

                    if forgeMultiPrompt.image_embeds is not None:
                        embeds = forgeMultiPrompt.image_embeds
                        cond = self.text_cond["crossattn"]
                        for e in embeds:
                            if thisStep >= e[1] * lastStep and thisStep <= e[2] * lastStep:
                                image_embeds = e[0].repeat_interleave(len(self.text_cond["crossattn"]), dim=0)

                                image_embeds *= (256 / 729) #?hmm, scale down to give prompt a chance
                                                            # 256 could be cond.shape[1]
                                                            # 729 could be image_embeds.shape[1]

                                cond = torch.cat([cond, image_embeds.to(cond.device)], dim=1)
                                #or blend?

                                del image_embeds
                        cond = torch.sum(cond, dim=0, keepdim=True)
                        self.text_cond["crossattn"] = cond

                    if forgeMultiPrompt.latent is not None:
                        if thisStep >= forgeMultiPrompt.start * lastStep and thisStep <= forgeMultiPrompt.end * lastStep:
                            latent_strength = forgeMultiPrompt.latent * forgeMultiPrompt.strength
                            shared.sd_model.forge_objects.unet.extra_concat_condition = latent_strength
                        else:
                            if use_flex2:
                                latent_strength = forgeMultiPrompt.latent.clone()
                                latent_strength[:, 17:, :, :] = 0.0
                            else:
                                latent_strength = forgeMultiPrompt.latent * 0.0
                            shared.sd_model.forge_objects.unet.extra_concat_condition = latent_strength

                on_cfg_denoiser(apply_control)

        return


    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            if params.sd_model.is_sdxl:
                StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.glc_backup_sdxl
            elif getattr(params.sd_model, 'is_sd3', False):
                StableDiffusion3.get_learned_conditioning = forgeMultiPrompt.glc_backup_sd3
            elif not shared.sd_model.is_webui_legacy_model():
                Flux.get_learned_conditioning = forgeMultiPrompt.glc_backup_flux

            memory_management.text_encoder_device = forgeMultiPrompt.text_encoder_device_backup
            memory_management.text_encoder_offload_device = forgeMultiPrompt.text_encoder_offload_device_backup

            if forgeMultiPrompt.sigmasBackup is not None:
                shared.sd_model.forge_objects.unet.model.predictor.sigmas = forgeMultiPrompt.sigmasBackup
                forgeMultiPrompt.sigmasBackup = None

            if forgeMultiPrompt.prediction_typeBackup is not None:
                params.sd_model.forge_objects.unet.model.predictor.prediction_type = forgeMultiPrompt.prediction_typeBackup
                forgeMultiPrompt.prediction_typeBackup = None

            shared.sd_model.forge_objects.unet.extra_concat_condition = None
            forgeMultiPrompt.image_embeds = None
            forgeMultiPrompt.latent = None
            remove_current_script_callbacks()

        return


    # simple composite is bad - obvious edge
    def postprocess_image (self, params, pp, *args):
        enabled = args[0]
        if enabled and not shared.sd_model.is_webui_legacy_model():
            # FluxFill composite with original to avoid vae round-trip errors
            use_flex2 = args[-6]
            if not hasattr(shared.sd_model.model_config.unet_config, 'depth') or shared.sd_model.model_config.unet_config['depth'] != 8:
                use_flex2 = False

            if use_flex2:
                fill_image = args[-5]
                fill_mask = args[-4]
            else:
                fill_image = args[-8]
                fill_mask = args[-7]
            if fill_image is not None and fill_mask is not None:
                if isinstance (fill_image, str):
                    fill_image = decode_base64_to_image(fill_image)
                if isinstance (fill_mask, str):
                    fill_mask = decode_base64_to_image(fill_mask)

                w = pp.image.size[0]
                h = pp.image.size[1]
                image = fill_image.resize((w, h))
                mask = fill_mask.resize((w, h))

                short_side = min(mask.size)
                dilation_size = int(0.05 * short_side) * 2 + 1
                mask = TF.gaussian_blur(mask.filter(ImageFilter.MaxFilter(dilation_size)), dilation_size)

                pp.image = Image.composite(pp.image, image, mask)

        return


    def post_sample (self, params, ps, *args):
        enabled = args[0]
        t_vae = args[1]
        if enabled and t_vae and not shared.sd_model.is_webui_legacy_model():
            forgeMultiPrompt.samples = ps.samples

        return


    def postprocess_batch(self, params, *args, **kwargs):
        enabled = args[0]
        t_vae = args[1]
        if enabled and t_vae and not shared.sd_model.is_webui_legacy_model():
            if forgeMultiPrompt.transparentVAE is None:
                forgeMultiPrompt.transparentVAE = FluxTransparentVAE(dtype=torch.float32)
                forgeMultiPrompt.transparentVAE.load_state_dict(torch.load('models/TransparentVAE.pth'), strict=False)
                forgeMultiPrompt.transparentVAE.eval()

            forgeMultiPrompt.transparentVAE.cuda()

            rgb = kwargs['images']

            image_count = len(rgb)
            for i in range(image_count):
                print (f"Flux Transparent VAE {i+1}/{image_count}", end="\r", flush=True)
                rgba = forgeMultiPrompt.transparentVAE.decode(rgb[i].unsqueeze(0), forgeMultiPrompt.samples[i:i+1, ...]).squeeze(0).cpu().numpy()

                rgba = 255.0 * numpy.moveaxis(rgba, 0, 2)
                rgba = rgba.round().astype(numpy.uint8)

                params.extra_result_images.append(rgba)

                info = f"{params.all_prompts[i]}\nNegative prompt: {params.all_negative_prompts[i]}\nSeed: {params.seeds[i]}, Steps: {params.steps}, CFG Scale: {params.cfg_scale}, Distilled CFG Scale: {params.distilled_cfg_scale}, Size: {params.width}x{params.height}, Sampler: {params.sampler_name}, Scheduler: {params.scheduler}, Model: {params.sd_model_name}"

                images.save_image(Image.fromarray(rgba, mode="RGBA"), params.outpath_samples, "", 0, "", "png", info=info, p=params, suffix="-transparent")
                del rgba

            print ("Flux Transparent VAE done  ", end="\r", flush=True)
            forgeMultiPrompt.samples = None
            forgeMultiPrompt.transparentVAE.cpu()
            torch.cuda.empty_cache()

        return
