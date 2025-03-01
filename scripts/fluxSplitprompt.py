from backend import memory_management
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.sdxl import StableDiffusionXL
try:
    from backend.diffusion_engine.sd35 import StableDiffusion3
except:
    StableDiffusion3 = None

import gradio
from gradio_rangeslider import RangeSlider
import torch, math, numpy
from modules import scripts, shared
from modules.ui_components import InputAccordion, ToolButton
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.sd_samplers_common import images_tensor_to_samples, approximation_indexes


import gc
from backend import memory_management
from modules_forge import main_entry


class forgeMultiPrompt(scripts.Script):
    sorting_priority = 0
    
    glc_backup_flux = None
    glc_backup_sdxl = None
    glc_backup_sd3 = None
    clearConds = False
    sigmasBackup = None
    prediction_typeBackup = None
    text_encoder_device_backup = None
    
    flux_use_T5 = True
    flux_use_CL = True
    SDXL_use_CL = True
    SDXL_use_CG = True
    SD3_use_CL = True
    SD3_use_CG = True
    SD3_use_T5 = True

    def __init__(self):
        if forgeMultiPrompt.glc_backup_flux is None:
            forgeMultiPrompt.glc_backup_flux = Flux.get_learned_conditioning
        if forgeMultiPrompt.glc_backup_sdxl is None:
            forgeMultiPrompt.glc_backup_sdxl = StableDiffusionXL.get_learned_conditioning
        if forgeMultiPrompt.glc_backup_sd3 is None and StableDiffusion3 is not None:
            forgeMultiPrompt.glc_backup_sd3 = StableDiffusion3.get_learned_conditioning
        if forgeMultiPrompt.text_encoder_device_backup is None:
            forgeMultiPrompt.text_encoder_device_backup = memory_management.text_encoder_device

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

    def patched_text_encoder_gpu2():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        else:
            return torch.cuda.current_device()
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

            if forgeMultiPrompt.SD3_use_T5 and shared.opts.sd3_enable_t5:
                cond_t5 = self.text_processing_engine_t5(T5prompt)
            else:
                cond_t5 = torch.zeros([np, 256, 4096])

        #   conds get concatenated later, so sizes of dimension 1 must match
        #   padding with zero
        pad = cond_g.size(1) - cond_l.size(1)
        if pad > 1:
            padding = (0,0, 0, pad, 0,0)
            cond_l = torch.nn.functional.pad (cond_l, padding, mode='constant', value=0)
        elif pad < 1:
            padding = (0,0, 0, -pad, 0,0)
            cond_g = torch.nn.functional.pad (cond_g, padding, mode='constant', value=0)


        cond_lg = torch.cat([cond_l, cond_g.to(cond_l.device)], dim=-1)
        cond_lg = torch.nn.functional.pad(cond_lg, (0, 4096 - cond_lg.shape[-1]))

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

        #   conds get concatenated later, so sizes of dimension 1 must match
        #   padding with zero
        pad = cond_g.size(1) - cond_l.size(1)
        if pad > 1:
            padding = (0,0, 0, pad, 0,0)
            cond_l = torch.nn.functional.pad (cond_l, padding, mode='constant', value=0)
        elif pad < 1:
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
            _ = gradio.Markdown(show_label=False, value='### multi-prompt (SDXL, SD3, Flux) separator keyword: **SPLIT** ###')

            with gradio.Row():
                te_device = gradio.Radio(label="device for text encoders", choices=["default", "cpu", "gpu", "gpu-2"], value="default")
                prediction_type = gradio.Dropdown(label='Set model prediction type', choices=['default', 'epsilon', 'const', 'v_prediction', 'edm'], value='default', type='value')

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

            _ = gradio.Markdown(show_label=False, value='#### Shift control for Flux and SD3. ####')
            with gradio.Row():
                shift = gradio.Slider(label='Shift - 0: use default.', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                max = gradio.Slider(label='Max Shift - 0: non-dynamic', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
            with gradio.Row():
                shiftHR = gradio.Slider(label='HighRes Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                maxHR = gradio.Slider(label='HighRes Max Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)

            with InputAccordion(False, label="FluxTools (Canny / Depth)") as ft_enabled:
                gradio.Markdown("Select canny or depth model in **Checkpoint** menu; add **VAE / Text Encoders** as needed.")
                gradio.Markdown("Use an appropriately *preprocessed* control image.")
                with gradio.Row():
                    with gradio.Column():
                        control_image = gradio.Image(label="Control image", type="pil", height=300, sources=["upload", "clipboard"])
                    with gradio.Column():
                        control_strength = gradio.Slider(label="Strength", minimum = 0.0, maximum = 2.0, step = 0.01, value=1.0)
                        control_time = RangeSlider(label="Start / End", minimum = 0.0, maximum = 1.0, step = 0.01, value=(0.0, 0.8))
                        image_info = gradio.Markdown("Control image aspect ratio: *no image*")
                        
                def update_info (image):
                    if image is None:
                        return "Control image aspect ratio: *no image*"
                    else:
                        return f"Control image aspect ratio: {image.size[0] / image.size[1]} ({image.size[0]} \u00D7 {image.size[1]})"

                control_image.change(fn=update_info, inputs=[control_image], outputs=[image_info], show_progress=False)


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

        enabled.change     (fn=clearCondCache, inputs=None, outputs=None)
        flux_use_T5.change (fn=clearCondCache, inputs=None, outputs=None)
        flux_use_CL.change (fn=clearCondCache, inputs=None, outputs=None)
        SDXL_use_CL.change (fn=clearCondCache, inputs=None, outputs=None)
        SDXL_use_CG.change (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_CL.change  (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_CG.change  (fn=clearCondCache, inputs=None, outputs=None)
        SD3_use_T5.change  (fn=clearCondCache, inputs=None, outputs=None)

        return enabled, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, ft_enabled, control_image, control_strength, control_time

    def after_extra_networks_activate(self, p, *script_args, **kwargs):
        enabled = script_args[0]
        if enabled:
            te_device = script_args[5]
            match te_device:
                case "gpu-2":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_gpu2
                case "gpu":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_gpu
                case "cpu":
                    memory_management.text_encoder_device = forgeMultiPrompt.patched_text_encoder_cpu
                case _:
                    pass

    def process(self, params, *script_args, **kwargs):
        enabled, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, ft_enabled, control_image, control_strength, control_time = script_args

        #   clear conds if usage has changed - must do this even if extension has been disabled
        if forgeMultiPrompt.clearConds == True:
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
            
            isMPModel = not ((params.sd_model.is_sd1 == True) or (params.sd_model.is_sd2 == True))
            if isMPModel:
                if params.sd_model.is_sdxl == True:
                    StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.patched_glc_sdxl
                    params.extra_generation_params.update({
                        "fmp_sdxlCL"    :   SDXL_use_CL,
                        "fmp_sdxlCG"    :   SDXL_use_CG,
                    })
                elif params.sd_model.is_sd3 == True:
                    StableDiffusion3.get_learned_conditioning = forgeMultiPrompt.patched_glc_sd3
                    params.extra_generation_params.update({
                        "fmp_sd3CL"    :   SD3_use_CL,
                        "fmp_sd3CG"    :   SD3_use_CG,
                        "fmp_sd3T5"    :   SD3_use_T5,
                    })
                else:
                    Flux.get_learned_conditioning = forgeMultiPrompt.patched_glc_flux
                    params.extra_generation_params.update({
                        "fmp_shift"     :   shift,
                        "fmp_max"       :   max,
                        "fmp_shiftHR"   :   shiftHR,
                        "fmp_maxHR"     :   maxHR,
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
        enabled, shift, max, shiftHR, maxHR, te_device, prediction_type, flux_use_T5, flux_use_CL, SDXL_use_CL, SDXL_use_CG, SD3_use_CL, SD3_use_CG, SD3_use_T5, ft_enabled, control_image, control_strength, control_time = script_args
        if enabled:
            # print (shared.sd_model.model_config.unet_config)

            if not shared.sd_model.is_webui_legacy_model() or params.sd_model.is_sd3:
                # fullfatFlux = False
                # if not fullfatFlux:
                    ##shared.sd_model.model_config.unet_config['depth'] = 8 # Flex, reduced to 8 double blocks
                    # shared.sd_model.forge_objects.unet.model.diffusion_model.double_blocks = shared.sd_model.forge_objects.unet.model.diffusion_model.double_blocks[0:8]

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

            if not ft_enabled or params.sd_model.is_webui_legacy_model() or control_image is None or control_strength == 0:
                return

            x = kwargs['x']

            n, c, h, w = x.size()

            image = control_image.resize((w*8, h*8))
            image = numpy.array(image)
            image = numpy.transpose(image, (2, 0, 1))
            image = torch.tensor(image).unsqueeze(0)

            latent = images_tensor_to_samples(image, approximation_indexes.get(shared.opts.sd_vae_encode_method), params.sd_model)
            latent *= control_strength
            forgeMultiPrompt.latent = latent
            
            forgeMultiPrompt.start = control_time[0]
            forgeMultiPrompt.end = control_time[1]
            forgeMultiPrompt.strength = control_strength

            def apply_control(self):
                lastStep = self.total_sampling_steps - 1
                thisStep = self.sampling_step
                
                if thisStep >= forgeMultiPrompt.start * lastStep and thisStep <= forgeMultiPrompt.end * lastStep:
                    latent_strength = forgeMultiPrompt.latent * forgeMultiPrompt.strength
                    shared.sd_model.forge_objects.unet.extra_concat_condition = latent_strength
                else:
                    latent_strength = forgeMultiPrompt.latent * 0.0
                    shared.sd_model.forge_objects.unet.extra_concat_condition = latent_strength

            on_cfg_denoiser(apply_control)

            return


    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            if params.sd_model.is_sdxl == True:
                StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.glc_backup_sdxl
            elif params.sd_model.is_sd3 == True:
                StableDiffusion3.get_learned_conditioning = forgeMultiPrompt.glc_backup_sd3
            elif not shared.sd_model.is_webui_legacy_model():
                Flux.get_learned_conditioning = forgeMultiPrompt.glc_backup_flux

            memory_management.text_encoder_device = forgeMultiPrompt.text_encoder_device_backup

            if forgeMultiPrompt.sigmasBackup != None:
                shared.sd_model.forge_objects.unet.model.predictor.sigmas = forgeMultiPrompt.sigmasBackup
                forgeMultiPrompt.sigmasBackup = None

            if forgeMultiPrompt.prediction_typeBackup != None:
                params.sd_model.forge_objects.unet.model.predictor.prediction_type = forgeMultiPrompt.prediction_typeBackup
                forgeMultiPrompt.prediction_typeBackup = None

        shared.sd_model.forge_objects.unet.extra_concat_condition = None
        remove_current_script_callbacks()

        return
