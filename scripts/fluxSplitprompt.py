from backend import memory_management
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.sdxl import StableDiffusionXL
import gradio
import torch, math
from modules import scripts, shared
from modules.ui_components import InputAccordion


class forgeMultiPrompt(scripts.Script):
    sorting_priority = 0
    glc_backup_flux = None
    glc_backup_sdxl = None
    clearConds = False
    sigmasBackup = None
    prediction_typeBackup = None

    def __init__(self):
        if forgeMultiPrompt.glc_backup_flux is None:
            forgeMultiPrompt.glc_backup_flux = Flux.get_learned_conditioning
        if forgeMultiPrompt.glc_backup_sdxl is None:
            forgeMultiPrompt.glc_backup_sdxl = StableDiffusionXL.get_learned_conditioning

    def splitPrompt (prompt, countTextEncoders):
        promptTE1 = []
        promptTE2 = []
#        promptTE3 = []

        for p in prompt:
            splitPrompt = p.split('SPLIT')
            
            countSplits = min (countTextEncoders, len(splitPrompt))
            match countSplits:
#                case 3:         #   sd3, included for future proofing
#                    promptTE1.append(splitPrompt[0].strip())
#                    promptTE2.append(splitPrompt[1].strip())
#                    promptTE3.append(splitPrompt[2].strip())
                case 2:         #   sdxl, flux, hunyuan future proofing or SD3 with incomplete SPLITs
                    promptTE1.append(splitPrompt[0].strip())
                    promptTE2.append(splitPrompt[1].strip())
#                    promptTE3.append(p)
                case 1:         #   sd1,    or Any if SPLIT not used
                    promptTE1.append(p)
                    promptTE2.append(p)
#                    promptTE3.append(p)
                case _:
                    promptTE1.append(p)
                    promptTE2.append(p)
#                    promptTE3.append(p)

        return promptTE1, promptTE2#, promptTE3

    @torch.inference_mode()
    def patched_glc_flux(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        #   make 2 prompt lists, split each prompt in original list based on 'SPLIT'
        CLIPprompt, T5prompt = forgeMultiPrompt.splitPrompt (prompt, 2)

        cond_l, pooled_l = self.text_processing_engine_l(CLIPprompt)
        cond_t5 = self.text_processing_engine_t5(T5prompt)
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

        #   make 2 prompt lists, split each prompt in original list based on 'SPLIT'
        CLIPLprompt, CLIPGprompt = forgeMultiPrompt.splitPrompt (prompt, 2)
        
        cond_l = self.text_processing_engine_l(CLIPLprompt)
        cond_g, clip_pooled = self.text_processing_engine_g(CLIPGprompt)

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

        crop_w = 0
        crop_h = 0
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
            crossattn=torch.cat([cond_l, cond_g], dim=2),
            vector=torch.cat([clip_pooled, flat], dim=1),
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
                _ = gradio.Markdown(show_label=False, value='### multi-prompt separator keyword: **SPLIT** ###')
                prediction_type = gradio.Dropdown(label='Set model prediction type', choices=['default', 'epsilon', 'const', 'v_prediction', 'edm'], value='default', type='value')

            _ = gradio.Markdown(show_label=False, value='#### Shift control for Flux, Simple scheduler only. ####')
            with gradio.Row():
                shift = gradio.Slider(label='Shift - 0: use default.', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                max = gradio.Slider(label='Max Shift - 0: non-dynamic', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
            with gradio.Row():
                shiftHR = gradio.Slider(label='HighRes Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)
                maxHR = gradio.Slider(label='HighRes Max Shift - 0: no change', minimum=0.0, maximum=12.0, step=0.01, value=0.0)


        self.infotext_fields = [
            (enabled, lambda d: d.get("fmp_enabled", False)),
            (shift,           "fmp_shift"),
            (max,             "fmp_max"),
            (shiftHR,         "fmp_shiftHR"),
            (maxHR,           "fmp_maxHR"),
            (prediction_type, "fmp_prediction"),
        ]

        def clearCondCache ():
            forgeMultiPrompt.clearConds ^= True      #   if False, set to True; if True then next Generate hasn't happened so safe to reset to False

        enabled.change (fn=clearCondCache, inputs=[], outputs=[])

        return enabled, shift, max, shiftHR, maxHR, prediction_type

    def process(self, params, *script_args, **kwargs):
        enabled, shift, max, shiftHR, maxHR, prediction_type = script_args

        #   clear conds if usage has changed - must do this even if extension has been disabled
        if forgeMultiPrompt.clearConds == True:
            params.clear_prompt_cache()
            forgeMultiPrompt.clearConds = False

        if enabled:
            params.extra_generation_params.update({
                "fmp_enabled"        :   enabled,
            })
            
            isMPModel = not ((params.sd_model.is_sd1 == True) or (params.sd_model.is_sd2 == True))
            if isMPModel:
                params.extra_generation_params.update({
                    "fmp_shift"          :   shift,
                    "fmp_max"            :   max,
                    "fmp_shiftHR"        :   shiftHR,
                    "fmp_maxHR"          :   maxHR,
                })
                if params.sd_model.is_sdxl == True:
                    StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.patched_glc_sdxl
                else:
                    Flux.get_learned_conditioning = forgeMultiPrompt.patched_glc_flux

            if prediction_type != 'default':
                self.prediction_typeBackup = params.sd_model.forge_objects.unet.model.predictor.prediction_type
                params.sd_model.forge_objects.unet.model.predictor.prediction_type = prediction_type

                params.extra_generation_params.update({
                    "fmp_prediction"     :   prediction_type,
                })


        return

    def process_before_every_sampling(self, params, *script_args, **kwargs):
        enabled, shift, max, shiftHR, maxHR, _ = script_args
        if enabled and not shared.sd_model.is_webui_legacy_model():
            self.sigmasBackup = shared.sd_model.forge_objects.unet.model.predictor.sigmas
            
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
                ts = sigma((torch.arange(1, 10000 + 1, 1) / 10000), thisShift, dynamic)
                shared.sd_model.forge_objects.unet.model.predictor.sigmas = ts


    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled:
            isMPModel = not ((params.sd_model.is_sd1 == True) or (params.sd_model.is_sd2 == True))
            if isMPModel:
                if params.sd_model.is_sdxl == True:
                    StableDiffusionXL.get_learned_conditioning = forgeMultiPrompt.glc_backup_sdxl
                else:
                    Flux.get_learned_conditioning = forgeMultiPrompt.glc_backup_flux

            if self.sigmasBackup != None:
                shared.sd_model.forge_objects.unet.model.predictor.sigmas = self.sigmasBackup
                self.sigmasBackup = None

            if self.prediction_typeBackup != None:
                params.sd_model.forge_objects.unet.model.predictor.prediction_type = self.prediction_typeBackup
                self.prediction_typeBackup = None

        return
