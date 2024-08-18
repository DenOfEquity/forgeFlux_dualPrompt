from backend.diffusion_engine.flux import Flux
import gradio
import torch
from modules import scripts

from backend import memory_management

class fluxSplitPrompt(scripts.Script):
    sorting_priority = 0
    glc_backup = None


    def __init__(self):
        fluxSplitPrompt.glc_backup = Flux.get_learned_conditioning

    @torch.inference_mode()
    def patched_glc(self, prompt: list[str]):
        #   first, unpatch in case of error
        Flux.get_learned_conditioning = fluxSplitPrompt.glc_backup
        
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

#        print (prompt)

        #   make 2 prompt lists, split each prompt in original list based on 'SPLIT'
        CLIPprompt = []
        T5prompt = []
        for p in prompt:
            splitPrompt = p.split('SPLIT')
            if len(splitPrompt) >= 2:
                CLIPprompt.append(splitPrompt[0].strip())
                T5prompt.append(splitPrompt[1].strip())
            else:
                CLIPprompt.append(p)
                T5prompt.append(p)

#        print (CLIPprompt)
#        print (T5prompt)
        
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

    def title(self):
        return "Split Prompt"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gradio.Accordion(open=False, label=self.title()):
            enabled_flux = gradio.Checkbox(value=False, label='Flux Enabled', info='Flux split prompt: 1st prompt tags, 2nd prompt natural language. Split with "SPLIT".')

        self.infotext_fields = [
            (enabled_flux, lambda d: d.get("fsp_enabled", False)),
        ]

        return [enabled_flux]

    def process(self, params, *script_args, **kwargs):
        enabled_flux = script_args[0]

        if enabled_flux:
            params.extra_generation_params.update({
                "fsp_enabled"        :   enabled_flux,
            })
            Flux.get_learned_conditioning = fluxSplitPrompt.patched_glc

        return
