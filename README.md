## multi-prompt extension for Flux and SDXL in Forge webUI ##
### intended as quick proof of concept only ###
#### yet somehow updated already ####

install:
**Extensions** tab, **Install from URL**, use URL for this repo

usage:
1. Enable the extension: **Multi-prompt** -> **Enabled** checkbox
2. use the Prompt textbox as normal
3. prompt for first text encoder
4. enter word **SPLIT**, use surrounding whitespace/newline as you like
5. prompt for second text encoder
6. Generate

Flux (CLIP-L / T5) assumed best usage is tags for first prompt, natural language for second.
SDXL (CLIP-L / CLIP-G) would typically be tags for both but depends on training: seems likely that same captions would be used for both CLIPs so use may be limited.

update 1:
* added sdXL, why not? In process, prepped code for possible future additions of SD3 / Hunyuan / others?.
* changed unpatch location. Previously unpatched ASAP therefore wouldn't have applied during hires fix (only relevant if hires fix had new prompt which included SPLIT). In practice, unlikely to be relevant at all.
* only patches if using appropriate models (Flux or sdXL). Previously always patched if extension enabled, which would have left the patch in place if using a model which didn't use the patched function which would have unpatched itself but couldn't.
* Force clearance of cached conds when extension enabled/disabled.
* for explanation of these egregious errors, see line 2 of this document.

update 2:
* added control for Shift parameter (affects calculation of sigmas used to guide denoising per step). This is for the *Simple* scheduler only.
