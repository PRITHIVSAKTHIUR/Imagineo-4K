import os
import random
import uuid
import json
import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from typing import Tuple

bad_words = json.loads(os.getenv('BAD_WORDS', "[]"))
bad_words_negative = json.loads(os.getenv('BAD_WORDS_NEGATIVE', "[]"))
default_negative = os.getenv("default_negative","")

def check_text(prompt, negative=""):
    for i in bad_words:
        if i in prompt:
            return True
    for i in bad_words_negative:
        if i in negative:
            return True
    return False

#Quality/Style-----------------------------------------------------------------------------------------------------------------------------------------------------------Quality/Style

style_list = [
    {
        "name": "3840 x 2160",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "2560 x 1440",
        "prompt": "hyper-realistic 4K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },

    {
        "name": "HD+",
        "prompt": "hyper-realistic 2K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    
    {
        "name": "Style Zero",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    
]

#Clgstyle--------------------------------------------------------------------------------------------------------------------------------------------------------------Clgstyle

collage_style_list = [
    {
        "name": "Hi-Res",
        "prompt": "hyper-realistic 8K image of {prompt}. ultra-detailed, lifelike, high-resolution, sharp, vibrant colors, photorealistic",
        "negative_prompt": "cartoonish, low resolution, blurry, simplistic, abstract, deformed, ugly",
    },
    {
        "name": "B & W",
        "prompt": "black and white collage of {prompt}. monochromatic, timeless, classic, dramatic contrast",
        "negative_prompt": "colorful, vibrant, bright, flashy",
    },
    {
        "name": "Polaroid",
        "prompt": "collage of polaroid photos featuring {prompt}. vintage style, high contrast, nostalgic, instant film aesthetic",
        "negative_prompt": "digital, modern, low quality, blurry",
    },
    {
        "name": "Watercolor",
        "prompt": "watercolor collage of {prompt}. soft edges, translucent colors, painterly effects",
        "negative_prompt": "digital, sharp lines, solid colors",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic collage of {prompt}. film stills, movie posters, dramatic lighting",
        "negative_prompt": "static, lifeless, mundane",
    },
    {
        "name": "Nostalgic",
        "prompt": "nostalgic collage of {prompt}. retro imagery, vintage objects, sentimental journey",
        "negative_prompt": "contemporary, futuristic, forward-looking",
    },
    {
        "name": "Vintage",
        "prompt": "vintage collage of {prompt}. aged paper, sepia tones, retro imagery, antique vibes",
        "negative_prompt": "modern, contemporary, futuristic, high-tech",
    },
    {
        "name": "Scrapbook",
        "prompt": "scrapbook style collage of {prompt}. mixed media, hand-cut elements, textures, paper, stickers, doodles",
        "negative_prompt": "clean, digital, modern, low quality",
    },
    {
        "name": "NeoNGlow",
        "prompt": "neon glow collage of {prompt}. vibrant colors, glowing effects, futuristic vibes",
        "negative_prompt": "dull, muted colors, vintage, retro",
    },
    {
        "name": "Geometric",
        "prompt": "geometric collage of {prompt}. abstract shapes, colorful, sharp edges, modern design, high quality",
        "negative_prompt": "blurry, low quality, traditional, dull",
    },
    {
        "name": "Thematic",
        "prompt": "thematic collage of {prompt}. cohesive theme, well-organized, matching colors, creative layout",
        "negative_prompt": "random, messy, unorganized, clashing colors",
    },

#DuoTones by Canva --------------------------------------------------------------------------------------------------------------- Alters only the i++ Part / not Zero Tones
    
    {
        "name": "Cherry",
        "prompt": "Duotone style Cherry tone applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Fuchsia",
        "prompt": "Duotone style Fuchsia tone applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Pop",
        "prompt": "Duotone style Pop tone applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Violet",
        "prompt": "Duotone style Violet applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Sea Blue",
        "prompt": "Duotone style Sea Blue applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Sea Green",
        "prompt": "Duotone style Sea Green applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Mustard",
        "prompt": "Duotone style Mustard applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Amber",
        "prompt": "Duotone style Amber applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Pomelo",
        "prompt": "Duotone style Pomelo applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Peppermint",
        "prompt": "Duotone style Peppermint applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Mystic",
        "prompt": "Duotone style Mystic tone applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Pastel",
        "prompt": "Duotone style Pastel applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Coral",
        "prompt": "Duotone style Coral applied to {prompt}",
        "negative_prompt": "",
    },
    {
        "name": "No Style",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    
]

#filters------------------------------------------------------------------------------------------------------------------------------------------------filters

filters = {
    "Vivid": {
        "prompt": "extra vivid {prompt}",
        "negative_prompt": "washed out, dull"
    },
    "Playa": {
        "prompt": "{prompt} set in a vast playa",
        "negative_prompt": "forest, mountains"
    },
    "Desert": {
        "prompt": "{prompt} set in a desert landscape",
        "negative_prompt": "ocean, city"
    },
    "West": {
        "prompt": "{prompt} with a western theme",
        "negative_prompt": "eastern, modern"
    },
    "Blush": {
        "prompt": "{prompt} with a soft blush color palette",
        "negative_prompt": "harsh colors, neon"
    },
    "Minimalist": {
        "prompt": "{prompt} with a minimalist design",
        "negative_prompt": "cluttered, ornate"
    },

    "Zero filter": {
        "prompt": "{prompt}",
        "negative_prompt": ""
    },
    
    
}

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
collage_styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in collage_style_list}
filter_styles = {k: (v["prompt"], v["negative_prompt"]) for k, v in filters.items()}

STYLE_NAMES = list(styles.keys())
COLLAGE_STYLE_NAMES = list(collage_styles.keys())
FILTER_NAMES = list(filters.keys())
DEFAULT_STYLE_NAME = "3840 x 2160"
DEFAULT_COLLAGE_STYLE_NAME = "Hi-Res"
DEFAULT_FILTER_NAME = "Zero filter"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    if style_name in styles:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    elif style_name in collage_styles:
        p, n = collage_styles.get(style_name, collage_styles[DEFAULT_COLLAGE_STYLE_NAME])
    elif style_name in filter_styles:
        p, n = filter_styles.get(style_name, filter_styles[DEFAULT_FILTER_NAME])
    else:
        p, n = styles[DEFAULT_STYLE_NAME]
    
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

if not torch.cuda.is_available():
    DESCRIPTION = "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Compile
if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning", # / SG161222/RealVisXL_V4.0 / SG161222/RealVisXL_V4.0_Lightning 
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        variant="fp16"
    ).to(device)

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        print("Loaded on Device!")

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")

def save_image(img, path):
    img.save(path)
#seeding
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    style: str = DEFAULT_STYLE_NAME,
    collage_style: str = DEFAULT_COLLAGE_STYLE_NAME,
    filter_name: str = DEFAULT_FILTER_NAME,
    grid_size: str = "2x2",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    use_resolution_binning: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    if check_text(prompt, negative_prompt):
        raise ValueError("Prompt contains restricted words.")
    
    if collage_style != "No Style":
        prompt, negative_prompt = apply_style(collage_style, prompt, negative_prompt)
    elif filter_name != "No Filter":
        prompt, negative_prompt = apply_style(filter_name, prompt, negative_prompt)
    else:
        prompt, negative_prompt = apply_style(style, prompt, negative_prompt)
    
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if not use_negative_prompt:
        negative_prompt = ""  # type: ignore
    negative_prompt += default_negative    

    grid_sizes = {
        "2x1": (2, 1),
        "1x2": (1, 2),
        "2x2": (2, 2),
        "2x3": (2, 3),
        "3x2": (3, 2),
        "1x1": (1, 1)
    }
    
    grid_size_x, grid_size_y = grid_sizes.get(grid_size, (2, 2))
    num_images = grid_size_x * grid_size_y

    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": 30,
        "generator": generator,
        "num_images_per_prompt": num_images,
        "use_resolution_binning": use_resolution_binning,
        "output_type": "pil",
    }
    
    torch.cuda.empty_cache()  # Clear GPU memory
    images = pipe(**options).images

    grid_img = Image.new('RGB', (width * grid_size_x, height * grid_size_y))

    for i, img in enumerate(images[:num_images]):
        grid_img.paste(img, (i % grid_size_x * width, i // grid_size_x * height))

    unique_name = str(uuid.uuid4()) + ".png"
    save_image(grid_img, unique_name)
    return [unique_name], seed

examples = [

    "Chocolate dripping from a donut against a yellow background, in the style of brocore, hyper-realistic oil --ar 2:3 --q 2 --s 750 --v 5  --ar 2:3 --q 2 --s 750 --v 5",
    "3d image, cute girl, in the style of Pixar --ar 1:2 --stylize 750, 4K resolution highlights, Sharp focus, octane render, ray tracing, Ultra-High-Definition, 8k, UHD, HDR, (Masterpiece:1.5), (best quality:1.5)",
    "Cold coffee in a cup bokeh --ar 85:128 --v 6.0 --style raw5,4k",
    "Food photography of a milk shake with flying strawberrys against a pink background, professionally studio shot with cinematic lighting. The image is in the style of a professional studio shot --ar 85:128 --v 6.0 --style raw"
    
]

css = '''
.gradio-container{max-width: 888px !important}
h1{text-align:center}

.submit-btn {
    background-color: #ec7063 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #e74c3c !important;
}
'''

with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Generate  as  (1024 x 1024)🍺", scale=0, elem_classes="submit-btn")
        
            with gr.Row(visible=True):
                grid_size_selection = gr.Dropdown(
                    choices=["2x1", "1x2", "2x2", "2x3", "3x2", "1x1"],
                    value="1x1",
                    label="Grid Size"
                )

            with gr.Row(visible=True):
                filter_selection = gr.Dropdown(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=FILTER_NAMES,
                    value=DEFAULT_FILTER_NAME,
                    label="Filter Type",
                )

            with gr.Row(visible=True):
                collage_style_selection = gr.Dropdown(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=COLLAGE_STYLE_NAMES,
                    value=DEFAULT_COLLAGE_STYLE_NAME,
                    label="Collage Template + Duotone Canvas",
                )

            with gr.Row(visible=True):
                style_selection = gr.Dropdown(
                    show_label=True,
                    container=True,
                    interactive=True,
                    choices=STYLE_NAMES,
                    value=DEFAULT_STYLE_NAME,
                    label="Quality Style",
                )

            with gr.Accordion("Advanced options", open=False):
                use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=True, visible=True)
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                    value="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                    visible=True,
                )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="Steps",
                        minimum=10,
                        maximum=60,
                        step=1,
                        value=30,
                    )
                with gr.Row():
                    num_images_per_prompt = gr.Slider(
                        label="Images",
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=2,
                    )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                    visible=True
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row(visible=True):
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1024,
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        value=6,
                    )

        with gr.Column(scale=2):
            result = gr.Gallery(label="Result", columns=1, show_label=False)

            gr.Examples(
                examples=examples,
                inputs=prompt,
                outputs=[result, seed],
                fn=generate,
                cache_examples=CACHE_EXAMPLES,
            )

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
            api_name=False,
        )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            use_negative_prompt,
            style_selection,
            collage_style_selection,
            filter_selection,
            grid_size_selection,
            seed,
            width,
            height,
            guidance_scale,
            randomize_seed,
        ],
        outputs=[result, seed],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=40).launch()
