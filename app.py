#path1.0398
import os
import random
import uuid
import json

import gradio as gr
import numpy as np
from PIL import Image
import spaces
import torch
from diffusers import DiffusionPipeline
from typing import Tuple

# BaseConditions
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
        "name": "3D Model",
        "prompt": "professional 3d model {prompt}. octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

collage_style_list = [


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
    "name": "StoryBook",
    "prompt": "storybook collage of {prompt}. whimsical illustrations, handwritten text, fairy tale motifs",
    "negative_prompt": "technical, sterile, devoid of imagination",
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
    "name": "ComicBook",
    "prompt": "comic book-style collage of {prompt}. dynamic panels, speech bubbles, bold lines, vibrant colors",
    "negative_prompt": "static, monotonous, muted colors",
    },

    {
        "name": "Retro Pop",
        "prompt": "retro pop art collage of {prompt}. bold colors, comic book style, halftone dots, vintage ads",
        "negative_prompt": "subdued colors, minimalist, modern, subtle",
    },


    {
        "name": "No Style",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
collage_styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in collage_style_list}
STYLE_NAMES = list(styles.keys())
COLLAGE_STYLE_NAMES = list(collage_styles.keys())
DEFAULT_STYLE_NAME = "3840 x 2160"
DEFAULT_COLLAGE_STYLE_NAME = "Cinematic"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    if style_name in styles:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    elif style_name in collage_styles:
        p, n = collage_styles.get(style_name, collage_styles[DEFAULT_COLLAGE_STYLE_NAME])
    else:
        p, n = styles[DEFAULT_STYLE_NAME]
    
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

DESCRIPTION = """"""
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>⚠️Running on CPU, This may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
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
        "num_inference_steps": 25,
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

    "Portrait of a beautiful woman in a hat, summer outfit, with freckles on her face, in a close up shot, with sunlight, outdoors, in soft light, with a beach background, looking at the camera, with high resolution photography, in the style of Hasselblad X2D50c --ar 85:128 --v 6.0 --style raw",
    "Dragon ball, portrait of dr goku, in the style of street art aesthetic, cute cartoonish designs, photo-realistic techniques, dark red, childhood arcadias, anime aesthetic, cartoon-like figures --ar 73:98 --stylize 750 --v 6"
    
]

css = '''
.gradio-container{max-width: 560px !important}
h1{text-align:center}
'''
with gr.Blocks(css=css, theme="xiaobaiyuan/theme_brief") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run")
        result = gr.Gallery(label="Grid", columns=1, preview=True)

    with gr.Row(visible=True):
        collage_style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=COLLAGE_STYLE_NAMES,
            value=DEFAULT_COLLAGE_STYLE_NAME,
            label="Collage Template",
        )
    with gr.Row(visible=True):
        grid_size_selection = gr.Dropdown(
            choices=["2x1", "1x2", "2x2", "2x3", "3x2", "1x1"],
            value="2x3",
            label="Grid Size"
        )
    with gr.Row(visible=True):
        style_selection = gr.Radio(
            show_label=True,
            container=True,
            interactive=True,
            choices=STYLE_NAMES,
            value=DEFAULT_STYLE_NAME,
            label="Style",
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
                maximum=30,
                step=1,
                value=15,
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
                step=8,
                value=1024,
            )
            height = gr.Slider(
                label="Height",
                minimum=512,
                maximum=2048,
                step=8,
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
    demo.queue(max_size=20).launch()