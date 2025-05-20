import gc

import gradio as gr
import numpy as np
import torch
from diffusers import FluxControlNetImg2ImgPipeline, FluxPipeline, FluxControlNetModel, FluxControlPipeline
from controlnet_aux import CannyDetector
from diffusers.utils import load_image
from PIL import Image
from image_gen_aux import DepthPreprocessor

# --------------------------------------------------------------


# Function to unload model
def unload_model(model_state):
    if model_state is not None:
        del model_state

    try: 
        for name in list(globals()):
            if name not in ["gc", "torch", "unload_all_models"]:  # giữ lại những gì cần
                del globals()[name]
    except:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    return None, None

# Function to load model
def load_model(
    mode,
    model_state,
    preproc_state,
    load_lora=False,
    lora_model_name="black-forest-labs/FLUX.1-Canny-dev-lora",
    ip_adapter_scale=0.9,
    ip_adapter_model_name="XLabs-AI/flux-ip-adapter",
    ip_adapter_weight_name="ip_adapter.safetensors",
):
    model_state, preproc_state = unload_model(model_state)
    
    if mode == "Text to Image":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        
        # Move all components to the same device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # If CUDA not available, use CPU with consistent dtype
            pipe = pipe.to("cpu")
            
        # if load_lora:
        #     pipe.load_lora_weights(
        #         lora_model_name,
        #         weight_name=lora_weight_name,
        #         adapter_name="custom_lora",
        #     )
        #     pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        return (
            pipe,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif mode == "Image to Image (Depth Control)":
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny-alpha", 
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        )
        
        # Move all components to the same device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # If CUDA not available, use CPU with consistent dtype
            pipe = pipe.to("cpu")
        
        # processor = DepthPreprocessor.from_pretrained(
        #     "LiheYoung/depth-anything-large-hf"
        # )

        # if load_lora:
        #     pipe.load_lora_weights(
        #         lora_model_name,
        #         weight_name=lora_weight_name,
        #         adapter_name="custom_lora",
        #     )
        #     pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        return (
            pipe,
            None,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif mode == "Image Control (Depth)":
        pipe = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        
        # Move all components to the same device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # If CUDA not available, use CPU with consistent dtype
            pipe = pipe.to("cpu")

        if load_lora:
            pipe.load_lora_weights(
                lora_model_name if lora_model_name else "black-forest-labs/FLUX.1-Canny-dev-lora",
                # weight_name=lora_weight_name,
                # adapter_name="custom_lora",
            )
            # pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        try:
            processor = DepthPreprocessor.from_pretrained(
                "LiheYoung/depth-anything-large-hf"
            )
        except:
            processor = None

        print("Processor", processor)
        return (
            pipe,
            processor,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif mode == "Image to Image (IP Adapter)":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        
        # Move all components to the same device
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # If CUDA not available, use CPU with consistent dtype
            pipe = pipe.to("cpu")
            
        # if load_lora:
        #     pipe.load_lora_weights(
        #         lora_model_name,
        #         weight_name=lora_weight_name,
        #         adapter_name="custom_lora",
        #     )
        #     pipe.set_adapters(["custom_lora"], adapter_weights=[lora_scale])
        pipe.load_ip_adapter(
            ip_adapter_model_name,
            weight_name=ip_adapter_weight_name,
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)
        return (
            pipe,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    else:
        return (
            None,
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )


def text_to_image_gr(
    model_state,
    prompt,
    guidance_scale,
    height,
    width,
    num_inference_steps,
    max_sequence_length,
):
    if model_state is None:
        return None
    out = model_state(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
    ).images[0]
    return out


def image_to_image_gr(
    model_state,
    preproc_state,
    init_image,
    control_image,
    prompt,
    guidance_scale,
    height,
    width,
    num_inference_steps,
    max_sequence_length,
    strength,
    seed,
):
    if model_state is None:
        return None
        
    # Process init_image
    if isinstance(init_image, np.ndarray):
        init_img = Image.fromarray(init_image.astype("uint8"))
    elif isinstance(init_image, Image.Image):
        init_img = init_image
    elif isinstance(init_image, str):
        init_img = load_image(init_image)
    else:
        raise ValueError("init_image must be a numpy array, PIL.Image, or string")
    
    # Ensure init_image is RGB
    if hasattr(init_img, "mode") and init_img.mode != "RGB":
        init_img = init_img.convert("RGB")
        
    # Process control_image (depth or canny)
    if isinstance(control_image, np.ndarray):
        ctrl_img = Image.fromarray(control_image.astype("uint8"))
    elif isinstance(control_image, Image.Image):
        ctrl_img = control_image
    elif isinstance(control_image, str):
        ctrl_img = load_image(control_image)
    else:
        raise ValueError("control_image must be a numpy array, PIL.Image, or string")
        
    # Ensure control_image is RGB
    if hasattr(ctrl_img, "mode") and ctrl_img.mode != "RGB":
        ctrl_img = ctrl_img.convert("RGB")
        
    # Apply depth processor if available
    if preproc_state is not None:
        ctrl_img = preproc_state(ctrl_img)[0]
        
        # Ensure control_image is always RGB 3 channel
        if isinstance(ctrl_img, np.ndarray):
            if ctrl_img.ndim == 2:  # grayscale
                ctrl_img = np.stack([ctrl_img] * 3, axis=-1)
            elif ctrl_img.shape[-1] == 1:
                ctrl_img = np.repeat(ctrl_img, 3, axis=-1)
            ctrl_img = Image.fromarray(ctrl_img.astype(np.uint8))
        elif "torch" in str(type(ctrl_img)):
            arr = ctrl_img.cpu().numpy()
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[0] == 1:
                arr = np.repeat(arr, 3, axis=0)
            arr = np.moveaxis(arr, 0, -1)  # CHW -> HWC
            ctrl_img = Image.fromarray(arr.astype(np.uint8))
        if hasattr(ctrl_img, "mode") and ctrl_img.mode != "RGB":
            ctrl_img = ctrl_img.convert("RGB")
            
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    image = model_state(
        prompt=prompt,
        image=init_img,
        control_image=ctrl_img,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
    ).images[0]
    return image


def image_to_image_ip_adapter_gr(
    model_state,
    init_image,
    prompt,
    negative_prompt,
    guidance_scale,
    height,
    width,
    seed,
):
    if model_state is None:
        return None
    if isinstance(init_image, np.ndarray):
        ip_adapter_image = Image.fromarray(init_image.astype("uint8"))
    elif isinstance(init_image, Image.Image):
        ip_adapter_image = init_image
    elif isinstance(init_image, str):
        ip_adapter_image = load_image(init_image)
    else:
        raise ValueError("init_image must be a numpy array, PIL.Image, or string")

    if hasattr(ip_adapter_image, "mode") and ip_adapter_image.mode != "RGB":
        ip_adapter_image = ip_adapter_image.convert("RGB")

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    image = model_state(
        width=width,
        height=height,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=guidance_scale,
        generator=generator,
        ip_adapter_image=ip_adapter_image,
    ).images[0]
    return image


def control_only_gr(
    model_state,
    preproc_state,
    control_image,
    prompt,
    guidance_scale,
    height,
    width,
    num_inference_steps,
    max_sequence_length,
    strength,
    seed,
):
    # if model_state is None:
    #     return None
        
    # Process control_image (depth or canny)
    if isinstance(control_image, np.ndarray):
        ctrl_img = Image.fromarray(control_image.astype("uint8"))
    elif isinstance(control_image, Image.Image):
        ctrl_img = control_image
    elif isinstance(control_image, str):
        ctrl_img = load_image(control_image)
    else:
        raise ValueError("control_image must be a numpy array, PIL.Image, or string")
        
    # Ensure control_image is RGB
    if hasattr(ctrl_img, "mode") and ctrl_img.mode != "RGB":
        ctrl_img = ctrl_img.convert("RGB")
        
    # Apply depth processor if available
    print(preproc_state, height, width)
    if preproc_state is not None:
        ctrl_img = preproc_state(ctrl_img)[0].convert("RGB")

    else:
        print("Before", ctrl_img.size)
        orig_w, orig_h = ctrl_img.size

        # Làm tròn chiều về bội số của 8 để tránh lỗi reshape
        def round_to_multiple(x, base=8):
            return base * round(x / base)

        width = round_to_multiple(orig_w)
        height = round_to_multiple(orig_h)

        ctrl_img = ctrl_img.resize((width, height))  # Resize trước để tránh bị mismatch
        processor = CannyDetector()
        ctrl_img = processor(ctrl_img, low_threshold=50, high_threshold=200,
                            detect_resolution=width, image_resolution=width)

        print("After", ctrl_img.size)

    print(height, width)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    
    image = model_state(
        prompt=prompt,
        control_image=ctrl_img,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return image


demo_css = """
.blinking {
    animation: blinker 1s linear infinite;
}
@keyframes blinker {
    50% { opacity: 0.3; }
}
"""

with gr.Blocks(css=demo_css) as demo:
    gr.Markdown("# FLUX.1")
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Dropdown(
                [
                    "Text to Image",
                    "Image to Image (Depth Control)",
                    "Image Control (Depth)",
                    "Image to Image (IP Adapter)",
                ],
                value="Text to Image",
                label="Mode",
                info="Choose the generation mode.",
            )
        with gr.Column(scale=1):
            lora_checkbox = gr.Checkbox(
                label="Load LoRA",
                value=False,
                visible=False
            )
            lora_model_box = gr.Textbox(
                label="LoRA Model",
                value="black-forest-labs/FLUX.1-Canny-dev-lora",
                visible=False,
                info="HuggingFace model repo or path for LoRA weights.",
            )
            # lora_weight_name_box = gr.Textbox(
            #     label="LoRA Weight Name",
            #     value="anime_lora.safetensors",
            #     visible=False,
            #     info="Name of the LoRA weight file.",
            # )
            ip_adapter_model_box_global = gr.Textbox(
                label="IP-Adapter Model",
                value="XLabs-AI/flux-ip-adapter",
                visible=False,
                info="HuggingFace model repo or path for IP-Adapter weights.",
            )
            ip_adapter_scale_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.01,
                label="IP-Adapter Scale",
                visible=False,
                info="Adjust the influence of the loaded IP-Adapter weights.",
            )
            ip_adapter_weight_name_box_global = gr.Textbox(
                label="IP-Adapter Weight Name",
                value="ip_adapter.safetensors",
                visible=False,
                info="Name of the IP-Adapter weight file.",
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("Load Model", size="lg", variant="primary")
            # Status message box below the Load Model button
            status_msg_box = gr.Textbox(
                label="Status", value="", interactive=False, visible=True
            )

    model_state = gr.State(None)
    preproc_state = gr.State(None)

    with gr.Column(visible=True) as txt2img_col:
        prompt = gr.Textbox(
            label="Prompt",
            value="A cat holding a sign that says hello world",
            info="Describe the image you want to generate.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale = gr.Slider(
                0,
                20,
                value=0.0,
                step=0.1,
                label="Guidance Scale",
            )
            height = gr.Slider(
                256,
                1536,
                value=640,
                step=8,
                label="Height",
            )
            width = gr.Slider(
                256,
                2048,
                value=640,
                step=8,
                label="Width",
            )
            num_inference_steps = gr.Slider(
                1,
                100,
                value=4,
                step=1,
                label="Num Inference Steps",
            )
            max_sequence_length = gr.Slider(
                32,
                512,
                value=256,
                step=8,
                label="Max Sequence Length",
            )
        gen_btn = gr.Button("Generate")
        img_out = gr.Image(label="Output Image")

    with gr.Column(visible=False) as img2img_col:
        with gr.Row():
            with gr.Column(scale=1):
                init_img = gr.Image(
                    label="Input Image",
                )
            with gr.Column(scale=1):
                control_img = gr.Image(
                    label="Control Image",
                )
        prompt2 = gr.Textbox(
            label="Prompt",
            value="",
            info="Describe the modifications or style for the output image.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale2 = gr.Slider(
                0,
                20,
                value=10.0,
                step=0.1,
                label="Guidance Scale",
            )
            height2 = gr.Slider(
                256,
                1536,
                value=640,
                step=8,
                label="Height",
            )
            width2 = gr.Slider(
                256,
                2048,
                value=640,
                step=8,
                label="Width",
            )
            num_inference_steps2 = gr.Slider(
                1,
                100,
                value=30,
                step=1,
                label="Num Inference Steps",
            )
            max_sequence_length2 = gr.Slider(
                32,
                512,
                value=256,
                step=8,
                label="Max Sequence Length",
            )
            strength2 = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.01,
                label="Strength",
            )
            seed2 = gr.Number(
                value=42,
                label="Seed (int)",
            )
        gen_btn2 = gr.Button("Generate")
        img_out2 = gr.Image(label="Output Image")
        # if not HAS_DEPTH:
        #     gr.Markdown(
        #         "<span style='color:red'>Missing package image_gen_aux or DepthPreprocessor! Please install to use Depth Control.</span>"
        #     )

    with gr.Column(visible=False) as ipadapter_col:
        init_img_ip = gr.Image(
            label="Input Image",
        )
        prompt_ip = gr.Textbox(
            label="Prompt",
            value="",
            info="Describe the modifications or style for the output image.",
        )
        with gr.Accordion("Advanced Options", open=False):
            negative_prompt_ip = gr.Textbox(
                label="Negative Prompt",
                value="",
                info="Describe what you do NOT want to see in the image.",
            )
            guidance_scale_ip = gr.Slider(
                0,
                20,
                value=4.0,
                step=0.1,
                label="Guidance Scale (true_cfg_scale)",
            )
            height_ip = gr.Slider(
                256,
                1536,
                value=640,
                step=8,
                label="Height",
            )
            width_ip = gr.Slider(
                256,
                2048,
                value=640,
                step=8,
                label="Width",
            )
            seed_ip = gr.Number(
                value=4444,
                label="Seed (int)",
            )
        gen_btn_ip = gr.Button("Generate")
        img_out_ip = gr.Image(label="Output Image")

    with gr.Column(visible=False) as control_only_col:
        control_only_img = gr.Image(
            label="Control Image",
        )
        prompt_ctrl = gr.Textbox(
            label="Prompt",
            value="A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts.",
            info="Describe the image you want to generate based on the control image.",
        )
        with gr.Accordion("Advanced Options", open=False):
            guidance_scale_ctrl = gr.Slider(
                0,
                20,
                value=10.0,
                step=0.1,
                label="Guidance Scale",
            )
            height_ctrl = gr.Slider(
                256,
                1536,
                value=640,
                step=8,
                label="Height",
            )
            width_ctrl = gr.Slider(
                256,
                2048,
                value=640,
                step=8,
                label="Width",
            )
            num_inference_steps_ctrl = gr.Slider(
                1,
                100,
                value=30,
                step=1,
                label="Num Inference Steps",
            )
            max_sequence_length_ctrl = gr.Slider(
                32,
                512,
                value=256,
                step=8,
                label="Max Sequence Length",
            )
            strength_ctrl = gr.Slider(
                0.0,
                1.0,
                value=0.5,
                step=0.01,
                label="Strength",
            )
            seed_ctrl = gr.Number(
                value=42,
                label="Seed (int)",
            )
        gen_btn_ctrl = gr.Button("Generate")
        img_out_ctrl = gr.Image(label="Output Image")

    def switch_mode(selected_mode):
        if selected_mode == "Text to Image":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                None,
                gr.Button(interactive=True, elem_classes=[], variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif selected_mode == "Image to Image (Depth Control)":
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                None,
                gr.Button(interactive=True, elem_classes=[], variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif selected_mode == "Image Control (Depth)":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                None,
                gr.Button(interactive=True, elem_classes=[], variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif selected_mode == "Image to Image (IP Adapter)":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                None,
                None,
                gr.Button(interactive=True, elem_classes=[], variant="primary"),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                None,
                gr.Button(interactive=True, elem_classes=[], variant="primary"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    mode.change(
        switch_mode,
        inputs=mode,
        outputs=[
            txt2img_col,
            img2img_col,
            control_only_col,
            ipadapter_col,
            model_state,
            preproc_state,
            load_btn,
            ip_adapter_model_box_global,
            ip_adapter_weight_name_box_global,
            status_msg_box,
            lora_checkbox,
            lora_model_box,
            ip_adapter_scale_slider,
        ],
    )

    def set_loading_msg():
        return gr.update(value="Loading model...")

    def set_loaded_msg():
        return gr.update(value="Done")

    def clear_loading_msg():
        return gr.update(value="")

    def start_loading():
        return gr.update(value="Loading model..."), gr.Button(
            interactive=False, variant="primary"
        )

    def handle_load_click(
        mode,
        model_state,
        preproc_state,
        lora_checkbox,
        lora_model_box,
        ip_adapter_model_box_global,
        ip_adapter_scale_slider,
        ip_adapter_weight_name_box_global,
    ):
        # Initial state: Loading message and disabled button
        yield (
            gr.update(value="Loading model..."),  # status_msg_box
            gr.Button(interactive=False, variant="primary"),  # load_btn
            gr.skip(),  # model_state
            gr.skip(),  # preproc_state
            gr.skip(),  # txt2img_col
            gr.skip(),  # img2img_col
            gr.skip(),  # control_only_col
            gr.skip(),  # ipadapter_col
        )

        # Load the model (this is the potentially long-running part)
        new_model_state, new_preproc_state, txt2img_viz, img2img_viz, control_only_viz, ipadapter_viz = (
            load_model(
                mode,
                model_state,
                preproc_state,
                lora_checkbox,
                lora_model_box,
                ip_adapter_scale_slider,
                ip_adapter_model_box_global,
                ip_adapter_weight_name_box_global,
            )
        )

        # Final state: Done message and re-enabled button, and update column visibility
        yield (
            gr.update(value="Done"),  # status_msg_box
            gr.Button(interactive=True, variant="primary"),  # load_btn
            new_model_state,  # model_state
            new_preproc_state,  # preproc_state
            txt2img_viz,
            img2img_viz,
            control_only_viz,
            ipadapter_viz,
        )

    # Hiện ô lora_model_box và lora_scale_slider khi lora_checkbox được tích
    def toggle_lora_controls(checked):
        return (
            gr.update(visible=checked)
        )
    
    def toggle_ip_adapter_controls(checked):
        return (
            gr.update(visible=checked),
            gr.update(visible=checked),
            gr.update(visible=checked),
        )
    
    lora_checkbox.change(
        toggle_lora_controls,
        inputs=lora_checkbox,
        outputs=[lora_model_box],
    )

    # Hiệu ứng loading cho nút Load Model (thêm class blinking)
    def unset_btn_loading():
        return gr.Button(interactive=True, elem_classes=[], variant="primary")

    load_btn.click(
        handle_load_click,
        inputs=[
            mode,
            model_state,
            preproc_state,
            lora_checkbox,
            lora_model_box,
            ip_adapter_model_box_global,
            ip_adapter_scale_slider,
            ip_adapter_weight_name_box_global,
        ],
        outputs=[
            status_msg_box,
            load_btn,
            model_state,
            preproc_state,
            txt2img_col,
            img2img_col,
            control_only_col,
            ipadapter_col,
        ],
    )

    gen_btn.click(
        text_to_image_gr,
        inputs=[
            model_state,
            prompt,
            guidance_scale,
            height,
            width,
            num_inference_steps,
            max_sequence_length,
        ],
        outputs=img_out,
    )
    gen_btn2.click(
        image_to_image_gr,
        inputs=[
            model_state,
            preproc_state,
            init_img,
            control_img,
            prompt2,
            guidance_scale2,
            height2,
            width2,
            num_inference_steps2,
            max_sequence_length2,
            strength2,
            seed2,
        ],
        outputs=img_out2,
    )
    gen_btn_ip.click(
        image_to_image_ip_adapter_gr,
        inputs=[
            model_state,
            init_img_ip,
            prompt_ip,
            negative_prompt_ip,
            guidance_scale_ip,
            height_ip,
            width_ip,
            seed_ip,
        ],
        outputs=img_out_ip,
    )
    gen_btn_ctrl.click(
        control_only_gr,
        inputs=[
            model_state,
            preproc_state,
            control_only_img,
            prompt_ctrl,
            guidance_scale_ctrl,
            height_ctrl,
            width_ctrl,
            num_inference_steps_ctrl,
            max_sequence_length_ctrl,
            strength_ctrl,
            seed_ctrl,
        ],
        outputs=img_out_ctrl,
    )

if __name__ == "__main__":
    demo.launch(share=True)
