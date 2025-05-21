import os
import torch
import numpy as np
from PIL import Image
import sys


from uno.flux.modules.conditioner import HFEmbedder
from uno.flux.pipeline import UNOPipeline, preprocess_ref
from uno.flux.util import configs, print_load_warning, set_lora
from safetensors.torch import load_file as load_sft
from huggingface_hub import snapshot_download

try:
    cache_dir = snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_files_only=True
    )
    print(f"Loaded cached repo at {cache_dir}")
except Exception as e:
    print("Cache not found, downloading from Hugging Face...")
    cache_dir = snapshot_download("black-forest-labs/FLUX.1-dev", local_files_only=False)
    print(f"Downloaded repo at {cache_dir}")

print("cache_dir", cache_dir)
t5_cache_dir = os.path.join(cache_dir, "text_encoder_2")
ae_cache_dir = os.path.join(cache_dir, "ae.safetensors")
flux_cache_path = os.path.join(cache_dir, "flux1-dev.safetensors")
vae_cache_path = os.path.join(cache_dir, "vae/diffusion_pytorch_model.safetensors")

print("t5_cache_dir", t5_cache_dir)
print("ae_cache_dir", ae_cache_dir)
print("flux_cache_path", flux_cache_path)
print("vae_cache_path", vae_cache_path)

try:
    lora_dir = snapshot_download("bytedance-research/UNO", local_files_only=True)
    print(f"Loaded cached repo at {lora_dir}")
except Exception as e:
    print("Cache not found, downloading from Hugging Face...")
    lora_dir = snapshot_download("bytedance-research/UNO", local_files_only=False)
    print(f"Downloaded repo at {lora_dir}")

lora_paths = os.path.join(lora_dir, "dit_lora.safetensors")

try:
    clip_path = snapshot_download("openai/clip-vit-large-patch14", local_files_only=True)
    print(f"Loaded cached repo at {clip_path}")
except Exception as e:
    print("Cache not found, downloading from Hugging Face...")
    clip_path = snapshot_download("openai/clip-vit-large-patch14", local_files_only=False)
    print(f"Downloaded repo at {clip_path}")

def custom_load_flux_model(model_path, device, use_fp8=True, lora_rank=512, lora_path=None):
    from uno.flux.model import Flux
    from uno.flux.util import load_model
    
    if use_fp8:
        params = configs["flux-dev-fp8"].params
    else:
        params = configs["flux-dev"].params
    
    with torch.device("meta" if model_path is not None else device):
        model = Flux(params)
    
    if os.path.exists(lora_path):
        print(f"Using only_lora mode with rank: {lora_rank}")
        model = set_lora(model, lora_rank, device="meta" if model_path is not None else device)
    
    if model_path is not None:
        print(f"Loading Flux model from {model_path}")
        print("Loading lora")
        lora_sd = load_sft(lora_path, device=str(device)) if lora_path.endswith("safetensors")\
            else torch.load(lora_path, map_location='cpu', weights_only=False)
        print("Loading main checkpoint")
        if model_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(model_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(model_path, device=str(device))
            
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(model_path, map_location='cpu', weights_only=False)
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)

    return model

def custom_load_ae(ae_path, device):
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.util import load_model
    
    ae_params = configs["flux-dev"].ae_params
    
    with torch.device("meta" if ae_path is not None else device):
        ae = AutoEncoder(ae_params)
    
    if ae_path is not None:
        print(f"Loading AutoEncoder from {ae_path}")
        if ae_path.endswith('safetensors'):
            sd = load_sft(ae_path, device=str(device))
        else:
            sd = torch.load(ae_path, map_location=str(device), weights_only=False)
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if len(missing) > 0:
            print(f"Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
        
        ae = ae.to(str(device))
    return ae

def custom_load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    return HFEmbedder(cache_dir, max_length=max_length, torch_dtype=torch.bfloat16, subfolder="text_encoder_2", local_files_only=True).to(device)

def custom_load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder(clip_path, max_length=77, torch_dtype=torch.bfloat16, cache_dir=clip_path).to(device)

class FluxModelLoader:
    def __init__(self):
        self.output_dir = os.getcwd()
        self.type = "UNO_MODEL"
        self.loaded_model = None

    @classmethod
    def INPUT_TYPES(cls):
        model_paths = flux_cache_path
        vae_paths = vae_cache_path
        
        return {
            "required": {
                "flux_model": (model_paths, ),
                "ae_model": (vae_paths, ),
                "use_fp8": ("BOOLEAN", {"default": False}),
                "offload": ("BOOLEAN", {"default": False}),
                "lora_model": (["None"] + lora_paths, ),
            }
        }

    RETURN_TYPES = ("UNO_MODEL",)
    RETURN_NAMES = ("uno_model",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    def load_model(self, flux_model_path=flux_cache_path, ae_model_path=vae_cache_path, use_fp8=False, offload=False, lora_model=None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        try:
            lora_model_path = None
            if lora_model is not None and lora_model != "None":
                lora_model_path = lora_dir
            
            print(f"Loading Flux model from: {flux_model_path}")
            print(f"Loading AE model from: {ae_model_path}")
            lora_rank = 512
            if lora_model_path:
                print(f"Loading LoRA model from: {lora_model_path}")
            
            class CustomUNOPipeline(UNOPipeline):
                def __init__(self, use_fp8, device, flux_path, ae_path, offload=False, 
                            lora_rank=512, lora_path=None):
                    self.device = device
                    self.offload = offload
                    self.model_type = "flux-dev-fp8" if use_fp8 else "flux-dev"
                    self.use_fp8 = use_fp8

                    self.clip = custom_load_clip(device=self.device)
                    self.t5 = custom_load_t5(device=self.device, max_length=512)
                    
                    self.ae = custom_load_ae(ae_cache_dir, device=self.device)
                    self.model = custom_load_flux_model(
                        flux_cache_path, 
                        device=self.device, 
                        use_fp8=use_fp8,
                        lora_rank=lora_rank,
                        lora_path=lora_paths
                    )
                    
            model = CustomUNOPipeline(
                use_fp8=use_fp8,
                device=device,
                flux_path=flux_model_path,
                ae_path=ae_model_path,
                offload=offload,
                lora_rank=lora_rank,
                lora_path=lora_paths,
            )
            
            self.loaded_model = model
            print(f"UNO model loaded successfully with custom models.")
            return (model,)
        except Exception as e:
            print(f"Error loading UNO model: {e}")
            import traceback
            traceback.print_exc()
            raise e

class FluxGenerate:
    def __init__(self):
        self.output_dir = os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uno_model": ("UNO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 3407}),
                "pe": (["d", "h", "w", "o"], {"default": "d"}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "UNO"

    def generate(self, uno_model, prompt, width, height, guidance, num_steps, seed, pe, 
                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None):
        # Make sure width and height are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Process reference images if provided
        ref_imgs = []
        ref_tensors = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        for ref_tensor in ref_tensors:
            if ref_tensor is not None:
                # Convert from tensor to PIL
                if isinstance(ref_tensor, torch.Tensor):
                    # Handle batch of images
                    if ref_tensor.dim() == 4:  # [batch, height, width, channels]
                        for i in range(ref_tensor.shape[0]):
                            img = ref_tensor[i].cpu().numpy()
                            ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                            # Determine reference size based on number of reference images
                            ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                            ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                            ref_imgs.append(ref_image_pil)
                    else:  # [height, width, channels]
                        img = ref_tensor.cpu().numpy()
                        ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        # Determine reference size based on number of reference images
                        ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                        ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                        ref_imgs.append(ref_image_pil)
                elif isinstance(ref_tensor, np.ndarray):
                    # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                    ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    # Determine reference size based on number of reference images
                    ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                    ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                    ref_imgs.append(ref_image_pil)
        
        try:
            # Generate image
            output_img = uno_model(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                ref_imgs=ref_imgs,
                pe=pe
            )
            
            return output_img
        except Exception as e:
            print(f"Error generating image with UNO: {e}")
            raise e