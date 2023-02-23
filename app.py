import os
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    # os.system("python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ./local.ckpt --dump_path sd_weights/")
    model = StableDiffusionPipeline.from_pretrained(
        "sd_weights/", 
        # safety_checker=None,
        revision="fp16",
        torch_dtype=torch.float16
    ).to("cuda")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    steps = model_inputs.get('inference_steps', 50)
    scale = model_inputs.get('guidance_scale', 5)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    fullPrompt = "shbdg " + prompt
    image = model(fullPrompt, num_inference_steps=steps, guidance_scale=scale).images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
     # Return the results as a dictionary
    return {'image_base64': image_base64}
