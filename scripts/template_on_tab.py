import gradio as gr
import torch
import dlib
import numpy as np
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DPMSolverMultistepScheduler, ControlNetModel
from detail_encoder.encoder_plus import detail_encoder
from spiga_draw import *
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector
from modules import script_callbacks
import os
from huggingface_hub import snapshot_download


torch.cuda.set_device(0)

processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")

def get_draw(pil_img, size):
    spigas = spiga_process(pil_img, detector)
    if not spigas:
        width, height = pil_img.size
        black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
        return black_image_pil
    else:
        spigas_faces = spiga_segmentation(spigas, size=size)
        return spigas_faces

#to download model
model_id = "runwayml/stable-diffusion-v1-5"
base_path = os.path.join(os.path.dirname(__file__), "..", "models")
makeup_encoder_path = os.path.join(base_path, "pytorch_model.bin")
id_encoder_path = os.path.join(base_path, "pytorch_model_1.bin")
pose_encoder_path = os.path.join(base_path, "pytorch_model_2.bin")
snapshot_download(repo_id="kigy1/Stable-Makeup" , local_dir=base_path)

Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, "openai/clip-vit-large-patch14", "cuda", dtype=torch.float32)
makeup_state_dict = torch.load(makeup_encoder_path)
id_state_dict = torch.load(id_encoder_path)
id_encoder.load_state_dict(id_state_dict, strict=False)
pose_state_dict = torch.load(pose_encoder_path)
pose_encoder.load_state_dict(pose_state_dict, strict=False)
makeup_encoder.load_state_dict(makeup_state_dict, strict=False)

pose_encoder.to("cuda")
id_encoder.to("cuda")
makeup_encoder.to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def model_call(id_image, makeup_image, num):
    try:
        detector = dlib.get_frontal_face_detector()
        image_array = np.array(id_image)
        face_locations = detector(image_array)
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0].top(), face_locations[0].right(), face_locations[0].bottom(), face_locations[0].left()
            margin = 50
            sstop = 50
            top = max(0, top - margin)
            right = min(image_array.shape[1], right + margin)
            bottom = min(image_array.shape[0], bottom + margin)
            left = max(0, left - margin)
            ttop = 0
            tright = 0
            tbottom = 0
            tleft = 0
            while True:
                if (top <= 0) or (left <= 0) or (right >= image_array.shape[1]) or (bottom >= image_array.shape[0]):
                    top = ttop
                    right = tright
                    bottom = tbottom
                    left = tleft
                    break
                else:
                    margin = 20
                    sstop += 20
                    if sstop >= 200:
                        break
                    else:
                        ttop = top
                        tright = right
                        tbottom = bottom
                        tleft = left
                        top = max(0, top - margin)
                        right = min(image_array.shape[1], right + margin)
                        bottom = min(image_array.shape[0], bottom + margin)
                        left = max(0, left - margin)
        face_image_pil = Image.fromarray(id_image.astype('uint8'), 'RGB')
        head_image = face_image_pil.crop((left, top, right, bottom))

        back_img_org = Image.fromarray(id_image.astype('uint8'), 'RGB')

        makeup_image_pil = Image.fromarray(makeup_image.astype('uint8'), 'RGB')

        head_image = head_image.resize((512, 512))
        makeup_image_pil = makeup_image_pil.resize((512, 512))
        back_img = back_img_org.resize((512, 512))

        pose_image = get_draw(id_image, size=512)

        result_img = makeup_encoder.generate(id_image=[head_image, pose_image], makeup_image=makeup_image_pil, guidance_scale=num, pipe=pipe)
        backgr = makeup_encoder.generate(id_image=[back_img, pose_image], makeup_image=makeup_image_pil, guidance_scale=num, pipe=pipe)

        width, height = right - left, bottom - top
        head_image_original_size = result_img.resize((width, height))

        finalsize = np.array(id_image)
        heightx, widthx, channelsx = finalsize.shape
        backgr_size = backgr.resize((widthx, heightx))

        backgr_array = np.array(backgr_size)
        face_with_makeup = Image.fromarray(backgr_array.astype('uint8'), 'RGB')

        face_with_makeup.paste(head_image_original_size, (left, top))

        return face_with_makeup
    except:
        id_image = Image.fromarray(id_image.astype('uint8'), 'RGB')
        org_size = id_image
        makeup_image = Image.fromarray(makeup_image.astype('uint8'), 'RGB')
        id_image = id_image.resize((512, 512))
        makeup_image = makeup_image.resize((512, 512))
        pose_image = get_draw(id_image, size=512)
        result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, guidance_scale=num, pipe=pipe)
        result_img = result_img.resize(org_size.size)
        return result_img


css = '''
#face_image_upload, #makeup_image_upload, #output_image {
    height: 400px;
    max-height: 400px;
    max-width: 350px;
}
#face_image_upload [data-testid="image"], #face_image_upload [data-testid="image"] > div,
#makeup_image_upload [data-testid="image"], #makeup_image_upload [data-testid="image"] > div,
#output_image [data-testid="image"], #output_image [data-testid="image"] > div {
    height: 400px;
    max-height: 400px;
    max-width: 350px;
}
#slider_column {
    width: 20%;  /* Adjust the width as needed */
}
'''

def create_ui():
    with gr.Blocks(css=css, analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image1 = gr.Image(label="Face Image", elem_id="face_image_upload").style(height=400)
                    image2 = gr.Image(label="Makeup Image", elem_id="makeup_image_upload").style(height=400)
            with gr.Column():
                number = gr.Slider(minimum=1.01, maximum=5, value=1.5, label="Makeup Guidance Scale")
                button = gr.Button("Run")
            with gr.Column():
                output = gr.Image(type="pil", label="Output Image", elem_id="output_image").style(height=400)

            def process_images(id_image, makeup_image, num):
                return model_call(id_image, makeup_image, num)

            button.click(process_images, inputs=[image1, image2, number], outputs=output)

        ui_component.title = "Stable-Makeup"
        ui_component.description = "Upload 2 images to see the model output. 1.05-1.15 is suggested for light makeup and 2 for heavy makeup"

    return [(ui_component, "Stable-Makeup", "Stable-Makeup")]



script_callbacks.on_ui_tabs(create_ui)
