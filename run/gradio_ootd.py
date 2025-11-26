LOGO_PATH = "brand_logo.png"
import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image, ImageOps

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ------------- 新增：將 logo 轉為 data URI（Base64），避免相對路徑導致瀏覽器無法載入） -------------
import base64, mimetypes
_logo_path = Path(__file__).parent / LOGO_PATH
if not _logo_path.exists():
    # fallback to current working dir or given path
    _logo_path = Path(LOGO_PATH)
logo_src = LOGO_PATH  # default: 原始路徑
try:
    if _logo_path.exists():
        _mime = mimetypes.guess_type(str(_logo_path))[0] or "image/png"
        _b64 = base64.b64encode(_logo_path.read_bytes()).decode("utf-8")
        logo_src = f"data:{_mime};base64,{_b64}"
except Exception:
    # 若任何步驟失敗，保留原始 LOGO_PATH（gr.HTML 仍可嘗試載入）
    logo_src = LOGO_PATH
# -----------------------------------------------------------------------------------------------

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC


openpose_model_dc = OpenPose(0)
parsing_model_dc = Parsing(0)
ootd_model_dc = OOTDiffusionDC(0)

# openpose_model_hd = OpenPose(0)
# parsing_model_hd = Parsing(0)
# ootd_model_hd = OOTDiffusionHD(0)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

example_path = os.path.join(os.path.dirname(__file__), 'examples')
model_hd = os.path.join(example_path, 'model/model_1.png')
garment_hd = os.path.join(example_path, 'garment/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/model_8.png')
garment_dc = os.path.join(example_path, 'garment/048554_1.jpg')


# =========================
# Full-body (DC) inference
# =========================
def process_dc(vton_img, garm_img, category, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    if category == 'Upper-body':
        category = 0
    elif category == 'Lower-body':
        category = 1
    else:
        category = 2

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))

        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(
            model_type, category_dict_utils[category], model_parse, keypoints
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images


# =========================
# Half-body (HD) inference
# =========================
def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'hd'
    category = 0  # only upper-body

    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))

        keypoints = openpose_model_hd(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_hd(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(
            model_type, category_dict_utils[category], model_parse, keypoints
        )
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_hd(
            model_type=model_type,
            category=category_dict[category],
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images


# =========================
# UI with Tabs
# =========================

block = gr.Blocks(css="custom.css", title="超會搭 - 虛擬換衣系統").queue()

with block:

    with gr.Row(elem_classes="header-bar"):
        # 將 gr.Image 放入 scale=0 的 Column，確保它只佔用最小空間
        with gr.Column(scale=0):
            # 改用 gr.HTML 直接插入 <img>，避免 gr.Image 產生外框
            gr.HTML(
                # 使用 logo_src（可能是 data URI），確保瀏覽器可直接顯示圖片
                f'<img src="{logo_src}" alt="logo" class="header-logo" '
                'style="display:block; max-height:56px; height:auto; border:0; object-fit:contain;" />'
            )
        # 標題 (佔用剩餘空間)
        gr.HTML(
    "<div class='header-title'>超會搭 — AI 虛擬換衣系統</div>"
    )
    # --------------------
    #      Tabs UI
    # --------------------
    with gr.Tabs():

        # FULL BODY TAB（已放在最前面）
        with gr.Tab("Full Body"):

            gr.Markdown("### Full-body 虛擬換衣（支援 Upper / Lower / Dress）")
            gr.Markdown("***請確保服裝類別一致，否則效果可能不佳***")

            with gr.Row():

                # Model
                with gr.Column():
                    vton_img_dc = gr.Image(
                        label="Model",
                        sources='upload',
                        type="filepath",
                        height=384,
                        value=model_dc
                    )
                    gr.Examples(
                        label="Example Models",
                        inputs=vton_img_dc,
                        examples_per_page=6,
                        examples=[
                            os.path.join(example_path, 'model/model_8.png'),
                            os.path.join(example_path, 'model/049447_0.jpg'),
                            os.path.join(example_path, 'model/049713_0.jpg'),
                            os.path.join(example_path, 'model/051482_0.jpg'),
                            os.path.join(example_path, 'model/051918_0.jpg'),
                            os.path.join(example_path, 'model/049205_0.jpg'),
                        ]
                    )

                # Garment
                with gr.Column():
                    garm_img_dc = gr.Image(
                        label="Garment",
                        sources='upload',
                        type="filepath",
                        height=384,
                        value=garment_dc
                    )
                    category_dc = gr.Dropdown(
                        label="Garment Category",
                        choices=["Upper-body", "Lower-body", "Dress"],
                        value="Upper-body"
                    )
                    gr.Examples(
                        label="Example Garments",
                        inputs=garm_img_dc,
                        examples_per_page=6,
                        examples=[
                            os.path.join(example_path, 'garment/048554_1.jpg'),
                            os.path.join(example_path, 'garment/049920_1.jpg'),
                            os.path.join(example_path, 'garment/051827_1.jpg'),
                            os.path.join(example_path, 'garment/053290_1.jpg'),
                            os.path.join(example_path, 'garment/053742_1.jpg'),
                            os.path.join(example_path, 'garment/00126_00.jpg'),
                        ]
                    )

                # Output
                with gr.Column():
                    result_gallery_dc = gr.Gallery(
                        label="Output",
                        show_label=False,
                        elem_id="gallery"
                    )

            # Parameters
            run_button_dc = gr.Button("Run Full Body")
            n_samples_dc = gr.Slider(1, 4, value=1, step=1, label="Images")
            n_steps_dc = gr.Slider(20, 40, value=20, step=1, label="Steps")
            image_scale_dc = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="Guidance scale")
            seed_dc = gr.Slider(-1, 2147483647, value=-1, step=1, label="Seed")

            ips_dc = [vton_img_dc, garm_img_dc, category_dc, n_samples_dc, n_steps_dc, image_scale_dc, seed_dc]
            run_button_dc.click(fn=process_dc, inputs=ips_dc, outputs=[result_gallery_dc])

        # HALF BODY TAB
        with gr.Tab("Half Body"):

            gr.Markdown("### Half-body 虛擬換衣（只支援 Upper-body）")

            with gr.Row():

                with gr.Column():
                    vton_img = gr.Image(
                        label="Model",
                        sources="upload",
                        type="filepath",
                        height=384,
                        value=model_hd
                    )
                    gr.Examples(
                        inputs=vton_img,
                        examples_per_page=10,
                        examples=[
                            os.path.join(example_path, 'model/model_1.png'),
                            os.path.join(example_path, 'model/model_2.png'),
                            os.path.join(example_path, 'model/model_3.png'),
                            os.path.join(example_path, 'model/model_4.png'),
                            os.path.join(example_path, 'model/01861_00.jpg'),
                        ]
                    )

                with gr.Column():
                    garm_img = gr.Image(
                        label="Garment",
                        sources="upload",
                        type="filepath",
                        height=384,
                        value=garment_hd
                    )
                    gr.Examples(
                        inputs=garm_img,
                        examples_per_page=10,
                        examples=[
                            os.path.join(example_path, 'garment/03244_00.jpg'),
                            os.path.join(example_path, 'garment/00126_00.jpg'),
                            os.path.join(example_path, 'garment/00470_00.jpg'),
                            os.path.join(example_path, 'garment/02015_00.jpg'),
                        ]
                    )

                with gr.Column():
                    result_gallery = gr.Gallery(
                        label="Output",
                        show_label=False,
                        elem_id="gallery"
                    )

            # Parameters
            run_button = gr.Button("Run Half Body")
            n_samples = gr.Slider(1, 4, value=1, step=1, label="Images")
            n_steps = gr.Slider(20, 40, value=20, step=1, label="Steps")
            image_scale = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="Guidance scale")
            seed = gr.Slider(-1, 2147483647, value=-1, step=1, label="Seed")

            ips = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
            run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])


# Launch UI
block.launch(server_name='0.0.0.0', server_port=7865, share=True)
