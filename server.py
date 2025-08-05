from fastapi import FastAPI, status, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_2.build_sam import build_sam2, build_sam2_video_predictor
from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator


import os
import cv2
import time
import torch
import shutil
import zipfile
import tempfile
import numpy as np
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
from base64 import b64encode, b64decode

# Map object label to color (lane: blue, drivable area: orange, others: tab20 colormap)
def get_mask(mask, obj_id=None):
    # Lane: blue, Drivable area: orange, else: tab20
    if obj_id == 1:  # Lane
        color = np.array([0.2, 0.4, 1.0, 0.7])  # blue RGBA
    elif obj_id == 2:  # Drivable area
        color = np.array([1.0, 0.5, 0.0, 0.7])  # orange RGBA
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id % 20
        color = np.array([*cmap(cmap_idx)[:3], 0.7])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def overlay_mask(image, masks, obj_ids=None):
    out = image.astype(np.float32)
    if obj_ids is None:
        obj_ids = [None] * len(masks)
    for mask_, obj_id in zip(masks, obj_ids):
        alpha = mask_[..., 3:]
        mask_rgb = mask_[..., :3] * 255
        out = out * (1 - alpha) + mask_rgb * alpha
    return out.astype(np.uint8)

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def read_content(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()



# --- FastAPI app definition and CORS middleware setup ---

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model setup: device, use_sam2, predictor, vid_predictor, mask_generator ---
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

use_sam2 = True
if not use_sam2:
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    print("Loading model")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    print("Finishing loading")
    predictor = SamPredictor(sam)
    mask_generator = SamAutomaticMaskGenerator(sam)

else:
    sam2_checkpoint = "sam2_hiera_tiny.pt"
    model_cfg = "sam2_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    vid_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = None
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    # --- SAM2 seg_propagation implementation ---
    def seg_propagation():
        global VIDEO_PATH, FPS, inference_state, vid_predictor
        # Get all frame paths in VIDEO_PATH
        frame_paths = sorted([os.path.join(VIDEO_PATH, f) for f in os.listdir(VIDEO_PATH) if f.endswith('.jpg')], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not frame_paths:
            raise RuntimeError("No frames found in video path.")
        # Propagate masks for all frames and store in inference_state['masks']
        mask_results = list(vid_predictor.propagate_in_video(inference_state))
        masks = []
        for frame_idx, obj_ids, video_res_masks in mask_results:
            # video_res_masks shape: (num_objects, H, W)
            # For single-object, squeeze to (H, W)
            if video_res_masks.shape[0] == 1:
                masks.append(video_res_masks[0])
            else:
                # For multi-object, create a single mask with object ids
                mask = np.zeros_like(video_res_masks[0], dtype=np.uint8)
                for i, obj_id in enumerate(obj_ids):
                    mask[video_res_masks[i] > 0.5] = obj_id
                masks.append(mask)
        inference_state['masks'] = masks
        if not masks or len(masks) != len(frame_paths):
            raise RuntimeError(f"Number of masks ({len(masks) if masks else 0}) does not match number of frames ({len(frame_paths)}).")
        # Overlay masks on frames
        out_frames = []
        for idx, (frame_path, mask) in enumerate(zip(frame_paths, masks)):
            frame = cv2.imread(frame_path)
            # If mask is multi-class, overlay each class with its color
            if mask.ndim == 2:
                mask = mask[None, ...]
            overlay = frame.copy().astype(np.float32)
            for obj_id in np.unique(mask):
                if obj_id == 0:
                    continue  # background
                mask_bin = (mask[0] == obj_id)
                if hasattr(mask_bin, 'cpu'):
                    mask_bin = mask_bin.cpu().numpy()
                mask_bin = mask_bin.astype(np.uint8)
                color = None
                if obj_id == 1:
                    color = np.array([0.2, 0.4, 1.0, 0.7])  # blue RGBA
                elif obj_id == 2:
                    color = np.array([1.0, 0.5, 0.0, 0.7])  # orange RGBA
                else:
                    cmap = plt.get_cmap("tab20")
                    cmap_idx = int(obj_id) % 20
                    color = np.array([*cmap(cmap_idx)[:3], 0.7])
                alpha = color[3]
                rgb = (color[:3] * 255).astype(np.uint8)
                mask_rgb = np.zeros_like(frame, dtype=np.float32)
                for c in range(3):
                    mask_rgb[..., c] = rgb[c] * mask_bin
                overlay = overlay * (1 - alpha * mask_bin[..., None]) + mask_rgb * (alpha * mask_bin[..., None])
            out_frames.append(overlay.astype(np.uint8))
        # Write output video
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        output_dir = './output'
        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        os.makedirs(output_dir, exist_ok=True)
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out_frames], fps=FPS or 25)
        clip.write_videofile(output_video_path, fps=FPS or 25, audio=False)
        print(f"[DEBUG] Wrote video to {output_video_path}")
        return output_video_path

# Define a palette for video segmentation
import random
palette = [random.randint(0, 255) for _ in range(256*3)]

input_point = []
input_label = []
masks = []

segmented_mask = []
interactive_mask = []
mask_input = None

GLOBAL_IMAGE = None
GLOBAL_MASK = None
GLOBAL_ZIPBUFFER = None

@app.post("/image")
async def process_images(
    image: UploadFile = File(...)
):
    global segmented_mask, interactive_mask
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER

    segmented_mask = []
    interactive_mask = []

    # Read the image and mask data as bytes
    image_data = await image.read()

    image_data = BytesIO(image_data)
    img = np.array(Image.open(image_data))
    print("get image", img.shape)
    GLOBAL_IMAGE = img[:,:,:-1]
    GLOBAL_MASK = None
    GLOBAL_ZIPBUFFER = None

    predictor.set_image(GLOBAL_IMAGE)

    # Return a JSON response
    return JSONResponse(
        content={
            "message": "Images received successfully",
        },
        status_code=200,
    )

from XMem import XMem, InferenceCore, image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis

torch.set_grad_enabled(False)

if not use_sam2:
    def seg_propagation(video_name, mask_name):
        # default configuration
        config = {
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'enable_long_term': True,
            'enable_long_term_count_usage': True,
            'num_prototypes': 128,
            'min_mid_term_frames': 5,
            'max_mid_term_frames': 10,
            'max_long_term_elements': 10000,
        }

        network = XMem(config, './XMem/saves/XMem.pth').eval().to(device)

        im = Image.open(mask_name).convert('L')
        im.putpalette(palette)
        mask = np.array(im)
        acc = 0
        for i in range(256):
            if np.sum(mask==i) == 0:
                acc += 1
                mask[mask==i] -= acc-1
            else:
                mask[mask==i] -= acc
        print(np.unique(mask))
        num_objects = len(np.unique(mask)) - 1

        st = time.time()
        # torch.cuda.empty_cache()

        processor = InferenceCore(network, config=config)
        processor.set_all_labels(range(1, num_objects+1)) # consecutive labels
        cap = cv2.VideoCapture(video_name)

        # You can change these two numbers
        frames_to_propagate = 1500
        visualize_every = 1

        current_frame_index = 0

        masked_video = []

        with torch.cuda.amp.autocast(enabled=True):
            while (cap.isOpened()):
                # load frame-by-frame
                _, frame = cap.read()
                if frame is None or current_frame_index > frames_to_propagate:
                    break

                # convert numpy array to pytorch tensor format
                frame_torch, _ = image_to_torch(frame, device=device)
                if current_frame_index == 0:
                    # initialize with the mask
                    mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
                    # the background mask is not fed into the model
                    prediction = processor.step(frame_torch, mask_torch[1:])
                else:
                    # propagate only
                    prediction = processor.step(frame_torch)

                # argmax, convert to numpy
                prediction = torch_prob_to_numpy_mask(prediction)

                if current_frame_index % visualize_every == 0:
                    visualization = overlay_davis(frame[...,::-1], prediction)
                    masked_video.append(visualization)

                current_frame_index += 1
        ed = time.time()

        print(f"Propagation time: {ed-st} s")

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

        audio = AudioFileClip(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_dir = './output'
        output_video_path = os.path.join(output_dir, 'output_video.mp4')
        os.makedirs(output_dir, exist_ok=True)
        clip = ImageSequenceClip(sequence=masked_video, fps=fps)
        # Set the audio of the new video to be the audio from the original video
        clip = clip.set_audio(audio)
        print(f"[DEBUG] Writing processed video to {output_video_path}")
        clip.write_videofile(output_video_path, fps=fps, audio=True)
        print(f"[DEBUG] Wrote video to {output_video_path}")
        return output_video_path

VIDEO_NAME = ""
VIDEO_PATH = ""
FPS = 0

@app.post("/video")
async def obtain_videos(
    video: UploadFile = File(...)
):
    # Read the video data as bytes
    video_data = await video.read()

    # Write the video data to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_data)
    temp_file.close()

    print(temp_file.name)

    global VIDEO_NAME, VIDEO_PATH, FPS, inference_state
    if VIDEO_NAME != "":
        os.unlink(VIDEO_NAME)
    VIDEO_NAME = temp_file.name

    if use_sam2:
        VIDEO_PATH = os.path.join('./output', VIDEO_NAME.split("/")[-1].split(".")[0])
        os.makedirs(VIDEO_PATH, exist_ok=True)
        assert os.path.exists(VIDEO_PATH)

        print("VIDEO_PATH", VIDEO_PATH)
        # save the video frames in jpg format
        cap = cv2.VideoCapture(VIDEO_NAME)
        frame_count = 0
        while (cap.isOpened()):
            # load frame-by-frame
            _, frame = cap.read()
            if frame is None:
                break
            cv2.imwrite(f"{VIDEO_PATH}/{frame_count}.jpg", frame)
            frame_count += 1
        FPS = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        # Defensive: set default FPS if not detected
        if not FPS or FPS == 0:
            FPS = 25

        inference_state = vid_predictor.init_state(video_path=VIDEO_PATH)
        vid_predictor.reset_state(inference_state)



    return JSONResponse(
        content={
            "message": "upload video successfully",
        },
        status_code=200,
    )

@app.post("/ini_seg")
async def process_videos(
    ini_seg: UploadFile = File(...)
):
    global VIDEO_NAME, VIDEO_PATH

    ini_seg_data = await ini_seg.read()
    if not ini_seg_data or len(ini_seg_data) < 10:
        raise HTTPException(status_code=400, detail="Uploaded mask is empty or invalid. Please draw a mask before segmenting the video.")

    tmp_seg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_seg_file.write(ini_seg_data)
    tmp_seg_file.close()

    print(tmp_seg_file.name)

    if VIDEO_NAME == "" and VIDEO_PATH == "":
        raise HTTPException(status_code=204, detail="No content")
    
    try:
        if not use_sam2:
            res_path = seg_propagation(VIDEO_NAME, tmp_seg_file.name)
        else:
            res_path = seg_propagation()
        print(f"[DEBUG] seg_propagation returned: {res_path}")
    except RuntimeError as e:
        if "No points are provided" in str(e):
            print(f"[ERROR] seg_propagation failed: {e}")
            return JSONResponse(
                content={
                    "error": "No points are provided. Please annotate at least one region (point, box, or mask) on the first frame before segmenting the video.",
                    "message": str(e)
                },
                status_code=400,
            )
        else:
            print(f"[ERROR] seg_propagation failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Video processing failed.")

    os.unlink(tmp_seg_file.name)

    # List all .mp4 files in ./output/ and /tmp/ for debugging
    import glob
    output_mp4s = glob.glob('./output/*.mp4')
    tmp_mp4s = glob.glob('/tmp/*.mp4')
    print(f"[DEBUG] .mp4 files in ./output/: {output_mp4s}")
    print(f"[DEBUG] .mp4 files in /tmp/: {tmp_mp4s}")

    # Always copy the latest .mp4 (excluding output_video.mp4) from ./output/ and /tmp/ to output_video.mp4
    import time, glob, shutil, os
    wait_path = './output/output_video.mp4'
    for _ in range(20):
        # Find all .mp4 files in ./output/ and /tmp/ except output_video.mp4
        mp4s = [f for f in glob.glob('./output/*.mp4') if os.path.basename(f) != 'output_video.mp4']
        mp4s += [f for f in glob.glob('/tmp/*.mp4')]
        if mp4s:
            latest_mp4 = sorted(mp4s, key=os.path.getmtime, reverse=True)[0]
            print(f"[DEBUG] Found latest processed video: {latest_mp4}")
            # Ensure output directory exists
            try:
                os.makedirs(os.path.dirname(wait_path), exist_ok=True)
            except Exception as e:
                print(f"[ERROR] Failed to create output directory: {e}")
            if not os.path.exists(wait_path) or os.path.getmtime(latest_mp4) > os.path.getmtime(wait_path):
                print(f"[DEBUG] Copying {latest_mp4} to {wait_path}")
                try:
                    shutil.copy(latest_mp4, wait_path)
                    print(f"[DEBUG] Copy succeeded. Exists after copy: {os.path.exists(wait_path)}")
                except Exception as e:
                    print(f"[ERROR] Failed to copy {latest_mp4} to {wait_path}: {e}")
        if os.path.exists(wait_path):
            print(f"[DEBUG] Confirmed {wait_path} exists before returning response.")
            break
        print(f"[DEBUG] Waiting for {wait_path} to be created...")
        time.sleep(0.5)
    else:
        print(f"[ERROR] {wait_path} was not created after seg_propagation! Check if seg_propagation failed or mask was empty.")


    # Final check: ensure output video exists and is a valid MP4 (nonzero size)
    if not os.path.exists(wait_path) or os.path.getsize(wait_path) < 1024:
        print(f"[ERROR] Output video {wait_path} is missing or too small. Returning bundled blank.mp4 for browser compatibility.")
        fallback_bundled = './assets/blank.mp4'  # Place a known-good blank.mp4 in assets for emergency fallback
        if os.path.exists(fallback_bundled):
            print(f"[DEBUG] Using bundled blank.mp4 from assets.")
            return FileResponse(
                fallback_bundled,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f'inline; filename="error.mp4"',
                    "Content-Type": "video/mp4"
                },
            )
        else:
            print(f"[ERROR] No fallback video available at all! Please add a blank.mp4 to ./assets/.")
            return JSONResponse(
                content={
                    "error": "Processed video was not generated or is invalid, and fallback video could not be created.",
                    "message": "No fallback video available. Please add a blank.mp4 to ./assets/."
                },
                status_code=500,
            )

    # Extra debug: print file size and confirm before returning
    print(f"[DEBUG] Returning video file: {wait_path}, size: {os.path.getsize(wait_path)} bytes")
    print(f"[DEBUG] VIDEO_NAME: {VIDEO_NAME}")
    print(f"[DEBUG] FileResponse headers: inline; filename=\"{VIDEO_NAME.split('/')[-1].split('.')[0]}.mp4\"")

    # Serve as inline for browser playback, not attachment, and force Content-Type for browser compatibility
    return FileResponse(
        wait_path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{VIDEO_NAME.split("/")[-1].split(".")[0]}.mp4"',
            "Content-Type": "video/mp4"
        },
    )

@app.post("/undo")
async def undo_mask():
    global segmented_mask
    # Only pop if there is something to undo
    if segmented_mask:
        segmented_mask.pop()
        return JSONResponse(
            content={
                "message": "Clear successfully",
            },
            status_code=200,
        )
    else:
        return JSONResponse(
            content={
                "message": "No mask to undo",
            },
            status_code=200,
        )


from fastapi import Request



# --- Enhanced /click endpoint to support object label (e.g., lane, drivable area) ---
@app.post("/click")
async def click_images(
    request: Request,
):
    global mask_input, interactive_mask, inference_state

    form_data = await request.form()
    type_list = [int(i) for i in form_data.get("type").split(',')]
    click_list = [int(i) for i in form_data.get("click_list").split(',')]
    # Accept object label (e.g., 1=lane, 2=drivable area) from frontend, default to 1 if not provided
    obj_label = int(form_data.get("obj_label", 1))
    point_coords = np.array(click_list, np.float32).reshape(-1, 2)
    point_labels = np.array(type_list).reshape(-1)

    print(f"[DEBUG] point_coords: {point_coords}, point_labels: {point_labels}, obj_label: {obj_label}")

    if (len(point_coords) == 1):
        mask_input = None

    try:
        if VIDEO_NAME == "":
            masks_, scores_, logits_ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                multimask_output=True,
            )
            best_idx = np.argmax(scores_)
            res = masks_[best_idx]
            mask_input = logits_[best_idx][None, :, :]
        else:
            # Prevent error if inference_state is not initialized (video not processed)
            if inference_state is None or getattr(inference_state, 'obj_id_to_idx', None) is None and (not hasattr(inference_state, 'get') or inference_state.get('obj_id_to_idx', None) is None):
                raise HTTPException(status_code=400, detail="Video inference state is not initialized. Please upload and process a video first, or use the tool on images or the first frame of a video only.")
            _, _, out_mask_logits = vid_predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_label,
                points=point_coords,
                labels=point_labels,
            )
            res = (out_mask_logits[0][0] > 0.0).cpu().numpy()
    except RuntimeError as e:
        if "set_image" in str(e):
            return JSONResponse(
                content={
                    "error": "No image or video has been uploaded. Please upload an image or video before using the click tool.",
                    "message": str(e)
                },
                status_code=400,
            )
        else:
            return JSONResponse(
                content={
                    "error": "Unexpected error during click mask prediction.",
                    "message": str(e)
                },
                status_code=500,
            )

    len_prompt = len(point_labels)
    len_mask = len(interactive_mask)
    # Store both mask and obj_label for later overlay
    if len_mask == 0 or len_mask < len_prompt:
        interactive_mask.append((res, obj_label))
    else:
        interactive_mask[len_prompt-1] = (res, obj_label)

    res_img = Image.fromarray(res)
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res_img),
            "message": "Images processed successfully"
        },
        status_code=200,
    )


# --- Enhanced finish_click to support multi-class masks ---
@app.post("/finish_click")
async def finish_interactive_click(
    mask_idx: int = Form(...),
):
    global segmented_mask, interactive_mask

    # Store both mask and obj_label
    segmented_mask.append(interactive_mask[mask_idx])
    interactive_mask = list()

    return JSONResponse(
        content={
            "message": "Finish successfully",
        },
        status_code=200,
    )
    

@app.post("/rect")
async def rect_images(
    start_x: int = Form(...), # horizontal
    start_y: int = Form(...), # vertical
    end_x: int = Form(...), # horizontal
    end_y: int = Form(...)  # vertical
):
    # Defensive: check if predictor has an image set
    try:
        masks_, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([[start_x, start_y, end_x, end_y]]),
            multimask_output=False
        )
    except RuntimeError as e:
        if "set_image" in str(e):
            return JSONResponse(
                content={
                    "error": "No image has been uploaded. Please upload an image before using the rectangle tool.",
                    "message": str(e)
                },
                status_code=400,
            )
        else:
            return JSONResponse(
                content={
                    "error": "Unexpected error during rectangle mask prediction.",
                    "message": str(e)
                },
                status_code=500,
            )
    res = Image.fromarray(masks_[0])
    print(masks_[0].shape)
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(res),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.post("/everything")
async def seg_everything():
    """
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
    """
    global GLOBAL_IMAGE, GLOBAL_MASK, GLOBAL_ZIPBUFFER
    if type(GLOBAL_MASK) != type(None):
        return JSONResponse(
            content={
                "masks": pil_image_to_base64(GLOBAL_MASK),
                "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
                "message": "Images processed successfully"
            },
            status_code=200,
        )


    masks = mask_generator.generate(GLOBAL_IMAGE)
    assert len(masks) > 0, "No masks found"

    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(len(sorted_anns))

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1 # color can only be in range [1, 255]
    
    res = Image.fromarray(img)
    GLOBAL_MASK = res

    # Save the original image, mask, and cropped segments to a zip file in memory
    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(GLOBAL_IMAGE)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Cut out the segmented regions as smaller squares
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))

            # Create a transparent image with the same size as the cropped image
            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))

            # Create a mask from the segmentation data and crop it
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))

            # Combine the cropped image with the transparent image using the mask
            result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)

            # Save the result to the zip file
            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())

    # move the file pointer to the beginning of the file so we can read whole file
    zip_buffer.seek(0)
    GLOBAL_ZIPBUFFER = zip_buffer

    # Return a JSON response
    return JSONResponse(
        content={
            "masks": pil_image_to_base64(GLOBAL_MASK),
            "zipfile": b64encode(GLOBAL_ZIPBUFFER.getvalue()).decode("utf-8"),
            "message": "Images processed successfully"
        },
        status_code=200,
    )

@app.get("/assets/{path}/{file_name}", response_class=FileResponse)
async def read_assets(path, file_name):
    return f"assets/{path}/{file_name}"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return read_content('segDrawer 2.html')

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=7860)

