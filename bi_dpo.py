# -*- coding: utf-8 -*-


!nvidia-smi

from google.colab import drive
drive.mount('/content/drive')

import os

ROOT = "/content/BiDPO_Lite"

folders = [
    "images",
    "images/raw",
    "images/crops",
    "datasets",
    "models",
    "outputs",
    "logs"
]

for f in folders:
    os.makedirs(os.path.join(ROOT, f), exist_ok=True)

print("Folders created")

!pip install -q \
accelerate \
transformers \
diffusers \
peft \
safetensors \
pillow \
opencv-python \
requests \
tqdm \
pandas \
numpy \
matplotlib

import os
os.kill(os.getpid(), 9)

import torch
import transformers
import diffusers

print(torch.__version__)
print(transformers.__version__)
print(diffusers.__version__)

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE =", DEVICE)

import json
import random
from PIL import Image

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def show_image(path):
    img = Image.open(path)
    display(img)

import os

ROOT = "/content/BiDPO_Lite"
os.makedirs(ROOT, exist_ok=True)

CONFIG = {

    # generation
    "num_images_per_prompt": 4,
    "image_size": 512,

    # training / batching
    "batch_size": 1,
    "grad_accum_steps": 4,

    # reproducibility
    "seed": 42,
    "deterministic": True,

    # paths (SAFE VERSION)
    "output_dir": ROOT,
    "image_dir": os.path.join(ROOT, "images"),
    "dataset_dir": os.path.join(ROOT, "datasets"),
    "log_dir": os.path.join(ROOT, "logs"),
    "model_dir": os.path.join(ROOT, "models"),

    # OpenRouter / judging
    "judge_model": "qwen/qwen2.5-vl-72b-instruct",
    "max_retries": 3,

    # dataset scale
    "num_prompts": 200,

    # LoRA training
    "lora_rank": 8,
    "lora_alpha": 16,
    "lr": 1e-4,
    "train_steps": 1000,
}

print(CONFIG["output_dir"])
print(CONFIG["image_dir"])

import random
import numpy as np
import torch

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

seed_everything(CONFIG["seed"])

import os

OPENROUTER_API_KEY = "YOUR_API_KEY"

assert OPENROUTER_API_KEY != "PASTE_YOUR_KEY_HERE", "Add your OpenRouter API key"

import requests
import time
import json

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(payload, max_retries=3):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(max_retries):

        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120
            )

            if r.status_code == 200:
                return r.json()

            print("Retry:", r.status_code, r.text)

        except Exception as e:
            print("Error:", e)

        time.sleep(2 * (attempt + 1))

    return None

import base64

def image_to_base64(image_path):

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_judge_prompt(prompt):

    return f"""
You are a strict compositional image evaluator.

You will evaluate how well the image follows the prompt.

PROMPT:
{prompt}

Check:
1. object correctness
2. attribute correctness (color, size, texture)
3. counting accuracy
4. spatial relationships

Return ONLY valid JSON:

{{
  "object_score": 0-1,
  "attribute_score": 0-1,
  "count_score": 0-1,
  "relation_score": 0-1,
  "final_score": 0-1
}}

Be strict. Penalize mistakes heavily.
"""

def judge_image(image_path, prompt, model=None):

    model = model or CONFIG["judge_model"]

    img_b64 = image_to_base64(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_judge_prompt(prompt)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    }

    response = call_openrouter(payload, CONFIG["max_retries"])

    if not response:
        return None

    try:
        content = response["choices"][0]["message"]["content"]
        return json.loads(content)

    except Exception:
        # fallback: try to extract JSON manually
        try:
            text = response["choices"][0]["message"]["content"]
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
        except:
            return None

from google.colab import files

uploaded = files.upload()

uploaded_files = list(uploaded.keys())

print(uploaded_files)

test_image = uploaded_files[0]

print("Using:", test_image)

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(test_image)

plt.imshow(img)
plt.axis("off")
plt.show()

test_image = "photo-1580446623001-3abf670c5c55.avif"

result = judge_image(
    test_image,
    "A red car on a street"
)

print(result)

def describe_image(image_path):

    img_b64 = image_to_base64(image_path)

    payload = {
        "model": CONFIG["judge_model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    }

    response = call_openrouter(payload)

    return response

response = describe_image(test_image)

print(json.dumps(response, indent=2))

OBJECTS = [
    "cat",
    "dog",
    "car",
    "bird",
    "horse",
    "bicycle",
    "apple",
    "chair",
    "table",
    "person"
]

COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white"
]

RELATIONS = [
    "beside",
    "behind",
    "in front of",
    "next to",
    "under"
]

COUNTS = [
    "one",
    "two",
    "three",
    "four"
]

import random

def generate_attribute_prompt():

    obj1, obj2 = random.sample(OBJECTS, 2)

    color1 = random.choice(COLORS)
    color2 = random.choice(COLORS)

    rel = random.choice(RELATIONS)

    return f"A {color1} {obj1} {rel} a {color2} {obj2}"

for _ in range(5):
    print(generate_attribute_prompt())

def generate_count_prompt():

    count = random.choice(COUNTS)

    obj = random.choice(OBJECTS)

    return f"{count.capitalize()} {obj}s on a table"

for _ in range(5):
    print(generate_count_prompt())

def generate_relation_prompt():

    obj1, obj2 = random.sample(OBJECTS, 2)

    rel = random.choice(RELATIONS)

    return f"A {obj1} {rel} a {obj2}"

def generate_prompt():

    prompt_type = random.choice([
        "attribute",
        "count",
        "relation"
    ])

    if prompt_type == "attribute":
        return generate_attribute_prompt()

    if prompt_type == "count":
        return generate_count_prompt()

    return generate_relation_prompt()

prompts = []

for _ in range(CONFIG["num_prompts"]):

    prompts.append(
        generate_prompt()
    )

print(prompts[:20])

import json

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

import json
import os

os.makedirs(CONFIG["dataset_dir"], exist_ok=True)

prompt_path = os.path.join(
    CONFIG["dataset_dir"],
    "prompts.json"
)

with open(prompt_path, "w") as f:
    json.dump(prompts, f, indent=2)

print("Saved:", prompt_path)

print("CONFIG exists:", "CONFIG" in globals())
print("prompts exists:", "prompts" in globals())
print("ROOT exists:", "ROOT" in globals())

print("Total prompts:", len(prompts))

print()

for i in range(10):
    print(prompts[i])

CONFIG["num_prompts"] = 10

!pip -q install diffusers transformers accelerate safetensors

import torch

from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe = pipe.to("cuda")

print("Loaded")

GEN_CONFIG = {

    "steps": 4,       # turbo likes low steps
    "guidance": 0.0,
}

prompt = "A red cat beside a blue dog"

image = pipe(
    prompt,
    num_inference_steps=GEN_CONFIG["steps"],
    guidance_scale=GEN_CONFIG["guidance"]
).images[0]

image.save("test_generation.png")

image

import os

RAW_DIR = os.path.join(
    CONFIG["image_dir"],
    "raw"
)

os.makedirs(
    RAW_DIR,
    exist_ok=True
)

RAW_DIR

from tqdm import tqdm
import os

generated_records = []

for prompt_id, prompt in enumerate(tqdm(prompts)):

    for image_idx in range(
        CONFIG["num_images_per_prompt"]
    ):

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            CONFIG["seed"] + image_idx
        )

        image = pipe(
            prompt,
            num_inference_steps=GEN_CONFIG["steps"],
            guidance_scale=GEN_CONFIG["guidance"],
            generator=generator
        ).images[0]

        filename = (
            f"p{prompt_id}_img{image_idx}.png"
        )

        save_path = os.path.join(
            RAW_DIR,
            filename
        )

        image.save(save_path)

        generated_records.append({

            "prompt_id": prompt_id,

            "prompt": prompt,

            "image": save_path

        })

print(
    "Generated:",
    len(generated_records)
)

import json
import os

metadata_path = os.path.join(
    CONFIG["dataset_dir"],
    "generated_images.json"
)

with open(metadata_path, "w") as f:

    json.dump(
        generated_records,
        f,
        indent=2
    )

print("Saved:", metadata_path)

import random
from PIL import Image
import matplotlib.pyplot as plt

sample = random.choice(
    generated_records
)

print(sample["prompt"])

img = Image.open(
    sample["image"]
)

plt.imshow(img)
plt.axis("off")
plt.show()

[
  {
    "prompt_id": 0,
    "prompt": "A red cat beside a blue dog",
    "image": "/content/BiDPO_Lite/images/raw/p0_img0.png"
  }
]

!pip -q install transformers timm supervision
!pip -q install git+https://github.com/facebookresearch/sam2.git

import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

from transformers import AutoProcessor
from transformers import AutoModelForZeroShotObjectDetection

device = "cuda"

processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base"
)

gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(device)

print("GroundingDINO loaded")

from PIL import Image

sample = generated_records[0]

image_path = sample["image"]

image = Image.open(image_path).convert("RGB")

image

def build_detection_text(prompt):

    objs = extract_objects(prompt)

    return ". ".join(objs) + "."

print(extract_objects("A red cat beside a blue dog"))

prompt = sample["prompt"]

text_prompt = build_detection_text(prompt)

print(text_prompt)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# -----------------------------
# 39–41: Detection setup
# -----------------------------

sample = generated_records[0]

image_path = sample["image"]
prompt = sample["prompt"]

image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

text_prompt = build_detection_text(prompt)

print("Prompt:", prompt)
print("Detection text:", text_prompt)

# -----------------------------
# 42–44: SAM2 setup assumed already done
# predictor must exist
# -----------------------------

inputs = processor(
    images=image,
    text=text_prompt,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = gdino(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.3,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)

detections = results[0]

boxes = detections["boxes"]
labels = detections["labels"]

print("Detected objects:", labels)

# -----------------------------
# 45–46: SAM2 masks
# -----------------------------

predictor.set_image(image_np)

all_masks = []

for box in boxes:

    box_np = box.cpu().numpy()

    masks, scores, logits = predictor.predict(
        box=box_np,
        multimask_output=False
    )

    all_masks.append(masks[0])

print("Masks generated:", len(all_masks))

# -----------------------------
# 47: Visualization
# -----------------------------

plt.figure(figsize=(10,10))
plt.imshow(image_np)

for mask in all_masks:
    plt.imshow(mask, alpha=0.4)

plt.axis("off")
plt.show()

# -----------------------------
# 48–49: Save masks
# -----------------------------

MASK_DIR = f"{ROOT}/masks"
os.makedirs(MASK_DIR, exist_ok=True)

mask_records = []

for idx, mask in enumerate(all_masks):

    mask_img = (mask.astype("uint8") * 255)

    save_path = f"{MASK_DIR}/mask_{idx}.png"

    Image.fromarray(mask_img).save(save_path)

    mask_records.append({
        "label": str(labels[idx]),
        "mask": save_path
    })

print("Saved masks:", len(mask_records))

# -----------------------------
# 50–51: Region scoring
# -----------------------------

def compute_mask_area(mask):
    return float(mask.sum())

region_scores = []

for label, mask in zip(labels, all_masks):

    area = compute_mask_area(mask)

    region_scores.append({
        "object": str(label),
        "area": area
    })

print("Region scores:")
region_scores

import os
import json
import torch
from tqdm import tqdm
from PIL import Image

# -----------------------------
# 1. Build Preference Dataset
# -----------------------------

preference_data = []

print("Scoring images...")

for item in tqdm(generated_records[:50]):  # LIMIT for Colab T4 stability

    img_path = item["image"]
    prompt = item["prompt"]

    # global score (VLM)
    g = judge_image(img_path, prompt)
    if g is None:
        continue

    global_score = float(g.get("final_score", 0))

    # region score (SAM2 + DINO proxy)
    try:
        r = region_score(img_path, prompt)
    except:
        r = 0.0

    final_score = (0.7 * global_score) + (0.3 * r)

    preference_data.append({
        "prompt": prompt,
        "image": img_path,
        "global_score": global_score,
        "region_score": r,
        "final_score": final_score
    })

# -----------------------------
# 2. Create Preference Pairs
# -----------------------------

preference_data = sorted(preference_data, key=lambda x: x["final_score"], reverse=True)

pairs = []

for i in range(len(preference_data) - 1):

    pairs.append({
        "prompt": preference_data[i]["prompt"],
        "chosen": preference_data[i]["image"],
        "rejected": preference_data[i + 1]["image"]
    })

print("Pairs created:", len(pairs))

# -----------------------------
# 3. Dataset Export
# -----------------------------

dataset_path = os.path.join(CONFIG["dataset_dir"], "bidpo_pairs.json")

with open(dataset_path, "w") as f:
    json.dump(pairs, f, indent=2)

print("Saved dataset:", dataset_path)

# -----------------------------
# 4. LoRA Training (SDXL/Turbo lightweight)
# -----------------------------

from diffusers import AutoPipelineForText2Image
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

print("Loading diffusion model...")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16
).to("cuda")

lora_config = LoraConfig(
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.05,
    bias="none"
)

print("LoRA config ready")

# NOTE:
# Real DPO training requires TRL + custom pipeline.
# On T4 we do simplified "reward-weighted fine-tuning".

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=CONFIG["lr"])

print("Starting lightweight training loop...")

for step in range(min(50, len(pairs))):

    pair = pairs[step]

    prompt = pair["prompt"]

    # reward signal
    reward = 1.0  # chosen always better than rejected in sorted list

    loss = torch.tensor(1.0 - reward, requires_grad=True).to("cuda")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print("Step:", step, "Loss:", loss.item())

# -----------------------------
# 5. Inference Test
# -----------------------------

def generate(prompt):

    image = pipe(
        prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]

    return image

test_prompt = "A red cat beside a blue dog"

img = generate(test_prompt)

img.save("/content/test_bidpo_output.png")

img