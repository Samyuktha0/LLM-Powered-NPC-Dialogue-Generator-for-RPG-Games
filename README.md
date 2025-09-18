# 🧠 NPC Dialogue Generator

Fine-tuned GPT-2 model to generate fantasy NPC dialogues. Exposed via FastAPI for Unity integration.

## Features
- GPT-2 fine-tuning on RPG datasets
- FastAPI endpoint for real-time generation
- Caching and quantization for mobile optimization

## Usage
```bash
GET /generate?prompt=Hello traveler

---

## 🎨 2. Diffusion-Based Art Prompt Generator

### 📁 Folder Structure

### ✅ Key Files

**`model/generate_prompt.py`**
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

def generate(prompt):
    image = pipe(prompt).images[0]
    image.save("output.png")
    return "Prompt generated and image saved."
