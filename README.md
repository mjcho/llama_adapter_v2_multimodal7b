# LLaMA-Adapter-V2 Multi-modal

*An adaptation of OpenGVLab's LLaMA-Adapter-V2 for screenshot understanding.*

## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n llama_adapter_v2 python=3.8 -y
  conda activate llama_adapter_v2
  pip install -r requirements.txt
  ```

## Inference

Here is a simple inference script for LLaMA-Adapter V2. The pre-trained model will be downloaded directly from [Github Release](https://github.com/OpenGVLab/LLaMA-Adapter/releases/tag/v.2.0.0).

```python
import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/path/to/LLaMA/" 

# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="llama-2-7b", device=device) # llama_type is concatenated after llama_dir, e.g., "/path/to/LLaMA/llama-2-7b"

prompt = llama.format_prompt("Please introduce this painting.")
img = Image.fromarray(cv2.imread("../docs/logo_v1.png"))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]

print(result)
```
