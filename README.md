# LLaMA-Adapter-V2 Multi-modal

## Notes
Copy of OpenGVLab's LLaMA-Adapter-V2 for screenshot understanding.

## Setup

* setup up a new conda env and install necessary packages.
  ```bash
  conda create -n llama_adapter_v2 python=3.8 -y
  conda activate llama_adapter_v2
  pip install -r requirements.txt
  ```

* Obtain the LLaMA backbone weights using [this form](https://forms.gle/jk851eBVbX1m5TAv5). Please note that checkpoints from unofficial sources (e.g., BitTorrent) may contain malicious code and should be used with care. Organize the downloaded file in the following structure
  ```
  /path/to/llama_model_weights
  ├── 7B
  │   ├── checklist.chk
  │   ├── consolidated.00.pth
  │   └── params.json
  └── tokenizer.model
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

## Evaluation
Check [eval.md](./docs/eval.md) for details.

## Online demo

We provide an online demo at [OpenGVLab](http://llama-adapter.opengvlab.com).

You can also start it locally with:
```bash
python gradio_app.py
```

## Models

You can check our models by running:
```python
import llama
print(llama.available_models())
```

Now we provide `BIAS-7B` which fine-tunes the `bias` and `norm` parameters of LLaMA, and `LORA-BIAS-7B` which fine-tunes the `bias`, `norm` and `lora` parameters of LLaMA. We will include more pretrained models in the future, such as the LoRA fine-tuning model `LORA-7B` and partial-tuning model `PARTIAL-7B`.

## Pre-traininig & Fine-tuning
See [train.md](docs/train.md)
