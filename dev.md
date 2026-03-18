# Dev Bundle
> Skills: Modal · PyTorch Lightning · Transformers · fastapi-mcp
> Usage: Paste this URL at session start for ML/dev/deployment tasks.

---

## 1. Modal — Serverless Cloud Compute
**Trigger:** "run on GPU", "cloud deployment", "serverless ML", "batch processing", "scale containers"
**Requires:** `uv pip install modal` · `modal token new` · Free $30/mo credits

```python
import modal

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "numpy")
)
app = modal.App("my-app", image=image)

# GPU function
@app.function(gpu="L40S", timeout=3600)
def train():
    import torch
    assert torch.cuda.is_available()
    # ... training code

# Parallel batch processing
@app.function(cpu=2.0, memory=4096)
def process(item: str): return item.upper()

@app.local_entrypoint()
def main():
    results = list(process.map(["a", "b", "c"]))  # auto-parallelized

# Persistent storage
vol = modal.Volume.from_name("my-vol", create_if_missing=True)
@app.function(volumes={"/data": vol})
def save(text):
    open("/data/out.txt", "w").write(text)
    vol.commit()

# Web endpoint
@app.function()
@modal.web_endpoint(method="POST")
def predict(data: dict): return {"result": data["input"].upper()}

# Scheduled job
@app.function(schedule=modal.Cron("0 2 * * *"))
def daily(): pass
```

**GPU tiers:** T4/L4 (cheap inference) · A10/A100 (training) · L40S (best cost/perf, 48GB) · H100/H200 (max perf)
**Multi-GPU:** `gpu="H100:8"` · **Autoscale:** `max_containers=100, min_containers=2`
**Run:** `modal run script.py` · **Deploy:** `modal deploy script.py`

---

## 2. PyTorch Lightning
**Trigger:** "train neural network", "LightningModule", "multi-GPU training", "DDP", "FSDP"
**Requires:** `uv pip install torch pytorch-lightning`

```python
import lightning as L
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader

class MyModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Linear(28*28, 10)

    def training_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self.net(x.flatten(1)), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        loss = F.cross_entropy(self.net(x.flatten(1)), y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# Train
trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu", devices=2,     # multi-GPU
    strategy="ddp",                    # DDP (<500M params) or "fsdp" (larger)
    precision="16-mixed",              # auto mixed precision
    callbacks=[
        L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss"),
        L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=3),
    ]
)
trainer.fit(MyModel(), train_loader, val_loader)
```

**Strategy guide:** DDP → models <500M · FSDP → 500M+ · DeepSpeed → cutting edge control
**Debug:** `Trainer(fast_dev_run=True)` runs 1 batch · use `self.device` not `.cuda()` · `seed_everything(42)` for reproducibility

---

## 3. Transformers (HuggingFace)
**Trigger:** "NLP", "LLM", "text generation", "classification", "fine-tune", "embeddings", "vision model"
**Requires:** `uv pip install torch transformers datasets accelerate`

```python
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ── Quick inference (pipeline API) ──
gen  = pipeline("text-generation",     model="gpt2")
cls  = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
qa   = pipeline("question-answering")
summ = pipeline("summarization",       model="facebook/bart-large-cnn")
img  = pipeline("image-classification",model="google/vit-base-patch16-224")

result = gen("The future of AI is", max_new_tokens=50, temperature=0.7)

# ── Custom model loading ──
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",        # auto GPU placement
    torch_dtype="auto",       # bfloat16 if supported
    load_in_4bit=True,        # quantization
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# ── Fine-tuning (Trainer API) ──
from transformers import Trainer, TrainingArguments
args = TrainingArguments(
    output_dir="./out", num_train_epochs=3,
    per_device_train_batch_size=8, eval_strategy="epoch",
    fp16=True, report_to="wandb"
)
trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

**Auth:** `export HUGGINGFACE_TOKEN="hf_..."` or `huggingface-cli login`
**Key model classes:** `AutoModelForCausalLM` · `AutoModelForSequenceClassification` · `AutoModelForTokenClassification` · `AutoModelForQuestionAnswering`

---

## 4. fastapi-mcp — FastAPI → MCP Auto-converter
**Trigger:** "FastAPI as MCP", "expose API to Claude", "MCP from existing API"
**Requires:** `pip install fastapi-mcp` · Python 3.10+

```python
from fastapi import FastAPI, Depends
from fastapi_mcp import FastApiMCP, AuthConfig

app = FastAPI()

# ── Basic setup (ASGI transport — no HTTP needed) ──
mcp = FastApiMCP(app)
mcp.mount_http()   # recommended · mounts at /mcp
# mcp.mount_sse()  # legacy SSE at /sse

# ── Custom HTTP client (remote API) ──
import httpx
mcp = FastApiMCP(app, http_client=httpx.AsyncClient(
    base_url="https://api.example.com", timeout=30.0
))

# ── Auth: token passthrough ──
mcp = FastApiMCP(app, auth_config=AuthConfig(
    dependencies=[Depends(verify_token)]
))

# ── Auth: full OAuth2 ──
mcp = FastApiMCP(app, auth_config=AuthConfig(
    issuer="https://auth.example.com/",
    authorize_url="https://auth.example.com/authorize",
    oauth_metadata_url="https://auth.example.com/.well-known/oauth-authorization-server",
    client_id="my-client-id", client_secret="my-client-secret",
    audience="my-audience", setup_proxies=True,
    dependencies=[Depends(verify_auth)]
))
mcp.mount_http()

# ── Separate MCP app from API app ──
api_app = FastAPI()   # your endpoints
mcp_app = FastAPI()   # dedicated MCP server
mcp = FastApiMCP(api_app)
mcp.mount_http(mcp_app)
# uvicorn main:api_app --port 8001 & uvicorn main:mcp_app --port 8000

# ── Add endpoints after creation ──
@app.get("/new/endpoint/", operation_id="new_endpoint")
async def new_endpoint(): return {"ok": True}
mcp.setup_server()  # refresh to include new routes
```

**MCP client config:**
```json
{ "mcpServers": { "my-api": { "url": "http://localhost:8000/mcp" } } }
```
**OAuth with mcp-remote:** use fixed port `8080` → callback URL = `http://127.0.0.1:8080/oauth/callback`
