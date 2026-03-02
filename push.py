import modal

app = modal.App("push-to-hub")

push_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.30.0")
)

HF_USER = "azatvaliev"
MODEL_NAME = "sieve-llama-3.2-1b"
BASE_MODEL = "meta-llama/Llama-3.2-1B"

@app.local_entrypoint()
def cli(volume_name: str):
    push.remote(volume_name)

@app.function(
    image=push_image,
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("DANGER_huggingface-WRITE-secret")],
)
def push(volume_name: str):
    import os
    from huggingface_hub import HfApi, ModelCard, ModelCardData, create_collection, add_collection_item

    vol = modal.Volume.from_name(volume_name)
    mount_path = "/checkpoints"
    os.makedirs(mount_path, exist_ok=True)

    # Only download merged model, adapter, and GGUF files (skip epoch checkpoints etc.)
    for entry in vol.listdir("/", recursive=True):
        is_relevant = (
            entry.path.startswith("merged/")
            or entry.path.startswith("adapter/")
            or entry.path.endswith(".gguf")
        )
        if not is_relevant:
            continue
        local_path = os.path.join(mount_path, entry.path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)

    api = HfApi()

    # Push merged HF model + adapter
    merged_path = os.path.join(mount_path, "merged")
    adapter_path = os.path.join(mount_path, "adapter")

    if os.path.isdir(merged_path):
        repo_id = f"{HF_USER}/{MODEL_NAME}"
        print(f"Pushing merged model to {repo_id}")
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=merged_path, repo_id=repo_id)

        if os.path.isdir(adapter_path):
            print(f"Pushing adapter to {repo_id}/adapter")
            api.upload_folder(
                folder_path=adapter_path,
                repo_id=repo_id,
                path_in_repo="adapter",
            )

        card = ModelCard(f"""---
{ModelCardData(
    base_model=BASE_MODEL,
    library_name="transformers",
    pipeline_tag="text-generation",
    license="llama3.2",
    language=["en"],
    tags=["llama", "llama-3", "fine-tune", "lora", "sql"],
).to_yaml()}
---

# {MODEL_NAME}

Fine-tune of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for SQL WHERE clause generation.

GGUF quantizations are available at [{HF_USER}/{MODEL_NAME}-GGUF](https://huggingface.co/{HF_USER}/{MODEL_NAME}-GGUF).

## Overview

This is a **completion model** (not instruct/chat). Given a PostgreSQL schema (DDL), a natural language filter as a `-- filter:` comment, and a `SELECT * FROM table ` prefix, the model completes with the appropriate `WHERE` clause.

### Example

**Input:**
```sql
CREATE TABLE "public"."product" (
  "id" bigint PRIMARY KEY,
  "name" text NOT NULL,
  "category" text NOT NULL,
  "price" numeric(10,2) NOT NULL,
  "in_stock" boolean NOT NULL DEFAULT true
);
-- filter: electronics under $50
SELECT * FROM product\x20
```

**Output:**
```sql
WHERE category LIKE '%electronics%' AND price < 50
```

## Training

- **Method**: LoRA (r=16, alpha=32, dropout=0.10) via TRL's SFTTrainer
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dataset**: 500 examples (completion-only loss)
- **Epochs**: 3
- **Hardware**: H100
- **Precision**: bf16

The `adapter/` directory contains the LoRA adapter weights.
""")
        card.push_to_hub(repo_id)

    # Push GGUF files
    gguf_files = [f for f in os.listdir(mount_path) if f.endswith(".gguf")]
    if gguf_files:
        gguf_repo = f"{HF_USER}/{MODEL_NAME}-GGUF"
        print(f"Pushing {len(gguf_files)} GGUF files to {gguf_repo}")
        api.create_repo(gguf_repo, exist_ok=True)

        for gguf in sorted(gguf_files):
            gguf_path = os.path.join(mount_path, gguf)
            size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
            print(f"  Uploading {gguf} ({size_mb:.0f} MB)")
            api.upload_file(
                path_or_fileobj=gguf_path,
                path_in_repo=gguf,
                repo_id=gguf_repo,
            )

        model_repo = f"{HF_USER}/{MODEL_NAME}"

        # Build quant table dynamically from files (exclude F16)
        recommendations = {
            "q4_k_m": "Best speed, still retains high quality",
            "q6_k": "Balanced, but not recommended",
            "q8_0": "Best quality, no perceptible loss from F16",
        }
        table_rows = []
        for f in sorted(gguf_files):
            quant = f.replace("model-", "").replace(".gguf", "")
            if quant == "f16":
                continue
            size_gb = os.path.getsize(os.path.join(mount_path, f)) / (1024 ** 3)
            rec = recommendations.get(quant, "")
            table_rows.append(f"| {f} | {quant.upper()} | {size_gb:.1f} GB | {rec} |")
        quant_table = "\n".join(table_rows)

        card = ModelCard(f"""---
{ModelCardData(
    base_model=model_repo,
    library_name="gguf",
    pipeline_tag="text-generation",
    license="llama3.2",
    language=["en"],
    tags=["llama", "llama-3", "gguf", "quantized", "sql"],
).to_yaml()}
---

# {MODEL_NAME}-GGUF

GGUF quantizations of [{model_repo}](https://huggingface.co/{model_repo}), a fine-tune of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for SQL WHERE clause generation.

See the [model card](https://huggingface.co/{model_repo}) for usage details and input/output format.

## Quantizations

| File | Quant | Size | Recommendation |
|------|-------|------|----------------|
{quant_table}

## Usage

```bash
# Download
huggingface-cli download {HF_USER}/{MODEL_NAME}-GGUF model-q4_k_m.gguf

# Run
llama-server --model model-q4_k_m.gguf -c 4096 -ngl 99 --port 8080
```
""")
        card.push_to_hub(gguf_repo)

    # Create/update collection
    print("Setting up collection")
    collection = create_collection(
        title="Sieve",
        description="SQL WHERE clause generation",
        exists_ok=True,
    )
    repo_ids = [f"{HF_USER}/{MODEL_NAME}", f"{HF_USER}/{MODEL_NAME}-GGUF"]
    for repo_id in repo_ids:
        add_collection_item(
            collection.slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )

    print("Done!")
