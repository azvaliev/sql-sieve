import modal

app = modal.App("llama-3.2-1b-finetune")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
      "git clone --depth=1 https://github.com/ggml-org/llama.cpp /llama.cpp",
      "pip install -r /llama.cpp/requirements/requirements-convert_hf_to_gguf.txt",
      "cmake -B /llama.cpp/build /llama.cpp",
      "cmake --build /llama.cpp/build --config Release -j$(nproc) --target llama-quantize",
    )
    .uv_pip_install(
        "accelerate==1.12.0",
        "datasets==4.6.1",
        "hf-transfer==0.1.9",
        "huggingface_hub==1.5.0",
        "peft==0.18.1",
        "torch==2.7.0",
        "transformers==5.2.0",
        "trl==0.29.0",
        "wandb==0.25.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"HF_HOME": "/model_cache"})
    .add_local_file("./data/sieve-training.v1.jsonl", remote_path="/data/sieve-training.v1.jsonl")
)

with train_image.imports():
    import datasets
    import wandb
    import torch
    import subprocess
    import transformers
    import warnings
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel

model_cache_volume = modal.Volume.from_name(
    "sieve-model-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "sieve-checkpoints", create_if_missing=True
)

# Modal constants
GPU_TYPE = "H100"
TIMEOUT_HOURS = 1
MAX_RETRIES = 2

# Training constants
max_seq_length = 4096
base_model = "meta-llama/Llama-3.2-1B"

@app.local_entrypoint()
def cli():
    main.remote()

@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        # Auto set HF_TOKEN to pull llama
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def main():
    warnings.filterwarnings("ignore", message="Mismatch between tokenized prompt")
    model, tokenizer = load_model()
    dataset = prepare_dataset()

    wandb.init(project='sieve',entity='azatvaliev')

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        processing_class = tokenizer,
        args = SFTConfig(
            output_dir="/checkpoints",
            report_to="wandb",
            # this seemed to deliver the best gains and not overfit too much
            # eval dataset was doing well
            num_train_epochs=3,
            per_device_train_batch_size=8, # h100 baby lfg
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="epoch",
            warmup_ratio=0.1,
            max_length = max_seq_length,
            bf16=True, #h100 baby lfg
            completion_only_loss=True #just focused on completions
        )
    )
    trainer.train()

    save_model(model, tokenizer)

    checkpoint_volume.commit()

def save_model(model, tokenizer):
    # save the LoRA adapter
    model.save_pretrained("/checkpoints/adapter") 

    # save the full merged model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("/checkpoints/merged")
    tokenizer.save_pretrained("/checkpoints/merged")

    # Convert HF model → GGUF F16
    subprocess.run([
        "python", "/llama.cpp/convert_hf_to_gguf.py",
        "/checkpoints/merged",
        "--outtype", "f16",
        "--outfile", "/checkpoints/model-f16.gguf",
    ], check=True)

    # Quantize F16 → Q8, Q6, Q4
    for quant in ["q8_0", "q6_k", "q4_k_m"]:
        subprocess.run([
            "/llama.cpp/build/bin/llama-quantize",
            "/checkpoints/model-f16.gguf",
            f"/checkpoints/model-{quant}.gguf",
            quant,
        ], check=True)

def load_model(): 
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = base_model,
    )

    model = get_peft_model(
        model = model,
        peft_config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            r=16,
            # Attention & MLP layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
    )

    model_cache_volume.commit()

    return model, tokenizer

def prepare_dataset():
    dataset = datasets.load_dataset(
        path="json",
        data_files="/data/sieve-training.v1.jsonl"
    )["train"]
    dataset = dataset.select_columns(["prompt","completion"])

    return dataset
