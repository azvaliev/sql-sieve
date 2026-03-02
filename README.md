# Sieve Training

Fine-tuning [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) for SQL WHERE clause generation, used by the Planorix sieve feature.

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
SELECT * FROM product
```

**Output:**
```sql
WHERE category LIKE '%electronics%' AND price < 50
```

## Training

- **Method**: LoRA (r=16, alpha=32, dropout=0.05) via TRL's SFTTrainer
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Dataset**: 500 examples (completion-only loss)
- **Epochs**: 3
- **Hardware**: H100 (via Modal)
- **Precision**: bf16

## Links

- [Collection](https://huggingface.co/collections/azatvaliev/sieve)
- [Model Card](https://huggingface.co/azatvaliev/sieve-llama-3.2-1b)
- [GGUF Quantizations](https://huggingface.co/azatvaliev/sieve-llama-3.2-1b-GGUF)

## Usage

```bash
# Train
modal run main.py

# Push to HF
modal run push.py --volume-name sieve-checkpoints
```
