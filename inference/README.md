# ⚡ Lightweight Inference Code for SenseNova-MARS

Tested on 4x H100 GPUs (2 for SenseNova-MARS-32B, 2 for Qwen/Qwen3-32B summarizer) with Azure OpenAI as the judge client.

## Setup

```bash
docker pull yxchng/sensenova-mars:v0.1
```

## Configuration

Fill in the following placeholders in `eval.sh` before running:
- `SERPER_API_KEY` - Your Serper API key for web search
- For `--judge-client azure`:
  - `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key
  - `AZURE_OPENAI_BASE_URL` - Your Azure OpenAI endpoint URL
  - `AZURE_API_VERSION` - Azure API version
- For `--judge-client openai`:
  - `OPENAI_API_KEY` - Your OpenAI API key

## Run

**Step 1: Start the model server**

```bash
bash serve_model.sh
```

**Step 2: Start the summarizer server**

```bash
bash serve_summarizer.sh
```

**Step 3: Run evaluation**

```bash
bash eval.sh
```

## Output Files

Results are saved to the output directory in multiple formats:

| File | Description |
|------|-------------|
| `*.jsonl` | Raw results in JSON Lines format for programmatic analysis |
| `*.html` | Same data as JSONL but formatted for easy viewing with embedded images |
| `images/` | Cropped images and search result images generated during inference |
| `summary.json` | Aggregated metrics including accuracy and per-benchmark breakdowns |
