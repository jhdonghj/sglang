# MinerU on SGLang

这是一份简化使用手册，说明如何用当前仓库原生运行 MinerU diffusion OCR 模型。

## 1. 环境

进入 SGLang 仓库后，确保：

```bash
export PATH=<your_venv>/bin:$PATH
export PYTHONPATH=python
```

如果需要新建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e "python[all]"
```

## 2. 模型目录要求

模型目录至少需要包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `added_tokens.json`
- processor 配置文件
- safetensors 权重文件

下面用 `<MODEL_PATH>` 表示模型目录。

## 3. 表格结构 Token 要求

MinerU 的表格输出会使用这些结构 token：

- `<ched>`
- `<ecel>`
- `<fcel>`
- `<lcel>`
- `<ucel>`
- `<xcel>`
- `<nl>`

这些 token 需要保留在输出里，不能被默认过滤掉。

模型目录里的 tokenizer 配置需要满足：

- `tokenizer_config.json` 中，这些 token 不应出现在 `additional_special_tokens`
- `special_tokens_map.json` 中，这些 token 不应出现在 `additional_special_tokens`
- `tokenizer_config.json -> added_tokens_decoder` 中，这些 token 的 `special` 应为 `false`
- `tokenizer.json -> added_tokens` 中，这些 token 的 `special` 应为 `false`

视觉相关控制 token 仍然应该保留为 special token，例如：

- `<|image_pad|>`
- `<|vision_start|>`
- `<|vision_end|>`
- `<|MASK|>`

## 4. 启动服务

推荐启动命令：

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path <MODEL_PATH> \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 1 \
  --dllm-algorithm LowConfidence \
  --mem-fraction-static 0.72 \
  --cuda-graph-max-bs 160
```

如果只想保守跑通，也可以关闭 cuda graph：

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 sglang serve \
  --model-path <MODEL_PATH> \
  --host 127.0.0.1 \
  --port 31000 \
  --tp-size 1 \
  --dllm-algorithm LowConfidence \
  --disable-cuda-graph \
  --attention-backend triton \
  --sampling-backend pytorch
```

## 5. 请求样例

下面用：

- `<BASE_URL>` 表示服务地址，例如 `http://127.0.0.1:31000`
- `<MODEL_PATH>` 表示模型目录
- `<IMAGE_PATH>` 表示图片路径

### 5.1 公式识别

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Formula Recognition:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 128,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode("utf-8"))
PY
```

### 5.2 表格识别

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Table Recognition:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 1024,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.read().decode("utf-8"))
PY
```

如果 tokenizer 配置正确，表格结果中会保留 `<fcel>`、`<nl>` 等结构 token。

### 5.3 布局分析

```bash
python - <<'PY'
import base64, json, pathlib, urllib.request

base_url = "<BASE_URL>/v1/chat/completions"
model = "<MODEL_PATH>"
img_path = pathlib.Path("<IMAGE_PATH>")
img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Layout Analysis:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 1024,
}

req = urllib.request.Request(
    base_url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    print(resp.read().decode("utf-8"))
PY
```

## 6. HF 对照脚本

如果你有一个 HF 对照脚本，可以用同一张图、同一类 prompt 做结果比对。

通用形式：

```bash
python <HF_DEMO_SCRIPT> \
  --model-path <MODEL_PATH> \
  --image-path <IMAGE_PATH> \
  --prompt-type table
```

## 7. 当前推荐配置

单卡、正确性优先时，推荐：

```bash
--tp-size 1
--dllm-algorithm LowConfidence
--mem-fraction-static 0.72
--cuda-graph-max-bs 160
```
