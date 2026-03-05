# Supported Models

aha supports a growing collection of state-of-the-art AI models across multiple domains.

## Language Model

| Model | Parameters | Description | Use Case | License |
|-------|-----------|-------------|----------|---------|
| **Qwen3-0.6B** | 0.6B | Latest generation | Advanced reasoning | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM4-0.5B** | 0.5B | Efficient lightweight | Edge deployment | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Vision & Multimodal

| Model | Parameters | Description | License |
|-------|-----------|-------------|---------|
| **Qwen2.5-VL-3B** | 3B | Image understanding | [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) |
| **Qwen2.5-VL-7B** | 7B | Image understanding | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-2B** | 2B | Enhanced multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-4B** | 4B | Enhanced multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-8B** | 8B | Enhanced multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-32B** | 32B | Enhanced multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-0.8B** | 0.8B | Native Multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-2B** | 2B | Native Multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-4B** | 4B | Native Multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-9B** | 9B | Native Multimodal | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## OCR

| Model | Languages | Type | Strength | License |
|-------|-----------|------|----------|---------|
| **PaddleOCR-VL** | 80+ | Lightweight | General documents | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Hunyuan-OCR** | Chinese | Deep learning | Complex layouts | [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |
| **DeepSeek-OCR** | Multi | Scene text | Natural images | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |

## Speech Recognition (ASR)

| Model | Parameters | Language | Real-time | Speed | License |
|-------|-----------|----------|-----------|-------|---------|
| **Fun-ASR-Nano-2512** | 2G | Chinese/English | Yes | Fast | Not Specified |
| **GLM-ASR-Nano-2512** | 4.5G | Chinese/English | Yes | Fast | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR-0.6B** | 0.6B | Chinese/English | Yes | Fast | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-ASR-1.7B** | 1.7B | Chinese/English | Yes | Fast | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |


## Audio Generation

| Model | Parameters | Type | Description | License |
|-------|-----------|------|-------------|---------|
| **VoxCPM-0.5B** | 0.5B | Voice Codec | Neural audio codec | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **VoxCPM1.5** | - | Voice Codec | Enhanced voice generation | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## Image Processing

| Model | Type | Description | License |
|-------|------|-------------|---------|
| **RMBG-2.0** | Background Removal | Remove image backgrounds | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) |

## Model Sources

Models are sourced from:

- [Hugging Face](https://huggingface.co) - Primary model hub
- [ModelScope](https://modelscope.cn) - Chinese model hub

## Adding New Models

See [Development Guide](./development.md) for instructions on adding new model integrations.

## License Compliance

**Important**: Each model has its own license. Please review the model's license before use in production. Some key considerations:

- **Apache 2.0**: Permissive, commercial-friendly
- **MIT**: Permissive, commercial-friendly
- **Qwen Research License**: Research use, may have restrictions
- **Tencent Hunyuan Community License**: Custom license, review terms
- **CC BY-NC 4.0**: Non-commercial only

Always verify license terms before deployment in production environments.

## Model Updates

Models updated from time to time. 

## Performance Benchmarks

Approximate inference speeds on CPU (M1 Pro):

| Model | Task | Tokens/sec |
|-------|------|------------|
| Qwen3-0.6B | Text | 40-50 |
| Qwen2.5-VL-3B | Vision | 20-30 |
| Qwen3-ASR-0.6B | ASR | 200-500x |
| VoxCPM-0.5B | TTS | Real-time |

*Benchmarks vary by hardware and input size.*
