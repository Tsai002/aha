# 支持的模型

aha 支持多个领域的最先进 AI 模型集合。

## 文本生成

| 模型 | 参数量 | 描述 | 使用场景 | 开源协议 |
|------|--------|------|----------|---------|
| **Qwen3-0.6B** | 0.6B | 最新一代 | 高级推理 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM4-0.5B** | 0.5B | 高效轻量级 | 边缘部署 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 视觉与多模态

| 模型 | 参数量 | 描述 | 开源协议 |
|------|--------|------|--------|---------|
| **Qwen2.5-VL-3B** | 3B | 图像理解 | [Qwen 研究许可协议](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) |
| **Qwen2.5-VL-7B** | 7B | 图像理解 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-2B** | 2B | 增强多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-4B** | 4B | 增强多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-8B** | 8B | 增强多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL-32B** | 32B | 增强多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-0.8B** | 0.8B | 原生多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-2B** | 2B | 原生多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-4B** | 4B | 原生多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5-9B** | 9B | 原生多模态 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## OCR

| 模型 | 语言 | 类型 | 优势 | 开源协议 |
|------|------|------|------|---------|
| **PaddleOCR-VL** | 80+ | 轻量级 | 通用文档 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Hunyuan-OCR** | 中文 | 深度学习 | 复杂布局 | [腾讯混元社区许可协议](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |
| **DeepSeek-OCR** | 多语言 | 场景文字 | 自然图像 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |

## 语音识别 (ASR)

| 模型 | 参数量 | 语言 | 实时 | 速度 | 开源协议 |
|------|--------|------|------|------|---------|
| **Fun-ASR-Nano-2512** | 2512M | 中/英 | 是 | 快速 | 未标明 |
| **GLM-ASR-Nano-2512** | 2512M | 中/英 | 是 | 快速 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR-0.6B** | 0.6B | 中/英 | 是 | 快速 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-ASR-1.7B** | 1.7B | 中/英 | 是 | 快速 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |


## 语音生成

| 模型 | 参数量 | 类型 | 描述 | 开源协议 |
|------|--------|------|------|---------|
| **VoxCPM-0.5B** | 0.5B | 语音编解码器 | 神经音频编解码 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **VoxCPM1.5** | - | 语音编解码器 | 增强语音生成 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 图像处理

| 模型 | 类型 | 描述 | 开源协议 |
|------|------|------|---------|
| **RMBG-2.0** | 背景移除 | 移除图像背景 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans) |

## 模型来源

模型来源：

- [Hugging Face](https://huggingface.co) - 主模型中心
- [ModelScope](https://modelscope.cn) - 中文模型中心

## 添加新模型

参见 [开发指南](./development.zh-CN.md) 了解添加新模型集成的说明。

## 许可证合规

**重要提示**：每个模型都有自己的许可证。在生产环境使用前请查看模型许可证。一些关键注意事项：

- **Apache 2.0**: 宽松许可，支持商业使用
- **MIT**: 宽松许可，支持商业使用
- **Qwen 研究许可协议**: 研究用途，可能有使用限制
- **腾讯混元社区许可协议**: 自定义许可，请查看条款
- **CC BY-NC 4.0**: 仅限非商业用途

在生产环境部署前，请务必验证许可证条款。

## 模型更新

模型不定期更新。

## 性能基准

CPU (M1 Pro) 上的近似推理速度：

| 模型 | 任务 | Tokens/秒 |
|------|------|-----------|
| Qwen3-0.6B | 文本 | 40-50 |
| Qwen2.5-VL-3B | 视觉 | 20-30 |
| Qwen3-ASR-0.6B | ASR | 200-500x |
| VoxCPM-0.5B | TTS | 实时 |

*基准测试因硬件和输入大小而异。*
