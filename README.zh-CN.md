<p align="center">
  <img src="assets/img/logo.png" alt="aha logo" width="120"/>
</p>

<p align="center">
  <a href="https://github.com/jhqxxx/aha/stargazers">
    <img src="https://img.shields.io/github/stars/jhqxxx/aha" alt="GitHub Stars">
  </a>
  <a href="https://github.com/jhqxxx/aha/issues">
    <img src="https://img.shields.io/github/issues/jhqxxx/aha" alt="GitHub Issues">
  </a>
  <a href="https://github.com/jhqxxx/aha/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/jhqxxx/aha" alt="GitHub License">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

# aha

**轻量 AI 推理引擎 —— 文本、视觉、语音与 OCR 一站式解决方案**

aha 是一款基于 Rust 和 Candle 框架构建的高性能跨平台 AI 推理引擎。将最先进的 AI 模型带到您的本地机器——无需 API 密钥，无需云依赖，纯粹、快速的 AI，直接在您的硬件上运行。

## 更新日志
### v0.2.1 (2026-03-05)
- 新增Qwen3.5 模型

### 2026-03-01
- 更新 interpolate.rs

### 2026-02-24
- 更新 candle 版本 0.9.2

### v0.2.0 (2026-02-05)
- 新增 Qwen3-ASR 语音识别模型

### v0.1.9 (2026-01-31)
- 新增 CLI `list` 子命令，显示支持的模型
- 新增 CLI 子命令结构支持（`cli`、`serv`、`download`、`run`）
- 修复 Qwen3VL thinking startswith bug
- 修复 `aha run` 多输入 bug

### v0.1.8 (2026-01-17)
- 新增 Qwen3 文本模型支持
- 新增 Fun-ASR-Nano-2512 语音识别模型
- 修复 ModelScope Fun-ASR-Nano 模型加载错误
- 使用 rubato 更新音频重采样

### v0.1.7 (2026-01-07)
- 新增 GLM-ASR-Nano-2512 语音识别模型
- 合并 Metal (GPU) 支持，适用于 Apple Silicon
- 新增动态主目录和模型下载脚本

**[查看完整更新日志](docs/changelog.zh-CN.md)** →

## 快速开始

### 安装

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
cargo build --release
```

**可选特性：**

```bash
# CUDA (NVIDIA GPU 加速)
cargo build --release --features cuda

# Metal (Apple GPU 加速，适用于 macOS)
cargo build --release --features metal

# Flash Attention (更快推理)
cargo build --release --features cuda,flash-attn

# FFmpeg (多媒体处理)
cargo build --release --features ffmpeg
```

### CLI 快速参考

```bash

# 列出所有支持的模型
aha list

# 仅下载模型
aha download -m qwen3asr-0.6b

# 下载模型并启动服务
aha -m qwen3asr-0.6b

# 直接运行推理（无需启动服务）
aha run -m qwen3asr-0.6b -i "audio.wav"

# 仅启动服务（模型已下载）
aha serv -m qwen3asr-0.6b -p 10100

```

### 对话

```bash
aha serv -m qwen3-0.6b -p 10100
```

然后使用统一(兼容 OpenAI)的 API：

```bash
curl http://localhost:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": false
  }'
```


### 支持的模型

| 类别 | 模型 |
|------|------|
| **文本** | Qwen3, MiniCPM4 |
| **视觉** | Qwen2.5-VL, Qwen3-VL |
| **OCR** | DeepSeek-OCR, Hunyuan-OCR, PaddleOCR-VL |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano,Qwen3-ASR |
| **音频** | VoxCPM, VoxCPM1.5 |
| **图像** | RMBG-2.0 (背景移除) |

## 文档

| 文档 | 描述 |
|------|------|
| [快速入门](docs/getting-started.zh-CN.md) | aha 入门指南 |
| [安装指南](docs/installation.zh-CN.md) | 详细安装说明 |
| [CLI 参考](docs/cli.zh-CN.md) | 命令行界面 |
| [API 文档](docs/api.zh-CN.md) | 库与 REST API |
| [支持的模型](docs/supported-models.zh-CN.md) | 可用的 AI 模型 |
| [核心概念](docs/concepts.zh-CN.md) | 架构与设计 |
| [开发指南](docs/development.zh-CN.md) | 贡献指南 |
| [更新日志](docs/changelog.zh-CN.md) | 版本历史 |

## 为什么选择 aha？
- **🚀 高性能推理** - 基于 Candle 框架，提供高效的张量计算和模型推理
- **🔧 统一接口** — 一个工具搞定文本、视觉、语音和 OCR
- **📦 本地优先** — 所有处理在本地运行，数据不离境
- **🎯 跨平台** — 支持 Linux、macOS 和 Windows
- **⚡ GPU 加速** — 可选 CUDA 支持以获得更快推理
- **🛡️ 内存安全** — Rust 构建，稳定可靠
- **🧠 注意力优化** - 可选 Flash Attention 支持，优化长序列处理

## 开发

### aha 作为库使用
> cargo add aha

```rust
# VoxCPM示例
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/openbmb/VoxCPM-0.5B/";
    
    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    
    let generate = voxcpm_generate.generate(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
        None,
        None,
        2,
        100,
        10,
        2.0,
        false,
        6.0,
    )?;

    let _ = save_wav(&generate, "voxcpm.wav")?;
    Ok(())
}
```


### 扩展新的模型 

- 在src/models/创建新模型文件
- 在src/models/mod.rs中导出
- 在src/exec/中添加支持cli运行模型推理
- 在tests/中添加测试和示例


## 特性

- 基于 Candle 框架的高性能推理
- 多模态模型支持（视觉、语言、语音）
- 简洁易用的 API 设计
- 最小化依赖，紧凑的二进制文件
- Flash Attention 支持长序列处理
- FFmpeg 支持多媒体处理

## 许可证

Apache-2.0 &mdash; 详见 [LICENSE](LICENSE)

## 致谢

- [Candle](https://github.com/huggingface/candle) - 优秀的 Rust 机器学习框架
- 所有模型作者和贡献者

---

<p align="center">
  <sub>由 aha 团队用 ❤️ 构建</sub>
</p>

<p align="center">
  <sub>我们持续扩展支持的模型列表，欢迎贡献！</sub>
</p>

<p align="center">
  <sub>如果这个项目对你有帮助，请给我们一个 ⭐ Star！</sub>
</p>
