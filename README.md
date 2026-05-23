# ComfyUI Easy SongGeneration

`ComfyUI-Easy-SongGeneration` 是一个面向 ComfyUI 的 SongGeneration 推理插件。它把仓库内精简后的 `songgeneration` 推理代码封装为 ComfyUI 自定义节点，可以在工作流中加载并推理 SongGeneration Checkpoint。

本插件侧重本地推理与 ComfyUI 工作流集成：去除了上游项目中训练、评估、微调和 Gradio GUI 相关内容，保留核心推理链路，并加入模型缓存、显存释放、分段加载、量化缓存、进度条和中文节点翻译。

## 运行需求

- ComfyUI。
- 支持 PyTorch 的 CUDA 或 HIP/ROCm GPU。
  - CPU 推理不受支持。
  - ROCm/HIP 环境需要安装 ROCm 版 PyTorch；在 PyTorch 中通常仍通过 `torch.cuda` 接口暴露设备。
- 可用显存取决于所选 SongGeneration 模型、生成时长、精度和是否启用量化。
- 已安装可用的 `torch`、`torchaudio`、`transformers`、`einops`、`pydantic`、`PyYAML`、`safetensors`、`tqdm` 等 ComfyUI 常见依赖。
- 如果要使用 Flash Attention，需要环境和 GPU 支持对应实现；不支持时请在加载节点中关闭 `Flash Attention`。

## 安装

进入 ComfyUI 的自定义节点目录，直接克隆本仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eastmoe/ComfyUI-Easy-SongGeneration.git
```

然后重启 ComfyUI。

本仓库已经把多个上游运行时依赖替换为本地轻量实现，`songgeneration/requirements.txt` 中主要保留依赖说明。不需要额外安装大量上游依赖；如果你的 ComfyUI 环境缺少基础依赖，请按 ComfyUI 当前环境的 PyTorch/CUDA/HIP 版本补齐。

## 模型放置

模型文件放在：

```text
ComfyUI/models/SongGeneration/
```

可以直接在 ComfyUI 里使用 `Easy SongGeneration - 下载模型` 节点自动下载。节点会从 `eastmoe/SongGeneration` 下载到上面的目录，其中 `common` 和 `third_party` 会始终下载，`SongGeneration-*` 模型目录按节点里的 `模型目录` 选择下载。下载源可选 `hf-mirror.com` 或 `huggingface.co`，默认不会覆盖大小一致的本地文件。

推荐结构：

```text
ComfyUI/models/SongGeneration/
  SongGeneration-v2-large/
    config.yaml
    model.pt
  common/
    model_1rvq/
      model_2_fixed.safetensors
    model_septoken/
      model_2.safetensors
    vae/
      stable_audio_1920_vae.json
      autoencoder_music_1320k.ckpt
  third_party/
    demucs/
      ckpt/
        htdemucs.pth
        htdemucs.yaml
```

加载节点会扫描 `ComfyUI/models/SongGeneration/` 下包含 `config.yaml` 和 `model.pt` 的子目录，因此可以同时保存多个模型：

```text
ComfyUI/models/SongGeneration/
  songgeneration_base/
    config.yaml
    model.pt
  songgeneration_large/
    config.yaml
    model.pt
  songgeneration_v2_large/
    config.yaml
    model.pt
```

运行时资源搜索顺序：

1. 当前选择的模型目录。
2. `ComfyUI/models/SongGeneration`。
3. 插件内置的 `songgeneration` 目录。

如果模型的 `config.yaml` 中引用了 `common/...` 或旧版 `ckpt/...` 一类相对路径，请把对应目录放到模型目录旁边，或放到 `ComfyUI/models/SongGeneration` 下。节点会兼容 `common` 与 `ckpt` 两种目录名：优先使用配置中写明的路径，找不到时会在两者之间自动 fallback。

使用外部音频作为参考音频时，需要 Demucs 文件：

```text
ComfyUI/models/SongGeneration/third_party/demucs/ckpt/htdemucs.pth
ComfyUI/models/SongGeneration/third_party/demucs/ckpt/htdemucs.yaml
```

自动参考风格会默认使用插件内置的：

```text
songgeneration/tools/new_auto_prompt.pt
```

也可以在模型根目录提供同名运行时资源。

### 自动下载模型

添加 `Easy SongGeneration - 下载模型` 节点后运行即可。常用选项：

- `下载源`：中国大陆网络通常可先选 `hf-mirror.com`，也可以切换为 `huggingface.co`。
- `模型目录`：选择要下载的模型目录；`common` 与 `third_party` 始终会一起下载。可选 `SongGeneration-v2-large`、`SongGeneration-base-full`、`SongGeneration-base-new`、`runtime-only` 或 `all`。
- `分支/版本`：默认 `main`。
- `覆盖已有文件`：关闭时会跳过大小一致的文件；下载中断后再次运行会继续使用 `.download` 临时文件续传。

下载完成后，在 `加载模型` 节点里刷新/重新打开模型下拉列表，即可选择刚下载的模型目录。

## 包含节点

节点分类默认为 `eastmoe/Comfy-Easy-SongGeneration`，以下是当前版本节点列表：

| 节点 | 作用 |
| --- | --- |
| `Easy SongGeneration - 下载模型` | 从 `eastmoe/SongGeneration` 自动下载 `common`、`third_party` 和选中的模型目录到 `ComfyUI/models/SongGeneration`。 |
| `Easy SongGeneration - 加载模型` | 从 `ComfyUI/models/SongGeneration` 加载模型，输出 `songgen_model` 和模型信息 JSON。 |
| `Easy SongGeneration - 释放模型` | 释放已加载模型，并可清理 CUDA/HIP 显存缓存。 |
| `Easy SongGeneration - 生成完整歌曲` | 生成包含人声和伴奏的混音歌曲，输出 `AUDIO` 和元数据。 |
| `Easy SongGeneration - 生成人声` | 生成人声轨，输出 `AUDIO` 和元数据。 |
| `Easy SongGeneration - 生成伴奏` | 生成伴奏或纯音乐，输出 `AUDIO` 和元数据。 |
| `Easy SongGeneration - 分轨生成` | 一次输出混音、人声、伴奏三路 `AUDIO`，并返回元数据。 |

加载模型节点的重要选项：

- `模型目录`：选择包含 `config.yaml` 和 `model.pt` 的模型目录。
- `版本`：可选 `auto`、`v1`、`v2`，`auto` 会根据模型目录名推断。
- `运行时根目录`：默认 `auto`，自动搜索模型目录、模型根目录和插件目录。
- `GPU ID`：`-1` 使用当前设备，也可以指定设备编号。
- `Flash Attention`：环境支持时可开启，否则关闭。
- `分段加载`：按模块加载和移动权重，降低加载阶段显存峰值。
- `量化格式`、`量化范围`、`重建量化缓存`：对 Linear 权重进行量化并缓存，缓存目录为 `ComfyUI/models/SongGeneration-cache`。
- `LLM 精度`、`Diffusion 精度`、`VAE 精度`：分别控制不同模块的计算精度。
- `重新加载`：忽略现有模型缓存，重新加载权重。

生成节点的重要输入：

- `歌词`：SongGeneration 段落格式歌词。
- `描述`：风格、情绪、乐器、人声等提示词，通常用逗号分隔。
- `种子`：`-1` 表示使用当前时间。
- `时长`：`0` 表示使用模型 `config.yaml` 中的 `max_dur`。
- `扩展步长`：长音频生成步长，通常保持默认值。
- `温度`、`CFG`、`Top K`、`Top P`、`采样`：采样与生成控制参数。
- `解码块大小`：Diffusion 解码时的 chunk size。
- `自动参考风格`：可选 `None`、`Pop`、`Latin`、`Rock`、`Electronic`、`Metal`、`Country`、`R&B/Soul`、`Ballad`、`Jazz`、`World`、`Hip-Hop`、`Funk`、`Soundtrack`、`Auto`。
- `参考音频`：可选 ComfyUI `AUDIO` 输入；连接后优先级高于自动参考风格。

歌词示例：

```text
[intro-short] ; [verse] Stars are waking in the rain. ; [chorus] We sing until the morning light. ; [outro-short]
```

## 对原项目的改造

- 初始化时引入上游 SongGeneration 代码。
- 移除训练、评估、微调相关代码和依赖，仅保留核心推理功能。
- 移除 Gradio/GUI 推理代码与相关依赖，转为保留命令行推理入口。
- 放宽部分依赖版本限制，减少与 ComfyUI 现有 Python 环境的冲突。
- 注释或移除 ComfyUI 已经提供的重复依赖。
- 用本地兼容实现替代多个上游依赖：
  - `alias-free-torch` 替换为 `songgeneration/alias_free_torch`。
  - `einops-exts` 替换为 `songgeneration/einops_exts.py`。
  - `julius` 替换为基于 `torchaudio.functional.resample` 的本地实现。
  - `k-diffusion` 替换为 `songgeneration/k_diffusion` 中推理所需子集。
  - `vector-quantize-pytorch` 替换为 Flow1dVAE 内本地 RVQ 实现。
  - `diffusers` 替换为 Flow1dVAE 内本地兼容子集。
  - `omegaconf` 替换为静态配置读取所需的本地兼容实现。
  - 移除未直接使用的 `openunmix`、`huggingface-hub`、`x-transformers`、`packaging`、`librosa`、`lameenc` 等依赖。
- 新增功能完整的推理 CLI，作为 可用于测试的本地推理入口。
- 新增 ComfyUI 节点实现，支持模型加载、模型释放、混音/人声/伴奏/分轨输出，并返回 ComfyUI 原生 `AUDIO`。
- 新增模型自动下载节点，支持从 `hf-mirror.com` 或 `huggingface.co` 下载 `eastmoe/SongGeneration` 中的必需运行时目录和选中的模型目录。
- 新增 Linear 权重量化、量化缓存、LLM/Diffusion/VAE 精度选择和分段加载，改善显存占用与加载体验。
- 新增 ComfyUI 右键菜单适配、中文节点翻译、加载与生成进度条，并支持 ComfyUI 中断回调。
- 将原单文件节点实现拆分为 `easy_songgeneration_nodes` 包，分离配置、运行时、模型加载、量化、进度和节点定义，便于维护。

## 目录结构

```text
ComfyUI-Easy-SongGeneration/
  __init__.py                         # ComfyUI 插件入口
  nodes.py                            # 兼容入口，导出节点映射
  README.md                           # 本说明文档
  local/
    zh-cn/
      nodes.json                      # ComfyUI 中文节点名称、提示和返回值翻译
  easy_songgeneration_nodes/
    __init__.py
    config.py                         # 节点常量、模型路径、量化/精度选项、翻译读取
    downloader.py                     # Hugging Face / hf-mirror 模型下载
    model.py                          # 模型加载、缓存、推理调度、显存释放
    nodes.py                          # ComfyUI 节点定义
    options.py                        # 生成参数数据结构
    progress.py                       # ComfyUI 进度条和中断桥接
    quantization.py                   # Linear 权重量化与缓存逻辑
    runtime.py                        # 路径解析、AUDIO 转换、临时音频写入
  songgeneration/
    README.md                         # 精简后 SongGeneration 推理代码说明
    generate.py                       # CLI/推理辅助入口
    inference.py                      # 推理相关代码
    comfy_progress.py                 # ComfyUI 进度 hook
    requirements.txt                  # 依赖变更说明
    requirements_nodeps.txt           # 已被本地替代的依赖说明
    tools/
      new_auto_prompt.pt              # 自动参考风格提示
    sample/                           # 示例歌词、描述和参考音频
    codeclm/                          # SongGeneration 核心模型代码
    conf/                             # 配置资源
    alias_free_torch/                 # 本地 alias-free-torch 兼容实现
    k_diffusion/                      # 本地 k-diffusion 推理子集
    img/                              # 原项目图片资源
  models/                             # 仓库内临时/本地模型目录，不建议提交大模型
  temp/                               # 临时文件目录
```

## 使用提示

- 推荐先使用 `加载模型` 节点加载一次模型，再把 `songgen_model` 输出连接到生成节点。
- 多个生成节点可以复用同一个 `songgen_model`，不必重复加载。
- 生成结束后如果需要释放显存，运行 `释放模型` 节点。
- 如果显存紧张，优先尝试开启 `分段加载`、降低精度或启用量化；如果音质或兼容性异常，先关闭 VAE 量化。
- 当 Flash Attention 报错或显卡不支持时，在加载节点中关闭 `Flash Attention`。
