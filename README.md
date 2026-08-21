# Efficiency Nodes for ComfyUI

这是一个面向 ComfyUI 的效率节点合集，在较少节点数量下完成模型加载、采样、LoRA/ControlNet 堆栈、XY Plot、高清修复和图片元数据保存等常用流程。

本仓库基于 [jags111/efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui) 扩展，增加了提示词整理、Anima/SDXL 提示词转换、图片提示词读取以及带 A1111/Civitai 元数据的图片保存节点。

## 安装

进入 ComfyUI 的 `custom_nodes` 目录：

```bash
git clone https://github.com/NaviVoid/efficiency-nodes-comfyui.git
cd efficiency-nodes-comfyui
python -m pip install -r requirements.txt
```

请使用运行 ComfyUI 的同一个 Python 环境安装依赖。安装或更新后需要重启 ComfyUI。

更新：

```bash
git pull
python -m pip install -r requirements.txt
```

## 主要节点

### 原有效率节点

| 分类 | 节点 |
| --- | --- |
| 加载 | `Efficient Loader`、`Eff. Loader SDXL` |
| 采样 | `KSampler (Efficient)`、`KSampler Adv. (Efficient)`、`KSampler SDXL (Eff.)` |
| 堆栈 | `LoRA Stacker`、`Control Net Stacker`、`Apply ControlNet Stack` |
| XY Plot | `XY Plot` 及 Seed、Steps、CFG、Sampler、Checkpoint、LoRA、ControlNet 等输入节点 |
| 脚本 | `Noise Control Script`、`HighRes-Fix Script`、`Tiled Upscaler Script` |
| 其他 | `Image Overlay`、`Pack SDXL Tuple`、`Unpack SDXL Tuple` |

### 本仓库扩展节点

| 节点 | 作用 |
| --- | --- |
| `Save Image (Efficient)` | 保存 PNG、JPEG、WebP，并写入 A1111/Civitai 格式生成参数 |
| `OrganizePrompt` | 清理、去重和规范化逗号分隔的提示词 |
| `AnimaToSDXLPrompt` | 去除 Anima tag 的 `@` 标记，转换为普通 SDXL 提示词 |
| `SDXLToAnimaPrompt` | 为每个 tag 添加 Anima 使用的 `@` 标记 |
| `ImageWithPrompt` | 从图片 `parameters` 元数据读取 positive、negative 和原始参数 |
| `PickImageWithPrompt` | 批量读取图片元数据并通过正则表达式提取文本 |
| `SaveImageWithMetadata` | 将已有 metadata 字符串写回单张 PNG，并支持动态路径与放大信息 |
| `RandomWeightedPrompt` | 从文本文件随机抽取词条并生成随机权重 |
| `StringListToWildcards` | 将字符串列表转换为 `{a\|b\|c}` wildcard 格式 |
| `SDupscaleTiledSize` | 根据图片尺寸和放大倍率计算 Ultimate SD Upscale 分块尺寸 |
| `Eff MosaicMask` | 仅对遮罩区域应用马赛克 |

## Save Image (Efficient)

该节点通过运行时采集获取工作流实际使用的参数，避免动态节点、pipe 或子图中的输入在保存时变成连线 ID。

可保存的信息包括：

- positive 和 negative prompt
- Steps、Sampler、Scheduler、CFG、Seed、图片尺寸
- Checkpoint 名称及 SHA-256 前 10 位 hash
- 实际加载的 LoRA、权重及 SHA-256 前 10 位 hash
- 可选 ComfyUI workflow
- Anima Artist Mixer 的 `base_prompt` 和 `artist_chain`

已处理的常见来源包括 Efficiency Nodes、EasyUse pipe、LoRA Manager stack、普通 Checkpoint/UNET/LoRA Loader，以及 `Anima Artist Pack (Split + Encode)`。

### Anima 工作流

使用 `Anima Artist Pack (Split + Encode)` 时：

- `base_prompt` 作为最终 positive prompt 保存。
- `artist_chain` 不混入 positive，而是写入 A1111 参数行的自定义字段 `Anima artist chain`。

示例：

```text
masterpiece, 1girl, <lora:anima-turbo-lora-v0.2:0.7>
Negative prompt: worst quality, lowres
Steps: 8, Sampler: Euler a Simple, CFG scale: 1.0, Seed: 1618, Size: 1280x1920, Model hash: d1d4fa3de4, Model: OBAnima_v20, Lora hashes: "anima-turbo-lora-v0.2: 1b55e40bdb", Anima artist chain: "0.8::@artist_a, 0.1::@artist_b"
```

提示词内部的换行会被转换为空格。自动补入的 LoRA 标签使用逗号与 positive 分隔。

### 文件格式

| 格式 | 元数据 | Workflow | 选项 |
| --- | --- | --- | --- |
| PNG | `parameters` 文本块 | 可选 | - |
| JPEG | EXIF | 不写入 | `quality` |
| WebP | EXIF | 可选 | `quality`、`lossless_webp` |

其他选项：

- `save_with_metadata`：是否保存生成参数。
- `embed_workflow`：是否嵌入 ComfyUI workflow。
- `add_counter_to_filename`：是否在文件名后添加递增编号。

## SaveImageWithMetadata

该节点用于读取图片、放大后继续保存原始 `parameters` 文本。保存时会将 `Size` 更新为最终图片尺寸。`filename_prefix` 支持与 `Save Image (Efficient)` 相同的路径变量；其中 `%width%`、`%height%` 使用最终保存图片尺寸，`%seed%`、`%model%`、`%pprompt%`、`%nprompt%` 从传入的 metadata 解析。

保存 PNG 时会按 A1111 infotext 字段补充放大信息：

- `Hires upscale`：从 metadata 中的原始 `Size` 和输出图片尺寸推断，或使用手动 `upscale_by`。
- `Hires upscaler`：默认从上游 upscale model loader 推断，也可通过 `upscale_model_name` 手动指定。

### 动态文件名和目录

`filename_prefix` 支持路径和变量，因此可以按日期、模型自动建立目录：

```text
civitai/%date:yyyy-MM-dd%/%model%/img
```

支持的变量：

| 变量 | 内容 |
| --- | --- |
| `%seed%` | Seed |
| `%width%` | 图片宽度 |
| `%height%` | 图片高度 |
| `%model%` | 不含扩展名的模型名称 |
| `%model:N%` | 模型名称前 N 个字符 |
| `%pprompt%` | Positive prompt |
| `%pprompt:N%` | Positive prompt 前 N 个字符 |
| `%nprompt%` | Negative prompt |
| `%nprompt:N%` | Negative prompt 前 N 个字符 |
| `%date:FORMAT%` | 日期时间，支持 `yyyy`、`yy`、`MM`、`dd`、`hh`、`mm`、`ss` |


## 说明

- `ImageWithPrompt` 和 `PickImageWithPrompt` 主要读取标准 A1111 `parameters` 格式。
- 模型和 LoRA hash 需要对应文件仍存在于 ComfyUI 模型目录中。
- 运行时元数据钩子在插件加载时安装，因此修改或更新插件后必须重启 ComfyUI。

## 致谢与许可

- 上游项目：[jags111/efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui)
- Save Image 行为参考：[comfyui-lora-manager](https://github.com/willmiao/ComfyUI-Lora-Manager)
- 许可证见 [LICENSE](LICENSE)
