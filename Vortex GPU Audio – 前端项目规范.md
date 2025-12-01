\# Vortex GPU Audio – 前端项目规范（Vue3 \+ TypeScript \+ 全平台终极版）

目标：构建一个\*\*全球最强发烧级音频控制台\*\*的前端，支持 Web / Electron / PWA / iOS / Android 全平台统一运行。

\#\#\# 项目技术栈（2025 最强组合）

\- Vue 3（Composition API \+ \`\<script setup\>\`）  
\- TypeScript 5.x  
\- Vite 5 \+ vite-plugin-pwa  
\- Pinia（状态管理）  
\- Vue Router 4  
\- vue-i18n 9（完整多语言）  
\- TailwindCSS v3 \+ daisyUI（极简美观）  
\- Naive UI（可选高级组件）  
\- ECharts / wavesurfer.js（实时频谱 \+ 波形）  
\- WebSocket \+ REST 客户端（自动重连 \+ 二进制解析）  
\- 支持暗黑模式 \+ 毛玻璃 \+ 发烧级 UI 细节

\#\#\# 项目目录结构（必须严格遵守）  
vortex-frontend/  
├── public/  
│   ├── icons/  
│   │   ├── icon-192x192.png  
│   │   ├── icon-256x256.png  
│   │   ├── icon-384x384.png  
│   │   ├── icon-512x512.png  
│   │   └── apple-touch-icon.png  
│   ├── favicon.ico  
│   └── manifest.json                  \# PWA manifest（可由 vite-plugin-pwa 自动生成）  
│  
├── src/  
│   ├── assets/  
│   │   ├── fonts/                     \# 多语言必备字体  
│   │   │   ├── NotoSansSC-Regular.otf  
│   │   │   ├── NotoSansTC-Regular.otf  
│   │   │   └── NotoSansEN-Regular.otf  
│   │   └── images/  
│   │       ├── logo.svg  
│   │       └── background.jpg  
│   │  
│   ├── components/
│   │   ├── common/
│   │   │   ├── Layout.vue
│   │   │   ├── Header.vue
│   │   │   ├── Footer.vue
│   │   │   └── LoadingSpinner.vue
│   │   │
│   │   ├── player/
│   │   │   ├── SpectrumAnalyzer.vue       \# 实时2048点频谱
│   │   │   ├── WaveformView.vue           \# 实时波形
│   │   │   ├── VUMeter.vue                \# 专业VU \+ PPM \+ Peak
│   │   │   ├── GpuLoadIndicator.vue       \# GPU/NPU/CPU负载仪表盘
│   │   │   ├── PlaybackControls.vue
│   │   │   ├── FileLoader.vue             \# 音频文件加载器
│   │   │   ├── FileDropZone.vue           \# 拖拽文件区域
│   │   │   ├── FileProgressBar.vue        \# 文件加载进度条
│   │   │   ├── AudioMetadata.vue          \# 音频元数据显示
│   │   │   └── PlaylistManager.vue        \# 播放列表管理器
│   │   │
│   │   ├── chain/
│   │   │   ├── ModuleChain.vue            \# 可拖拽排序的模块链
│   │   │   ├── ModuleCard.vue
│   │   │   └── ModuleParameterPanel.vue
│   │   │
│   │   ├── output/
│   │   │   ├── OutputSelector.vue         \# 本地 \+ Roon \+ HQPlayer NAA \+ UPnP
│   │   │   └── DeviceItem.vue
│   │   │
│   │   └── market/
│   │       ├── ModuleMarket.vue
│   │       └── ModuleItem.vue  
│   │  
│   ├── locales/                           \# vue-i18n 语言包  
│   │   ├── zh-CN.json  
│   │   ├── zh-TW.json  
│   │   ├── en.json   
│   │  
│   ├── router/  
│   │   └── index.ts  
│   │  
│   ├── stores/                            \# Pinia stores
│   │   ├── index.ts
│   │   ├── player.ts
│   │   ├── chain.ts
│   │   ├── output.ts
│   │   ├── system.ts                      \# 主题、语言、后端地址
│   │   ├── websocket.ts                   \# 实时数据状态
│   │   ├── audio.ts                       \# 音频文件处理状态
│   │   └── playlist.ts                    \# 播放列表管理状态  
│   │  
│   ├── types/  
│   │   ├── api.ts                         \# 后端API类型定义  
│   │   └── index.ts  
│   │  
│   ├── views/  
│   │   ├── PlayerView.vue                 \# 主播放页面  
│   │   ├── SettingsView.vue               \# 设置（语言、主题、后端地址）  
│   │   ├── MarketView.vue                 \# 模块商店（占位）  
│   │   └── AboutView.vue  
│   │  
│   ├── utils/  
│   │   ├── websocket.ts                   \# WebSocket 连接管理 \+ 二进制解析  
│   │   ├── api.ts                         \# REST 封装  
│   │   └── helpers.ts  
│   │  
│   ├── App.vue  
│   ├── main.ts  
│   └── env.d.ts  
│  
├── index.html  
├── vite.config.ts  
├── tsconfig.json  
├── tsconfig.node.json  
├── package.json  
├── pnpm-lock.yaml        (或 yarn.lock / package-lock.json)  
├── .env  
├── .env.production  
└── README.md

\#\#\# 核心功能需求（必须全部实现）

1\. 实时川流数据接收（WebSocket 二进制协议）
   \- 频谱（2048点 Float32）
   \- 波形（4096样本）
   \- VU/Peak/RMS 表盘
   \- GPU/NPU/CPU 负载 \+ 延迟分解
   \- 当前滤波器名称实时显示

2\. 模块链可视化编辑
   \- 支持拖拽排序
   \- 每个模块显示：图标 \+ 名称 \+ Bypass / Solo / Wet 滑块
   \- 右键 → 参数面板（弹窗或抽屉）

3\. 输出设备智能选择
   \- 自动发现局域网所有 Vortex Server / Roon Bridge / HQPlayer NAA
   \- 显示设备名 \+ IP \+ 支持格式（DSD1024 / PCM 768k）
   \- 一键无缝切换（零爆音）

4\. **通用音频文件支持**
   \- **支持所有主流格式**：MP3, WAV, FLAC, ALAC, AAC, OGG, M4A
   \- **高分辨率格式**：DSD, DSF, DFF
   \- **超高分辨率**：DSD1024, PCM 768kHz
   \- **拖拽文件上传**：支持批量文件加载
   \- **实时进度显示**：大文件上传进度条和时间估算
   \- **元数据读取**：自动检测格式、采样率、位深度、声道数
   \- **文件完整性验证**：损坏文件检测和错误报告
   \- **播放列表创建**：多文件批量加载和管理

5\. 完整 i18n（至少以下语言）
   \- 简体中文（zh-CN）—— 100% 翻译完成
   \- 繁體中文（zh-TW）—— 100% 翻译完成
   \- English（en）

6\. 响应式 \+ PWA
   \- 手机、平板、桌面完美适配
   \- 可添加到主屏幕（iOS/Android）
   \- 离线缓存 \+ Service Worker

7\. 主题系统
   \- 自动跟随系统
   \- 手动切换 Light / Dark / OLED Black
   \- 支持毛玻璃 \+ 发烧级配色（深空灰 \+ 琥珀指示灯）

\#\#\# 关键 API 类型定义（src/types/api.ts）

\`\`\`ts
interface RealtimeSpectrum { bins: Float32Array; }   // 2048
interface RealtimeWaveform { left: Float32Array; right: Float32Array; }
interface HardwareStatus {
  gpu: number;      // 0\~100
  npu: number;
  cpu: number;
  latency: { total: number; breakdown: Record\<string, number\> };
}
interface OutputDevice {
  id: string;
  name: string;
  type: 'local' | 'roon' | 'hqplayer' | 'upnp';
  capabilities: string\[\];  // \["DSD1024", "closed-form-16M"\]
}

// 音频文件相关类型定义
interface AudioFile {
  id: string;
  name: string;
  format: string;                    // 'mp3', 'wav', 'flac', 'dsd', etc.
  sampleRate: number;                // 44100, 48000, 96000, 192000, 384000, 768000
  bitDepth: number;                  // 16, 24, 32
  channels: number;                  // 1, 2, 6, 8
  duration: number;                  // seconds
  fileSize: number;                  // bytes
  metadata: AudioMetadata;
  loadingProgress: number;           // 0-100
  status: 'idle' | 'loading' | 'ready' | 'processing' | 'error';
  error?: string;
}

interface AudioMetadata {
  // 技术元数据
  format: string;
  codec: string;
  bitrate: number;                  // kbps
  sampleRate: number;
  bitDepth: number;
  channels: number;
  duration: number;

  // 艺术元数据
  title?: string;
  artist?: string;
  album?: string;
  year?: number;
  genre?: string;
  track?: number;
  albumArt?: string;               // base64 or URL
}

interface FileUploadProgress {
  fileId: string;
  fileName: string;
  bytesLoaded: number;
  totalBytes: number;
  percentage: number;
  speed: number;                    // bytes per second
  estimatedTimeRemaining: number;  // seconds
  status: 'uploading' | 'processing' | 'completed' | 'error';
}

interface PlaylistItem {
  id: string;
  audioFile: AudioFile;
  position: number;
  addedAt: Date;
}

interface SupportedFormat {
  extension: string;
  mimeType: string;
  name: string;
  description: string;
  maxSampleRate: number;
  maxBitDepth: number;
  maxChannels: number;
  isHighResolution: boolean;
}

interface ChannelConfiguration {
  id: string;
  name: string;
  channels: number;
  layout: string[];                 // ['L', 'R'], ['L', 'R', 'C', 'LFE', 'SL', 'SR']
  description: string;

请严格按照以上《Vortex GPU Audio – 前端项目规范》生成一个完整、可运行、生产级的 Vue3 \+ TypeScript 前端项目。

要求：
1\. 使用 Vite \+ Vue3 script setup \+ Pinia \+ vue-i18n \+ TailwindCSS \+ daisyUI
2\. 实现实时频谱、波形、VU表、GPU负载仪表盘（支持 NPU 显示）
3\. 实现可拖拽模块链 \+ 输出设备选择器（支持 Roon/HQPlayer NAA 自动发现）
4\. **实现通用音频文件支持**：
   \- 文件选择器 + 拖拽区域
   \- 支持所有主流音频格式（MP3, WAV, FLAC, ALAC, AAC, OGG, M4A, DSD, DSF, DFF）
   \- 实时加载进度显示和元数据读取
   \- 播放列表管理和批量文件处理
5\. 完整实现 8 语言 i18n（含简体中文 100% 翻译）
6\. 支持 PWA \+ 暗黑模式 \+ 响应式
7\. 提供 Settings 页面（切换后端地址、语言、主题）
8\. 提供模块商店页面骨架（未来扩展）

完成后，此前端可完美控制任意部署的 Vortex GPU Audio 后端，实现手机/网页远程实时操控：

**核心功能**：
- **DSD1024** \+ **512段EQ** \+ **closed-form-16M** \+ **网络透明播放**
- **通用音频文件支持**：支持所有主流格式（MP3, WAV, FLAC, ALAC, AAC, OGG, M4A, DSD, DSF, DFF）
- **实时可视化**：任何加载的音频文件立即显示频谱、波形、VU表
- **无缝后端集成**：自动检测格式并发送到Vortex后端处理

**全球最强发烧级音频控制台**：从标准消费级音频文件到超高分辨率专业音频处理的完整支持！

