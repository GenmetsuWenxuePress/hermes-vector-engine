# BGE-M3 Vulkan GPU Acceleration on Windows 11 / WSL2

Guide for deploying local `llama-server` on Windows with Vulkan GPU acceleration (`-ngl 99`), serving embeddings to WSL2 over localhost.

---

## 1. Why Windows Vulkan + WSL2 Topology?

1. **AMD iGPU / APU Limitation in WSL2**:
   * AMD Radeon iGPUs (780M / 880M / RDNA3) do NOT have native ROCm support inside WSL2.
   * Direct CPU inference takes ~15-20s per batch and ramps fan noise.
2. **The Vulkan Solution**:
   * Windows native Vulkan driver exposes the full hardware compute capability of the AMD Radeon GPU.
   * WSL2 with mirrored networking (`networkingMode=mirrored`) connects to `127.0.0.1:8081` with near-zero latency (~0.2ms).

---

## 2. Windows Deployment Steps

### Step 1: Download llama.cpp Vulkan Release
Download the pre-compiled `llama-bXXXX-bin-win-vulkan-x64.zip` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases).

Extract to `C:\Users\<username>\llama.cpp\bin\`.

### Step 2: Download BGE-M3 GGUF Model
Download `bge-m3-Q4_K_M.gguf` (approx 360MB) from HuggingFace to `C:\Users\<username>\llama.cpp\models\`.

### Step 3: Run Server with Full GPU Offload
```powershell
# Run from PowerShell on Windows:
C:\Users\<username>\llama.cpp\bin\llama-server.exe `
  -m C:\Users\<username>\llama.cpp\models\bge-m3-Q4_K_M.gguf `
  --embeddings `
  --host 127.0.0.1 `
  --port 8081 `
  -b 2048 `
  -ub 2048 `
  -ngl 99 `
  -t 4
```

---

## 3. WSL2 Health Check
From inside WSL2:
```bash
curl -s --noproxy '*' http://127.0.0.1:8081/health
# Expected: {"status":"ok"}

curl -s --noproxy '*' http://127.0.0.1:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test embedding", "model": "bge-m3"}'
```
