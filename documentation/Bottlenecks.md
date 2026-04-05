---
title: "Bottlenecks"
category: "wLLm"
tags: []
status: "Active"
created: "2026-04-01"
---
making wLLM as fast as vLLM or Ollama requires me to make a purpose built intermediary that would have to handle and sit between the program and CUDA kernels basically the GPU

## The Zero-Day Wedge
As of now, wLLM's **only** functional advantage over incumbents (Ollama, LM Studio) is **Zero-Day Model Support**. 
- Because we use raw HuggingFace weights without conversion (GGUF/Exl2), we can run any new architecture the moment it drops.
- All other speed/memory advantages belong to more complex engines with custom C++/CUDA kernels.