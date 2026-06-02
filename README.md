# 🧠 BiDPO-Lite

A lightweight, Colab-friendly implementation of **Region-aware Preference Optimization for Text-to-Image Models**, inspired by BiDPO-style research.

This project builds a complete pipeline for:
- prompt generation
- image synthesis
- vision-language judging
- region-aware scoring
- preference dataset creation
- lightweight DPO-style training

---

## 🚀 Overview

BiDPO-Lite improves compositional text-to-image generation by combining:

- 🎨 Diffusion models (SDXL-Turbo)
- 🧠 Vision-Language Models (OpenRouter)
- 📍 Region-aware reasoning (SAM / heuristic proxy)
- ⚖️ Preference Optimization (DPO-style training)

---

## 🔁 Pipeline
Prompt Generator
↓
Diffusion Model (SDXL-Turbo)
↓
Multiple Candidate Images
↓
Vision-Language Model Judge (OpenRouter)
↓
Region-aware Scoring (Objects + Segments)
↓
Preference Pair Construction
↓
Lightweight DPO Training
↓
Improved Image Generator


---

## 📌 Features

- Automatic compositional prompt generation
- Multi-sample image generation per prompt
- Vision-based scoring using OpenRouter models
- Region-aware object evaluation
- Preference dataset creation (chosen vs rejected pairs)
- Lightweight LoRA / DPO-style training
- Fully runnable on free Colab T4 GPU

---


---

## ⚙️ Installation

### Clone repository
```bash
git clone https://github.com/your-username/BiDPO-Lite.git
cd BiDPO-Lite
```
