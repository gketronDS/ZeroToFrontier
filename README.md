# ZeroToFrontier
A structured path to the skills used at Anthropic, OpenAI, and the teams building the most capable AI systems in the world.
The Map
This guide is organized into six phases. Each phase builds on the last. The estimated times assume 20-30 hours/week of focused work. If you're full-time, compress accordingly.

The core insight from watching how frontier labs operate: the best researchers are systems thinkers. They understand the math, but they also understand the hardware, the data pipeline, the training infrastructure, and the evaluation methodology. This guide covers the full stack.


Phase 0: Foundations (2-4 weeks)
If you can already train a neural net from scratch in PyTorch, skip to Phase 1.
Build Neural Nets From Scratch
Andrej Karpathy's "Neural Networks: Zero to Hero" (https://karpathy.ai/zero-to-hero.html)

This is the single best entry point into deep learning that exists. Karpathy was a founding member of OpenAI, led Tesla's computer vision team, and built Stanford's first deep learning course. The series takes you from implementing backpropagation on scalar values (micrograd) through building a GPT from scratch.

Watch and code along with every video in order:

micrograd — Build a tiny autograd engine. You will understand backpropagation at the level of individual operations. This is the foundation everything else rests on.
makemore series (Parts 1-5) — Build increasingly sophisticated character-level language models: bigrams → MLPs → WaveNet-style models. You'll internalize how neural nets learn distributions over sequences.
Building GPT from scratch — Implement a transformer. By the end you'll understand attention, layer normalization, residual connections, and the training loop.
Tokenization — Build a BPE tokenizer from scratch. Understand why tokenization decisions affect everything downstream.

Karpathy's "Deep Dive into LLMs like ChatGPT" YouTube video is also excellent as a conceptual overview of the full LLM pipeline from pre-training through RLHF.
Supplementary Foundations
3Blue1Brown's neural network series (YouTube) — Best visual intuitions for how neural nets learn, gradient descent, and backpropagation.
Linear Algebra Done Right by Sheldon Axler, Chapters 1-7 — You need comfort with vector spaces, linear maps, eigenvalues, and inner product spaces. Most of deep learning is linear algebra with nonlinear activation functions.
The Matrix Cookbook (free PDF, Petersen & Pedersen) — Keep this as a reference. You'll need matrix calculus identities constantly.


Phase 1: PyTorch Internals and Training Infrastructure (3-4 weeks)
The goal: understand what happens between your Python code and the GPU.
PyTorch Deep Dive
Don't just use PyTorch — understand it.

torch.compile and the compiler stack. Read the PyTorch 2.0 blog post ("PyTorch 2.0: Our next generation release") and understand the three components: TorchDynamo (captures the Python computation graph), TorchInductor (generates optimized kernels), and the intermediate representations. Run torch.compile on a simple model with TORCH_LOGS="output_code" set and read the generated Triton kernels. This is where you start understanding what the framework does for you.

Profiling. Learn torch.profiler and NVIDIA Nsight Systems. Profile a training step of a small transformer. Identify where time is spent: is it in attention? In the optimizer step? In data loading? In CPU-GPU synchronization? Most "slow" training runs are slow because of avoidable bottlenecks in data movement, not because of bad kernels.

Mixed precision training. Understand FP32, FP16, BF16, and FP8. Read the NVIDIA mixed precision training documentation. Understand why loss scaling is needed for FP16, why BF16 avoids this problem, and when FP8 (Transformer Engine) applies. This is not optional — every frontier lab trains in mixed precision.

Distributed training. Read the PyTorch FSDP (Fully Sharded Data Parallel) tutorial. Understand the difference between data parallelism (same model, different data on each GPU), tensor parallelism (split a single layer across GPUs), and pipeline parallelism (different layers on different GPUs). You don't need to implement these from scratch, but you need to know which strategy to use when and why.
Key Resources
PyTorch documentation — The official tutorials are good. Work through "Introduction to Distributed Data Parallel", "Getting Started with FSDP", and "Automatic Mixed Precision".
Stas Bekman's "Machine Learning Engineering" (free online book) — Outstanding practical guide to the engineering side of ML. Covers multi-GPU training, debugging, performance optimization.
The "Efficient Training on a Single GPU" page from HuggingFace docs — Practical tricks that matter: gradient accumulation, gradient checkpointing, batch size tuning.


Phase 2: Understanding the GPU (4-6 weeks)
The goal: build a mental model of the hardware that runs your models.
GPU Architecture from First Principles
Start with the NVIDIA CUDA C Programming Guide, Chapters 1-5. Not a summary, not a blog post — the actual documentation. It's well-written and covers the execution model (grids → blocks → threads → warps), the memory hierarchy (global → L2 → shared → registers), and synchronization.

Key concepts to internalize:

A GPU is a throughput machine, not a latency machine. It hides memory latency by running thousands of threads simultaneously. An individual thread is slow; the aggregate throughput is massive.
The memory hierarchy is everything. Global memory (HBM): ~2TB/s bandwidth on H100, but 100s of cycles latency. Shared memory (SRAM): ~30TB/s bandwidth, ~20 cycles latency. Registers: instant. Almost every GPU optimization is about keeping data in the faster levels.
Warps. 32 threads execute in lockstep (SIMT). Warp divergence (threads in the same warp taking different branches) is expensive. Coalesced memory access (threads in a warp accessing contiguous memory) is essential.
Occupancy. The ratio of active warps to maximum possible warps per SM. Higher occupancy generally means better latency hiding, but it's not the only factor.
Write CUDA Kernels
Simon Boehm's "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (https://siboehm.com/articles/22/CUDA-MMM)

This is the single most important hands-on tutorial for GPU programming. Boehm (now at Anthropic) walks through optimizing a matrix multiply kernel step by step:

Naive kernel → 1.3% of cuBLAS
Coalesced global memory access → significant speedup
Shared memory tiling → avoid redundant global loads
1D block tiling → better work distribution
2D warp tiling → more compute per memory load
Vectorized loads (float4) → 4x memory bandwidth per instruction
Final kernel → ~80-95% of cuBLAS

The accompanying code is at https://github.com/siboehm/SGEMM_CUDA. Clone it. Run it. Modify it. Break it and fix it.

Why matmul? Because matmul IS deep learning. The forward pass of a transformer is: LayerNorm → QKV projection (matmul) → Attention → Output projection (matmul) → MLP up-projection (matmul) → activation → MLP down-projection (matmul). If you understand how to make matmul fast, you understand the core computational bottleneck of every model at every frontier lab.

Anastasia Salykova's advanced SGEMM tutorial (https://salykova.github.io/sgemm-gpu) goes further, covering inlined PTX, async memory copies, and double-buffering. This bridges the gap between educational kernels and production-grade CUTLASS code.
FlashAttention: The Most Important Kernel Innovation
Read these papers in order:

FlashAttention (Dao et al., 2022) — https://arxiv.org/abs/2205.14135. The key insight: standard attention materializes the full N×N attention matrix in slow HBM. FlashAttention tiles the computation so it stays in fast SRAM. The math is identical. The compute is identical. It's 2-4x faster because of WHERE the data lives. This is a pure IO-awareness optimization.

FlashAttention-2 (Dao, 2023) — Better parallelism and work partitioning. Restructures the algorithm to parallelize over the sequence length dimension.

FlashAttention-3 (Dao et al., 2024) — https://tridao.me/blog/2024/flash3/. Exploits Hopper-specific features: warp specialization, async memory copies via TMA, GEMM-softmax pipelining. Reaches 75% of theoretical peak on H100.

"We Reverse-Engineered Flash Attention 4" (Modal blog, 2025) — https://modal.com/blog/reverse-engineer-flash-attention-4. Outstanding walkthrough of FA4's Blackwell-optimized kernel. The key architecture: five specialized warp roles (load, MMA, softmax, correction, store) coordinated through async pipelines. Surprisingly understandable if you have concurrent programming experience.

After reading the papers, implement FlashAttention in Triton using this tutorial: "Understanding Flash Attention: Writing the Algorithm from Scratch in Triton" by Alex Dremov (https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/). The Triton implementation is more readable than the CUDA version and uses notation closer to the paper.
Triton: GPU Kernels in Python
OpenAI Triton tutorials (https://triton-lang.org/main/getting-started/tutorials/)

Triton lets you write GPU kernels in Python-like syntax that compile to the same PTX assembly as CUDA C. Start with:

Vector addition — understand the programming model
Fused softmax — your first real optimization (fusing reduces memory round-trips)
Matrix multiplication — same concepts as Simon Boehm's tutorial, but in Python
Flash Attention — the Triton tutorial on fused attention (tutorial 06)

For research work, Triton-level performance is sufficient. The last 10-15% of performance that requires raw CUDA matters at production scale but not for proving a concept.
NVIDIA Architecture Specifics
Know the differences between GPU generations:

Ampere (A100): Introduced async memory copies (cp.async), TF32 tensor cores, 40MB L2 cache. The workhorse of 2021-2024 training.
Hopper (H100): Transformer Engine (FP8 with dynamic scaling), TMA (hardware-managed async loads), Thread Block Clusters for cross-SM communication. Where FlashAttention-3 gets its biggest wins.
Blackwell (B200/GB200): Second-gen Transformer Engine, wider tensor cores, more SRAM. FA4 is optimized for this.

For your work, understanding Ampere deeply is sufficient. H100/Blackwell optimizations matter at BFL/Anthropic/OpenAI scale.


Phase 3: Transformers and Language Models (4-6 weeks)
The goal: understand every component of a modern LLM at the implementation level.
Build an LLM from Raw C/CUDA
Karpathy's llm.c (https://github.com/karpathy/llm.c)

This is a complete GPT-2 training implementation in ~1,000 lines of C/CUDA. No PyTorch, no Python, no frameworks. Every operation — attention, layernorm, GELU, Adam optimizer — is implemented as a raw kernel. Reading this code connects everything: the math you learned in Phase 0, the GPU concepts from Phase 2, and the model architecture.

Karpathy reproduced GPT-2 (124M) training for $20 using this code. It's also ~7% faster than PyTorch Nightly. That gap teaches you how much overhead frameworks add.

The parallel Python reference implementation (train_gpt2.py) is essentially a cleaned-up nanoGPT. Compare the two implementations side by side.
Key Papers
Read these papers carefully, with pen and paper:

"Attention Is All You Need" (Vaswani et al., 2017) — The original transformer paper. Even in 2026, this is required reading.

"Language Models are Few-Shot Learners" (GPT-3, Brown et al., 2020) — Established scaling laws and in-context learning. Pay attention to the training infrastructure section.

"Training Compute-Optimal Large Language Models" (Chinchilla, Hoffmann et al., 2022) — The scaling law paper that changed how every lab allocates compute between model size and data.

"LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023) — Clean architecture with RoPE, RMSNorm, GQA. The architecture most open models are based on.

"Constitutional AI" (Bai et al., 2022, Anthropic) — How Anthropic approaches alignment. Understand RLHF and CAI if you want to understand the post-training pipeline.

"Scaling Laws for Neural Language Models" (Kaplan et al., 2020, OpenAI) — The empirical foundation for why scaling works and how to predict performance from compute budgets.
Architecture Deep Dives
Understand the specific components that distinguish modern architectures from the original transformer:

RoPE (Rotary Position Embeddings) — Why they replaced learned position embeddings, how they encode relative position through rotation matrices.
GQA (Grouped Query Attention) — Why sharing KV heads across query heads reduces memory without sacrificing quality.
RMSNorm vs LayerNorm — Why RMSNorm is faster and works just as well.
SwiGLU — Why gated activations outperform ReLU/GELU in the MLP.
MoE (Mixture of Experts) — How sparse models like Qwen 3.6 activate only 3B of 35B parameters per token. Load balancing, routing, expert capacity.


Phase 4: Vision and Multimodal Models (3-4 weeks)
The goal: understand visual intelligence at the architectural level, relevant to your chart understanding project.
Vision Transformers
"An Image Is Worth 16x16 Words" (ViT, Dosovitskiy et al., 2020) — The paper that brought transformers to vision. Understand patch embeddings, how a 224x224 image becomes a sequence of 196 tokens, and why this works.

"DINOv2" (Oquab et al., 2023, Meta) — Self-supervised vision representation learning. Relevant because Andy Blattmann referenced this class of representation learning models in the podcast (he mentions "dino" specifically).

"SigLIP" (Zhai et al., 2023) — The vision-language contrastive model used in many VLMs. Understand how CLIP-style models align image and text representations.
Vision-Language Models
"LLaVA: Visual Instruction Tuning" (Liu et al., 2023) — The simplest architecture for a VLM: vision encoder → projection layer → LLM. This is the template most open VLMs follow.

"Qwen-VL" and "Qwen2-VL" technical reports — Understand how the model you'll likely fine-tune handles vision. Dynamic resolution, native multimodal integration.

"PaliGemma" (Beyer et al., 2024, Google) — A clean multimodal model designed for fine-tuning. Good reference for how to add vision capabilities to a language model efficiently.
Diffusion Models (for understanding the BFL pipeline)
"Denoising Diffusion Probabilistic Models" (Ho et al., 2020) — The foundation of modern image generation.

"High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach, Blattmann et al., 2022) — The Latent Diffusion paper. This is Andy and team's key contribution: compress images to a learned latent space, train the diffusion model there. Orders of magnitude more efficient.

"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser, Kulal, Blattmann et al., 2024) — The Flux architecture paper. Flow matching instead of diffusion, transformer backbone instead of U-Net.

"SelfFlow" (BFL, March 2026) — The multimodal representation alignment paper Andy discussed in the podcast. How to align generative model representations with representation learning objectives across modalities.
Chart Understanding Specifically
"MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering" (Liu et al., 2023, Google) — Chart derendering as a pre-training task. The closest existing work to your project idea.

"DePlot: One-shot Visual Language Reasoning by Plot-to-Table Translation" (Liu et al., 2023, Google) — Two-stage pipeline: extract table from chart, then reason with an LLM.

"CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs" (Yue et al., NeurIPS 2024) — The benchmark that exposed the generalization gap: performance drops 34.5% with minor chart perturbations.


Phase 5: Training and Evaluation at Scale (3-4 weeks)
The goal: understand how frontier labs run training and evaluate models.
Pre-training
"OLMo: Accelerating the Science of Language Models" (Groeneveld et al., 2024, AI2) — Fully open pre-training pipeline. Data, code, weights, logs, everything published. Study the training infrastructure section.
"The Pile" and "RedPajama" and "FineWeb" data documentation — Understand how training data is curated, deduplicated, and quality-filtered. Data quality is arguably the most important factor in model quality.
Post-training (RLHF, DPO, and beyond)
"Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022, OpenAI) — The InstructGPT paper. How RLHF works in practice.
"Direct Preference Optimization" (Rafailov et al., 2023) — The simpler alternative to RLHF that avoids training a separate reward model.
Karpathy's "2025 LLM Year in Review" — His overview of how RLVR (Reinforcement Learning with Verifiable Rewards) became the dominant post-training paradigm. Models improve by solving problems with objectively verifiable answers (math, code), which forces them to develop reasoning traces.
Evaluation
"Holistic Evaluation of Language Models" (HELM, Liang et al., 2022) — Comprehensive eval framework.
MMMU / MMMU-Pro papers — The multimodal benchmarks we discussed. Understand the methodology: heterogeneous image types, descriptive vs. reasoning questions, why the Pro version is much harder.
"Adding Error Bars to Evals" (2024) — Statistical rigor in evaluation. Understand confidence intervals and why single-number benchmark comparisons can be misleading.


Phase 6: Research and Frontier Skills (Ongoing)
The goal: develop the taste and methodology to do original work.
Developing Research Taste
Read arXiv daily. Subscribe to cs.CV, cs.CL, cs.LG on arXiv. Use Semantic Scholar or Papers With Code to track what's getting cited. Build the habit of reading 2-3 abstracts per day and 1-2 full papers per week.
Reproduce results. Pick a paper, implement it from scratch, and match their numbers. This is the single best way to develop research skills. You'll discover that papers omit crucial details, and learning to fill those gaps is the skill.
The Bitter Lesson by Rich Sutton (2019) — A one-page essay that's arguably the most important piece of writing in AI. The lesson: methods that leverage computation scale better than methods that leverage human knowledge. Every time researchers have tried to build in human knowledge, compute-based methods eventually win.
Communities
GPU MODE Discord — Community of people writing GPU kernels, discussing optimization, sharing implementations. This is where Simon Boehm, Tri Dao, and other kernel authors hang out.
Karpathy's Zero to Hero Discord — Community around the learning materials.
EleutherAI Discord — Open-source LLM research community.
Tools You Should Know
Weights & Biases (wandb) — Experiment tracking. Every frontier lab uses something like this.
NVIDIA Nsight Systems / Nsight Compute — GPU profiling. Nsight Systems for system-level timeline, Nsight Compute for kernel-level analysis.
vLLM — The standard inference engine. Understanding its architecture (PagedAttention, continuous batching) teaches you how production serving works.


The Meta-Curriculum
The resources above give you technical depth. But frontier labs also select for something harder to teach: the ability to identify and attack the right problem.

From the Andy Blattmann podcast, several principles emerged:

Focus beats breadth. BFL succeeded by obsessing over one domain (image generation) and being 10x better, not by trying to do everything at once.

The feedback loop is the product. It's not about building one model. It's about building a system that produces better models repeatedly. Data → training → deployment → context feedback → better data → better training.

Don't panic when competitors launch. Assess the landscape calmly, find the unsolved problem in your domain expertise, and execute. BFL shipped Kontext 60 days after a competitor launched something that looked better.

Verification is the bottleneck. Wherever you can measure progress objectively (code: unit tests, math: correct answers, robotics: physical constraints), progress will be fast. Wherever verification requires human judgment (aesthetics, preference), progress is slower. The most valuable research contributions create new ways to verify.

Open weights win when preferences are heterogeneous. Different users want different things from the same model. Open weights let them customize.

This last point is directly relevant to your chart understanding project. Chart interpretation has objectively verifiable ground truth (you know the underlying data), which means you can build a rigorous evaluation framework. That's rare in vision research, and it's exactly the kind of contribution that moves the field.


Recommended Order for Someone Starting Today
If you're starting from a solid Python / basic ML background:

Weeks 1-2: Karpathy Zero to Hero (Phase 0) Weeks 3-4: PyTorch internals, profiling, mixed precision (Phase 1) Weeks 5-8: Simon Boehm matmul tutorial + FlashAttention papers (Phase 2) Weeks 9-12: llm.c + transformer papers + architecture deep dives (Phase 3) Weeks 13-15: VLM papers + chart understanding papers (Phase 4) Weeks 16-18: Start your chart derendering project Ongoing: Phase 5-6 in parallel with project work

Total ramp time to being dangerous: ~4 months of focused work.

Total ramp time to contributing at the frontier: 12-18 months, because the last mile is developing research taste through doing original work, failing, and iterating.


Key Links (Quick Reference)
Video Courses

Karpathy Zero to Hero: https://karpathy.ai/zero-to-hero.html
Karpathy "Deep Dive into LLMs": YouTube

GPU Programming

Simon Boehm CUDA Matmul: https://siboehm.com/articles/22/CUDA-MMM
Boehm's code: https://github.com/siboehm/SGEMM_CUDA
Salykova's Advanced SGEMM: https://salykova.github.io/sgemm-gpu
Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

From-Scratch Implementations

llm.c: https://github.com/karpathy/llm.c
nanoGPT: https://github.com/karpathy/nanoGPT
micrograd: https://github.com/karpathy/micrograd
minbpe: https://github.com/karpathy/minbpe

FlashAttention

FA1 paper: https://arxiv.org/abs/2205.14135
FA2 paper: https://tridao.me/publications/flash2/flash2.pdf
FA3 blog: https://tridao.me/blog/2024/flash3/
FA4 reverse-engineered: https://modal.com/blog/reverse-engineer-flash-attention-4
FA in Triton walkthrough: https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/
Code: https://github.com/Dao-AILab/flash-attention

Communities

GPU MODE Discord: https://discord.gg/gpumode
Karpathy Zero to Hero Discord: linked from karpathy.ai

Textbooks

"Programming Massively Parallel Processors" — Kirk & Hwu
"Linear Algebra Done Right" — Axler
"Deep Learning" — Goodfellow, Bengio, Courville (free online)

