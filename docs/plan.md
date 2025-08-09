

# AI Study Plan

Consolidates three study plans:
[`docs/plan basic.md`](plan-basic.md), [`docs/plan-opus4.1.md`](plan-opus4.1.md) (comprehensive), and [`docs/plan-sonnet4.md`](plan-sonnet4.md) (advanced/production).

It preserves all topics and resources. When content becomes too dense or duplicative, it is marked as Optional so nothing is lost.

## How to use this plan

- Icons used throughout:
   - 🔹 Core: recommended for everyone; forms the main track
   - 🧩 Optional: valuable but not required; pick based on time/interest
   - 🚀 Advanced: assumes strong background and/or higher compute
   - 💸 Budget: guidance and fallbacks to run on free/low-cost tiers
   - 🔹 🚀 Core + Advanced: combined items spanning both levels

## Overview

- Prerequisites: Basic Python, high school math; 🚀 items assume solid PyTorch, CUDA/DDP, and some production experience
- Duration: 26 weeks (approx. 6 months) with 🧩/🚀 modules you can include or skip
- Study time: 15–25 hours/week (peak weeks up to 30+)
- Compute posture: Free-first; scale up only when needed. A full compute strategy and budget fallbacks are provided at the end and referenced inline as 💸.

## Contents
<a id="contents"></a><a id="top"></a>

<details>
<summary>Phase 1: Foundations & Computer Vision (Weeks 1–12)</summary>

- [Week 1: AI Foundations & Neural Networks](#week-1)
- [Week 2: Python Mastery & Dev Tools](#week-2)
- [Week 3: PyTorch Deep Dive & Cloud Training](#week-3)
- [Week 4: First End-to-End Training Loop](#week-4)
- [Week 5: CNNs & Backpropagation Deep Dive](#week-5)
- [Week 6: Advanced CNN Architectures](#week-6)
- [Week 7: Training Optimization & Regularization](#week-7)
- [Week 8: One Cycle Policy & Advanced Training](#week-8)
- [Week 9: ImageNet-Scale & Distributed Training](#week-9)
- [Week 10: Computer Vision Applications](#week-10)
- [Week 11: Generative Models Intro (VAE/GAN)](#week-11)
- [Week 12: CV Capstone](#week-12)

</details>

<details>
<summary>Phase 2: Transformers & Large Language Models (Weeks 13–18)</summary>

- [Week 13: Transformer Architecture Deep Dive](#week-13)
- [Week 14: Embeddings & Tokenization](#week-14)
- [Week 14b: Retrieval-Augmented Generation (RAG)](#week-14b)
- [Week 15: LLM Training (GPT)](#week-15)
- [Week 16: LLM Optimization & Evaluation](#week-16)
- [Week 17: Supervised Fine-tuning & Instruction Tuning](#week-17)
- [Week 18: Quantization & Model Compression](#week-18)

</details>

<details>
<summary>Phase 3: Advanced Applications (Weeks 19–22)</summary>

- [Week 19: Vision–Language Models (CLIP)](#week-19)
- [Week 20: Reinforcement Learning Fundamentals](#week-20)
- [Week 21: Advanced RL (PPO/SAC/TD3/DDPG)](#week-21)
- [Week 22: RLHF & Alignment](#week-22)

</details>

<details>
<summary>Phase 4: Large-Scale Training & Production (Weeks 23–24)</summary>

- [Week 23: Large-Scale Model Training](#week-23)
- [Week 24: Serving, Optimization, and MLOps](#week-24)

</details>

<details>
<summary>Phase 5: Capstone & Portfolio (Weeks 25–26)</summary>

- [Capstone & Portfolio](#phase-5)

</details>

<details>
<summary>Compute Resource Strategy (Free → Low-cost → Credits)</summary>

- [Compute Resource Strategy](#compute)

</details>

<details>
<summary>Study Schedule, Success Strategies, and Milestones</summary>

- [Study Schedule, Success Strategies, and Milestones](#schedule)

</details>

<details>
<summary>Resources Summary</summary>

- [Resources Summary](#resources)

</details>

---

<a id="phase-1"></a>

## Phase 1: Foundations & Computer Vision (Weeks 1–12)

<a id="week-1"></a>
### Week 1: AI Foundations & Neural Networks 🔹
**Study Time:** ~15–18h
**Objectives:** neural net intuition, dev environment, first MLP

Theory:
- 🔹 3Blue1Brown Neural Networks (visual intuition) — 4h: [https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- 🧩 Fast.ai Lesson 1 (practical intro) — 2h: [https://course.fast.ai/Lessons/lesson1.html](https://course.fast.ai/Lessons/lesson1.html)
- 🧩 MIT 6.034 Neural Networks (theory) — 2h: [https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/video_galleries/lecture-videos/](https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/video_galleries/lecture-videos/)

Practical:
- 🔹 PyTorch 60-min Blitz: [https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- 🔹 Implement perceptron and MLP (NumPy) for XOR
- 🧩 Gradio/Streamlit quick UI: [https://www.gradio.app/](https://www.gradio.app/) | [https://streamlit.io/](https://streamlit.io/)

Deliverables:
- NumPy MLP + PyTorch re-implementation, repo initialized with README

💸: CPU ok; get Colab/Kaggle accounts set up.

[Back to Contents](#contents)

---

<a id="week-2"></a>
### Week 2: Python Mastery & Dev Tools 🔹
**Study Time:** ~12–20h
**Objectives:** Python for ML, Git/GitHub, reproducible env

Theory/Practical:
- 🔹 Real Python advanced features (OOP, decorators, generators): [https://realpython.com/learning-paths/python-fundamentals/](https://realpython.com/learning-paths/python-fundamentals/)
- 🔹 Git/GitHub tutorial: [https://www.youtube.com/watch?v=RGOj5yH7evk](https://www.youtube.com/watch?v=RGOj5yH7evk)
- 🧩 Google ML Crash Course (Python sections): [https://developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)
- 🧩 Matplotlib gallery: [https://matplotlib.org/stable/gallery/index.html](https://matplotlib.org/stable/gallery/index.html)

Deliverables:
- Data analysis pipeline (NumPy/Matplotlib), GitHub repo with branches/PRs, devcontainer/conda/venv

💸: CPU only; focus on tooling and hygiene.

[Back to Contents](#contents)

---

<a id="week-3"></a>
### Week 3: PyTorch Deep Dive & Cloud Training 🔹
**Study Time:** ~18–22h
**Objectives:** tensors/autograd, custom Dataset/DataLoader, GPU basics

Theory/Practical:
- 🔹 PyTorch Tutorials (fundamentals): [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
- 🔹 PyTorch Examples: [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
- 🧩 Colab guide: [https://colab.research.google.com/notebooks/intro.ipynb](https://colab.research.google.com/notebooks/intro.ipynb)

Assignments:
- Linear/logistic regression, feedforward net in PyTorch
- Custom dataset/dataloader; train small CNN on CIFAR-10

💸: Start using free GPU tiers; compare CPU vs GPU timings.

[Back to Contents](#contents)

---

<a id="week-4"></a>
### Week 4: First End-to-End Training Loop 🔹
**Study Time:** ~20h
**Objectives:** robust training loop, validation/test, augmentation, logging

Theory/Practical:
- 🔹 PyTorch training loop tutorial: [https://pytorch.org/tutorials/beginner/introyt/trainingyt.html](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- 🧩 DeepLearning.AI specialization (audit): [https://www.coursera.org/learn/deep-neural-network](https://www.coursera.org/learn/deep-neural-network)
- 🧩 Albumentations docs: [https://albumentations.ai/](https://albumentations.ai/)
- 🧩 PyTorch Lightning intro: [https://lightning.ai/docs/pytorch/stable/](https://lightning.ai/docs/pytorch/stable/)

Project:
- CIFAR-10 ResNet-18 with aug, LR scheduling, early stopping; W&B logging [https://wandb.ai/](https://wandb.ai/)

💸: Kaggle GPUs; checkpoint runs.

[Back to Contents](#contents)

---

<a id="week-5"></a>
### Week 5: CNNs & Backpropagation Deep Dive 🔹
**Study Time:** ~18–22h
**Objectives:** conv operations, manual backprop intuition, visualize grads

Theory:
- 🔹 Stanford CS231n CNN lectures: [http://cs231n.stanford.edu/schedule.html](http://cs231n.stanford.edu/schedule.html)
- 🧩 3Blue1Brown backprop: [https://www.youtube.com/watch?v=Ilg3gGewQ5U](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- 🧩 Convolution arithmetic: [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

Practical:
- Implement 2D conv + maxpool from scratch
- Build CNN for CIFAR-10; visualize feature maps and gradients

[Back to Contents](#contents)

---

<a id="week-6"></a>
### Week 6: Advanced CNN Architectures 🔹
**Study Time:** ~22–25h
**Objectives:** ResNet/DenseNet/EfficientNet, transfer learning

Theory/Practical:
- 🔹 ResNet paper: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
- 🧩 EfficientNet paper: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
- 🔹 timm models: [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- 🔹 Transfer learning tutorial: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

Assignments:
- Implement ResNet blocks from scratch
- Compare multiple CNNs on ImageNet subset

[Back to Contents](#contents)

---

<a id="week-7"></a>
### Week 7: Training Optimization & Regularization 🔹
**Study Time:** ~20–24h
**Objectives:** optimizers, schedulers, BN, regularization, monitoring

Theory/Practical:
- 🔹 DL Book Ch.8 (optimization): [https://www.deeplearningbook.org/contents/optimization.html](https://www.deeplearningbook.org/contents/optimization.html)
- 🧩 Adam paper: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
- 🧩 BatchNorm paper: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
- 🧩 LR finder: [https://github.com/davidtvs/pytorch-lr-finder](https://github.com/davidtvs/pytorch-lr-finder)

Assignments:
- Implement/compare Adam, AdamW, RMSprop, SAM/Lion 🧩
- LR scheduling experiments; dashboard for monitoring

[Back to Contents](#contents)

---

<a id="week-8"></a>
### Week 8: One Cycle Policy & Advanced Training 🔹
**Study Time:** ~18–22h
**Objectives:** OneCycleLR, gradient clipping, stability, scaling awareness

Theory/Practical:
- 🔹 OneCycleLR (PyTorch): [https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- 🧩 Leslie Smith cyclical LR: [https://arxiv.org/abs/1506.01186](https://arxiv.org/abs/1506.01186)
- 🧩 One Cycle policy: [https://arxiv.org/abs/1708.07120](https://arxiv.org/abs/1708.07120)

Project:
- Train CIFAR-10 to ≥94% in <100 epochs with OneCycleLR

[Back to Contents](#contents)

---

<a id="week-9"></a>
### Week 9: ImageNet-Scale & Distributed Training 🔹 🚀
**Study Time:** ~25–30h
**Objectives:** DDP, data loading optimizations, mixed precision

Theory/Practical:
- 🔹 PyTorch DDP tutorial: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- 🧩 Large minibatch ImageNet best practices: [https://arxiv.org/abs/1812.01187](https://arxiv.org/abs/1812.01187)
- 🧩 Mixed precision (AMP): [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)

Projects:
- 🔹💸 Train ResNet-50 on ImageNet-100/Imagenette with AMP + grad accumulation
- 🚀🧩 Full ImageNet from scratch with multi-GPU (document costs and infra)

💸: Use subsets, multiple sessions with checkpointing, and credits (see strategy).

[Back to Contents](#contents)

---

<a id="week-10"></a>
### Week 10: Computer Vision Applications 🧩
**Study Time:** ~20h
Object detection, segmentation, real-time demos

Resources:
- YOLO paper: [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640) | YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- Detectron2: [https://detectron2.readthedocs.io/](https://detectron2.readthedocs.io/)
- U-Net: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

[Back to Contents](#contents)

---

<a id="week-11"></a>
### Week 11: Generative Models Intro (VAE/GAN) 🧩
**Study Time:** ~22h
Resources:
- VAE paper: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114) | PyTorch VAE: [https://github.com/pytorch/examples/tree/main/vae](https://github.com/pytorch/examples/tree/main/vae)
- GAN paper: [https://arxiv.org/abs/1406.2661](https://arxiv.org/abs/1406.2661) | DCGAN: [https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

[Back to Contents](#contents)

---

<a id="week-12"></a>
### Week 12: CV Capstone 🧩
Pick one domain project (traffic analysis, medical imaging, agriculture, retail). Include custom data, deployment, and evaluation.

[Back to Contents](#contents)

---

[Back to top](#top)

<a id="phase-2"></a>

## Phase 2: Transformers & LLMs (Weeks 13–18)

<a id="week-13"></a>
### Week 13: Transformer Architecture Deep Dive 🔹
**Study Time:** ~20–25h
**Objectives:** attention, multi-head attention, positional encodings

Resources:
- 🔹 Attention Is All You Need: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- 🔹 Illustrated Transformer: [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- 🧩 Annotated Transformer: [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
- 🧩 3Blue1Brown on attention: [https://www.youtube.com/watch?v=eMlx5fFNoYc](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- 🧩 PyTorch transformer tutorial: [https://pytorch.org/tutorials/beginner/transformer_tutorial.html](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

Assignments:
- Implement multi-head attention + positional encodings from scratch
- Build encoder–decoder; visualization of attention maps

[Back to Contents](#contents)

---

<a id="week-14"></a>
### Week 14: Embeddings & Tokenization 🔹
**Study Time:** ~20h
Resources:
- Word2Vec: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781) | HF embeddings tutorial: [https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
- BPE/SentencePiece: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909) | [https://arxiv.org/abs/1808.06226](https://arxiv.org/abs/1808.06226)
- HF Tokenizers: [https://huggingface.co/docs/tokenizers/python/latest/](https://huggingface.co/docs/tokenizers/python/latest/)
- 🧩 Karpathy tokenizer video: [https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE)

Assignments:
- Train BPE tokenizer on corpus; embeddings; semantic search mini-project

[Back to Contents](#contents)

---

<a id="week-14b"></a>
### Week 14b: Retrieval-Augmented Generation (RAG) 🧩
**Study Time:** ~18h
Resources:
- LangChain RAG tutorial: [https://python.langchain.com/docs/tutorials/rag/](https://python.langchain.com/docs/tutorials/rag/)
- A Simple Guide to RAG: [https://www.manning.com/books/a-simple-guide-to-retrieval-augmented-generation](https://www.manning.com/books/a-simple-guide-to-retrieval-augmented-generation)

Projects:
- Build a RAG pipeline with multiple retrievers and reranking; add context compression and query optimization

[Back to Contents](#contents)

---

<a id="week-15"></a>
### Week 15: LLM Training (GPT) 🔹 🚀
**Study Time:** ~28h
Resources:
- GPT/GPT-2 papers: [https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Chinchilla scaling: [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- nanoGPT: [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- Karpathy’s GPT from scratch: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Projects:
- 🔹💸 Train 20–50M param toy GPT on TinyStories/Wiki subset with AMP + grad accumulation
- 🚀🧩 125M+ params pretraining; document infra and costs

[Back to Contents](#contents)

---

<a id="week-16"></a>
### Week 16: LLM Optimization & Evaluation 🔹
**Study Time:** ~25h
Resources:
- Megatron-LM: [https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053) | ZeRO: [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)
- DeepSpeed tutorials: [https://www.deepspeed.ai/tutorials/](https://www.deepspeed.ai/tutorials/)
- HF Accelerate: [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
- HELM: [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110) | OpenAI Evals: [https://github.com/openai/evals](https://github.com/openai/evals)

Assignments:
- Mixed precision, grad accumulation, model parallelism basics
- Build evaluation suite (perplexity, GLUE/SuperGLUE where applicable)

[Back to Contents](#contents)

---

<a id="week-17"></a>
### Week 17: Supervised Fine-tuning & Instruction Tuning 🔹
**Study Time:** ~24h
Resources:
- InstructGPT: [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- PEFT/LoRA: [https://github.com/huggingface/peft](https://github.com/huggingface/peft) | [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA paper: [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

Assignments:
- Fine-tune a pretrained model (SFT) on instruction data; implement LoRA and compare with full FT

[Back to Contents](#contents)

---

<a id="week-18"></a>
### Week 18: Quantization & Model Compression 🔹
**Study Time:** ~22–26h
Resources:
- PyTorch Quantization/QAT: [https://pytorch.org/docs/stable/quantization.html](https://pytorch.org/docs/stable/quantization.html)
- BitsAndBytes (8-bit): [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- AutoGPTQ: [https://github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- Knowledge Distillation: [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
- 🧩 QAT deep dive (TorchAO/IBM guides)

Project:
- Implement PTQ + QAT on transformer; compare FP32/FP16/INT8 latency & accuracy

[Back to Contents](#contents)

---

[Back to top](#top)

<a id="phase-3"></a>

## Phase 3: Advanced Applications (Weeks 19–22)

<a id="week-19"></a>
### Week 19: Vision–Language Models (CLIP) 🔹
**Study Time:** ~20–26h
Resources:
- CLIP paper: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- OpenCLIP: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- 🧩 DALL·E: [https://arxiv.org/abs/2102.12092](https://arxiv.org/abs/2102.12092), Flamingo: [https://arxiv.org/abs/2204.14198](https://arxiv.org/abs/2204.14198)

Assignments:
- Implement CLIP-style contrastive training; zero-shot classification; VQA 🧩

[Back to Contents](#contents)

---

<a id="week-20"></a>
### Week 20: Reinforcement Learning Fundamentals 🧩
Resources:
- Sutton & Barto (Ch.1–6): [http://incompleteideas.net/book-the-book-2nd.html](http://incompleteideas.net/book-the-book-2nd.html)
- David Silver Lectures: [https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
- OpenAI Spinning Up: [https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)

Assignments:
- Tabular Q-learning/SARSA; DQN via Stable-Baselines3 (Optional)

[Back to Contents](#contents)

---

<a id="week-21"></a>
### Week 21: Advanced RL (PPO/SAC/TD3/DDPG) 🧩
Resources:
- PPO: [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347), SAC: [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)
- CleanRL: [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- Ray RLlib: [https://docs.ray.io/en/latest/rllib/](https://docs.ray.io/en/latest/rllib/)

[Back to Contents](#contents)

---

<a id="week-22"></a>
### Week 22: RLHF & Alignment 🔹
Resources:
- HF RLHF Guide: [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf) | TRL: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- Reward Modeling: [https://arxiv.org/abs/1909.12917](https://arxiv.org/abs/1909.12917)
- Constitutional AI: [https://arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)

Project:
- Implement reward model + PPO loop for a small SFT model; compare RLHF vs SFT

[Back to Contents](#contents)

---

### Optional Modules for Phase 3 (choose per interest and time)

- Advanced Multi-Modal Architectures 🧩
  - Resources: CMU MultiModal ML [https://cmu-mmml.github.io/spring2023/](https://cmu-mmml.github.io/spring2023/), DeepLearning.AI multimodal short course [https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/](https://www.deeplearning.ai/short-courses/building-multimodal-search-and-rag/), Awesome Multimodal ML [https://github.com/pliang279/awesome-multimodal-ml](https://github.com/pliang279/awesome-multimodal-ml)
  - Projects: Multimodal transformer (text+image+audio); cross-modal retrieval; multimodal RAG

- Diffusion Models 🧩
  - Resources: Stanford CS236 (selected) [https://deepgenerativemodels.github.io/](https://deepgenerativemodels.github.io/), Diffusion overview [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - Projects: DDPM from scratch; conditional/latent diffusion; text-to-image guidance

- Meta-Learning & Few-Shot Adaptation 🧩
  - Resources: Stanford CS330 [https://cs330.stanford.edu/](https://cs330.stanford.edu/), Few-Shot on PwC [https://paperswithcode.com/task/few-shot-learning](https://paperswithcode.com/task/few-shot-learning)
  - Projects: MAML from scratch; few-shot adapters for LLMs; rapid domain adaptation

- AI Safety & Interpretability 🧩
  - Resources: Interpretable ML Class [https://interpretable-ml-class.github.io/](https://interpretable-ml-class.github.io/), Duke XAI Specialization [https://www.coursera.org/specializations/explainable-artificial-intelligence-xai](https://www.coursera.org/specializations/explainable-artificial-intelligence-xai)
  - Projects: Mechanistic interpretability for transformers; explanation tooling; red teaming and safety evals

---

[Back to top](#top)

<a id="phase-4"></a>

## Phase 4: Large-Scale Training & Production (Weeks 23–24)

<a id="week-23"></a>
### Week 23: Large-Scale Model Training 🚀
Resources:
- Megatron-DeepSpeed: [https://github.com/microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
- FairScale: [https://github.com/facebookresearch/fairscale](https://github.com/facebookresearch/fairscale)
- Colossal-AI: [https://github.com/hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI)
- HF parallelism guide: [https://huggingface.co/docs/transformers/perf_train_gpu_many](https://huggingface.co/docs/transformers/perf_train_gpu_many)

Assignments:
- 3D parallelism (data/tensor/pipeline) on 7B-class toy config [simulated/budgeted]
- Monitoring & logging for long runs; checkpointing strategy

💸: Use distillation and QLoRA if full pretraining isn’t feasible.

[Back to Contents](#contents)

---

<a id="week-24"></a>
### Week 24: Serving, Optimization, and MLOps 🔹 🚀
Resources:
- NVIDIA Triton Inference Server: [https://docs.nvidia.com/deeplearning/triton-inference-server/](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- TensorRT: [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) | Torch-TensorRT: [https://pytorch.org/TensorRT/](https://pytorch.org/TensorRT/)
- ONNX Runtime perf: [https://onnxruntime.ai/docs/performance/model-optimizations/](https://onnxruntime.ai/docs/performance/model-optimizations/)
- MLOps Zoomcamp: [https://github.com/DataTalksClub/mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- BentoML: [https://github.com/bentoml/BentoML](https://github.com/bentoml/BentoML) | Seldon Core: [https://github.com/SeldonIO/seldon-core](https://github.com/SeldonIO/seldon-core)

Projects:
- Optimize transformer for 5–10x inference speedup (quantization + TensorRT/ONNX)
- Deploy via Triton/BentoML; add monitoring, drift detection, and CI/CD

[Back to Contents](#contents)

---

[Back to top](#top)

<a id="phase-5"></a>

## Phase 5: Capstone & Portfolio (Weeks 25–26) 🔹

Pick one comprehensive project and deliver production-grade assets:

Options:
1) Multimodal AI Assistant (text+image+voice), with RAG and RLHF
2) LLM From Scratch (domain-specific): pretrain → SFT → RLHF → quantize → serve
3) Production Recommender or CV system with A/B testing and autoscaling
4) Research Reproduction + Extension (novel contribution)

Final deliverables:
- Complete codebase with tests and docs
- Live demo (HF Spaces/Cloud) + API endpoint
- Technical report (15–20 pages) + video walkthrough (10–15 min)
- Benchmarks and ablation studies

---

[Back to top](#top)

<a id="compute"></a>

## Compute Resource Strategy 💸

Tier 1 (Free):
- Google Colab Free (T4/K80/P100), 12h sessions
- Kaggle Kernels (30–41h/week, P100), 9h sessions
- Paperspace Free (limited)

Tier 2 (Low-cost):
- Colab Pro (~$10/mo), longer sessions, better GPUs
- Paperspace Pro (~$8/mo) A4000
- RunPod serverless (~$0.19–3.19/h)

Tier 3 (Credits — apply early, Week 1):
- AWS Research Credits (up to ~$5k)
- Google Cloud Research (student/faculty tiers)
- Azure for Students ($100)

Heavy weeks guidance:
- Week 9 (ImageNet): use ImageNet-100/Imagenette; AMP + grad accumulation; multiple sessions + checkpointing
- Week 15 (LLM pretraining): 20–50M param toy models; stream datasets; mixed precision; resumeable training
- Week 23 (7B+ systems): prioritize distillation and QLoRA; focus on infra + correctness over raw scale

---

[Back to top](#top)

<a id="schedule"></a>

## Study Schedule & Success Strategies

Weekly structure (~20h avg):
- Theory/Research (5h), Hands-on (10h), Community (2h), Docs/Reflection (3h)

Success tips:
1) Apply for credits in Week 1; approvals can take 90–120 days
2) Checkpoint all long runs; automate resume
3) Implement from scratch first; then lean on libraries
4) Track experiments (W&B) and write weekly notes
5) Engage with communities (Discord, forums); ask/answer questions

---

[Back to top](#top)

## Success Metrics & Milestones

- Week 6: Train ResNet-50 on ImageNet subset with documented pipeline
- Week 10–12: Deployed CV app or capstone-lite
- Week 15–16: Train and evaluate a small GPT; implement mixed precision + accumulation
- Week 18: QAT/quantization results with latency/accuracy trade-offs
- Week 19: Working CLIP zero-shot system
- Week 24: 5–10x inference speedup + production deployment
- Week 26: Portfolio-ready capstone with live demo and report

---

[Back to top](#top)

<a id="resources"></a>

## Resources Summary

Reference Course Syllabus:
- [ERA V4 Course Syllabus (PDF)](ERA+V4+Course+Syllabus.pdf)

Primary platforms:
- Fast.ai, 3Blue1Brown, Hugging Face, Papers with Code, Google Colab, Kaggle

Essential tools:
- PyTorch, Transformers, W&B, Docker, Git/GitHub, Streamlit/Gradio
- DeepSpeed, Accelerate, PEFT, BitsAndBytes, AutoGPTQ
- Triton, TensorRT/Torch-TensorRT, ONNX Runtime, BentoML, Seldon Core

Reference implementations:
- PyTorch Examples, nanoGPT, OpenCLIP, Megatron-DeepSpeed, FairScale, Colossal-AI

---

## Notes on Optionality & Advanced Paths

- All 🧩 items preserve breadth from the comprehensive and advanced plans without bloating the core track.
- 🚀 items aggregate the production/deployment/scaling focus from `plan-sonnet4.md`.
- 💸 fallbacks consolidate the compute guidance from `plan basic.md` and are referenced where runs are heavy.

This consolidated plan retains all core topics and resources from the three originals, with clear markings so you can tailor depth and compute to your context without losing information.

# End
