# Free ERA V4 Alternative Study Plan

## Contents

- [Course Overview and Philosophy](#course-overview-and-philosophy)
- [Phase 1: Foundations and Neural Networks (Weeks 1-6)](#phase-1-foundations-and-neural-networks-weeks-1-6)
- [Phase 2: Advanced Vision and Training (Weeks 7-9)](#phase-2-advanced-vision-and-training-weeks-7-9)
- [Phase 3: Transformers and LLM Foundations (Weeks 10-13)](#phase-3-transformers-and-llm-foundations-weeks-10-13)
- [Phase 4: Advanced LLM Techniques (Weeks 14-15)](#phase-4-advanced-llm-techniques-weeks-14-15)
- [Phase 5: Reinforcement Learning (Weeks 16-18)](#phase-5-reinforcement-learning-weeks-16-18)
- [Phase 6: Capstone and Advanced Implementation (Weeks 19-20)](#phase-6-capstone-and-advanced-implementation-weeks-19-20)
- [Free Compute Resource Strategy](#free-compute-resource-strategy)
- [Study Schedule and Time Management](#study-schedule-and-time-management)

The ERA V4 course from The School of AI represents one of the most ambitious educational programs in deep learning, uniquely training students to build 70B parameter models from scratch. This comprehensive 20-week study plan provides free alternatives that match ERA V4's progression and hands-on philosophy, focusing on **real implementation over theory** and **production-scale techniques**.

## Course Overview and Philosophy

ERA V4's revolutionary approach combines foundational learning with unprecedented scale - full ImageNet training, complete quantization-aware training (not shortcuts like LoRA), and end-to-end 70B model pretraining. This study plan maintains that ambitious scope using carefully selected free resources, efficient training techniques, and creative compute solutions.

**Total Time Commitment**: 15-25 hours per week over 20 weeks (300-500 hours total)
**Hardware Requirements**: Start with free GPU tiers, progress to cloud credits and efficient training methods
**Expected Outcome**: Production-level AI engineering skills typically found only in advanced research labs

## Phase 1: Foundations and Neural Networks (Weeks 1-6)

### Week 1: Introduction to AI, Neural Networks and Development Tools
**Learning Objectives**: Build foundational understanding and set up complete development environment

**Primary Resources**:
- **3Blue1Brown Neural Networks Series** (4 hours): Visual foundation for neural network intuition
- **Andrew Ng's Machine Learning Specialization** (start Course 1): Modern Python-based introduction
- **Development Setup**: Google Colab Pro setup, Kaggle account, GitHub workflow

**Hands-on Exercises**:
- Implement perceptron from scratch in NumPy
- Build simple neural network for XOR problem
- Set up cloud development environment with GPU access

**Time Commitment**: 15-18 hours
**Compute Requirements**: CPU sufficient, begin familiarizing with Colab interface

### Week 2: Python Essentials, Version Control, and Web Development Basics
**Learning Objectives**: Master Python ecosystem for AI/ML and establish professional development practices

**Primary Resources**:
- **DeepLearning.AI - AI Python for Beginners** (6 hours): AI-assisted Python programming
- **Google ML Crash Course** - Python sections (8 hours): NumPy, Pandas, TensorFlow basics
- **freeCodeCamp Git/GitHub Course** (selected sections): Professional version control

**Hands-on Exercises**:
- Build custom recipe generator using Python and APIs
- Create personal GitHub portfolio with first ML project
- Deploy simple web app using Streamlit

**Time Commitment**: 16-20 hours
**Compute Requirements**: CPU, focus on environment setup and tooling

### Week 3: PyTorch Fundamentals and Cloud Training
**Learning Objectives**: Deep mastery of PyTorch framework and cloud-based training workflows

**Primary Resources**:
- **PyTorch Official Tutorials** - "Deep Learning with PyTorch: A 60 Minute Blitz" (3 hours)
- **PyTorch Beginner Tutorial Series** (GitHub: yunjey/pytorch-tutorial) (8 hours)
- **AWS/GCP Free Tier Setup**: Apply for academic credits, understand cloud training basics

**Hands-on Exercises**:
- Implement linear regression, logistic regression, and feedforward network in PyTorch
- Train first model on cloud (Google Colab)
- Build custom dataset and dataloader from scratch

**Time Commitment**: 18-22 hours  
**Compute Requirements**: Start using free GPU tiers regularly

### Week 4: Building First Neural Network and Training on Cloud
**Learning Objectives**: End-to-end neural network implementation with cloud-based training

**Primary Resources**:
- **Andrej Karpathy's "Neural Networks: Zero to Hero"** - Micrograd episode (2.5 hours)
- **PyTorch Lightning Tutorial** (4 hours): Professional training loops and logging
- **Kaggle Learn - Deep Learning Course** (4 hours): Practical implementation focus

**Hands-on Exercises**:
- Build neural network from scratch following micrograd approach
- Implement MNIST classifier with proper train/validation/test splits
- Set up experiment tracking with Weights & Biases (free tier)

**Time Commitment**: 20-24 hours
**Compute Requirements**: GPU required for larger experiments, Kaggle 30h/week quota

### Week 5: CNNs and Backpropagation
**Learning Objectives**: Deep understanding of convolutional operations and gradient flow

**Primary Resources**:
- **Stanford CS231n Lectures** - CNN Architecture (3 hours video + notes)
- **Andrej Karpathy's Backpropagation Tutorial** (2 hours)
- **3Blue1Brown Backpropagation Video** (17 minutes): Mathematical intuition

**Hands-on Exercises**:
- Implement 2D convolution and max pooling from scratch
- Build CNN for CIFAR-10 with manual backpropagation tracking
- Visualize feature maps and gradients throughout network

**Time Commitment**: 18-22 hours
**Compute Requirements**: GPU recommended for CIFAR-10 experiments

### Week 6: In-Depth CNN Coding Practice
**Learning Objectives**: Extensive hands-on CNN implementation and optimization

**Primary Resources**:
- **Fast.ai Practical Deep Learning** - Lesson 1-2 (6 hours): State-of-the-art CNN training
- **PyImageSearch CNN Tutorial** (3 hours): Production-ready implementation patterns
- **CS231n Assignment 2** (CNN implementation): Academic rigor with full solutions available

**Hands-on Exercises**:
- Implement ResNet blocks from scratch
- Train state-of-the-art image classifier using transfer learning
- Experiment with data augmentation and regularization techniques

**Time Commitment**: 22-25 hours
**Compute Requirements**: Consistent GPU access needed, consider upgrading to paid tiers

## Phase 2: Advanced Vision and Training (Weeks 7-9)

### Week 7: Advanced CNN Architectures and Training
**Learning Objectives**: Master modern CNN architectures and training techniques

**Primary Resources**:
- **Stanford CS231n** - Architecture lectures and readings (5 hours)
- **Fast.ai Part 2** - CNN foundations from scratch (8 hours)
- **Papers with Code** - ResNet, EfficientNet, Vision Transformer implementations

**Hands-on Exercises**:
- Implement ResNet-50 from scratch in PyTorch
- Compare performance of different architectures on custom dataset
- Analyze computational efficiency and parameter counts

**Time Commitment**: 20-24 hours
**Compute Requirements**: Multi-hour training sessions, GPU essential

### Week 8: One Cycle Policy and CoreSet Training
**Learning Objectives**: Advanced optimization techniques and data efficiency methods

**Primary Resources**:
- **Fast.ai Lesson** on One Cycle Policy (2 hours)
- **CRAIG CoreSets Paper** + implementation study (4 hours)
- **Leslie Smith's Papers** on learning rate scheduling (research reading)

**Hands-on Exercises**:
- Implement One Cycle Policy from scratch
- Apply CoreSets methodology to reduce CIFAR-10 training data by 50%
- Compare training efficiency with and without data efficiency techniques

**Time Commitment**: 18-22 hours
**Compute Requirements**: Extended training experiments, consider cloud credits

### Week 9: Multi-GPU Training of ResNet from Scratch on Full ImageNet
**Learning Objectives**: Production-scale distributed training (ERA V4's signature achievement)

**Primary Resources**:
- **PyTorch Lightning Multi-GPU Tutorial** (6 hours)
- **ImageNet Training Papers** - "Accurate, Large Minibatch SGD" (research study)
- **Distributed Training Best Practices** - PyTorch documentation (4 hours)

**Hands-on Exercises**:
- **Major Project**: Train ResNet-18 on ImageNet subset using distributed training
- Implement gradient accumulation to simulate large batch training on single GPU
- Use efficient data loading and mixed precision training

**Time Commitment**: 25-30 hours (includes long training runs)
**Compute Requirements**: **Critical Week** - requires substantial GPU resources or creative solutions using free tiers

**Note**: This week requires the most compute resources. Free alternatives: 
- Use ImageNet subset (10% of data) 
- Multiple Colab sessions with checkpointing
- Apply for AWS/GCP research credits before this week

## Phase 3: Transformers and LLM Foundations (Weeks 10-13)

### Week 10: Introduction to Transformers and Emergent Abilities in LLMs
**Learning Objectives**: Deep understanding of transformer architecture and attention mechanisms

**Primary Resources**:
- **Stanford CS224n** - Transformer lectures (4 hours video)
- **Transformer Explainer** (interactive): Live GPT-2 visualization in browser
- **DeepLearning.AI - How Transformer LLMs Work** (2 hours): Hands-on attention implementation

**Hands-on Exercises**:
- Implement attention mechanism from scratch
- Build complete transformer encoder/decoder
- Analyze attention patterns in pretrained models

**Time Commitment**: 18-22 hours
**Compute Requirements**: Moderate GPU usage for transformer experiments

### Week 11: Embeddings, Tokenization, and CoreSets
**Learning Objectives**: Master text preprocessing and efficient data representation

**Primary Resources**:
- **Andrej Karpathy's Tokenizer Video** (2.5 hours): Build BPE tokenizer from scratch  
- **Hugging Face Tokenizers Documentation** (4 hours): Production tokenizer training
- **Word2Vec and Modern Embeddings Tutorial** (3 hours)

**Hands-on Exercises**:
- Train custom BPE tokenizer on domain-specific corpus
- Implement word embeddings from scratch using skip-gram
- Apply CoreSets to reduce training data for language models

**Time Commitment**: 16-20 hours
**Compute Requirements**: CPU sufficient for most tokenization work

### Week 12: Transformer Architectures, Multi-Head Attention and LLM Training
**Learning Objectives**: Complete transformer implementation and small-scale LLM training

**Primary Resources**:
- **Andrej Karpathy's "Let's build GPT"** (2 hours): From-scratch GPT implementation
- **nanoGPT Repository Study** (4 hours): Clean, educational GPT implementation
- **Hugging Face NLP Course** - Transformer section (6 hours)

**Hands-on Exercises**:
- Implement multi-head attention and positional encoding
- Train GPT-2 Small (124M parameters) from scratch on OpenWebText
- Fine-tune pretrained transformers for downstream tasks

**Time Commitment**: 22-26 hours
**Compute Requirements**: Significant GPU time for LLM training, use efficient techniques

### Week 13: Optimization, RoPE, CoreSets and LLM Evaluations
**Learning Objectives**: Advanced LLM techniques and comprehensive evaluation methods

**Primary Resources**:
- **RoPE (Rotary Position Embedding) Paper** + implementations (3 hours)
- **LLM Evaluation Benchmarks** - GLUE, SuperGLUE, BIG-bench (4 hours)
- **Advanced Optimization for LLMs** - AdamW, learning rate schedules (3 hours)

**Hands-on Exercises**:
- Implement RoPE and compare with standard positional encoding
- Evaluate trained models on multiple benchmarks
- Apply advanced optimization techniques to improve training stability

**Time Commitment**: 20-24 hours
**Compute Requirements**: Extended evaluation runs, consider batch processing

## Phase 4: Advanced LLM Techniques (Weeks 14-15)

### Week 14: Full Quantization-Aware Training (Real QAT, not LoRA shortcuts)
**Learning Objectives**: Production-grade model compression and deployment optimization

**Primary Resources**:
- **PyTorch QAT Official Tutorial** (4 hours): Complete QAT workflow
- **TorchAO QAT API Documentation** (3 hours): Production-ready implementation
- **IBM QAT Guide** (2 hours): Theory and straight-through estimator

**Hands-on Exercises**:
- **Major Project**: Implement full QAT on transformer model (not PEFT shortcuts)
- Compare accuracy and inference speed: FP32 vs FP16 vs INT8
- Deploy quantized model with measurable latency improvements

**Time Commitment**: 24-28 hours
**Compute Requirements**: Extensive training and evaluation cycles

**Note**: This week differentiates from typical courses that only teach LoRA/PEFT. Focus on real quantization-aware training as done in production systems.

### Week 15: CLIP and Vision-Language Models
**Learning Objectives**: Multi-modal AI systems and cross-modal understanding

**Primary Resources**:
- **OpenAI CLIP Paper** + official implementation (4 hours)
- **Hugging Face CLIP Documentation** (3 hours): Production usage patterns
- **CLIP from Scratch Tutorial** (5 hours): Complete implementation walkthrough

**Hands-on Exercises**:
- Implement CLIP architecture from scratch
- Train small CLIP model on custom image-text dataset
- Build zero-shot classification system using pretrained CLIP

**Time Commitment**: 20-24 hours
**Compute Requirements**: Multi-modal training requires substantial GPU resources

## Phase 5: Reinforcement Learning (Weeks 16-18)

### Week 16: Reinforcement Learning 101
**Learning Objectives**: RL fundamentals and basic algorithm implementation

**Primary Resources**:
- **David Silver's RL Course** - Lectures 1-5 (6 hours video)
- **OpenAI Spinning Up** - Introduction and VPG implementation (8 hours)
- **DeepMind RL Course** - Selected lectures (4 hours)

**Hands-on Exercises**:
- Implement tabular Q-learning for grid world
- Build policy gradient algorithm (VPG) from scratch
- Train agent on classic control problems (CartPole, MountainCar)

**Time Commitment**: 18-22 hours
**Compute Requirements**: CPU sufficient for basic RL, GPU helpful for deep RL

### Week 17: Continuous Action Spaces and Advanced RL
**Learning Objectives**: Advanced RL algorithms for complex environments

**Primary Resources**:
- **David Silver's RL Course** - Lectures 6-10 (6 hours)
- **OpenAI Spinning Up** - DDPG, TD3, SAC implementations (10 hours)
- **Berkeley CS294-112** - Selected advanced lectures (4 hours)

**Hands-on Exercises**:
- Implement DDPG for continuous control
- Train agents on MuJoCo environments (or alternatives)
- Compare policy-based vs value-based methods empirically

**Time Commitment**: 22-26 hours
**Compute Requirements**: GPU recommended for complex environments

### Week 18: RLHF, GPO and Instruction Fine-Tuning for LLMs
**Learning Objectives**: Cutting-edge techniques for aligning language models with human preferences

**Primary Resources**:
- **DeepLearning.AI RLHF Course** (4 hours): Hands-on RLHF with Llama 2
- **Hugging Face RLHF Blog Post** (2 hours): Complete 3-step RLHF process
- **Chip Huyen's RLHF Guide** (3 hours): Practical implementation insights

**Hands-on Exercises**:
- **Major Project**: Implement complete RLHF pipeline on small language model
- Train reward model from human preference data  
- Apply PPO for language model fine-tuning

**Time Commitment**: 26-30 hours
**Compute Requirements**: Substantial resources for LLM + RL training combined

## Phase 6: Capstone and Advanced Implementation (Weeks 19-20)

### Week 19: Pretraining a 70B LLM End-to-End + Instruction Tuning
**Learning Objectives**: ERA V4's flagship achievement - training production-scale language models

**Primary Resources**:
- **Alternative Approach**: Use efficient training methods since 70B training requires massive resources
- **NVIDIA Minitron Distillation** (4 hours): Efficient large model training
- **QLoRA Implementation** (6 hours): Memory-efficient fine-tuning 
- **Model Parallelism Papers** (research study): Understanding large model training

**Hands-on Exercises**:
- **Capstone Project Option A**: Train 7B model using QLoRA techniques
- **Capstone Project Option B**: Implement model distillation from 70B to smaller model
- **Capstone Project Option C**: Multi-stage training pipeline with efficiency optimizations

**Free Resource Strategy for 70B Alternative**:
- Use model distillation from existing 70B models
- Apply QLoRA for efficient fine-tuning of large models
- Implement training pipeline that could scale to 70B with more resources
- Focus on the engineering and techniques rather than raw parameter count

**Time Commitment**: 30-35 hours
**Compute Requirements**: **Maximum resource week** - use all available free credits and creative solutions

### Week 20: Capstone Integration and Deployment
**Learning Objectives**: Complete project integration and real-world deployment

**Primary Resources**:
- **Hugging Face Spaces** (3 hours): Model deployment and demos
- **Fast.ai Deployment Lessons** (4 hours): Production considerations
- **MLOps Best Practices** (selected topics, 4 hours)

**Hands-on Exercises**:
- Complete capstone project with full documentation
- Deploy final model as interactive web application
- Create comprehensive GitHub portfolio showcasing all 20 weeks of work
- Present project to online AI community for feedback

**Time Commitment**: 25-30 hours
**Compute Requirements**: Focus on optimization and deployment rather than training

## Free Compute Resource Strategy

### Tier 1: Free Resources (Start Here)
- **Google Colab Free**: 12-hour sessions, shared GPUs (K80/T4/P100)
- **Kaggle Kernels**: 30-41 hours/week, Tesla P100, 9-hour sessions
- **Paperspace Gradient Free**: Limited GPU time, good for experiments

### Tier 2: Low-Cost Options ($10-50/month)
- **Google Colab Pro**: $10/month, better GPUs, longer sessions
- **Paperspace Pro**: $8/month, unlimited A4000 access
- **RunPod Community**: $0.19-3.19/hour, pay-per-use serverless

### Tier 3: Academic Credits (Apply Early!)
- **AWS Research Credits**: Up to $5,000 for students
- **Google Cloud Research**: Up to $1,000/year PhD, $5,000 faculty  
- **Azure for Students**: $100 free credits

### Creative Solutions for Resource-Intensive Weeks
**Week 9 (ImageNet Training)**:
- Train on ImageNet subset (10-20% of data)
- Use multiple Colab sessions with checkpointing
- Implement gradient accumulation to simulate large batches

**Week 19 (70B Model)**:
- Focus on distillation from existing large models
- Use QLoRA for parameter-efficient training
- Demonstrate understanding through smaller-scale implementation

## Study Schedule and Time Management

### Weekly Structure (20 hours average)
- **Theory and Research** (5 hours): Video lectures, paper reading
- **Hands-on Implementation** (10 hours): Coding, experimentation  
- **Community Engagement** (2 hours): Discord discussions, forums
- **Documentation and Reflection** (3 hours): GitHub updates, learning notes

### Success Strategies
1. **Join AI Communities Early**: Discord servers, Reddit communities for support
2. **Apply for Cloud Credits Week 1**: 90-120 day processing times
3. **Checkpoint Everything**: Free GPU sessions have time limits
4. **Focus on Understanding**: Implement from scratch before using libraries
5. **Document Journey**: Create portfolio showcasing 20-week progression

### Expected Learning Outcomes

By Week 20, you will have:
- **Production-Scale Training Experience**: Multi-GPU distributed training on large datasets
- **Complete LLM Pipeline Knowledge**: From tokenization through deployment
- **Advanced Optimization Techniques**: QAT, mixed precision, efficient training
- **Multi-Modal AI Capabilities**: Vision-language models and applications
- **RLHF Implementation Skills**: Cutting-edge alignment techniques
- **Professional AI Engineering Portfolio**: 20 weeks of documented projects

This study plan provides a comprehensive, free alternative to ERA V4 that maintains the course's ambitious scope and hands-on philosophy. The combination of world-class educational resources, creative compute solutions, and structured progression creates an educational experience comparable to the best paid programs in AI.

**Total Investment**: 300-500 hours over 20 weeks + $50-200 for compute resources
**Expected ROI**: Production-level AI engineering skills typically acquired only in advanced research labs or expensive bootcamps

The key to success lies in consistent execution, active community participation, and creative problem-solving when facing compute constraints. This path transforms motivated learners into AI engineers capable of training and deploying state-of-the-art models at scale.