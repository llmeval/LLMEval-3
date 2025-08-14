<div align="center">
<h2>LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models</h2>

[![Paper](https://img.shields.io/badge/Paper-Arxiv-blue.svg?style=for-the-badge)](https://arxiv.org/abs/2508.05452)


</div>

<div align="center">
<img src=".\pic\llmeval-logo.png" alt="llmeval-logo" style="zoom:50%;" />
</div>

> **Note:** For the Chinese version of this README, please refer to [README_zh.md](README_zh.md).

## 📚 Benchmark Content and Format

LLMEval-3 focuses on evaluating professional knowledge capabilities, covering 13 academic disciplines as defined by the Ministry of Education: Philosophy, Economics, Law, Education, Literature, History, Science, Engineering, Agriculture, Medicine, Military Science, Management, and Arts. It includes over 50 sub-disciplines and a total of approximately 200,000 standardized generative question-answering items (we will continue to expand the question bank to 1 million).

<div align="center">
<img src=".\pic\subjects.PNG" alt="Academic Disciplines Coverage" style="zoom:80%;" />
</div>

Question sources mainly include **undergraduate homework**, **undergraduate mid-term and final exams**, and **graduate entrance exams**. To prevent large models from being exposed to a significant portion of the evaluation data during pre-training, LLMEval-3 sources its questions from non-public channels where possible. The data is in PDF and Word formats, which undergo OCR and data cleaning before being formatted. A standardized interface is provided for different question types to enable a fully automated process for the models under test.

Unlike other knowledge benchmarks that use a multiple-choice format, LLMEval-3 treats all questions as **generative knowledge question-answering**. It includes a variety of formats such as short answer, calculation, true/false, analysis, and essay questions. Compared to standardized multiple-choice questions, the generative format used in LLMEval-3 better reflects real-world user needs and the language capabilities of the models.

## 🔬 Methodology

### Evaluation Pipeline

Preventing cheating is a key consideration for LLMEval-3. Existing public benchmarks suffer from test set leakage, which can lead to unfair practices like "leaderboard hacking" or score inflation. In LLMEval-3, each participating system must complete 1,000 questions randomly sampled from the total question bank. **For models from the same institution, we ensure that the questions are not repeated in subsequent evaluations**. The evaluation is conducted online, with questions sent sequentially in a single round; the next question is only sent after the previous one has been answered, preventing malicious crawling.

This round of evaluation uses an automated scoring method, with GPT-4 Turbo as the current evaluation model. Each question is scored on a scale of 0-3 points. The scoring focuses on the core correctness of the answer and the validity of the explanation, with core correctness being the primary metric. The evaluation prompt used is as follows:

```text
Please evaluate the following response from the LLM regarding a discipline-specific question based on the following criteria. You must score it on a scale of 0, 1, 2 or 3 stars:

Overall Rating:
0 stars indicate wrong answer with a wrong explanation
1 star indicates wrong answer but a partially reasonable explanation
2 stars indicate a correct answer with a partially reasonable explanation
3 stars indicate a correct answer with a reasonable explanation

User: {question}

LLM:{answer_from_llm}

The correct answer to user's question is: {correct_answer}

You must provide your feedback in the following format:
{"Overall Rating":numbers of its stars(int)}
```

### Scoring

To mitigate systematic bias introduced by randomly sampling 1,000 questions, LLMEval-3 uses both **relative scores** and **absolute scores**.

**Relative Score Calculation:**
Given the rapid development of large language model technology, we introduce a relative score to measure the gap between a model and the current state-of-the-art performance. We select the top-performing model on the leaderboard as the SOTA baseline, which is currently Doubao-1.5-Thinking-Pro:

$$R_{\text{SOTA}}^{\text{model}}=\frac{S_{model}}{S_\text{sotamodel}} \times 100 $$

**Absolute Score Calculation:**
The absolute score represents the model's raw performance on N=1,000 questions. It is calculated by normalizing each question's score (0-3 points) to a 0-100 scale:

$$S_{model}=\sum_{i=1}^N{\frac{s_i}{s_{max}} \times 100} \quad (1)$$

Where $s_i$ is the score for question i, and $s_{max}=3$.

**Scoring Notes:** $S_{model}$ is the absolute score (0-100 scale), $R_{\text{SOTA}}^{\text{model}}$ is the relative score (with the SOTA model as the 100% baseline), and discipline-specific scores use a 10-point scale.


## 🏆 Current Leaderboard (As of August 2025)

### 📋 Overall Scores

| Model Name | Organization | Access Type | Release Date | Relative Score | Absolute Score |
|------------|--------------|-------------|-----------------|----------------|----------------|
| Doubao-1.5-Thinking-Pro | ByteDance | API | 2025.4.15 | 100.00 | 93.67 |
| DeepSeek-R1 | DeepSeek | API | 2025.5.28 | 97.40 | 91.23 |
| Gemini-2.5-Pro-Preview | Google | API | 2025.6.5 | 97.22 | 91.07 |
| Gemini-2.5-Pro-Preview-Thinking | Google | API | 2025.6.5 | 97.15 | 91.00 |
| DeepSeek-V3 | DeepSeek | API | 2025.3.24 | 96.48 | 90.37 |
| Qwen3-235B | Alibaba Cloud | API | 2025.4.29 | 96.44 | 90.33 |
| Doubao-1.5-Pro-256K | ByteDance | API | 2025.1.15 | 95.69 | 89.63 |
| QwQ-32B | Alibaba Cloud | API | 2025.3.6 | 94.52 | 88.54 |
| GPT-5 | OpenAI | API | 2025.8.7 | 93.84 | 87.9 |
| O1-2024-12-17 | OpenAI | API | 2024.12.17 | 93.35 | 87.43 |
| Gemini-2.5-Flash-Thinking | Google | API | 2025.4.17 | 92.74 | 86.87 |
| Qwen3-32B | Alibaba Cloud | API | 2025.4.29 | 92.21 | 86.37 |
| Claude-Sonnet-4-Thinking | Anthropic | API | 2025.5.14 | 91.03 | 85.27 |
| Claude-Sonnet-4 | Anthropic | API | 2025.5.14 | 91.00 | 85.23 |
| GPT-4o-Search-Preview | OpenAI | API | 2024.11.20 | 89.40 | 83.73 |
| GLM-4-32B | Tsinghua&Zhipu.AI | API | 2025.4.14 | 88.43 | 82.83 |
| GPT-4o-2024-11-20 | OpenAI | API | 2024.11.20 | 88.08 | 82.50 |
| Gemini-1.5-Pro | Google | API | 2024.2.14 | 85.92 | 80.47 |
| Qwen2.5-32B-Instruct | Alibaba Cloud | API | 2024.9.19 | 85.07 | 79.68 |
| O3-Mini | OpenAI | API | 2025.1.29 | 84.13 | 78.80 |
| Qwen-Turbo-1101 | Alibaba Cloud | API | 2024.11.1 | 83.71 | 78.41 |
| Claude-3.5-Sonnet | Anthropic | API | 2024.10.22 | 83.38 | 78.10 |
| O1-Mini-2024-09-12 | OpenAI | API | 2024.9.12 | 78.93 | 73.93 |
| Claude-3-Haiku | Anthropic | API | 2024.3.7 | 62.95 | 58.97 |
| LLaMA-3.2-90B-Vision-Instruct | Meta | API | 2024.9.25 | 61.74 | 57.83 |
| LLaMA-3.3-70B | Meta | API | 2024.12.6 | 60.85 | 57.00 |
| Phi-3-Medium-128K-Instruct | Microsoft | API | 2024.5.3 | 36.94 | 34.60 |
| GPT-4 Turbo(gpt-4-1106-preview) | OpenAI | API | 2023.11.6 | 78.56 | 73.6 |
| GPT-4-0125-Preview | OpenAI | API | 2024.1.26 | 76.44 | 71.6 |
| Baidu-4.0 | Baidu | API | 2023.10.17 | 75.09 | 70.33 |
| Yi-34B-Chat | 01.AI | API | 2023.11.24 | 70.17 | 65.70 |
| Baidu-3.5 | Baidu | API | 2023.7.6 | 69.14 | 64.73 |
| ChatGLM-Pro | Tsinghua&Zhipu.AI | API | 2023.9.25 | 69.14 | 64.73 |
| GPT-4-0613 | OpenAI | API | 2023.6.13 | 66.17 | 61.97 |
| iFlytek Spark v3.0 | iFlytek | API | 2023.10.24 | 65.64 | 61.47 |
| Nanbeige-Plus | NanBeiGe LLM Lab | API | 2023.12.1 | 65.14 | 61.00 |
| Baichuan2-13B-Chat | Baichuan | Weights | 2023.9.6 | 58.31 | 54.6 |
| Gemini-Pro | Google | API | 2023.12.13 | 58.20 | 54.5 |
| Qwen-Plus | Alibaba Cloud | API | 2023.11.1 | 56.60 | 53.0 |
| Qwen-Turbo | Alibaba Cloud | API | 2023.9.1 | 55.78 | 52.23 |
| Nanbeige-16B | NanBeiGe LLM Lab | API | 2023.11.19 | 55.46 | 51.93 |
| GPT-3.5-Turbo | OpenAI | API | 2023.6.13 | 55.42 | 51.9 |
| MiniMax-Abab5 | MiniMax | Weights | 2023.8.31 | 55.33 | 51.83 |
| Mixtral-8x7B-Instruct | Mistral AI | Weights | 2023.12.11 | 51.69 | 48.4 |
| ChatGLM2-6B | Tsinghua&Zhipu.AI | Weights | 2023.6.25 | 42.32 | 39.63 |
| Ziya-v1.1-13B | IDEA | Weights | 2023.6.7 | 40.18 | 37.63 |
| InternLM-Chat-7B | Shanghai AI Lab&SenseTime | Weights | 2023.7.6 | 38.73 | 36.27 |
| Linly-Chinese-LLaMA-2-13B-HF | National Engineering Lab | Weights | 2023.7.25 | 37.06 | 34.7 |
| BELLE-LLaMA2-13B-Chat-0.4M | LianjiaTech | Weights | 2023.7.6 | 36.28 | 33.97 |
| LLaMA-2-7B-Chat-HF | Meta | Weights | 2023.7.18 | 25.24 | 23.63 |

### 📊 Discipline-Specific Performance

| Model Name | Overall | Engineering | Economics | Education | Law | Literature | Management | Science | History | Medicine | Military |
|------------|---------|-------------|-----------|-----------|-----|------------|------------|---------|---------|----------|----------|
| Doubao-1.5-Thinking-Pro | 93.67 | 9.47 | 9.67 | 9.43 | 9.77 | 8.93 | 9.53 | 9.23 | 9.70 | 8.97 | 8.97 |
| DeepSeek-R1 | 91.23 | 9.47 | 9.43 | 9.27 | 9.37 | 8.83 | 9.37 | 9.03 | 9.53 | 8.50 | 8.43 |
| Gemini-2.5-Pro-Preview | 91.07 | 9.20 | 9.47 | 9.20 | 9.30 | 8.43 | 9.63 | 9.07 | 9.40 | 8.50 | 8.87 |
| Gemini-2.5-Pro-Preview-Thinking | 91.00 | 9.13 | 9.50 | 9.37 | 9.47 | 8.40 | 9.63 | 9.20 | 9.27 | 8.30 | 8.73 |
| DeepSeek-V3 | 90.37 | 9.30 | 9.57 | 8.93 | 9.23 | 8.60 | 9.13 | 8.97 | 9.47 | 8.83 | 8.33 |
| Qwen3-235B | 90.33 | 9.23 | 9.43 | 9.03 | 9.50 | 8.23 | 9.43 | 8.97 | 9.17 | 8.73 | 8.60 |
| Doubao-1.5-Pro-256K | 89.63 | 8.83 | 9.03 | 9.13 | 9.43 | 8.57 | 9.27 | 8.83 | 9.10 | 8.60 | 8.83 |
| QwQ-32B | 88.54 | 8.30 | 9.46 | 9.23 | 9.33 | 7.83 | 9.46 | 8.65 | 9.27 | 8.57 | 8.43 |
| GPT-5 | 87.9 | 8.83 | 9.37 | 8.90 | 8.87 | 8.10 | 9.10 | 8.90 | 9.03 | 8.50 | 8.30 |
| O1-2024-12-17 | 87.43 | 8.90 | 9.30 | 8.67 | 8.77 | 7.73 | 9.27 | 8.90 | 8.97 | 8.17 | 8.77 |
| Gemini-2.5-Flash-Thinking | 86.87 | 8.67 | 9.27 | 8.70 | 9.00 | 7.80 | 8.93 | 8.90 | 9.00 | 8.03 | 8.57 |
| Qwen3-32B | 86.37 | 8.43 | 9.10 | 8.57 | 9.10 | 7.77 | 9.47 | 8.67 | 9.30 | 7.70 | 8.27 |
| Claude-Sonnet-4-Thinking | 85.27 | 8.57 | 9.00 | 8.63 | 8.73 | 7.57 | 9.10 | 8.93 | 8.70 | 7.97 | 8.07 |
| Claude-Sonnet-4 | 85.23 | 8.57 | 8.80 | 8.50 | 8.70 | 7.80 | 9.03 | 8.80 | 8.80 | 8.17 | 8.07 |
| GPT-4o-Search-Preview | 83.73 | 8.27 | 8.77 | 8.43 | 8.67 | 7.77 | 8.80 | 8.20 | 8.73 | 8.27 | 7.83 |
| GLM-4-32B | 82.83 | 7.77 | 8.97 | 8.33 | 8.33 | 7.03 | 9.13 | 8.27 | 8.77 | 8.23 | 8.00 |
| GPT-4o-2024-11-20 | 82.50 | 7.90 | 8.67 | 8.30 | 8.33 | 7.17 | 8.97 | 8.57 | 8.67 | 7.63 | 8.30 |
| Gemini-1.5-Pro | 80.47 | 8.13 | 8.45 | 8.30 | 8.37 | 7.04 | 8.17 | 8.43 | 8.50 | 7.48 | 7.60 |
| Qwen2.5-32B-Instruct | 79.68 | 7.70 | 8.57 | 8.33 | 8.33 | 6.70 | 8.50 | 8.17 | 7.70 | 7.60 | 8.08 |
| O3-Mini | 78.80 | 7.97 | 8.60 | 8.30 | 8.20 | 6.73 | 8.57 | 8.53 | 7.17 | 7.03 | 7.70 |
| Qwen-Turbo-1101 | 78.41 | 7.97 | 8.37 | 8.03 | 8.23 | 6.40 | 8.50 | 8.10 | 7.50 | 7.27 | 8.05 |
| Claude-3.5-Sonnet | 78.10 | 7.97 | 8.53 | 8.27 | 7.93 | 7.03 | 8.50 | 8.00 | 7.57 | 6.70 | 7.60 |
| O1-Mini-2024-09-12 | 73.93 | 7.27 | 8.43 | 7.90 | 7.53 | 6.27 | 8.27 | 8.17 | 6.43 | 6.63 | 7.03 |
| Claude-3-Haiku | 58.97 | 5.80 | 6.60 | 6.97 | 6.63 | 4.83 | 5.93 | 6.33 | 4.80 | 5.23 | 5.83 |
| LLaMA-3.2-90B-Vision-Instruct | 57.83 | 5.63 | 6.33 | 6.20 | 5.80 | 4.73 | 6.10 | 6.57 | 5.03 | 5.27 | 6.17 |
| LLaMA-3.3-70B | 57.00 | 5.80 | 6.90 | 5.63 | 5.70 | 5.47 | 5.70 | 6.30 | 4.70 | 4.87 | 5.93 |
| Phi-3-Medium-128K-Instruct | 34.60 | 2.27 | 4.17 | 3.70 | 4.23 | 2.87 | 4.50 | 3.57 | 3.20 | 2.27 | 3.83 |
| GPT-4 Turbo(gpt-4-1106-preview) | 73.6 | 6.97 | 8.17 | 8.33 | 7.8 | 6.0 | 7.57 | 8.13 | 7.0 | 6.43 | 7.2 |
| GPT-4-0125-Preview | 71.6 | 6.9 | 7.4 | 8.03 | 7.3 | 6.0 | 7.47 | 7.63 | 6.87 | 6.33 | 7.67 |
| Baidu-4.0 | 70.33 | 7.27 | 7.23 | 7.67 | 7.43 | 5.63 | 6.47 | 6.8 | 7.63 | 7.8 | 6.4 |
| Yi-34B-Chat | 65.70 | 5.77 | 6.63 | 7.37 | 7.53 | 5.47 | 5.77 | 5.47 | 7.47 | 6.3 | 7.93 |
| Baidu-3.5 | 64.73 | 6.2 | 6.7 | 7.8 | 6.83 | 5.2 | 5.5 | 6.0 | 7.23 | 6.57 | 6.7 |
| ChatGLM-Pro | 64.73 | 5.9 | 7.07 | 7.03 | 7.9 | 5.43 | 6.33 | 5.0 | 6.67 | 5.97 | 7.43 |
| GPT-4-0613 | 61.97 | 6.5 | 6.73 | 6.6 | 6.73 | 5.43 | 6.1 | 6.47 | 5.3 | 5.2 | 6.9 |
| iFlytek Spark v3.0 | 61.47 | 5.77 | 6.5 | 7.27 | 7.3 | 5.7 | 5.9 | 5.03 | 6.5 | 5.23 | 6.27 |
| Nanbeige-Plus | 61.00 | 5.78 | 5.57 | 6.77 | 7.37 | 5.37 | 5.93 | 5.45 | 6.3 | 5.67 | 6.77 |
| Baichuan2-13B-Chat | 54.6 | 4.47 | 5.53 | 7.4 | 6.9 | 4.63 | 4.8 | 4.33 | 6.23 | 4.6 | 5.7 |
| Gemini-Pro | 54.5 | 4.87 | 5.43 | 7.07 | 6.43 | 5.10 | 4.5 | 4.65 | 6.33 | 4.42 | 5.7 |
| Qwen-Plus | 53.0 | 4.4 | 5.1 | 6.53 | 6.53 | 5.0 | 4.77 | 4.87 | 5.17 | 5.13 | 5.5 |
| Qwen-Turbo | 52.23 | 4.1 | 6.07 | 6.63 | 6.43 | 4.43 | 4.53 | 4.97 | 5.27 | 4.37 | 5.43 |
| Nanbeige-16B | 51.93 | 4.37 | 5.3 | 6.5 | 6.3 | 3.97 | 4.7 | 4.07 | 5.9 | 4.73 | 6.1 |
| GPT-3.5-Turbo | 51.9 | 4.97 | 5.37 | 6.4 | 6.47 | 4.43 | 4.67 | 5.43 | 4.2 | 4.37 | 5.6 |
| MiniMax-Abab5 | 51.83 | 3.87 | 5.63 | 6.87 | 6.97 | 4.33 | 4.4 | 2.93 | 6.13 | 4.27 | 6.43 |
| Mixtral-8x7B-Instruct | 48.4 | 4.27 | 5.47 | 6.47 | 6.4 | 3.13 | 4.5 | 5.07 | 3.57 | 4.37 | 5.17 |
| ChatGLM2-6B | 39.63 | 2.33 | 3.77 | 5.97 | 6.13 | 2.83 | 3.83 | 2.6 | 3.8 | 4.0 | 4.37 |
| Ziya-v1.1-13B | 37.63 | 2.77 | 3.97 | 5.17 | 5.33 | 2.8 | 3.77 | 2.53 | 3.7 | 3.03 | 4.57 |
| InternLM-Chat-7B | 36.27 | 2.63 | 3.67 | 4.87 | 5.57 | 3.17 | 3.33 | 2.33 | 4.03 | 3.13 | 3.53 |
| Linly-Chinese-LLaMA-2-13B-HF | 34.7 | 2.2 | 3.77 | 4.5 | 5.0 | 2.43 | 3.33 | 2.53 | 3.9 | 2.5 | 4.53 |
| BELLE-LLaMA2-13B-Chat-0.4M | 33.97 | 2.57 | 3.07 | 4.93 | 4.73 | 2.83 | 3.8 | 2.43 | 3.33 | 2.4 | 3.87 |
| LLaMA-2-7B-Chat-HF | 23.63 | 1.53 | 3.43 | 3.0 | 3.73 | 1.73 | 2.43 | 1.97 | 2.17 | 0.8 | 2.83 |

*Note: Discipline scores are on a 10-point scale*

The performance distribution over time for the currently ranked models is shown in the figure below:

<div align="center">
<img src=".\pic\trend_of_model_series.png" alt="Model Performance Trends" style="zoom:80%;" />
</div>

For more experimental details and analysis, please refer to our [paper](https://arxiv.org/abs/2508.05452).


## 📞 Contact Us

This project is open to the public, and we welcome you to participate in our evaluation.

Institutional evaluation requires certification. After registering an account, please contact the administrators for verification and to apply for evaluation permissions.

Unless there are special circumstances, all evaluation results will be added to the leaderboard upon completion.

- **Website**: [http://llmeval.com/](http://llmeval.com/)
- **Email**: mingzhang23@m.fudan.edu.cn
- **WeChat**: zanyingluan



---

<div align="center">

**LLMEval-3** | Building the Future of LLM Evaluation

</div>
