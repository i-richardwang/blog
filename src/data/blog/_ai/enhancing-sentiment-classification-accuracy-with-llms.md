---
title: 利用大模型提升情感分类任务准确性
author: Richard Wang
pubDatetime: 2024-02-20T04:06:31Z
slug: enhancing-sentiment-classification-accuracy-with-llms
featured: false
draft: false
tags:
  - LLM
  - NLP
  - Python
  - LangChain
description: 通过实际案例探讨大模型在文本情感分类任务中相比传统NLP方法的优势。
---

情感分类是自然语言处理（NLP）中的一项基础任务，旨在识别文本中所表达的情感倾向（如正向、负向、中性）。在企业管理、市场分析等领域，对用户评论、员工反馈等文本进行情感分析具有重要价值。传统的NLP方法（如基于词袋模型、TF-IDF或简单规则的方法）以及专门训练的情感分类模型在处理某些复杂文本时仍可能表现不佳，特别是对于包含反讽、隐晦表达或混合情感的长文本。

大型语言模型（LLM）的快速发展为解决这些挑战提供了新的途径。LLMs凭借其强大的上下文理解能力和生成能力，在处理语言的细微差别方面展现出显著优势。

本文将分享一个实际测试案例，探讨使用自部署的Qwen1.5-14B大模型进行员工调研主观回复的情感分类任务，并与传统NLP方法进行比较。测试结果初步表明，大模型在此类任务上具有明显的准确性优势。

## 实验设置

为了评估大模型在情感分类任务上的表现，我们进行了一个小规模的对比实验。

### 1. 任务定义

任务目标是对员工在敬业度调研中提交的主观回复文本进行情感分类。根据文本内容，将其归类为**正向**、**负向**或**中性**三个类别之一。

### 2. 数据集说明

我们构建了一个小型的虚构数据集用于本次测试。该数据集包含20条模拟的员工回复文本，通过人工编写并辅以ChatGPT生成，旨在覆盖不同类型的情感表达，包括一些可能对传统方法构成挑战的样本（如反话）。

> **注意：** 由于数据集规模较小且为虚构数据，本次测试结果的普适性有限，主要用于探索性验证和方法展示。

### 3. 评分规则

情感识别任务天然存在一定的主观性，尤其是在区分中性与弱正向/负向情感时，不同的人工标注者也可能产生分歧。为了更客观地评估模型性能，我们采用了差异化的赋分制：

*   **完全一致：** 模型预测结果与人工标注（Ground Truth）完全相同，得 **1** 分。
*   **轻微偏差：** 模型将中性判断为正向或负向，或将正向/负向判断为中性，得 **0.5** 分。这种情况被视为可接受的偏差。
*   **完全相反：** 模型将正向判断为负向，或将负向判断为正向，得 **0** 分。这种情况被视为严重错误。

最终得分通过计算所有样本得分的平均值得到。

## 测试结果对比

通过对20条样本进行测试，我们得到了以下对比结果：

| 模型        | 平均得分 | 严重识别错误占比 (正负向混淆) |
| :---------- | :------- | :---------------------------- |
| 大模型 (Qwen1.5-14B) | 0.950    | 0%                            |
| 传统NLP     | 0.775    | 15%                           |

从结果中可以看出：

1.  **大模型表现更优：** 大模型的平均得分显著高于传统NLP方法。
2.  **大模型鲁棒性更强：** 在本次测试中，大模型没有出现将正向情感误判为负向（或反之）的严重错误，而传统NLP方法出现了15%的此类错误。这表明大模型在理解复杂或模糊情感表达方面可能更具优势。

## 代码复现 (基于LangChain)

以下是使用LangChain框架调用大模型完成情感分类任务的核心代码片段。

> **版本提示：** 以下示例代码基于 LangChain 0.1.0 版本之前的用法编写。在更新版本的 LangChain 中，部分API可能已发生变化（推荐使用 LCEL 写法），直接运行时可能会出现警告或错误。

> **Prompt Engineering 提示：** 提示词（Prompt）的设计对大模型任务的效果至关重要。例如，在本次测试中，我们发现要求模型直接输出`正向`、`负向`、`中性`的准确性，优于要求其输出对应的英文`positive`、`negative`、`neutral`。

### 1. 环境准备 (省略)

此处省略导入所需库（如`pandas`, `langchain`, `json`, `tqdm`）以及初始化大模型实例（`llm`）的过程。假设 `llm` 实例已准备就绪。

### 2. 使用LangChain构建分类任务链

```python
import pandas as pd
import json
from tqdm import tqdm
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# 假设 llm 实例已定义并初始化

# 1. 定义期望的输出格式
response_schemas = [
    ResponseSchema(name="ID", description="原始数据中的员工ID"),
    ResponseSchema(name="sentiment_class_llm", description="大模型的情感分类结果，必须是 '正向'、 '中性' 或 '负向' 中的一个"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# 2. 设计提示词模板
classification_prompt_template = """
作为一个经验丰富的NLP专家，你的任务是分析员工敬业度调研中的主观回复内容。请严格按照以下指示操作：

1.  **情感分类**：仔细阅读员工的回复，判断其表达的核心情感倾向。将情感归类为“正向”、“中性”或“负向”。
    *   请特别注意识别可能存在的反话或讽刺，并根据其真实意图进行分类。
    *   分类必须是“正向”、“中性”或“负向”这三个选项之一。
2.  **依据事实**：你的判断必须严格基于提供的员工回复内容，避免进行任何无关的推测或联想。

请处理以下员工回复：
员工ID与员工回复: >>>{answer}<<<

请按照以下JSON格式输出结果:
{format_instructions}
"""

# 3. 创建PromptTemplate实例
classification_prompt = PromptTemplate(
    input_variables=['answer'],
    template=classification_prompt_template,
    partial_variables={"format_instructions": format_instructions}
)

# 4. 创建LLMChain实例
classification_chain = LLMChain(
    llm=llm,
    prompt=classification_prompt
)

print("LangChain 任务链已构建完成。")
```

### 3. 执行分类并将结果合并

```python
# 读取包含员工回复的CSV文件
subjective_answers_df = pd.read_csv("data/subjective_answers_train.csv") # 假设有此文件，包含 'ID' 和 '回复内容' 列，以及 'sentiment_class_true' (人工标注) 和 'sentiment_class_nlp' (传统NLP结果) 列

# 创建空的DataFrame用于存储大模型结果
result_llm = pd.DataFrame()

# 遍历数据集，使用大模型进行分类
print("开始使用大模型进行情感分类...")
for i in tqdm(range(len(subjective_answers_df))):
    # 提取ID和回复内容，格式化为JSON字符串输入给模型
    answer_data = subjective_answers_df.iloc[i, :2] # 假设前两列是 ID 和 回复内容
    answer_json = answer_data.to_json(force_ascii=False) # 使用 force_ascii=False 保留中文

    try:
        # 调用LLMChain执行分类
        sentiment_class_result = classification_chain.run(answer=answer_json)

        # 解析模型返回的JSON字符串
        # 注意：这里的解析方式依赖于模型输出的具体格式，可能需要调整
        # 假设模型输出包含 ```json ... ``` 代码块
        if '```json' in sentiment_class_result:
             sentiment_class_result_json = sentiment_class_result.split('```json')[1].split('```')[0].strip()
        else: # 备用解析，如果模型直接返回JSON
             sentiment_class_result_json = sentiment_class_result.strip()

        sentiment_class_result_df = pd.DataFrame([json.loads(sentiment_class_result_json)])
        result_llm = pd.concat([result_llm, sentiment_class_result_df], ignore_index=True)

    except Exception as e:
        print(f"处理第 {i} 条数据时出错: {e}")
        # 可以选择记录错误或填充默认值
        error_df = pd.DataFrame([{'ID': answer_data['ID'], 'sentiment_class_llm': '错误'}])
        result_llm = pd.concat([result_llm, error_df], ignore_index=True)


# 将大模型分类结果合并回原始DataFrame
result_final = subjective_answers_df.merge(result_llm, on='ID', how='left')

print("\n分类完成，结果已合并。")
# print(result_final.head()) # 可以取消注释查看合并后的部分结果
```

*输出示例:*
```text
开始使用大模型进行情感分类...
100%|██████████| 20/20 [00:25<00:00,  1.28s/it]

分类完成，结果已合并。
```

### 4. 计算评分

```python
# 定义情感标签到数值的映射，方便计算差异
# 注意：包含了中英文标签以增加兼容性
encoding_dict = {'正向': 0, 'positive': 0, '中性': 1, 'neutral': 1, '负向': 2, 'negative': 2}

result_score = result_final.copy()

# 应用编码，处理可能存在的NaN或错误值
result_score['sentiment_class_true_encoded'] = result_score['sentiment_class_true'].map(encoding_dict)
result_score['sentiment_class_nlp_encoded'] = result_score['sentiment_class_nlp'].map(encoding_dict)
result_score['sentiment_class_llm_encoded'] = result_score['sentiment_class_llm'].map(encoding_dict)

# 填充无法编码的值（例如模型返回'错误'或非预期标签）为 -1 或其他标记值，避免计算错误
result_score.fillna({'sentiment_class_true_encoded': -1,
                     'sentiment_class_nlp_encoded': -1,
                     'sentiment_class_llm_encoded': -1}, inplace=True)

# 计算预测结果与真实标签之间的绝对差值
# 仅对有效编码的样本进行计算
valid_llm_mask = result_score['sentiment_class_llm_encoded'] != -1
valid_nlp_mask = result_score['sentiment_class_nlp_encoded'] != -1
valid_true_mask = result_score['sentiment_class_true_encoded'] != -1

result_score['llm_diff'] = -1 # Default value
result_score.loc[valid_llm_mask & valid_true_mask, 'llm_diff'] = abs(result_score['sentiment_class_true_encoded'] - result_score['sentiment_class_llm_encoded'])

result_score['nlp_diff'] = -1 # Default value
result_score.loc[valid_nlp_mask & valid_true_mask, 'nlp_diff'] = abs(result_score['sentiment_class_true_encoded'] - result_score['sentiment_class_nlp_encoded'])

# 定义差值到得分的映射
score_mapping = {0: 1, 1: 0.5, 2: 0, -1: 0} # -1 (无效/错误) 也计0分

# 计算每个样本的得分
result_score['llm_score'] = result_score['llm_diff'].map(score_mapping)
result_score['nlp_score'] = result_score['nlp_diff'].map(score_mapping)

# 计算模型的平均得分 (仅基于有效预测)
llm_avg_score = result_score.loc[valid_llm_mask & valid_true_mask, 'llm_score'].mean()
nlp_avg_score = result_score.loc[valid_nlp_mask & valid_true_mask, 'nlp_score'].mean()

# 计算严重错误（正负向识别相反，即差值为2）的占比
total_valid_samples = (valid_true_mask).sum() # 以真实标签有效为基准
count_wrong_llm = (result_score['llm_diff'] == 2).sum()
llm_percentage_wrong = (count_wrong_llm / total_valid_samples) * 100 if total_valid_samples > 0 else 0

count_wrong_nlp = (result_score['nlp_diff'] == 2).sum()
nlp_percentage_wrong = (count_wrong_nlp / total_valid_samples) * 100 if total_valid_samples > 0 else 0

print(f"大模型平均得分: {llm_avg_score:.3f}")
print(f"传统NLP平均得分: {nlp_avg_score:.3f}")
print(f"大模型严重错误占比: {llm_percentage_wrong:.1f}%")
print(f"传统NLP严重错误占比: {nlp_percentage_wrong:.1f}%")
```

### 5. 输出格式化结果

```python
from tabulate import tabulate

# 准备用于表格展示的数据
data = [
    ["大模型 (Qwen1.5-14B)", f"{llm_avg_score:.3f}", f"{llm_percentage_wrong:.1f}%"],
    ["传统NLP", f"{nlp_avg_score:.3f}", f"{nlp_percentage_wrong:.1f}%"]
]

headers = ["模型", "平均得分", "严重识别错误占比"]
table = tabulate(data, headers, tablefmt="github") # 使用 github 风格的 markdown 表格

print("\n模型性能对比结果:")
print(table)
```

*输出示例:*
```text
模型性能对比结果:
| 模型                 |   平均得分 |   严重识别错误占比 |
|----------------------|------------|--------------------|
| 大模型 (Qwen1.5-14B) |      0.950 |               0.0% |
| 传统NLP              |      0.775 |              15.0% |
```

## 结论与讨论

本次基于小型虚构数据集的测试初步表明，大型语言模型（如Qwen1.5-14B）在处理员工主观回复这类可能包含复杂情感表达的文本时，相比传统NLP方法具有更高的准确性和鲁棒性，尤其在避免严重的情感方向误判方面表现突出。

尽管受限于数据集规模，这一结果仍然提示我们，LLM在需要深度语义理解和细微情感辨析的应用场景中潜力巨大。未来可以在更大、更真实的业务数据集上进行验证，并进一步探索通过优化提示词工程、模型微调等方式，持续提升LLM在特定领域情感分类任务上的表现。

对于需要高精度情感分析的场景，例如深入理解员工心声、精准把握客户反馈等，引入大模型技术有望带来显著的价值提升。