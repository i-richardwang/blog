---
title: 简单实现基于文档的 RAG 问答
author: Sat Naing
pubDatetime: 2024-01-06T04:06:31Z
slug: basic-rag-implementation
featured: false
draft: false
tags:
  - TypeScript
  - Astro
description: 介绍构建一个基础 RAG 工作流的思路与步骤，以腾讯财报为例，实现基于特定文档的自动问答。
---

在数据分析工作中，尤其是在进行行业对标和竞品分析时，我们需要频繁地从各类研究报告中提取关键信息，如竞争对手的财务数据、战略动向等。面对信息量大、结构各异的文档（特别是 PDF 格式的财报、研报），手动查找和提取信息不仅效率低下，还容易出错。

为了应对这一挑战，我们探索并构建了一个基础的 RAG (Retrieval-Augmented Generation) 流程。本文将以腾讯 2023 年第四季度及全年业绩报告为例，展示如何通过 RAG 实现从文档中**自动检索相关信息**并**生成针对性答案**，旨在提高特定场景下信息获取的效率。

## 核心思路与步骤

RAG 的核心思路是结合**信息检索 (Retrieval)** 和**文本生成 (Generation)**。首先，从文档库中精确找到与用户问题相关的文本片段；然后，将这些片段作为上下文，交给大语言模型 (LLM) 来生成最终答案。这使得答案既能基于文档的真实内容，又能以自然语言的形式呈现。

实现一个基础的 RAG 流程，通常包含以下关键步骤：

### 1. 文档加载：获取信息源

流程的第一步是将目标文档加载到处理环境中。我们需要将非结构化的文档内容（如 PDF）转换为可处理的文本格式。

:::note
针对不同的文档格式，有多种加载工具可供选择。例如，对于 PDF 文件，`PDFPlumberLoader` 是一个常用的选项，能够较好地提取文本内容。选择合适的加载器是确保后续处理质量的基础。
:::

```python
# 示例：使用 PDFPlumberLoader 加载财报 PDF
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("腾讯2023年第四季度及全年业绩.pdf")
data = loader.load() # data 包含了从 PDF 提取的文本和元数据
```

### 2. 文本分片：适应模型限制与优化检索

由于大语言模型通常存在输入长度限制（Context Window），我们无法将整篇长文档一次性输入。同时，为了让检索更聚焦，需要将文档分割成较小的、有意义的文本块（Chunks）。

这里选用 `RecursiveCharacterTextSplitter`，它会尝试按段落、句子等递归地分割文本，以保持语义的相对完整性。分片的大小 (`chunk_size`) 和相邻分片间的重叠 (`chunk_overlap`) 是需要权衡的参数。

:::tip
**分片策略考量：**
*   **`chunk_size`**: 决定了每个文本块包含多少信息。较大的 `chunk_size` 能提供更丰富的上下文，但可能增加 LLM 处理的负担，也可能在检索时因为包含过多无关信息而降低精度（大海捞针效应）。
*   **`chunk_overlap`**: 保留相邻文本块之间的部分重叠内容，有助于防止在分割点切断重要的语义联系。

理想的参数需要根据文档特性和具体任务进行实验和调整，目标是在**上下文完整性**和**检索精确性**之间找到平衡。
:::

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 设置分片大小为 800 字符，重叠 100 字符
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = text_splitter.split_documents(data)

print(f"文档被分割为 {len(splits)} 个片段")
# Output: 文档被分割为 76 个片段
```

### 3. 向量化与存储：实现语义检索

为了让机器能够理解文本内容并进行**基于语义的相似度匹配**（而非简单的关键词匹配），我们需要将文本片段转换为向量表示（Embeddings）。每个向量捕捉了对应文本块的核心语义。

我们选用 `bge-large-zh` 作为 Embedding 模型，该模型在中文文本处理上表现较好。随后，将这些文本向量存储到向量数据库中（本示例使用 `Chroma` 作为内存型向量库进行演示），以便后续快速检索。

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 初始化 Embedding 模型 (这里假设模型已下载到本地或可访问)
# model_name = "BAAI/bge-large-zh-v1.5"
# model_kwargs = {'device': 'cpu'} # 或 'cuda'
# encode_kwargs = {'normalize_embeddings': True}
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
# 注意: 实际运行时需要配置好 embeddings 对象

# 将分片文档及其向量存入 Chroma
# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings) # embeddings 对象需已初始化
```
*（为保持简洁，Embedding 模型初始化的代码已注释，实际使用时需确保 `embeddings` 对象可用）*

### 4. 文档检索：定位相关上下文

当用户提出问题时，首先将问题同样进行向量化，然后在向量数据库中搜索与之**语义最相似**的文本片段。这一步是 RAG 中 "R" (Retrieval) 的核心。

我们构建一个检索器 (`retriever`)，它可以根据用户问题从向量存储中找出最相关的 `k` 个文档片段。

```python
# 假设 vectorstore 已成功创建
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 设置为返回最相关的 3 个片段

# 示例：检索与 "腾讯全年收入" 相关的信息
query = "腾讯全年的收入是多少"
relevant_docs = retriever.get_relevant_documents(query)

# relevant_docs 将包含 3 个与问题最相关的 Document 对象 (文本片段及其元数据)
# (输出内容同原稿，此处省略以保持简洁)
```
可以看到，检索器成功找到了包含收入数据的相关段落。

### 5. 结合 LLM 生成答案：合成最终结果

最后一步是 RAG 中的 "G" (Generation)。我们将上一步检索到的相关文档片段 (`relevant_docs`) 作为**上下文信息**，连同用户的原始问题 (`query`)，一起提供给大语言模型 (LLM)。

通过设计一个合适的提示词 (Prompt)，我们指示 LLM **仅根据提供的上下文信息**来回答问题。这样做可以有效**减少模型幻觉**，确保答案来源于指定的文档。

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 假设 llm 对象已初始化 (例如: from langchain_openai import ChatOpenAI; llm = ChatOpenAI(model="gpt-3.5-turbo"))

# Prompt 模板：指导 LLM 如何利用上下文回答问题
template = """
你是一个问答助手。请根据下面提供的上下文信息来回答问题。如果你在上下文中找不到答案，就回答 "根据提供的文档信息，我无法回答该问题"。请使用简洁的语言回答，最多不超过三句话。

上下文:
{context}

问题: {question}

答案:
"""
custom_rag_prompt = PromptTemplate.from_template(template)

# 辅助函数：将检索到的文档列表格式化为单一字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG 链：检索 -> 格式化 -> Prompt -> LLM -> 输出解析
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm # llm 对象需已初始化
    | StrOutputParser()
)

# 执行问答
question = "腾讯全年的收入是多少"
answer = rag_chain.invoke(question)
print(answer)
```

```text title="预期输出示例"
根据提供的文档信息，腾讯在截至二零二三年十二月三十一日止年度的收入为人民币6,090亿元。
```
这个回答直接、准确地从检索到的上下文中提取并总结了信息。

## 总结与讨论

本文演示了一个基础的 RAG 流程，通过加载文档、分片、向量化存储、语义检索和结合 LLM 生成答案，实现了针对特定文档的自动化问答。这种方法相比手动查阅，在处理大量或结构化较差的文档时，有望显著提升信息获取效率。

需要强调的是，这只是一个**极简的实现示例**。在实际应用中，往往面临更复杂的挑战：
*   **文档规模与多样性：** 处理海量、多格式、质量不一的文档。
*   **问题复杂度：** 用户问题可能更开放、模糊或需要多步推理。
*   **效果优化：** 基础 RAG 的效果可能不满足要求，需要引入更高级的技术，如：
    *   **查询转换 (Query Transformation):** 对用户问题进行改写、扩展或分解，以提升检索效果。
    *   **混合检索 (Hybrid Search):** 结合向量检索与传统关键词检索。
    *   **重排序 (Re-ranking):** 对初步检索结果进行二次排序，提高最终送入 LLM 的上下文质量。
    *   **Prompt 工程优化：** 调整提示词结构，甚至上下文在 Prompt 中的位置。

构建一个鲁棒、高效的 RAG 系统是一个持续迭代和优化的过程。但理解其核心步骤和基本原理，是迈向更复杂应用的基础。