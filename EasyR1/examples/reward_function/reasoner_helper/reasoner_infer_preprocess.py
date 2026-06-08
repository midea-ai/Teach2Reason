Prompt_en = """
You will be given one or more chest X-ray images and an exam question. First, carefully inspect the X-ray image(s), then combine your observations with the question to reason step by step and produce a final answer.

Important requirements:
- In your reasoning, it is recommended (but not mandatory) to follow a natural sequence such as: “inspect the image(s) → describe key findings → analyze these findings in the context of the question → reach a conclusion”. The exact wording can be fully free-form and does not need to follow any fixed template.
- The reasoning should clearly explain what you see on the image(s), what these findings imply, and how they lead step by step to your final answer. Do not invent imaging findings or test results that are not actually supported by the X-ray image(s).

Additional notes:
- Question type: {question_type}.
- The expected answer format for this type is described as follows (this is only to constrain the format of <answer>; you do not need to mention it in your reasoning):
{question_type_desc}

Question:
{question}

Based on what you see on the chest X-ray image(s) and the question, produce your response as a single XML structure in the following format:

```xml
<response>
    <reason>(fill in the complete reasoning here, in natural English as a continuous description)</reason>
    <answer>(fill in the final answer here, and make sure its format matches the requirement)</answer>
</response>
````

Do not output anything outside this XML code block.

Your Output:
"""

Prompt_zh = """
你将看到一张或多张胸部X光图像，以及一道试题。请先认真观察胸片影像，再结合题干内容，完成推理并给出最后的答案。

重要要求：

* 在解题思路中，建议按照“观察影像 → 描述关键所见 → 结合题干问题进行分析 → 得出结论”的自然顺序展开，但具体表述可以自由发挥，不需要固定为某格式。
* 解题思路应能说明：你在影像中看到了什么，这些所见意味着什么，以及它们是如何一步步指向你的最终答案的；不要虚构不存在的影像所见或检查结果。

其他说明：

* 题型：{question_type}。
* 该题型的最终答案表达形式简要说明如下（仅用于约束 <answer> 的格式，不必在解题思路中提及）：
  {question_type_desc}

试题如下：
{question}

请根据你看到的胸片影像和题干完成作答，最终只输出一个 XML 结构，格式为：

```xml
<response>
    <reason>（在这里填写完整的解题思路，用自然中文连续描述）</reason>
    <answer>（在这里填写最终答案，格式必须符合要求）</answer>
</response>
```

不要输出任何 XML 代码块以外的内容。

你的输出为：
"""