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

question_type_dic = {
    "zh": {
        "open_ended": "开放式问答题",
        "closed_ended": "是否式问答题",
        "single_choice": "单选题",
        "multi_choice": "多选题",
    },
    "en": {
        "open_ended": "Open Ended Question",
        "closed_ended": "Yes/No Question",
        "single_choice": "Single Choice Question",
        "multi_choice": "Multi Choice Question",
    }
}

question_type_desc_dic = {
    "zh": {
        "open_ended": "简短文字答案",
        "closed_ended": "回答“是。”或“否。”",
        "single_choice": "- 只填入正确选项的字母，例如：A 或 B 或 C 或 D。\n- 不加中括号、不加引号、不加其它说明。",
        "multi_choice": "- 使用类似 Python 列表的字符串形式，列出所有正确选项的字母，例如：\n  ['A', 'C'] 或 ['A', 'C', 'D']\n- 字母必须大写，使用单引号包裹，选项之间用逗号+空格分隔。",
    },

    "en": {
        "open_ended": "A short free-text answer.",
        "closed_ended": 'Answer either "Yes." or "No." (with a period).',
        "single_choice": "- Provide only the letter of the correct option, e.g., A or B or C or D.\n- Do not use brackets, quotes, or any additional text.",
        "multi_choice": "- Use a Python list–like string containing all correct option letters, e.g.,\n  ['A', 'C'] or ['A', 'C', 'D']\n- Letters must be uppercase, wrapped in single quotes, separated by comma + space.",
    }
}


def judge_language_zh_or_eng(text):
    if not text:
        return "en"

    chinese_count = 0
    total_count = len(text)

    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1

    chinese_ratio = chinese_count / total_count

    if chinese_ratio > 0.1:
        return "zh"
    else:
        return "en"


def preprocess(*, question, question_type, **kwargs):
    """
    适用于: open_ended, closed_ended, single_choice, multi_choice
    适用于: en/zh
    将question, answer, report等信息处理成teacher模型的输入，从而可以得到teacher输出的CoT
    """
    lang = judge_language_zh_or_eng(question)
    prompt = Prompt_en if lang == "en" else Prompt_zh
    res = prompt.replace("{question}", question)
    res = res.replace("{question_type}", question_type_dic[lang][question_type])
    res = res.replace("{question_type_desc}", question_type_desc_dic[lang][question_type])
    return res
