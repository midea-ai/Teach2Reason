Prompt_zh = """
你将看到一道人为设计的胸片相关试题、对应的 CXR（胸片）图像、一份胸片医学报告，以及这道题的标准答案。现在需要你补全一段合理的解题思路（推理过程）。

重要要求：
- 在“解题思路”的文字中，要假装自己只看到了题目和胸片本身，而没有看到医学报告、也不知道标准答案。
- 医学报告和标准答案只作为你在后台的参考，用来保证你的推理方向和细节是医学上合理、并且最终结论与标准答案一致。
- 不要在解题思路中提到“报告写了……”“根据报告……”“标准答案是……”这类表述，也不要显得是在倒着解释答案。
- 在解题思路中，建议按照“观察影像 → 描述关键所见 → 结合题干问题进行分析 → 得出结论”的自然顺序展开，但具体表述可以自由发挥，不需要固定为某格式。

其他说明：
- 题型：{question_type}；
- 该题型的最终答案表达形式简要说明为（仅供你在心里对齐标准答案形式，不必在推理中提及）：
{question_type_desc}

题目如下：
{question}

标准答案如下（只用于校正你的最终结论，不要在文字中引用）：
{answer}

对应的胸片医学报告如下（只用于帮助你理解病例真相，不要在文字中显式引用或转述）：
{report}

请根据“像是在读片答题”的方式，写出一段完整的解题思路，并按下面格式输出唯一的 XML：

```xml
<response>（在这里填写完整的解题思路，用自然中文连续描述，不提及报告和“标准答案”）</response>
```

你的输出为：
"""

Prompt_en = """
You will be given a chest X-ray exam question, the corresponding CXR (chest X-ray) image, a radiology report for the same case, and the official correct answer. Your task is to reconstruct a reasonable reasoning process for solving this question.

**Important requirements:**

- In the written reasoning, you must pretend that you only see the question and the chest X-ray itself. You do *not* see the radiology report, and you do *not* know the official answer in advance.
- The radiology report and the official answer are only for you to use “in the background”: they help you keep your reasoning medically sound and ensure that your final conclusion is consistent with the official answer.
- Do **not** mention phrases like “the report says…”, “according to the report…”, “because the correct answer is…” in your reasoning, and do not make it look like you are reverse-explaining a known answer.
- In the reasoning, it is recommended (but not strictly required) to follow a natural sequence such as:
  “inspect the image → describe key findings → combine with the question to analyze → reach a conclusion”.
  The exact wording and structure can be fully free-form; there is no fixed template.

**Additional notes:**

* Question type: {question_type}
* The expected answer format for this type is (for your internal alignment only; do not mention it in the reasoning):
  {question_type_desc}

Question:
{question}

Official correct answer (only for aligning your final conclusion; do not quote it in the reasoning text):
{answer}

Radiology report for this chest X-ray (only to help you understand the true case; do not explicitly quote or paraphrase it in the reasoning text):
{report}

Please write a complete reasoning process as if you were answering the question while reading the film, and output exactly one XML block in the following format:

```xml
<response>(write the full reasoning here in continuous natural English, without mentioning the report or the “official answer”)</response>
```

Your output:
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


def preprocess(*, question, answer, report, question_type, **kwargs):
    """
    适用于: open_ended, closed_ended, single_choice, multi_choice
    适用于: en/zh
    将question, answer, report等信息处理成teacher模型的输入，从而可以得到teacher输出的CoT
    """
    lang = judge_language_zh_or_eng(question)
    prompt = Prompt_en if lang == "en" else Prompt_zh

    res = prompt.replace("{report}", report)
    res = res.replace("{question}", question)
    res = res.replace("{answer}", answer)

    res = res.replace("{question_type}", question_type_dic[lang][question_type])
    res = res.replace("{question_type_desc}", question_type_desc_dic[lang][question_type])
    return res
