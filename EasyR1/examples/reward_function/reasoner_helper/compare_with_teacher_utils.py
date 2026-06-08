import random
import re
import xmltodict

Prompt_zh = """
你将看到一道人为设计的胸片相关试题、一份对应的胸片医学报告、该题的标准答案，以及两份不同的解题思路（记为 A 和 B）。这两份解题思路都尝试解释：在看到胸片和题干的前提下，如何得出给定的标准答案。

你的任务：在充分理解病例和题目的基础上，对比解题思路 A 与解题思路 B 的优劣，选择其中总体上“更好的一份”，并给出你的评估理由。

【可用信息】
- 胸片医学报告（作为病例真实情况和影像所见的参考）：
{report}

- 试题题干：
{question}

- 该题的标准答案：
{answer}

- 解题思路 A：
{reason_a}

- 解题思路 B：
{reason_b}

【总体评估原则】
你评估“更好”的标准，必须基于内容质量本身，而不是篇幅、字数、句子多少或表面详细程度。
- 不可以因为某一份解题思路更长、写得更多、表述更繁复，就直接判定其更好；
- 也不可以因为某一份更短，就直接判定其更差；
- 如果一份思路虽然较短，但更准确、更贴合影像事实、推理更关键更连贯，则它可以优于更长的一份；
- 如果一份思路虽然较长，但包含冗余、重复、无关信息、臆测内容或错误推理，则不能因为“更详细”而给它更高评价；
- 你只能根据内容的正确性、相关性、推理质量、信息节制和表达清晰度来比较优劣，而不能把“长度”本身作为加分项或扣分项。

【评估时需要重点考虑的维度】
你需要综合以下方面来判断哪一份解题思路更好（允许二者都有缺点，选择相对更优的一方）：

1. 正确性与一致性
   - 是否能够在逻辑上支持标准答案 {answer}，而不是偏离或暗示其他答案；
   - 是否与医学报告所体现的影像学事实和诊断方向基本一致；
   - 是否尽量避免明显的医学错误、违背影像学常识的推理。

2. 依据与推理链条
   - 是否清晰地基于胸片上“可能会看到的关键所见”来展开推理，而不是仅仅重复答案或空泛评价；
   - 是否体现了合理的推理顺序，例如：
     “观察影像 → 描述关键异常/部位/范围 → 结合题干问题 → 排除不合适的可能 → 得出结论”；
   - 推理是否连贯、有因有果，而不是东一块西一块。

3. 信息利用与节制
   - 是否重点围绕与本题相关的关键所见和关键信息展开，而不是大量引入与本题无关的内容；
   - 是否避免明显“胡编”不存在的影像特征或检查信息；
   - 如果引用到类似医学报告中的信息，其内容是否与报告精神相符，且不会直接暴露“我在看报告”，而更像是从片子读出来的结论；
   - 是否避免为了显得“更完整”而堆砌冗余内容。内容相关、有效比内容多更重要。

4. 表达清晰度
   - 语言是否通顺清晰、容易理解；
   - 是否有条理地说明“为什么是这个答案”，而不是仅给出结论；
   - 表达简洁但有效，不因过度展开而削弱重点。

【比较时的特别注意事项】
- 请进行“质量比较”，不是“长度比较”；
- 篇幅长短只能被视为表面现象，不能作为优劣判断依据；
- 只有当更长的内容确实提供了更多正确、相关、必要且有助于推理的有效信息时，才可以因此体现出优势；
- 如果较长内容只是重复、空泛扩写、加入无关信息，或增加未经支持的细节，则这些都应视为缺点而不是优点；
- 如果 A 与 B 内容质量相近，应优先选择更准确、更聚焦、更有关键依据的一方，而不是默认选择更长的一方。

【输出要求】
1. 你需要在心里先综合以上维度，对解题思路 A 和 B 做出判断，选出“总体上更好的一份”；
2. 然后用 XML 格式输出你的决策过程和最终选择，根标签为 <response>，内部包含：
   - <reason>：用自然中文简要说明你是如何比较 A 和 B 的，在哪些方面 A 更好或 B 更好，并最终为什么选中某一方；
   - <result>：只写 “A” 或 “B” 两种之一，对应你认为更好的那一份解题思路。

注意：
- 不要输出任何 XML 结构之外的文字；
- <result> 中只能是单个大写字母 A 或 B，不加引号，不加其他说明；
- 在 <reason> 中不要把“篇幅更长/更短”本身当作主要理由，除非你明确指出其内容层面的实际优点或缺点。

请按以下格式输出：

```xml
<response>
    <reason>（在这里简要说明比较过程和理由，例如：哪一份更符合报告和标准答案、推理是否连贯、是否有明显医学错误等）</reason>
    <result>（在这里只写 A 或 B，对应你认为更好的那一份解题思路）</result>
</response>
````

你的比较结果为：
"""

Prompt_en = """
You will be given a chest X-ray exam question, a corresponding radiology report, the official correct answer, and two different reasoning traces (labeled A and B). Both reasoning traces attempt to explain how one could arrive at the official answer based on the chest X-ray and the question.

Your task: after fully understanding the case and the question, compare reasoning A and reasoning B, decide which one is overall better, and briefly explain your judgment.

【Available information】

* Radiology report (reflects the true findings and diagnostic thinking for this case):
  {report}

* Exam question:
  {question}

* Official correct answer:
  {answer}

* Reasoning A:
  {reason_a}

* Reasoning B:
  {reason_b}

【Overall evaluation principle】
Your judgment of which reasoning is “better” must be based on content quality, not on length, word count, number of sentences, or superficial level of detail.

* Do not judge a reasoning trace as better simply because it is longer, more verbose, or appears more detailed;
* Do not judge a reasoning trace as worse simply because it is shorter;
* A shorter reasoning trace can be better if it is more accurate, more consistent with the imaging findings, and more logically focused;
* A longer reasoning trace should not be rewarded if it contains redundancy, repetition, irrelevant discussion, unsupported speculation, or incorrect reasoning;
* You must compare A and B based only on the quality of their content: correctness, relevance, reasoning quality, restraint, and clarity. Length itself must not be treated as an advantage or disadvantage.

【Evaluation criteria】
When comparing A and B, you should consider the following aspects. It is allowed that both have flaws; you only need to choose the relatively better one.

1. Correctness and consistency

   * Does the reasoning logically support the official answer {answer}, instead of drifting toward a different answer?
   * Is it broadly consistent with the radiology report in terms of imaging findings and diagnostic direction?
   * Does it avoid obvious medical errors or reasoning that conflicts with basic chest imaging knowledge?

2. Use of evidence and reasoning chain

   * Does it clearly build on plausible key findings that one could see on the chest X-ray, rather than just restating the answer or giving vague comments?
   * Does it show a reasonable reasoning flow, for example:
     “inspect the image → describe key abnormalities/locations/extent → relate to the question → rule out less likely options → reach the conclusion”?
   * Is the reasoning coherent and causal, rather than fragmented and disjointed?

3. Information use and restraint

   * Does it focus on the findings and information that are actually relevant to this specific question, instead of introducing large amounts of unrelated content?
   * Does it avoid clearly inventing findings or test results that are not supported by the image or the case?
   * If it implicitly aligns with ideas present in the radiology report, does it remain compatible with the spirit of the report, and does it sound like it could come from reading the film rather than obviously “quoting the report”?
   * Does it avoid unnecessary elaboration added merely to sound more complete? Relevant and useful content matters more than more content.

4. Clarity of expression

   * Is the language clear and easy to understand?
   * Does it clearly explain why the official answer makes sense, rather than merely restating the conclusion?
   * Is it concise but effective, without losing focus through excessive expansion?

【Special instructions for comparison】

* This is a quality comparison, not a length comparison;
* Length should be treated only as a surface characteristic, not as a scoring factor by itself;
* A longer reasoning trace only deserves credit if the additional content is genuinely correct, relevant, necessary, and helpful to the reasoning process;
* If the extra length mainly consists of repetition, vague expansion, irrelevant content, or unsupported details, that should count against it rather than in its favor;
* If A and B are close in quality, prefer the one that is more accurate, more focused, and better grounded in key findings, rather than defaulting to the longer one.

【Output requirements】

1. Internally compare reasoning A and B using the above criteria, and decide which one is overall better.
2. Then output your decision and justification in XML format, with root tag <response> and two child tags:

   * <reason>: briefly explain how you compared A and B, in which aspects A is better or B is better, and why you finally chose one over the other.
   * <result>: write only a single capital letter, “A” or “B”, indicating which reasoning you judge to be better overall.

Important:

* Do NOT output anything outside the XML structure;
* Inside <result>, output only a single capital letter A or B, without quotes or any extra text;
* In <reason>, do not use “longer” or “shorter” itself as the main justification unless you explicitly connect it to an actual content-level strength or weakness.

Please output in the following format:

```xml
<response>
    <reason>(briefly describe your comparison and justification, e.g., which reasoning is more consistent with the report and the official answer, which has a more coherent chain of thought, whether there are obvious medical mistakes, etc.)</reason>
    <result>(write A or B here, corresponding to the reasoning you consider better)</result>
</response>
```

Your output:
"""


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


def preprocess(*, question, answer, report, reason, teacher_reasons, **kwargs):
    """
    预处理《比较reasoner和teacher的CoT哪个更好》llm judgement的输入
    其中gt是reasoner的CoT对应的字母索引

    适用于：single_choice, multi_choice, open_ended, closed_ended
    适用于：en/zh
    """
    lang = judge_language_zh_or_eng(question)
    prompt = Prompt_en if lang == "en" else Prompt_zh
    res = prompt.replace("{question}", question)
    res = res.replace("{answer}", answer)
    res = res.replace("{report}", report)
    results = []

    for teacher_reason in teacher_reasons:
        ground_truth = random.choice(["A", "B"])
        if ground_truth == "A":
            this_res = res.replace("{reason_a}", reason).replace("{reason_b}", teacher_reason)
        else:  # ground_truth == "B"
            this_res = res.replace("{reason_b}", reason).replace("{reason_a}", teacher_reason)
        this_item = {
            "prompt": this_res,
            "gt": ground_truth,
            "teacher_reason": teacher_reason,
            "reasoner_reason": reason
        }
        results.append(this_item)
    return results


def postprocess(response):
    def escape_label_content(xml_string, tag_start, tag_end):
        def escape_content(match):
            content = match.group(1)

            escaped_content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"',
                                                                                                              '&quot;').replace(
                "'", '&apos;')
            pattern = "<text>{}</text>".replace("<text>", tag_start).replace("</text>", tag_end)
            return pattern.format(escaped_content)

        escaped_string = re.sub(r'<text>(.*?)</text>'.replace("<text>", tag_start).replace("</text>", tag_end),
                                escape_content, xml_string, flags=re.DOTALL)
        return escaped_string

    response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

    response_list = response.split("```")
    string = ""
    for line in response_list:
        if "<response>" in line and "</response>" in line and "<result>" in line and "</result>" in line:
            string = line

    string = escape_label_content(string, "<result>", "</result>")
    string = escape_label_content(string, "<reason>", "</reason>")
    try:
        dic = xmltodict.parse(string)
        entity = dic["response"]
        result = str(entity["result"])
        reason = entity.get("reason", "Error.")
    except Exception as e:
        if ">a" in response.lower():
            result = "A"
        elif ">b" in response.lower():
            result = "B"
        else:
            result = "A"  # 默认为A

        reason = f"Parsed Failed: '{e}'."
    item = {
        "response": response,
        "result": result,
        "reason": reason,
    }
    return item
