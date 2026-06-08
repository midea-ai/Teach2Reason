import re
import xmltodict

Prompt_zh = """
你现在扮演“一名胸部影像学考试的阅卷老师”。

你的任务：
给定一份【胸片医学报告】、一条【开放题试题题干】、这道题的【标准答案】以及【某位学生的作答】，
判断该学生答案是否可以视为“答对”（即医学上正确并且满足题干要求），并给出简要理由。

【信息说明】

1. 胸片医学报告（Report）
   - 视为本题的客观事实基础。
   - 如果标准答案或学生答案与报告发生明显冲突，应以报告为最高优先级判断“是否合理”。

2. 标准答案（Ground Truth）
   - 由命题老师给出的参考答案，反映本题希望考察的“核心要点”。
   - 你需要从标准答案中抽取出关键医学信息（例如：异常名称、诊断名称、部位、病因、范围、严重程度、处理建议等）。

3. 学生答案（Pred）
   - 可能与标准答案在措辞上不同，但只要医学含义等价或高度接近、且不与报告矛盾，就可以判定为正确。
   - 也可能出现部分正确、部分错误或严重偏离。

【判分原则】

请根据以下原则，判断学生答案是否算“答对”，并最终给出二元结果（yes/no）：

1. 关注题干要求的“信息类型”
   - 先看题干问的是什么：例如“最可能的诊断”、“主要异常所见”、“病变部位”、“可能病因”、“严重程度”、“下一步处理”等；
   - 标准答案反映出题干真正希望考察的那一类信息。

2. 识别标准答案中的“核心要点”
   - 标准答案可能包含 1 个或多个关键点：
     - 若题干只要求“主要诊断/主要异常/最重要改变”，学生答案至少应覆盖这一“主核心点”，可适度忽略次要细节；
     - 若题干明确要求“列出所有主要异常/请完整列出……”之类，则学生遗漏任何一个重要要点时，应倾向判为不完全正确（result = "no"）。

3. 如何判定“学生答对”（result = "yes"）
   - 学生答案在医学含义上与标准答案的核心要点一致或高度接近，可以接受：
     - 同义词、近义表达、不同但等价的专业术语；
     - 更简短但保留关键意义的表述；
   - 学生答案不应：
     - 给出与报告或标准答案明显矛盾的诊断/异常/部位/病因；
     - 只给出极为模糊的描述（如“有问题”“有异常”）而缺少题干所要求的具体信息；
     - 在关键点上明显错误（例如把“左侧”说成“右侧”，把“不张”说成“气胸”，把“轻度”说成“重度”等）。

4. 如何判定“学生答错”（result = "no"）
   - 学生答案遗漏了题干要求的关键要点，或者主要内容方向错误；
   - 学生答案给出的核心内容与标准答案/报告的事实相矛盾；
   - 学生仅给出过于笼统、含糊的回答，无法认为其真正掌握了标准答案中的具体知识点；
   - 学生答案混入严重错误信息，即便部分内容接近正确，也不视为“整体正确”。

5. 宽严尺度
   - 不要求字面完全一致，只看医学含义；
   - 对轻微措辞差异和不影响判断的小瑕疵应宽松处理；
   - 对核心诊断方向、关键部位、主要异常类型、关键病因等错误应严格处理。

【输出格式要求】

1. 请先在内部完成推理和比较，不要把中间思考过程写出来。
2. 最终回答时，只输出一个 XML 代码块，用 ```xml 包裹，形式如下：

```xml
<response>
    <reason>这里用2-5句中文简要说明你为什么认为学生答案算对或不算对，重点提及：题干要什么、标准答案的核心要点是什么、学生答案与其是否匹配、有无明显错误或遗漏。</reason>
    <result>yes 或 no（二选一，小写）</result>
</response>
````

3. 严禁输出任何 XML 代码块以外的内容，不要多写解释文字或其他代码块。

---

【胸片医学报告】
{report}

---

【试题题干（开放式问答）】
{question}

---

【标准答案（命题老师提供）】
{ground_truth}

---

【学生作答】
{pred}

---

请根据上述信息和判分原则，判断该学生作答是否可以视为“答对”，
并严格按照输出格式要求给出你的 XML 答案。

你的判断为：
"""

Prompt_en = """
You are now acting as a grading examiner for chest radiology open-ended exam questions.

Your task:
Given a 【chest X-ray report】, an 【open-ended question】, a 【reference (ground truth) answer】 and a 【student answer】,
decide whether the student’s answer should be considered correct, and briefly explain your reasoning.

【Information】

1. Chest X-ray report (Report)
   - This is the factual basis for the case.
   - If the reference answer or the student answer conflicts with the report, the report has the highest priority for judging medical correctness.

2. Reference answer (Ground Truth)
   - Provided by the instructor; it reflects the key information that the question is intended to test.
   - You should extract the core medical elements from it (e.g. abnormal findings, diagnosis, location, cause, extent, severity, management advice).

3. Student answer (Pred)
   - The wording may differ from the reference answer.
   - As long as the medical meaning is equivalent or very close, and it does not contradict the report, it can be graded as correct.

【Grading principles】

You must decide a binary result (yes/no) based on the following:

1. Focus on what the question is asking for
   - First, understand what type of information the question explicitly asks:
     - e.g. “most likely diagnosis”, “main abnormal finding”, “location of the lesion”, “possible cause”, “severity”, “management step”, etc.
   - The reference answer shows what the instructor really wants the student to provide.

2. Identify the core elements in the reference answer
   - The reference answer may contain one or several key points:
     - If the question asks for “main diagnosis / main abnormality / most important change”, the student answer should at least cover that main core element; minor omissions may be acceptable.
     - If the question explicitly asks for “all major abnormalities / list all …”, then missing any important element should usually be treated as not fully correct (result = "no").

3. When to mark the student as correct (result = "yes")
   - The student’s answer matches the core meaning of the reference answer:
     - Synonyms, paraphrases, and equivalent medical terminology are acceptable;
     - Shorter wording is acceptable if it still captures the key medical content.
   - The student answer must NOT:
     - Present a diagnosis/finding/location/cause that clearly contradicts the report or the reference answer;
     - Be overly vague (e.g. “something is wrong”, “there is an abnormality”) while the question expects a specific diagnosis or finding;
     - Make clear mistakes on crucial aspects (e.g. wrong side, wrong structure, wrong type of pathology, wrong severity grade).

4. When to mark the student as incorrect (result = "no")
   - The student answer misses the key information required by the question;
   - The main content of the student answer conflicts with the reference answer and/or the report;
   - The answer is too general or ambiguous and does not demonstrate actual understanding of the required specific point;
   - The answer contains serious medical errors, even if some parts sound similar to the reference.

5. Leniency vs strictness
   - Do not require exact wording; judge based on medical meaning.
   - Be tolerant of minor phrasing differences that do not affect correctness.
   - Be strict regarding the main diagnostic direction, key location, main abnormality type, key cause, and other crucial elements.

【Output format】

1. First complete your reasoning internally; do NOT output your intermediate thoughts.
2. Then output exactly one XML code block, wrapped in ```xml, with the following structure:

```xml
<response>
    <reason>Use 2–5 sentences in English to briefly explain why you judged the student answer as correct or incorrect. Mention what the question asks for, what the core elements of the reference answer are, and whether the student answer matches them or contains clear errors/omissions.</reason>
    <result>yes or no (lowercase only)</result>
</response>
````

3. Do NOT output anything outside this XML code block. No extra explanations, no additional code blocks.

---

【Chest X-ray report】
{report}

---

【Open-ended question】
{question}

---

【Reference (ground truth) answer】
{ground_truth}

---

【Student answer】
{pred}

---

Based on the above information and grading principles, decide whether the student’s answer should be considered correct,
and return your XML response strictly following the required format.

Your Prediction:
"""


class OpenEndedJudgeFormatter:
    """
    支持中英文
    支持 single_choice, multi_choice, closed_ended, open_ended
    """

    def format(self, pred, ground_truth, question, report):
        lang = self.judge_language_zh_or_eng(report)
        if lang == "zh":
            meta_prompt = Prompt_zh
        else:
            meta_prompt = Prompt_en
        prompt = meta_prompt.replace("{pred}", pred)
        prompt = prompt.replace("{ground_truth}", ground_truth)
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{report}", report)
        return prompt

    @staticmethod
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

    def parse(self, response):
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line and "<result>" in line and "</result>" in line:
                string = line

        string = self.escape_label_content(string, "<result>", "</result>")
        string = self.escape_label_content(string, "<reason>", "</reason>")
        try:
            dic = xmltodict.parse(string)
            entity = dic["response"]
            result = str(entity["result"])
            reason = entity.get("reason", "Error.")
        except Exception as e:
            if ">no<" in response.lower():
                result = "no"
            else:
                result = "yes"
            reason = "Parsed Failed."
        if "no" in result.lower():
            result = False
        else:
            result = True
        item = {
            "response": response,
            "result": result,
            "reason": reason,
        }

        return item

    @staticmethod
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
