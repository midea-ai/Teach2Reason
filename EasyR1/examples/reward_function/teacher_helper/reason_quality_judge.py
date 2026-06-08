Prompt_zh = """
你将看到一道人为设计的胸片相关试题、一份对应的胸片医学报告、该题的标准答案，以及一段解题思路（推理过程）。

你的任务：综合这些信息，判断这段解题思路是否“合格”，并给出理由。这里的“合格”，是指这段解题思路是否可以被当作一个可信、可参考、且设定上自洽的标准解题过程。

特别注意：
你在评估时可以看到医学报告和标准答案，但“待评估的解题思路”本身应当被视为：**只能看到胸片图像和题目问题，看不到医学报告，也看不到标准答案**。  
因此，你要重点判断这段解题思路是否真正符合这种“只基于图像和问题作答”的设定，而不是借用了报告或答案的信息视角。

【可用信息】

- 胸片医学报告（仅供你作为评估依据使用，可视为该病例的影像学“真相”）：
{report}

- 试题题干：
{question}

- 该题的标准答案（仅供你作为评估依据使用）：
{answer}

- 解题思路（待评估）：
{reason}

【评估时请重点考虑以下维度】

1. 与标准答案的一致性
   - 这段解题思路最终指向的结论，是否与给定的标准答案 {answer} 一致（至少在语义上不矛盾）？
   - 推理过程中是否清楚说明“为什么这个答案合理”，而不是反向暗示其他答案更合理？

2. 与病例/报告的医学一致性
   - 解题思路中提到的影像学所见、异常、诊断倾向等，是否大体符合医学报告的内容和方向？
   - 是否出现与报告明显相反或严重违背胸片常识的结论（例如把明显正常当成严重异常，或把报告明确存在的异常说成完全不存在）？

3. 推理链条是否完整、合理
   - 是否体现出一个真实的解题过程，而不是只给出结论或一句话解释？
   - 是否大致包含以下要素中的大部分：
     “观察影像 → 描述关键所见 → 结合题干问题进行分析/排除 → 得出符合标准答案的结论”？
   - 推理是否前后连贯、有因有果，而不是东一句西一句、难以理解其逻辑？

4. 信息来源与视角是否合规（高优先级）
   这是本任务中的重点维度。你要判断这段解题思路是否真正像“只能看到胸片图像和题目问题的人”写出来的。

   4.1 显式泄漏（严重问题，通常应判为 no）
   - 是否直接出现表明其看过医学报告或标准答案的说法，例如：
     “根据报告……”“报告提示……”“报告写明……”“影像报告显示……”
     “标准答案是……”“已知答案……”“根据答案可知……”
     或其他等价表达。
   - 只要出现这类明显暴露信息来源的表述，通常应直接视为不合格。

   4.2 隐式泄漏 / 改写式照搬（严重时应判为 no）
   - 虽然没有直接说“根据报告/答案”，但是否明显沿用了报告原文的措辞、结构、罗列顺序、结论表达方式，使其读起来更像是在复述或轻度改写报告，而不是在模拟读片推理？
   - 是否呈现出“先知道结论，再回头拼凑理由”的痕迹，而不是从影像所见自然推到结论？
   - 如果这种痕迹明显，也应判为不合格。

   4.3 信息是否来自“图像可见 + 题干可得”
   - 解题思路中使用的信息，是否主要来自胸片上合理可见的影像学征象，以及题干中给出的信息？
   - 是否凭空引入题干和胸片中都无法支持的内容，例如：
     额外的化验结果、CT/超声/病理结果、详细病史、治疗经过、随访结论，或其他并非由胸片直接支持的信息？
   - 若大量依赖这些额外信息，说明其并非在模拟“只看图像和题目”的推理，应视为不合格或明显扣分。

   4.4 允许与不允许的边界
   - 允许：解题思路与报告在医学事实上保持一致，只要这些内容像是根据胸片影像征象和题目分析自然推导出来的。
   - 不允许：虽然医学上与报告一致，但表达方式明显暴露其依据是“报告文本”或“标准答案”，而不是“图像观察 + 问题分析”。

   4.5 本项优先级
   - 第 4 项优先级高于其他项。
   - 即使这段解题思路医学上大体正确、也支持标准答案，只要存在明显的信息来源泄漏、报告视角、答案视角、改写式照搬、或使用了不应知道的额外信息，也应优先判为 **no**。

5. 表达清晰度
   - 语言是否通顺清楚，普通医学背景的人是否大致能读懂“推理在说什么”？
   - 是否紧扣本题所问，没有长篇大论无关内容？

6. 语言质量与可读性
   - 文字是否通顺，语法是否基本正确，是否存在明显错字、乱词、语序混乱到影响理解的情况？
   - 检查解题思路文本的主要语言是否为中文：
     - 可以包含少量英文专业名词或缩写（如 COPD、ARDS 等）；
     - 如果解题思路的大部分内容为非中文（例如主要是英文、夹杂大量乱码或不可读字符），则应视为不合格。

【合格/不合格 的判定原则（你来做最终判断）】

你可以综合以上维度，自行判断是否“合格”：
- 若整体上推理逻辑合理、医学上基本正确、能支持标准答案、符合“只能看图像和题目”的设定、没有明显报告/答案泄漏痕迹，并且语言质量基本可接受，则判为 **yes（合格）**。
- 若存在严重问题，例如明显违背报告和常识、根本无法支持标准答案、逻辑极其混乱、直接或间接暴露“看过报告/知道答案”、大量使用并非由胸片和题干可得的信息，或者语言严重混乱/主要为非中文导致难以阅读，则判为 **no（不合格）**。
- 可以容忍一些小问题或表述不完美；关键看整体是否可以作为一个可信的、设定自洽的“标准解题过程”。

【输出要求】

1. 请先在心里完成评估，然后用 XML 格式给出你的判断和理由。
2. 根标签为 <response>，内部包含两个子标签：
   - <reason>：用自然中文简要说明你认为合格或不合格的原因。请特别指出：
     - 是否支持标准答案；
     - 是否与报告医学事实一致；
     - 是否存在“看过报告/答案”的显式或隐式痕迹；
     - 是否主要基于胸片可见征象和题干信息进行推理。
   - <result>：写 yes 或 no，表示你对这段解题思路的最终判定。

注意：
- 不要输出任何 XML 结构以外的文字；
- <result> 中只能是小写的 yes 或 no，不能带引号或其他字符。

请按以下格式输出：

```xml
<response>
    <reason>（在这里说明你判断的依据和理由，尤其要说明：是否符合标准答案和报告、推理是否完整连贯、有没有明显或隐性的“看报告/看答案”痕迹、以及是否像一个只能看胸片和题目的人写出来的）</reason>
    <result>（在这里填写 yes 或 no）</result>
</response>
```

你的预测为：
"""

Prompt_en = """
You will be given a chest X-ray exam question, a corresponding radiology report, the official correct answer, and one reasoning trace for solving this question.

Your task: using all this information, decide whether this reasoning trace is “acceptable” or “not acceptable”, and explain your judgment. Here, “acceptable” means that the reasoning can reasonably serve as a trustworthy, instructive, and setting-consistent solution process.

Important:
Although you are allowed to see the radiology report and the official answer for evaluation purposes, the reasoning trace being evaluated should be treated as if it was written by someone who could see only the chest X-ray image and the question, **not** the radiology report and **not** the official answer.  
Therefore, you must judge not only whether the reasoning is medically correct, but also whether it stays faithful to this constraint and does not leak report/answer knowledge.

【Available information】

- Radiology report (for evaluation only; this is the imaging “ground truth” of the case):
{report}

- Exam question:
{question}

- Official correct answer (for evaluation only):
{answer}

- Reasoning to be evaluated:
{reason}

【Evaluation criteria – key dimensions】

1. Consistency with the official answer
   - Does the reasoning ultimately support the given official answer {answer}, or at least not contradict it in meaning?
   - Does it explain why this answer is reasonable, rather than suggesting that another answer would actually be better?

2. Medical consistency with the case/report
   - Do the imaging findings, abnormalities, and diagnostic tendencies mentioned in the reasoning broadly match the report in terms of key facts and direction?
   - Does it avoid conclusions that are clearly opposite to the report or strongly inconsistent with basic chest imaging knowledge?

3. Completeness and plausibility of the reasoning chain
   - Does it show a genuine reasoning process, rather than just a conclusion or a one-sentence explanation?
   - Does it include most of the following elements:
     “inspect the image → describe key findings → analyze in light of the question → rule out less likely possibilities → reach a conclusion consistent with the official answer”?
   - Is the reasoning coherent and causal, rather than fragmented or logically unclear?

4. Source-faithfulness and perspective compliance (high priority)
   This is a core criterion. You must judge whether the reasoning truly reads like it was produced by someone who had access only to the chest X-ray image and the question.

   4.1 Explicit leakage (serious problem; usually no)
   - Does the reasoning explicitly reveal access to the report or the answer, with phrases such as:
     “the report states...”, “according to the report...”, “the radiology report shows...”
     “the correct answer is...”, “given the official answer...”, “based on the answer...”
     or equivalent expressions?
   - If such explicit leakage appears, it should usually be judged as not acceptable.

   4.2 Implicit leakage / paraphrased copying (serious cases should be no)
   - Even if it does not explicitly mention the report or answer, does it closely mirror the wording, structure, ordering, or conclusion style of the report, so that it reads like a paraphrase of the report rather than image-based reasoning?
   - Does it show signs of answer-first backward justification, i.e., starting from a known conclusion and then assembling supporting statements afterward, instead of naturally reasoning from image findings to the conclusion?
   - If this pattern is clear, it should be judged as not acceptable.

   4.3 Information should come from “image-observable findings + question”
   - Is the reasoning mainly based on findings that could reasonably be observed on the chest X-ray, together with information available from the question?
   - Does it avoid introducing unsupported extra information, such as:
     lab values, CT/ultrasound/pathology results, detailed clinical history, treatment course, follow-up conclusions, or other facts that are not justified by the chest X-ray and the question?
   - If it substantially depends on such extra information, it is not faithfully simulating image-only reasoning.

   4.4 Boundary between allowed and not allowed
   - Allowed: the reasoning may be medically consistent with the report, as long as it reads like a natural inference from image findings and the question.
   - Not allowed: the reasoning may still be medically correct, but if it clearly relies on report wording, report perspective, answer perspective, or other unavailable information, then it is not acceptable.

   4.5 Priority of this criterion
   - Criterion 4 has higher priority than the others.
   - Even if the reasoning is medically correct and supports the official answer, if it clearly leaks report/answer knowledge, paraphrases the report, uses the report’s perspective, or relies on information unavailable from the image and question, it should be judged **no**.

5. Clarity of explanation
   - Is the language clear enough that someone with basic medical knowledge can understand what the reasoning is saying?
   - Does it stay focused on the question rather than wandering into irrelevant discussion?

6. Language quality and readability
   - Is the text reasonably fluent, with basically correct grammar and without so many typos, nonsense words, or scrambled syntax that it becomes hard to understand?
   - Check whether the main language of the reasoning is English:
     - A small amount of non-English terminology or abbreviations is acceptable;
     - If most of the reasoning is in another language or is largely gibberish/garbled, it should be judged as not acceptable.

【Decision rule】

You should integrate all the dimensions above and make the final judgment:

- Judge **yes** only if the reasoning is overall medically sound, logically coherent, supports the official answer, remains faithful to the “image + question only” setting, shows no clear report/answer leakage, and has acceptable language quality.
- Judge **no** if there are serious problems, such as strong conflict with the report and basic medical knowledge, failure to support the official answer, severely chaotic logic, explicit or implicit signs of having seen the report/answer, substantial reliance on unavailable information, or very poor readability.
- Minor imperfections are acceptable. The key question is whether this could reasonably serve as a trustworthy and setting-consistent standard reasoning process.

【Output requirements】

1. First complete your internal evaluation, then output your judgment in XML format.
2. Use <response> as the root tag, with two child tags:
   - <reason>: briefly explain, in natural English, why the reasoning is acceptable or not acceptable. Be sure to mention:
     - whether it supports the official answer;
     - whether it is medically consistent with the report;
     - whether there are explicit or implicit signs that it saw the report/answer;
     - whether it mainly reasons from image-observable findings and the question.
   - <result>: write yes or no.

Important:
- Do NOT output anything outside the XML structure.
- Inside <result>, output only the lowercase word yes or no, with no quotes and no extra characters.

Please output in the following format:

```xml
<response>
    <reason>(Explain your basis here, especially whether it matches the official answer and report, whether the reasoning is coherent, whether there are explicit or implicit signs of report/answer leakage, and whether it truly reads like image-only reasoning.)</reason>
    <result>(write yes or no here)</result>
</response>
```

Your Predict:
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


def preprocess(*, report, question, answer, reason, **kwargs):
    lang = judge_language_zh_or_eng(report)
    prompt = Prompt_zh if lang == "zh" else Prompt_en
    res = prompt.replace("{report}", report)
    res = res.replace("{question}", question)
    res = res.replace("{answer}", answer)
    res = res.replace("{reason}", reason)
    return res
