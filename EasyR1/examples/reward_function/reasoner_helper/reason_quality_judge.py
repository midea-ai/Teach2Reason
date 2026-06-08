Prompt_zh = """
你将看到一道人为设计的胸片相关试题、一份对应的胸片医学报告、该题的标准答案，以及一段解题思路（推理过程）。

你的任务：综合这些信息，判断这段解题思路是否“合格”，并给出理由。这里的“合格”，可以理解为：这段解题思路是否可以被当作一个靠谱、可供参考的标准解题过程。

【可用信息】

- 胸片医学报告（可视为该病例的影像学“真相”）：
{report}

- 试题题干：
{question}

- 该题的标准答案：
{answer}

- 解题思路（待评估）：
{reason}

【评估时请重点考虑以下维度】

1. 与标准答案的一致性
   - 这段解题思路最终指向的结论，是否与给定的标准答案 {answer} 一致（至少在语义上不矛盾）？
   - 推理过程中是否清楚地说明“为什么这个答案是合理的”，而不是反向暗示其他答案更合理？

2. 与病例/报告的医学一致性
   - 解题思路中提到的影像学所见、异常、诊断倾向等，是否大体符合医学报告的内容和方向？
   - 是否出现与报告明显相反或严重违背胸片常识的结论（例如把明显正常当成严重异常，或把报告明确存在的异常说成完全不存在）？

3. 推理链条是否完整、合理
   - 是否体现出合理的解题过程，而不是只给出一个结论或一句话解释？
   - 是否大致包含以下要素中的大部分：
     “观察影像 → 描述关键所见 → 结合题干问题进行分析/排除 → 得出符合标准答案的结论”？
   - 推理是否前后连贯、有因有果，而不是东一句西一句、难以理解其逻辑？

4. 信息使用是否得当
   - 是否主要依赖于题目和胸片可能出现的影像学所见进行推理，而不是大量胡编不存在的检查结果或影像征象？
   - 是否避免明显照抄医学报告的措辞，或者在文字中直接暴露“我看到了报告/标准答案”（例如出现“报告写明……”“标准答案是……”等字样）？
   - 即使在精神上与报告内容一致，也应该更像是根据胸片读片得到的推断，而不是把报告原文当作理由。

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
- 若整体上推理逻辑合理、医学上基本正确、能支持标准答案，且没有出现严重的违背设定（例如明显引用报告原文、明说自己知道标准答案等），并且语言质量基本可接受，则判为 **yes（合格）**。
- 若存在严重问题，例如明显违背报告和常识、根本无法支持标准答案、逻辑极其混乱、直接在文字中暴露“我知道报告/答案所以倒着解释”，或者语言严重混乱/主要为非中文导致难以阅读，则判为 **no（不合格）**。
- 可以容忍一些小问题或表述不完美；关键看整体是否可以作为一个可信的“标准解题过程”。

【输出要求】

1. 请先在心里完成评估，然后用 XML 格式给出你的判断和理由。
2. 根标签为 <response>，内部包含两个子标签：
   - <reason>：用自然中文简要说明你认为合格或不合格的原因，可以同时指出优点和主要问题；
   - <result>：写 yes 或 no，表示你对这段解题思路的最终判定。

注意：
- 不要输出任何 XML 结构以外的文字；
- <result> 中只能是小写的 yes 或 no，不能带引号或其他字符。

请按以下格式输出：

```xml
<response>
    <reason>（在这里说明你判断的依据和理由，例如：是否符合标准答案和报告、推理是否完整连贯、有没有明显医学错误、“看报告/看答案”的痕迹、以及语言是否通顺且主要为中文等）</reason>
    <result>（在这里填写 yes 或 no）</result>
</response>
```

你的预测为：
"""

Prompt_en = """
You will be given a chest X-ray exam question, a corresponding radiology report, the official correct answer, and one reasoning trace (chain of thought) for solving this question.

Your task: using all this information, decide whether this reasoning trace is “acceptable” or “not acceptable”, and explain your judgment. Here, “acceptable” means that the reasoning can reasonably serve as a reliable, instructive solution process for this question.

【Available information】

- Radiology report (can be regarded as the imaging “ground truth” for this case):
{report}

- Exam question:
{question}

- Official correct answer:
{answer}

- Reasoning to be evaluated:
{reason}

【Evaluation criteria – key dimensions】

1. Consistency with the official answer
   - Does the reasoning ultimately support the given official answer {answer} (at least not contradict it in meaning)?
   - Does it clearly explain why this answer is reasonable, rather than implicitly suggesting that some other answer would be more appropriate?

2. Medical consistency with the case/report
   - Do the imaging findings, abnormalities, diagnostic tendencies, etc. mentioned in the reasoning broadly match the radiology report in terms of direction and key facts?
   - Does it avoid conclusions that are clearly opposite to the report or strongly violate basic chest imaging knowledge (e.g., treating obviously normal structures as severe pathology, or claiming an abnormality is completely absent when the report clearly states it is present)?

3. Completeness and plausibility of the reasoning chain
   - Does it present a genuine reasoning process, rather than just a bare conclusion or a single sentence explanation?
   - Does it cover most of the following elements:
     “inspect the image → describe key findings → analyze in light of the question → rule out less likely possibilities → arrive at a conclusion that matches the official answer”?
   - Is the reasoning coherent and causal, instead of fragmented and logically unclear?

4. Proper use of information
   - Does it mainly rely on findings that could reasonably be observed on the chest X-ray and on the information in the question, rather than inventing non-existent tests or imaging features?
   - Does it avoid obviously copying phrases from the radiology report, or explicitly revealing that it saw the report/answer (e.g., “the report states…”, “the correct answer is…”)? 
   - Even if it is aligned with the spirit of the report, it should read like reasoning based on image interpretation, not like quoting the report as the main justification.

5. Clarity of explanation
   - Is the language clear enough that someone with basic medical background can understand what the reasoning is saying?
   - Does it stay focused on what this particular question is asking, without long digressions into irrelevant content?

6. Language quality and readability
   - Is the text reasonably fluent, with basically correct grammar, and without so many typos, nonsense words, or scrambled word order that it becomes hard to understand?
   - Check whether the main language of the reasoning is English:
     - It may include some non-English technical terms or abbreviations, but the majority of the text should be readable English.
     - If most of the reasoning is in another language, or consists largely of gibberish/garbled characters, it should be judged as not acceptable.

【Decision rule (you make the final call)】

You should integrate all the above dimensions and decide whether the reasoning is “acceptable”:

- If overall the reasoning is medically sound, logically reasonable, supports the official answer, does not seriously violate the constraints (e.g., does not blatantly quote the report or admit to knowing the official answer), and the language quality is basically acceptable, you may judge it as **yes (acceptable)**.
- If there are serious problems—such as strong conflict with the report and basic medical knowledge, failure to support the official answer, extremely chaotic or incoherent logic, explicit signs of “I saw the report/answer and I’m explaining backwards”, or very poor language/mostly non-English so that it is hard to read—then you should judge it as **no (not acceptable)**.
- Minor imperfections or slightly suboptimal wording are allowed; the key question is whether it can reasonably function as a trustworthy “standard reasoning process” for this question.

【Output requirements】

1. First, complete your internal evaluation, then provide your judgment and justification in XML format.
2. Use <response> as the root tag, with two child tags:
   - <reason>: briefly explain, in natural English, why you consider the reasoning acceptable or not acceptable. You may mention both strengths and major flaws.
   - <result>: write yes or no, indicating your final judgment on the reasoning trace.

Important:
- Do NOT output anything outside the XML structure.
- Inside <result>, you must output only the lowercase word yes or no, without quotes or any other characters.

Please output in the following format:

```xml
<response>
    <reason>(Explain your basis and reasoning here, e.g., whether it matches the official answer and the report, whether the reasoning is complete and coherent, whether there are obvious medical errors or signs of having seen the report/answer, and whether the language is readable and mainly in English.)</reason>
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
