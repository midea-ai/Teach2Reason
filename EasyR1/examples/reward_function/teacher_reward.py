import os
import re
import json
import uuid
import os.path as osp
from typing import List, Dict
from importlib.util import spec_from_file_location, module_from_spec
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

STORAGE_PATH = os.getenv("STORAGE_PATH")
EXP_NAME = os.getenv("EXP_NAME") or "teacher_temp_results"
if STORAGE_PATH is None:
    STORAGE_PATH = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

TEMP_RESULTS_DIR = osp.join(STORAGE_PATH, "all_temp_results", EXP_NAME)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

CURRENT_ITER = 0

reasoner_helper_dir = osp.join(osp.dirname(osp.abspath(__file__)), "reasoner_helper")
teacher_helper_dir = osp.join(osp.dirname(osp.abspath(__file__)), "teacher_helper")


def import_module(module_name, helper_dir):
    path = osp.join(helper_dir, module_name + ".py")
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


response_parser = import_module("teacher_response_parse", teacher_helper_dir).Parser()
teacher_inputs_formatter = import_module("teacher_inputs_format", reasoner_helper_dir).Formatter()
chatbot = import_module("service_utils", reasoner_helper_dir).Chatbot()
teacher_inputs_preprocess = import_module("teacher_infer_preprocess", reasoner_helper_dir).preprocess
compare_utils = import_module("compare_with_teacher_utils", reasoner_helper_dir)
reason_judge_preprocess = import_module("reason_quality_judge", teacher_helper_dir).preprocess


def compute_judgement_score(data):
    model_inputs = [{
        "prompt": reason_judge_preprocess(**item)
    } for item in data]
    scores = chatbot.llm_judge(model_inputs)
    return scores


def calculate_comparison_scores(data, compare_results):
    compare_info = defaultdict(list)
    for item in compare_results:
        idx = item["idx"]
        compare_info[idx].append(item)
    results = []
    for item in data:
        idx = item["idx"]
        compare_res = compare_info.get(idx,
                                       [{"result": False}])  # wrong prediction don't need calculate comparison scores
        scores = [_["result"] for _ in compare_res]
        score = sum(scores) / len(scores) if scores else 0
        item["comparison_score"] = score
        item["comparison_response"] = compare_res
        results.append(item)
    return results


def parse_reward(data):
    result = defaultdict(list)
    score_info_list = [_["score_info"] for _ in data]
    for score_info in score_info_list:
        for k, v in score_info.items():
            result[k].append(v)
    res = {k: float(np.mean(v)) for k, v in result.items()}

    return res


def check_text(text: str) -> bool:
    """
    如果文本中出现类似下面的结构，则返回 False，否则返回 True：
    - the report
    - the medical report
    - the final medical report
    - the X-ray report
    - the chest X-ray report
    - this report's / that medical report's

    规则：
    - report 前允许 0~2 个词
    - 前缀限定为 the / a / an / this / that / these / those
    - 中间词可包含连字符或撇号
    - 支持 report's / report’s
    """
    word = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
    pattern = re.compile(
        rf"\b(?:the|a|an|this|that|these|those)"
        rf"(?:\s+{word}){{0,2}}"
        rf"\s+report(?:['’]s)?\b",
        re.IGNORECASE
    )
    return pattern.search(text) is None


def save_this_iter_rollout(rollout_data, split):
    global CURRENT_ITER
    CURRENT_ITER += 1
    file_path = osp.join(TEMP_RESULTS_DIR, f"iter_{CURRENT_ITER}_{split}.json")
    items = rollout_data
    reward_info = parse_reward(items)
    items = sorted(items, key=lambda x: x["sample_id"])
    result = {
        "reward_info": reward_info,
        "results": items
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"[reward] ERROR saving rollout data: {e}")
        return False


def compute_score(reward_inputs) -> List[Dict[str, float]]:
    responses = [_["response"] for _ in reward_inputs]
    meta_data_list = [_["meta_data"] for _ in reward_inputs]

    data = []
    teacher_infer_inputs = {}

    for q_id, (response, meta_data) in enumerate(zip(responses, meta_data_list)):
        sample_id = meta_data["idx"]
        uid = str(uuid.uuid4())
        question_type = meta_data["question_type"]
        question_info = meta_data["question_info"]  # question, answer, choices
        question, answer = teacher_inputs_formatter.format(question_info, question_type)
        report = meta_data["report"]
        data_source = meta_data["data_source"]
        split = meta_data["split"]

        success_flag, answer_info = response_parser.parse(response)

        this_item = {
            "sample_id": sample_id,
            "idx": uid,
            "question_type": question_type,
            "report": report,
            "question": question,
            "answer": answer,  # 字符串格式的标准答案
            "reason": answer_info["reason"],
            "question_info": question_info,  # 包含了题目和标准答案
            "answer_info": answer_info,  # reasoner的回答
            "success_flag": success_flag,
            "response": response,
            "data_source": data_source,
            "split": split,
        }
        data.append(this_item)

        teacher_inputs = teacher_inputs_preprocess(
            question=question,
            answer=answer,
            report=report,
            question_type=question_type,
        )
        teacher_infer_inputs[sample_id] = {"prompt": teacher_inputs, "images": meta_data["images"]}

    teacher_infer_inputs_sample_ids = list(teacher_infer_inputs.keys())
    teacher_infer_inputs_item_list = [teacher_infer_inputs[k] for k in teacher_infer_inputs_sample_ids]
    valid_flags = [_["success_flag"] for _ in data]
    split = "train" if data[0]["split"] == "train" else "val"

    with ThreadPoolExecutor(max_workers=2) as ex:
        teacher_cots = ex.submit(chatbot.teacher_infer, teacher_infer_inputs_item_list)
        reason_judge_results = compute_judgement_score(data)
        teacher_cots = teacher_cots.result()

    teacher_cot_outputs = {k: v for k, v in zip(teacher_infer_inputs_sample_ids, teacher_cots)}
    # prepare compare inputs
    total_compare_inputs = []
    for item in data:
        teacher_res = teacher_cot_outputs[item["sample_id"]]
        idx = item["idx"]
        compare_input_list = compare_utils.preprocess(**dict(
            question=item["question"],
            answer=item["answer"],
            report=item["report"],
            reason=item["answer_info"]["reason"],
            teacher_reasons=[_["reason"] for _ in teacher_res["response_list"]],
        ))
        for compare_input in compare_input_list:
            compare_input["idx"] = idx
        total_compare_inputs.extend(compare_input_list)

    print("Start compare cots...")
    compare_results = chatbot.compare_with_teacher_cots(total_compare_inputs)
    print("End compare cots...")
    data = calculate_comparison_scores(data, compare_results)

    scores = []
    for i, this_item in enumerate(data):
        valid_flag = valid_flags[i]
        reason_judge_result = reason_judge_results[i]
        this_item["reason_judge_result"] = reason_judge_result
        reason_judge_score = int(reason_judge_result["result"] and check_text(this_item["reason"]))
        comparison_score = this_item["comparison_score"]
        final_score = (comparison_score + 1) * int(valid_flag)

        score_info = {
            "comparison_score": comparison_score,
            "judge_score": reason_judge_score,  # cot是否合格
            "format": int(valid_flag),
            "overall": final_score,
        }

        scores.append(score_info)
        this_item["score_info"] = score_info
    save_this_iter_rollout(data, split)
    return scores
