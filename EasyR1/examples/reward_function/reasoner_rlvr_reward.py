import os
import json
import uuid
import os.path as osp
from typing import List, Dict
from importlib.util import spec_from_file_location, module_from_spec
from collections import defaultdict
import numpy as np

STORAGE_PATH = os.getenv("STORAGE_PATH")
EXP_NAME = os.getenv("EXP_NAME") or "reasoner_rlvr_temp_results"
if STORAGE_PATH is None:
    STORAGE_PATH = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

TEMP_RESULTS_DIR = osp.join(STORAGE_PATH, "all_temp_results", EXP_NAME)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

CURRENT_ITER = 0

reasoner_helper_dir = osp.join(osp.dirname(osp.abspath(__file__)), "reasoner_helper")


def import_module(module_name):
    path = osp.join(reasoner_helper_dir, module_name + ".py")
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


response_parser = import_module("reasoner_response_parse").Parser()
teacher_inputs_formatter = import_module("teacher_inputs_format").Formatter()
chatbot = import_module("service_utils").Chatbot()


def parse_acc(data):
    result = defaultdict(list)
    for item in data:
        question_type = item["data_source"]  # data_source为单位，而非question_type
        flag = item["success_flag"] * item["score_info"]["answer_correctness"]
        result[question_type].append(flag)
    res = {k: float(np.mean(v)) for k, v in result.items()}
    return res


def parse_reward(data):
    result = defaultdict(list)
    score_info_list = [_["score_info"] for _ in data]
    for score_info in score_info_list:
        for k, v in score_info.items():
            result[k].append(v)
    res = {k: float(np.mean(v)) for k, v in result.items()}
    return res


def save_this_iter_rollout(rollout_data, split):
    global CURRENT_ITER
    CURRENT_ITER += 1
    file_path = osp.join(TEMP_RESULTS_DIR, f"iter_{CURRENT_ITER}_{split}.json")
    items = rollout_data
    acc_info = parse_acc(items)
    reward_info = parse_reward(items)

    result = {
        "accuracy_info": acc_info,
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

    for q_id, (response, meta_data) in enumerate(zip(responses, meta_data_list)):
        sample_id = meta_data["idx"]
        uid = str(uuid.uuid4())
        question_type = meta_data["question_type"]
        question_info = meta_data["question_info"]  # question, answer, choices
        question, answer = teacher_inputs_formatter.format(question_info, question_type)
        report = meta_data["report"]
        data_source = meta_data["data_source"]
        split = meta_data["split"]

        success_flag, answer_info = response_parser.parse(response, question_type)

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

    valid_flags = [_["success_flag"] for _ in data]

    split = "train" if data[0]["split"] == "train" else "val"

    print("Start calculate accuracy...")
    data = chatbot.accuracy_score(data)
    print("End calculate accuracy...")

    scores = []
    for i, this_item in enumerate(data):
        valid_flag = valid_flags[i]

        answer_correctness_score = int(this_item["accuracy_result"])  # 0 or 1

        final_score = (answer_correctness_score + 1) * int(valid_flag)

        score_info = {
            "answer_correctness": answer_correctness_score,
            "format": int(valid_flag),
            "overall": final_score
        }
        scores.append(score_info)
        this_item["score_info"] = score_info
    save_this_iter_rollout(data, split)
    return scores
