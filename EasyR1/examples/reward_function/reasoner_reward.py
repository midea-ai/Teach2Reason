import os
import json
import uuid
import os.path as osp
from typing import List, Dict
from importlib.util import spec_from_file_location, module_from_spec
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np

STORAGE_PATH = os.getenv("STORAGE_PATH")
EXP_NAME = os.getenv("EXP_NAME") or "reasoner_temp_results"
if STORAGE_PATH is None:
    STORAGE_PATH = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

TEMP_RESULTS_DIR = osp.join(STORAGE_PATH, "all_temp_results", EXP_NAME)
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

CURRENT_ITER = 0

reasoner_helper_dir = osp.join(osp.dirname(osp.abspath(__file__)), "reasoner_helper")
scpo_helper_dir = osp.join(osp.dirname(osp.abspath(__file__)), "scpo_score")


def import_module(module_name, helper_dir=reasoner_helper_dir):
    path = osp.join(helper_dir, module_name + ".py")
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


response_parser = import_module("reasoner_response_parse").Parser()
teacher_inputs_formatter = import_module("teacher_inputs_format").Formatter()
chatbot = import_module("service_utils").Chatbot()
teacher_inputs_preprocess = import_module("teacher_infer_preprocess").preprocess
compare_utils = import_module("compare_with_teacher_utils")
apply_bonus_and_compute_advantages = import_module("build_advantage",
                                                   scpo_helper_dir).build_reasoner_training_signals_batch


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

    comparison_scores = []
    for item in data:
        flag = item["success_flag"] * item["score_info"]["answer_correctness"] > 0
        if flag:
            comparison_score = item["comparison_score"]
            comparison_scores.append(comparison_score)
    res["comparison_score"] = float(np.mean(comparison_scores))
    return res


def save_this_iter_rollout(rollout_data, split):
    global CURRENT_ITER
    CURRENT_ITER += 1
    file_path = osp.join(TEMP_RESULTS_DIR, f"iter_{CURRENT_ITER}_{split}.json")
    items = rollout_data
    acc_info = parse_acc(items)
    reward_info = parse_reward(items)
    items = sorted(items, key=lambda x: x["sample_id"])
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
        # 先获得teacher输出的CoT
        teacher_cots = ex.submit(chatbot.teacher_infer, teacher_infer_inputs_item_list)

        print("Start calculate accuracy...")
        data = chatbot.accuracy_score(data)
        print("End calculate accuracy...")

        teacher_cots = teacher_cots.result()  # 获得teacher的cot（每个item 10个）

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
        accuracy_result = item["accuracy_result"]
        if accuracy_result or split != "train":  # 只测预测正确的样本的compare score
            total_compare_inputs.extend(compare_input_list)

    print("Start compare cots...")
    compare_results = chatbot.compare_with_teacher_cots(total_compare_inputs)
    print("End compare cots...")
    data = calculate_comparison_scores(data, compare_results)

    ## calculate lambda for each group
    reward_info = defaultdict(list)
    comp_info = defaultdict(list)
    idx_info = defaultdict(list)
    sample_ids = set()

    for i, this_item in enumerate(data):
        sample_id = this_item["sample_id"]
        idx = this_item["idx"]
        sample_ids.add(sample_id)
        ans_score = int(this_item["accuracy_result"])
        comp_score = this_item["comparison_score"]
        fmt_score = int(valid_flags[i])
        adv_base_score = (ans_score + 1) * fmt_score
        idx_info[sample_id].append(idx)
        reward_info[sample_id].append(adv_base_score)
        comp_info[sample_id].append(comp_score)

    sample_ids = list(sample_ids)
    comp_matrix = [comp_info[sample_id] for sample_id in sample_ids]
    reward_base_matrix = [reward_info[sample_id] for sample_id in sample_ids]

    adv_res = apply_bonus_and_compute_advantages(
        base_rewards_batch=reward_base_matrix,
        comp_scores_batch=comp_matrix,
        gamma=0.3,
        alpha_pos=0.9,
        alpha_neg=0.9,
    )

    a_base_batch = adv_res["a_base_batch"]
    bonus_batch = adv_res["bonus_batch"]
    final_adv_batch = adv_res["final_adv_batch"]
    group_info_batch = adv_res["group_info"]
    bonus_info = {}
    a_base_info = {}
    final_adv_info = {}
    is_degenerate_info = {}
    for i, sample_id in enumerate(sample_ids):
        is_degenerate_info[sample_id] = group_info_batch[i]["is_degenerate"]
        for j, idx in enumerate(idx_info[sample_id]):
            bonus_info[idx] = float(bonus_batch[i][j])
            a_base_info[idx] = float(a_base_batch[i][j])
            final_adv_info[idx] = float(final_adv_batch[i][j])

    # end of lambda calculations

    scores = []
    for i, this_item in enumerate(data):
        valid_flag = valid_flags[i]
        idx = this_item["idx"]
        sample_id = this_item["sample_id"]
        is_degenerate_flag = is_degenerate_info[sample_id]
        bonus_value = bonus_info[idx]
        a_base_value = a_base_info[idx]
        final_adv_value = final_adv_info[idx]

        answer_correctness_score = int(this_item["accuracy_result"])  # 0 or 1
        comparison_score = this_item["comparison_score"]

        score_info = {
            "answer_correctness": answer_correctness_score,
            "format": int(valid_flag),
            "overall": final_adv_value if not is_degenerate_flag else a_base_value,  # 根据是否degenerate决定final adv
        }

        if split != "train":
            score_info["comparison_score"] = comparison_score
            score_info["overall"] = (answer_correctness_score + 1) * int(valid_flag)  # 准确率
        scores.append(score_info)
        this_item["score_info"] = score_info
        this_item["bonus_value"] = bonus_value
        this_item["adv_value"] = final_adv_value
        this_item["a_base_value"] = a_base_value
    save_this_iter_rollout(data, split)
    return scores
