import argparse
import os
import os.path as osp
import json
from typing import Dict, List
from omegaconf import OmegaConf
from format_utils.qa_format import Formatter
import multiprocessing as mp
from io_utils import ensure_dir, load_prediction_items


def worker_process(
        gpu_id: str,
        samples: List[Dict],
        kwargs,
        save_dir,
):
    """
    单个GPU的worker进程
    """

    if len(samples) == 0:
        return

    from handlers.judge_handler import QwenChatbotHandler
    from format_utils.open_ended_utils import OpenEndedJudgeFormatter

    # 设置CUDA可见设备（只对当前进程生效）
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["VLLM_PORT"] = str(60000 + int(gpu_id) * 50 + 5)

    llm = QwenChatbotHandler(**kwargs)
    formatter = Formatter()
    open_ended_formatter = OpenEndedJudgeFormatter()

    model_inputs = []
    for sample in samples:
        question_type = sample["question_type"]
        question_info = sample["question_info"]
        answer_info = sample["answer_info"]
        idx = sample["idx"]
        question, answer = formatter.format(question_info, question_type)
        prompt = open_ended_formatter.format(question=question, pred=answer_info["answer"],
                                             ground_truth=question_info["answer"])
        model_inputs.append({"prompt": prompt, "idx": idx})

    model_outputs = llm.process_batch(model_inputs)
    results = []
    for i, item in enumerate(samples):
        responses = model_outputs[i]
        evaluate_results = []
        for response in responses:
            evaluate_result = open_ended_formatter.parse(response)
            evaluate_result["gt"] = item["question_info"]["answer"]
            evaluate_result["pred"] = item["answer_info"]["answer"]
            evaluate_results.append(evaluate_result)
        item["evaluate_result"] = evaluate_results
        results.append(item)

    with open(osp.join(save_dir, f"results_{gpu_id}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def run_multi_gpu(samples, args):
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)

    print(f"\n多GPU数据并行模式，使用 {n_gpus} 张GPU: {gpu_ids}")

    samples_per_gpu = []
    for i in range(n_gpus):
        gpu_samples = samples[i::n_gpus]
        samples_per_gpu.append(gpu_samples)
        print(f"  GPU {gpu_ids[i]}: {len(gpu_samples)} 个样本")

    save_dir = args.save_dir
    ensure_dir(save_dir)

    ctx = mp.get_context("spawn")  # 关键
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, samples_per_gpu[i], OmegaConf.to_container(args.qwen_chatbot), save_dir),
        )
        p.start()
        processes.append(p)

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True

    if failed:
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise RuntimeError("至少一个 worker 失败，已终止其余 worker。")


def process_samples(samples):
    open_ended_samples = []
    other_samples = []
    for item in samples:
        question_type = item["question_type"]
        if question_type == "open_ended":
            open_ended_samples.append(item)
            continue

        gt = item["question_info"]["answer"]
        pred = item["answer_info"]["answer"]
        success_flag = item["answer_info"]["success_flag"]
        if question_type == "single_choice" or question_type == "closed_ended":
            result = gt == pred and success_flag
            item["evaluate_result"] = {
                "result": result,
                "pred": pred,
                "gt": gt,
            }
        else:  # multi choice
            gt = ",".join(sorted(gt))
            pred = ",".join(sorted(pred))
            result = gt == pred and success_flag
            item["evaluate_result"] = {
                "result": result,
                "pred": item["answer_info"]["answer"],
                "gt": item["question_info"]["answer"]
            }
        other_samples.append(item)
    return open_ended_samples, other_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='输入配置文件路径')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Reasoner 推理结果目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='验证结果输出目录；默认写入 pred_dir/evaluate_results')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    src_path = args.pred_dir
    samples = load_prediction_items(src_path)
    open_ended_samples, other_samples = process_samples(samples)
    save_dir = args.output_dir or osp.join(src_path, "evaluate_results")
    ensure_dir(save_dir)
    config.save_dir = save_dir

    run_multi_gpu(open_ended_samples, config)

    with open(osp.join(save_dir, "other_samples.json"), "w", encoding="utf-8") as f:
        json.dump(other_samples, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
