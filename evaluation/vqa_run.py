import argparse
import os
import os.path as osp
import json
from typing import Dict, List
from omegaconf import OmegaConf
from format_utils.qa_format import Formatter
from format_utils.reasoner_infer_preprocess import preprocess
import multiprocessing as mp
from tqdm import tqdm
from io_utils import (
    ensure_dir,
    infer_output_dir_name,
    load_json_or_jsonl,
    normalize_image_paths,
    resolve_input_file,
    validate_infer_samples,
)

CHUNK_SIZE = 256


def chunk_list(lst, chunk_size):
    """
    将列表按照指定大小分割成多个子列表

    参数:
    lst: 要分割的原始列表
    chunk_size: 每个子列表的大小

    返回:
    分割后的子列表列表
    """
    if not isinstance(lst, list):
        raise TypeError("第一个参数必须是列表")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size必须是正整数")

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def worker_process(
        gpu_id: str,
        samples: List[Dict],
        kwargs,
        save_dir,
):
    """
    单个GPU的worker进程
    """
    if not samples:
        return

    from handlers.reasoner_handler import ReasonerHandler
    from format_utils.reasoner_response_parse import Parser

    # 设置CUDA可见设备（只对当前进程生效）
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["VLLM_PORT"] = str(60000 + int(gpu_id) * 50)

    llm = ReasonerHandler(**kwargs)
    formatter = Formatter()
    parser = Parser()

    model_inputs = []
    for sample in samples:
        question_type = sample["question_type"]
        question_info = sample["question_info"]
        idx = sample["idx"]
        question, answer = formatter.format(question_info, question_type)
        prompt = preprocess(question=question, question_type=question_type)
        sample["images"] = normalize_image_paths(
            sample["images"],
            replace_src=kwargs.get("image_path_replace_src"),
            replace_dst=kwargs.get("image_path_replace_dst"),
            max_images=kwargs.get("max_images_per_sample", 2),
        )
        model_inputs.append({"prompt": prompt, "images": sample["images"], "idx": idx})

    model_inputs_list = chunk_list(model_inputs, CHUNK_SIZE)
    samples_list = chunk_list(samples, CHUNK_SIZE)
    for model_inputs, this_samples in tqdm(zip(model_inputs_list, samples_list), total=len(model_inputs_list)):
        model_outputs = llm.process_batch(model_inputs)
        for i, item in enumerate(this_samples):
            response_list = model_outputs[i]["responses"]
            question_type = item["question_type"]
            res_list = []
            for response in response_list:
                success_flag, this_res = parser.parse(response, question_type)
                this_res["success_flag"] = success_flag
                res_list.append(this_res)
            item["response_list"] = res_list
            item["answer_info"] = res_list[0]
            idx = item["idx"]

            this_dst_path = osp.join(save_dir, f"{idx}.json")
            with open(this_dst_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=4)


def run_multi_gpu(raw_samples, args):
    gpu_ids = list(dict.fromkeys([gid.strip() for gid in args.gpu_ids.split(',')]))
    n_gpus = len(gpu_ids)

    print(f"\n多GPU数据并行模式，使用 {n_gpus} 张GPU: {gpu_ids}")

    save_dir = args.result_dir or osp.join(args.save_dir, args.exp_name)
    ensure_dir(save_dir)

    samples = []
    for sample in raw_samples:
        idx = sample["idx"]
        p = osp.join(save_dir, f"{idx}.json")
        if not osp.exists(p):
            samples.append(sample)

    samples_per_gpu = []
    for i in range(n_gpus):
        gpu_samples = samples[i::n_gpus]
        samples_per_gpu.append(gpu_samples)
        print(f"  GPU {gpu_ids[i]}: {len(gpu_samples)} 个样本")

    ctx = mp.get_context("spawn")  # 关键
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, samples_per_gpu[i], OmegaConf.to_container(args.model), save_dir),
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


def preprocess_samples(samples: List[Dict]):
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='输入配置文件路径')
    parser.add_argument('--input-file', type=str, required=True,
                        help='待推理数据文件路径，支持 .json / .jsonl')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='推理结果输出目录；默认写入 config.save_dir 下')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="模型路径")
    parser.add_argument('--max-images-per-sample', type=int, default=2,
                        help='每个样本最多使用多少张图片，默认 2')
    parser.add_argument('--image-path-replace-src', type=str, default=None,
                        help='可选：将图片路径中的旧前缀替换为新前缀')
    parser.add_argument('--image-path-replace-dst', type=str, default=None,
                        help='可选：图片路径新前缀')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    exp_name = infer_output_dir_name(args.model)
    config.exp_name = exp_name
    config.model.model_path = args.model
    config.model.max_images_per_sample = args.max_images_per_sample
    config.model.image_path_replace_src = args.image_path_replace_src
    config.model.image_path_replace_dst = args.image_path_replace_dst

    src_path = resolve_input_file(args.input_file)

    samples = load_json_or_jsonl(src_path)
    validate_infer_samples(samples)
    samples = preprocess_samples(samples)

    dataset_name = osp.splitext(osp.basename(src_path))[0]
    save_root = args.output_dir or osp.join(config.save_dir, dataset_name)
    config.save_dir = save_root
    config.result_dir = args.output_dir

    run_multi_gpu(samples, config)


if __name__ == '__main__':
    main()
