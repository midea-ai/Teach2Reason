import os
import re
import uuid
import requests
import xmltodict
import os.path as osp
from importlib.util import spec_from_file_location, module_from_spec


def import_module(module_name):
    reasoner_formater_dir = osp.dirname(osp.abspath(__file__))
    path = osp.join(reasoner_formater_dir, module_name + ".py")
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Chatbot:

    def __init__(self):
        self.host = os.getenv("REWARD_SERVICE_HOST", "127.0.0.1")
        self.port = int(os.getenv("REWARD_SERVICE_PORT", "8000"))
        self.judge_handler = "qwen_chatbot"  # llm_as_judgement
        self.teacher_handler = "teacher_inference"  # teacher inference CoTs

        self.open_ended_format_helper = import_module("open_ended_utils").OpenEndedJudgeFormatter()
        self.comparison_response_postprocess = import_module("compare_with_teacher_utils").postprocess

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

    def postprocess_judgement(self, response):
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

    def postprocess_teacher_response(self, response):
        raw_response = response
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line:
                string = line
        string = self.escape_label_content(string, "<response>", "</response>")
        try:
            dic = xmltodict.parse(string)
            res = dic["response"].strip()
        except Exception as e:
            res = response.replace("```xml", "").replace("```", "").strip()
            res = res.replace("<response>", "").replace("</response>", "").strip()
        return {"success_flag": True, "reason": res, "response": raw_response}

    def llm_judge(self, batch, handler_name=None):
        """
        batch: [{prompt: xxx}, ...]
        """
        if not batch:
            return []
        handler_name = handler_name or self.judge_handler
        url = f"http://{self.host}:{self.port}/{handler_name}"
        payload = {
            "items": batch
        }

        try:
            # 这里可以视情况调大 timeout（LLM 推理可能比较慢）
            resp = requests.post(url, json=payload, timeout=6000)
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as e:
            # 如果整体请求失败，就给所有样本返回一个兜底结果
            print(f"[reward] ERROR calling qwen judgement service: {e}")
            fallback = []
            for _ in batch:
                item = {
                    "response": "",
                    "result": False,
                    "reason": "",
                }
                fallback.append(item)
            return fallback

        results = resp_json.get("results", [])
        results = [self.postprocess_judgement(r) for r in results]

        if len(results) != len(batch):
            # 长度对不上，说明服务端出了什么事，做一个保守兜底
            print(
                f"[reward] WARNING: reward service returned {len(results)} results "
                f"for {len(batch)} inputs, using fallback for missing."
            )
            fixed = []
            for idx, item in enumerate(batch):
                if idx < len(results) and isinstance(results[idx], dict):
                    r = results[idx]
                else:
                    r = {
                        "response": "Respond Failed.",
                        "result": False,
                        "reason": "",
                    }
                fixed.append(r)
            return fixed
        return results

    def teacher_infer(self, batch):
        """
        调用RayRewardServer的teacher接口，拿到每个question的10个cot样本
        参数：
            data: List[Dict]，每个元素形如：
                {
                    "prompt": xxxx,
                    "images": [xxx]
                }
        返回：
            {
                "prompt": xxxx,
                "response_list": [
                    {
                        "success_flag": True,
                        "reason": teacher预测的CoT,
                        "response": teacher的response
                    }
                ]
            }
        """
        if not batch:
            return []
        url = f"http://{self.host}:{self.port}/{self.teacher_handler}"
        payload = {
            "items": batch
        }
        try:
            # 这里可以视情况调大 timeout（LLM 推理可能比较慢）
            resp = requests.post(url, json=payload, timeout=6000)
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as e:
            print(f"[reward] ERROR calling reward service handler '{self.teacher_handler}': {e}")
            fallback = []
            for item in batch:
                item["response_list"] = [{"success_flag": False, "reason": "", "response": "Error when posting."}] * 10
                fallback.append(item)
            return fallback

        results = resp_json.get("results", [])

        if len(results) != len(batch):
            # 长度对不上，说明服务端出了什么事，做一个保守兜底
            print(
                f"[reward] WARNING: reward service returned {len(results)} results "
                f"for {len(batch)} inputs, using fallback for missing."
            )
            fixed = []
            for idx, item in enumerate(batch):
                if idx < len(results) and isinstance(results[idx], dict):
                    response_list = results[idx]["responses"]
                    res_list = []
                    for response in response_list:
                        this_res = self.postprocess_teacher_response(response)
                        res_list.append(this_res)
                    item["response_list"] = res_list
                else:
                    item["response_list"] = [{"success_flag": False, "reason": "",
                                              "response": "Error when posting."}] * 10
                fixed.append(item)
            return fixed

        total_res = []
        for idx, item in enumerate(batch):
            response_list = results[idx]["responses"]
            res_list = []
            for response in response_list:
                this_res = self.postprocess_teacher_response(response)
                res_list.append(this_res)
            item["response_list"] = res_list
            total_res.append(item)
        return total_res

    def compare_with_teacher_cots(self, batch, handler_name=None):
        """
        input:
            batch: List[Dict]
                {
                    "prompt": xxxx,
                    "gt": xxx,
                    "idx": idx,
                }
        """
        if not batch:
            return []
        handler_name = handler_name or self.judge_handler
        url = f"http://{self.host}:{self.port}/{handler_name}"
        payload = {
            "items": batch
        }
        try:
            # 这里可以视情况调大 timeout（LLM 推理可能比较慢）
            resp = requests.post(url, json=payload, timeout=6000)
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as e:
            # 如果整体请求失败，就给所有样本返回一个兜底结果
            print(f"[reward] ERROR calling qwen compare with teacher service: {e}")
            fallback = []
            for _ in batch:
                item = {
                    "idx": _["idx"],
                    "gt": _["gt"],
                    "teacher_reason": _["teacher_reason"],
                    "reasoner_reason": _["reasoner_reason"],
                    "pred": "A",
                    "response": "Post Failed.",
                    "result": False,
                    "reason": "",
                }
                fallback.append(item)
            return fallback

        results = resp_json.get("results", [])
        if len(results) != len(batch):
            # 长度对不上，说明服务端出了什么事，做一个保守兜底
            print(
                f"[reward] WARNING: reward service returned {len(results)} results "
                f"for {len(batch)} inputs, using fallback for missing."
            )
            fixed = []
            for idx, item in enumerate(batch):
                if idx < len(results) and isinstance(results[idx], str):
                    response = results[idx]
                    res = self.comparison_response_postprocess(response)
                    res = {
                        "idx": item["idx"],
                        "gt": item["gt"],
                        "teacher_reason": item["teacher_reason"],
                        "reasoner_reason": item["reasoner_reason"],
                        "pred": res["result"],
                        "response": res["response"],
                        "result": res["result"] == item["gt"],
                        "reason": res["reason"],
                    }
                else:
                    res = {
                        "idx": item["idx"],
                        "gt": item["gt"],
                        "teacher_reason": item["teacher_reason"],
                        "reasoner_reason": item["reasoner_reason"],
                        "pred": "A",
                        "response": "Post Failed.",
                        "result": False,
                        "reason": "",
                    }
                fixed.append(res)
            return fixed
        total_res = []
        for idx, item in enumerate(batch):
            response = results[idx]
            res = self.comparison_response_postprocess(response)
            res = {
                "idx": item["idx"],
                "gt": item["gt"],
                "teacher_reason": item["teacher_reason"],
                "reasoner_reason": item["reasoner_reason"],
                "pred": res["result"],
                "response": res["response"],
                "result": res["result"] == item["gt"],
                "reason": res["reason"],
            }
            total_res.append(res)
        return total_res

    def accuracy_score(self, batch, handler_name=None):
        """
        batch:
            {
                "idx": idx,
                "question_info": question_info,  # 包含了标准答案
                "question_type": question_type,
                "report": report,
                "answer_info": answer_info,  # 包含了模型预测结果
                "question": question,
                "answer": answer,  # 字符串形式的标准答案
            }
        """
        open_ended_qa_list = []
        handler_name = handler_name or self.judge_handler
        for i, item in enumerate(batch):
            question_info = item["question_info"]
            question_type = item["question_type"]
            response_info = item["answer_info"]
            answer = question_info["answer"]  # ground truth
            question = question_info["question"]
            report = item["report"]

            if question_type != "open_ended":  # 直接比对
                pred = response_info["answer"]
                gt = question_info["answer"]
                if isinstance(pred, (str, bool)):  # closed_ended, single_choice
                    item["accuracy_result"] = pred == gt
                elif isinstance(pred, list):  # multi_choice
                    item["accuracy_result"] = set(pred) == set(gt)
                if not item["success_flag"]:
                    item["accuracy_result"] = False  # 算错
            else:  # open_ended
                key = str(uuid.uuid4())
                item["accuracy_result"] = key
                open_ended_info = {
                    "key": key,
                    "prompt": self.open_ended_format_helper.format(
                        response_info["answer"],
                        answer,
                        question,
                        report
                    )
                }
                open_ended_qa_list.append(open_ended_info)

        open_ended_dic = {}
        open_ended_results = self.llm_judge(open_ended_qa_list, handler_name)
        for open_ended_info, open_ended_result in zip(open_ended_qa_list, open_ended_results):
            # open_ended_result:
            # {
            #    {
            #         "response": "",
            #         "result": False,
            #         "reason": "",
            #     }
            # }
            open_ended_dic[open_ended_info.pop("key")] = open_ended_result

        for i, item in enumerate(batch):
            question_type = item["question_type"]
            if question_type == "open_ended":
                key = item["accuracy_result"]
                if key in open_ended_dic:
                    item["accuracy_result"] = open_ended_dic[key]["result"]
                    item["open_ended_response"] = open_ended_dic[key]
        return batch
