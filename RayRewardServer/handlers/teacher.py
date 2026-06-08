from .base import BaseHandler
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO
import math
from PIL.Image import Image as ImageObject
from typing import Any, Union


class TeacherHandler(BaseHandler):
    def __init__(self, model_path, tensor_parallel_size, max_model_len, gpu_mem_util,
                 max_tokens, temperature, top_p, num_samples,
                 min_pixels, max_pixels):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            dtype="bfloat16",
            limit_mm_per_prompt={"image": 2},
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0,
            n=num_samples,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def process_image(
            self, image: Union[dict[str, Any], ImageObject, str]
    ) -> ImageObject:
        min_pixels = self.min_pixels
        max_pixels = self.max_pixels
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        image.load()  # avoid "Too many open files" errors
        if max_pixels is not None and (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if min_pixels is not None and (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def build_prompt_and_images(
            self,
            image_list,
            prompt,
    ):
        if isinstance(image_list, str):
            image_list = [image_list]
        image_content = []
        image_inputs = []
        for image_path in image_list:
            image = self.process_image(Image.open(image_path).convert('RGB'))
            image_content.append({
                "type": "image",
                "image": image,
            })
            image_inputs.append(image)

        messages = [
            {"role": "user", "content": image_content + [{"type": "text", "text": prompt}]}
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        results = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs}
        }
        return results

    def process_batch(self, batch):
        prompts = [item['prompt'] for item in batch]
        image_list = [item['images'] for item in batch]

        valid_chats = [self.build_prompt_and_images(imgs, p) for imgs, p in
                       zip(image_list, prompts)]
        responses = self.llm.generate(valid_chats, sampling_params=self.sampling_params, use_tqdm=True)
        results = []
        for i, item in enumerate(batch):
            response_list = responses[i]
            raw_response_list = [out.text for out in response_list.outputs]
            item["responses"] = raw_response_list
            results.append(item)
        return results

    def health(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "你好。"}]}
        ]

        # 使用 processor 的 chat template 格式化
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        dummy_inputs = {"prompt": prompt}
        try:
            _ = self.llm.generate([dummy_inputs], sampling_params=self.sampling_params, use_tqdm=True)
            ok = True
            error = None
        except Exception as e:
            ok = False
            error = f"{type(e).__name__}: {e}"

        return {
            "ok": ok,
            "error": error,
        }
