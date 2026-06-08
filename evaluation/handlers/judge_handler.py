from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm


class QwenChatbotHandler:
    def __init__(
            self,
            model_path, tensor_parallel_size, max_model_len, gpu_mem_util,
            max_tokens, temperature, top_p,
            n,
            batch_size,
    ):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )
        self.batch_size = batch_size

    @staticmethod
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

    def process_batch(self, batch):

        total_model_inputs = []
        for item in batch:
            prompt = item["prompt"]
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking_mode='off',
            )
            total_model_inputs.append(text)

        if self.batch_size > 0:
            batch_list = self.chunk_list(total_model_inputs, self.batch_size)
        else:
            batch_list = [total_model_inputs]
        total_results = []

        for micro_batch in tqdm(batch_list):
            outputs = self.llm.generate(micro_batch, self.sampling_params, use_tqdm=False)
            outputs = [[candidate.text for candidate in item.outputs] for item in outputs]
            total_results.extend(outputs)
        return total_results
