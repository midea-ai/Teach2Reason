from openai import OpenAI
from tqdm import tqdm


class APIQwenChatbotHandler:
    def __init__(
            self,
            model_name, base_url, api_key,
            max_tokens, temperature, top_p,
            n=5,
    ):
        self.llm = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.sampling_params = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )

    def chat(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        resp = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
            **self.sampling_params,
        )
        res = [_.message.content for _ in resp.choices]
        return res

    def process_batch(self, batch):
        results = []
        for item in tqdm(batch):
            prompt = item["prompt"]
            response = self.chat(prompt)
            results.append(response)
        return results
