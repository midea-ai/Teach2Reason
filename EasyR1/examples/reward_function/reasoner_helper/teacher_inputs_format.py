class Formatter:
    """
    适用于：single_choice, multi_choice, closed_ended, open_ended
    适用于：zh、en
    将各种题型内容转换为question, answer两个字符串，方便后续放置在各个prompt中
    """

    def format(self, question_info, question_type):
        assert question_type in ["multi_choice", "single_choice", "closed_ended", "open_ended"]
        question, answer = getattr(self, f"{question_type}_format")(question_info)
        return question, answer

    def single_choice_format(self, question_info):
        question = question_info["question"].strip()
        answer = question_info["answer"].strip()
        choices = question_info["choices"]

        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = """
Question: {question}
Choices: {choices}
"""
        else:
            format_prompt = """
问题: {question}
选项: {choices}
"""
        options = []
        for option, choice in choices.items():
            this_option = f"{option}: {choice}"
            options.append(this_option)
        options = sorted(options, key=lambda x: x.split(":")[0])
        choices = ", ".join(options)
        question = format_prompt.replace("{question}", question).replace("{choices}", choices)
        return question, answer

    def multi_choice_format(self, question_info):
        question = question_info["question"].strip()
        answer = question_info["answer"]
        choices = question_info["choices"]

        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            format_prompt = """
Question: {question}
Choices: {choices}
"""
        else:
            format_prompt = """
问题: {question}
选项: {choices}
"""
        options = []
        for option, choice in choices.items():
            this_option = f"{option}: {choice}"
            options.append(this_option)
        options = sorted(options, key=lambda x: x.split(":")[0])
        choices = ", ".join(options)
        question = format_prompt.replace("{question}", question).replace("{choices}", choices)
        answer = f"[{', '.join(answer)}]"
        return question, answer

    def closed_ended_format(self, question_info):
        question = question_info["question"].strip()
        answer = question_info["answer"]
        assert isinstance(answer, bool), f'Error: "{answer}" is not bool'
        lang = self.judge_language_zh_or_eng(question)
        if lang == "en":
            answer = "yes." if answer else "no."
        else:
            answer = "是。" if answer else "否。"
        return question, answer

    def open_ended_format(self, question_info):
        question = question_info["question"].strip()
        answer = question_info["answer"].strip()
        return question, answer

    @staticmethod
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
