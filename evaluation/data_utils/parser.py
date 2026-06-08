import re


class Parser:
    pattern_options = re.compile(r'([A-Z]):\s*(.*?)(?=\s*,\s*[A-Z]:|$)')

    def parse(self, question, answer, question_type):
        try:
            res = getattr(self, f"parse_{question_type}")(question, answer)
        except Exception as e:
            return None
        return res

    def parse_closed_ended(self, question, answer):
        assert "Please answer the question with \"Yes\" or \"No\":" in question, question
        question = question.replace("Please answer the question with \"Yes\" or \"No\":", "").strip()
        result = True
        if "no" in answer.lower():
            result = False
        question_info = {
            "question": question,
            "answer": result,
            "question_type": "closed_ended",
        }
        return question_info

    def parse_open_ended(self, question, answer):
        assert "Please answer the question:" in question, question
        question = question.replace("Please answer the question:", "").strip()
        question_info = {
            "question": question,
            "answer": answer,
            "question_type": "open_ended",
        }
        return question_info

    def parse_single_choice(self, question, answer):
        assert "Question:" in question and "Options:" in question, question
        question = question.replace("Question:", "<split>").strip().replace("Options:", "<split>").strip()
        question, choices = question.split("<split>")[-2:]
        question = question.strip()
        choices = self.extract_choices(choices)
        answer = answer.strip()
        question_info = {
            "question": question,
            "answer": answer,
            "choices": choices,
            "question_type": "single_choice",
        }
        return question_info

    def parse_multi_choice(self, question, answer):
        assert "Question:" in question and "Options:" in question, question
        question = question.replace("Question:", "<split>").strip().replace("Options:", "<split>").strip()
        question, choices = question.split("<split>")[-2:]
        question = question.strip()
        choices = self.extract_choices(choices)
        answer = [_.strip() for _ in answer.split(",")]
        for c in answer:
            assert c in choices, f"{c} not in {choices}"
        question_info = {
            "question": question,
            "answer": answer,
            "choices": choices,
            "question_type": "multi_choice",
        }
        return question_info

    @staticmethod
    def extract_choices(text):
        """
        最终推荐方案：处理各种情况
        """
        # 移除外部的方括号
        text = re.sub(r'^\[|\]$', '', text.strip())

        # 使用正则表达式匹配所有选项
        # 解释：匹配 [A-Z]: 后跟任意字符（非贪婪），直到遇到下一个 [A-Z]: 或字符串结束

        options = {}
        for match in re.finditer(Parser.pattern_options, text):
            label = match.group(1)
            content = match.group(2).strip()
            # 移除可能的多余逗号
            if content.endswith(','):
                content = content[:-1].strip()
            options[label] = content

        return options
