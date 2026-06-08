import re
import xmltodict


class Parser:
    """
    适用于single_choice, multi_choice, closed_ended, open_ended
    适用于zh/en
    response的格式应该为：
    ```xml
    <response>
        <reason>...</reason>
        <answer>...</answer>
    </response>
    ```
    将reasoner的response解析成reason+answer的形式
    """

    DEFAULT_RESULT = {
        "single_choice": {
            "reason": "reason",
            "answer": "A",
        },
        "multi_choice": {
            "reason": "reason",
            "answer": ["A", "B", "C"],
        },
        "closed_ended": {
            "reason": "reason",
            "answer": True
        },
        "open_ended": {
            "reason": "reason",
            "answer": "answer"
        }
    }

    def parse(self, response, question_type):
        try:
            xml_string = self.extract_xml(response)
            dic = xmltodict.parse(xml_string)["response"]
            reason = dic["reason"].strip()
            answer = dic["answer"].strip()
        except Exception as e:
            res = self.DEFAULT_RESULT[question_type].copy()
            res["exception"] = f"invalid xml response: {e}"
            return False, res

        try:
            answer = getattr(self, f"{question_type}_parse")(answer)
        except Exception as e:
            res = self.DEFAULT_RESULT[question_type].copy()
            res["exception"] = f"invalid answer response: {e}"
            return False, res

        res = {
            "reason": reason,
            "answer": answer,
        }
        return True, res

    def single_choice_parse(self, answer_string):
        answer = answer_string.strip()
        assert len(answer) == 1, f"'{answer}' is not valid for single choice response."
        return answer_string

    def multi_choice_parse(self, answer_string):
        answer = sorted(eval(answer_string.strip()))
        assert isinstance(answer, list), f"'{answer_string}' is not valid for multi choice response."
        for a in answer:
            assert len(a) == 1, f"'{a}' is not a valid option for multi choice response."
        return answer

    def closed_ended_parse(self, answer_string):
        answer = True
        for c in ["no", "否", "不"]:
            if c in answer_string.lower():
                answer = False
                break
        return answer

    def open_ended_parse(self, answer_string):
        return answer_string.strip()

    @staticmethod
    def extract_xml(response):
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line and "<reason>" in line and "</reason>" in line:
                string = line
        string = Parser.escape_label_content(string, "<reason>", "</reason>")
        string = Parser.escape_label_content(string, "<answer>", "</answer>")
        return string

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
