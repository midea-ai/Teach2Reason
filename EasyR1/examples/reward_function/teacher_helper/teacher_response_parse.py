import re
import xmltodict


class Parser:
    """
    用于将teacher的response成功解析，
    """

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

    def parse(self, response):
        raw_response = response
        response = response.replace("```xml", "```").replace("```Xml", "```").replace("```XML", "```")

        response_list = response.split("```")
        string = ""
        for line in response_list:
            if "<response>" in line and "</response>" in line:
                string = line
        string = self.escape_label_content(string, "<response>", "</response>")
        string = string.strip()

        if not string:
            return False, {"reason": string, "response": raw_response}
        try:
            dic = xmltodict.parse(string)
            res = dic["response"]
        except Exception as e:
            res = response.replace("```xml", "").replace("```", "").strip()
            return False, {"reason": res, "response": raw_response}

        return True, {"reason": res, "response": raw_response}
