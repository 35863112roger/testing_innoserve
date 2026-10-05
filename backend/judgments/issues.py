"""Behavior-to-search vocabulary. Search suggestions, not legal determinations."""

"""
用途：
- 從風險留言與摘要找出線索詞。
- 判斷可能的議題方向。
- 提供法律詞與行為同義詞。
- 協助產生司法院查詢條件。
"""

ISSUES = [
    {"name": "性意味言語／影像", "label": "性騷擾", "cues": ["裸照", "胸部", "奶子", "奶頭", "上床", "做愛", "摸你", "摸奶", "約炮", "性騷擾"],
     "legal": ["性騷擾"], "behavior": ["言語", "訊息", "照片"]},
    {"name": "貶抑／辱罵", "label": "侮辱貶抑", "cues": ["廢物", "白痴", "垃圾", "智障", "低能", "醜八怪", "去死", "婊子", "破麻", "沒人要"],
     "legal": ["公然侮辱", "妨害名譽"], "behavior": ["辱罵", "貶抑", "羞辱"]},
    {"name": "威脅人身安全", "label": "威脅恐嚇", "cues": ["打死", "殺你", "堵你", "弄死", "砍你", "殺全家", "讓你死", "找人打", "小心一點"],
     "legal": ["恐嚇", "恐嚇危害安全"], "behavior": ["威脅", "加害", "生命"]},
    {"name": "排擠／群體羞辱", "label": "霸凌風險", "cues": ["排擠", "大家不要理", "沒人想理", "每天嘲笑", "公審", "帶頭嘲笑", "霸凌"],
     "legal": ["霸凌", "公然侮辱", "侵權行為"], "behavior": ["排擠", "嘲笑", "羞辱"]},
    {"name": "反覆聯絡／跟蹤", "label": "一般騷擾", "cues": ["一直私訊", "每天傳", "不要再聯絡", "一直傳", "不停傳", "跟蹤", "糾纏", "騷擾"],
     "legal": ["跟蹤騷擾", "騷擾"], "behavior": ["反覆", "聯絡", "訊息"]},
]


def issue_profile(text):
    return [{**issue, "matched": [word for word in issue["cues"] if word in text.lower()]}
            for issue in ISSUES if any(word in text.lower() for word in issue["cues"])]
