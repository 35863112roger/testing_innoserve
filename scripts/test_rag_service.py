import json

from backend.rag.service import get_rag_service


service = get_rag_service()

result = service.analyze_thread(
    thread_id="manual-test",
    root_text="原始貼文正在討論某位使用者於社群上的發言。",
    statistics={
        "general": 20,
        "friendly": 2,
        "harassment": 2,
        "cyberbullying": 1,
    },
    risk_nodes=[
        {
            "node_id": "comment-1",
            "label": "harassment",
            "confidence": 0.94,
            "parent_text": "我只是表達自己的意見。",
            "text": "你這種人根本沒有資格講話。",
        },
        {
            "node_id": "comment-2",
            "label": "cyberbullying",
            "confidence": 0.89,
            "parent_text": "請不要再攻擊當事人。",
            "text": "大家每天一起去他的帳號留言。",
        },
        {
            "node_id": "comment-3",
            "label": "harassment",
            "confidence": 0.82,
            "parent_text": "這件事情可以理性討論。",
            "text": "看到你就噁心，快消失。",
        },
    ],
)

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )
)
