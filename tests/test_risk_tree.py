from unittest.mock import patch

from frontend.risk_tree import build_risk_subtree, render_risk_tree


def test_only_risk_branches_and_their_ancestors_are_kept() -> None:
    comments = [
        {"target_id": "a", "parent_id": "root", "depth": 1, "label": "general", "text": "A"},
        {"target_id": "b", "parent_id": "a", "depth": 2, "label": "harassment", "text": "B"},
        {"target_id": "c", "parent_id": "root", "depth": 1, "label": "friendly", "text": "C"},
        {"target_id": "d", "parent_id": "c", "depth": 2, "label": "cyberbullying", "text": "D"},
        {"target_id": "e", "parent_id": "root", "depth": 1, "label": "general", "text": "E"},
    ]

    tree = build_risk_subtree(
        comments,
        "root",
        "https://www.threads.com/@u/post/root",
        root_username="u",
        root_text="測試原始貼文",
    )

    assert {node["id"] for node in tree["nodes"]} == {"root", "a", "b", "c", "d"}
    assert {node["id"] for node in tree["nodes"] if node["label"] == "context"} == {"a", "c"}
    assert {(edge["from"], edge["to"]) for edge in tree["edges"]} == {
        ("root", "a"),
        ("a", "b"),
        ("root", "c"),
        ("c", "d"),
    }
    assert tree["risk_count"] == 2
    assert tree["context_count"] == 2
    assert tree["nodes"][0]["username"] == "u"
    assert tree["nodes"][0]["text"] == "測試原始貼文"


def test_no_risk_comment_produces_an_empty_tree() -> None:
    tree = build_risk_subtree(
        [{"target_id": "a", "parent_id": "root", "label": "friendly"}],
        "root",
        "https://www.threads.com/@u/post/root",
    )

    assert tree == {"nodes": [], "edges": [], "risk_count": 0, "context_count": 0}


def test_unclassified_direct_parent_is_kept_from_thread_structure() -> None:
    comments = [
        {
            "target_id": "risk-depth-2",
            "parent_id": "plain-depth-1",
            "depth": 2,
            "label": "harassment",
            "text": "風險留言",
        }
    ]
    thread_nodes = [
        {"id": "root", "parent_id": None, "depth": 0, "text": "原始貼文"},
        {
            "id": "plain-depth-1",
            "parent_id": "root",
            "depth": 1,
            "username": "parent",
            "text": "沒有被判定為風險的直接父留言",
        },
        {
            "id": "risk-depth-2",
            "parent_id": "plain-depth-1",
            "depth": 2,
            "username": "child",
            "text": "風險留言",
        },
    ]

    tree = build_risk_subtree(
        comments,
        "root",
        "https://www.threads.com/@u/post/root",
        thread_nodes=thread_nodes,
    )

    assert {node["id"] for node in tree["nodes"]} == {
        "root",
        "plain-depth-1",
        "risk-depth-2",
    }
    assert next(
        node for node in tree["nodes"] if node["id"] == "plain-depth-1"
    )["label"] == "context"
    assert tree["edges"] == [
        {"from": "root", "to": "plain-depth-1"},
        {"from": "plain-depth-1", "to": "risk-depth-2"},
    ]


def test_root_is_connected_directly_to_depth_one_risk_comment() -> None:
    comments = [
        {
            "target_id": "risk-depth-1",
            "parent_id": "root",
            "depth": 1,
            "label": "cyberbullying",
            "text": "直接回覆原始貼文的風險留言",
        }
    ]
    thread_nodes = [
        {"id": "root", "parent_id": None, "depth": 0, "text": "原始貼文"},
        {
            "id": "risk-depth-1",
            "parent_id": "root",
            "depth": 1,
            "text": "直接回覆原始貼文的風險留言",
        },
    ]

    tree = build_risk_subtree(
        comments,
        "root",
        "https://www.threads.com/@u/post/root",
        thread_nodes=thread_nodes,
    )

    assert [node["id"] for node in tree["nodes"]] == ["root", "risk-depth-1"]
    assert tree["edges"] == [{"from": "root", "to": "risk-depth-1"}]
    assert tree["context_count"] == 0


def test_renderer_uses_css_arrow_instead_of_sanitized_svg() -> None:
    tree = build_risk_subtree(
        [
            {
                "target_id": "risk",
                "parent_id": "root",
                "depth": 1,
                "label": "cyberbullying",
                "text": "風險留言",
            }
        ],
        "root",
        "https://www.threads.com/@u/post/root",
    )
    rendered: list[str] = []

    with patch("frontend.risk_tree.st.html", side_effect=rendered.append):
        render_risk_tree(tree)

    assert "<svg" not in rendered[0]
    assert 'class="edge-arrow"' in rendered[0]
    assert "border-left-color:#dc2626" in rendered[0]
