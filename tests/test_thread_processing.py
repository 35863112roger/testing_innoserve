from backend.threads.processing import (
    build_reply_samples,
    build_thread_text,
    clean_node_text,
    thread_tree_quality_error,
)


def sample_tree() -> dict:
    return {
        "conversation_root_id": "root",
        "nodes": [
            {
                "post_id": "root",
                "parent_id": None,
                "node_type": "root_post",
                "depth": 0,
                "username": "po",
                "text_clean": "原始貼文",
                "context_path_ids": ["root"],
            },
            {
                "post_id": "reply-1",
                "parent_id": "root",
                "node_type": "reply",
                "depth": 1,
                "username": "alice",
                "text_clean": "第一則留言",
                "context_path_ids": ["root", "reply-1"],
            },
            {
                "post_id": "reply-2",
                "parent_id": "reply-1",
                "node_type": "reply",
                "depth": 2,
                "username": "bob",
                "text_clean": "回覆第一則留言",
                "context_path_ids": ["root", "reply-1", "reply-2"],
            },
        ],
    }


def test_summary_format_matches_training_shape() -> None:
    text, truncated, used = build_thread_text(sample_tree(), max_nodes=10)
    assert text == (
        "[root_post] @po: 原始貼文\n"
        "  [reply] @alice: 第一則留言\n"
        "    [reply] @bob: 回覆第一則留言"
    )
    assert not truncated
    assert used == 3


def test_each_reply_gets_context_target_pair() -> None:
    samples, skipped = build_reply_samples(sample_tree())
    assert not skipped
    assert len(samples) == 2
    assert samples[0]["context_text"] == "[ROOT] 原始貼文"
    assert samples[0]["target_text"] == "第一則留言"
    assert samples[1]["context_text"] == "[ROOT] 原始貼文\n[PARENT] 第一則留言"
    assert samples[1]["target_text"] == "回覆第一則留言"


def test_clean_node_text_removes_leading_threads_date() -> None:
    node = {
        "username": "alice",
        "text_clean": "alice\n08/14/25\n這是留言內容",
    }

    assert clean_node_text(node) == "這是留言內容"


def test_clean_node_text_keeps_date_inside_actual_content() -> None:
    node = {
        "username": "alice",
        "text_clean": "活動日期\n08/14/25",
    }

    assert clean_node_text(node) == "活動日期 08/14/25"


def test_quality_accepts_high_exclusion_ratio_caused_by_node_limit() -> None:
    tree = {
        "nodes": [
            {"post_id": str(index)}
            for index in range(10)
        ],
        "stats": {
            "raw_discovered_node_count": 137,
            "excluded_unresolved_count": 127,
            "excluded_processed_count": 0,
            "excluded_unprocessed_count": 127,
            "processed_detail_pages": 10,
            "stopped_by_max_nodes": True,
        },
    }

    assert thread_tree_quality_error(tree) is None


def test_quality_rejects_low_retention_even_when_node_limit_was_reached() -> None:
    tree = {
        "nodes": [
            {"post_id": str(index)}
            for index in range(3)
        ],
        "stats": {
            "raw_discovered_node_count": 137,
            "excluded_unresolved_count": 134,
            "excluded_processed_count": 27,
            "excluded_unprocessed_count": 107,
            "processed_detail_pages": 30,
            "stopped_by_max_nodes": True,
        },
    }

    assert thread_tree_quality_error(tree) is not None
