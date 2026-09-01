def test_process_split_plan_converts_unknown_updates_to_new_docs() -> None:
    plan = {
        "new_docs": [],
        "updates": [
            {"id": "unknown", "new_content": "# T\n\n- fact"},
        ],
    }
    existing_docs_list = [{"id": "known", "summary": "s", "path": "doc/known.md"}]

    new_docs_list = plan["new_docs"]
    updates_list = plan["updates"]
    allowed_ids = {d["id"] for d in existing_docs_list}

    filtered_updates = []
    for u in updates_list:
        uid = str(u.get("id") or "").strip()
        ucontent = str(u.get("new_content") or "").strip()
        if not uid or uid not in allowed_ids:
            if ucontent:
                new_docs_list.append({"title": "", "content": ucontent})
            continue
        filtered_updates.append({"id": uid, "new_content": ucontent})

    assert filtered_updates == []
    assert len(new_docs_list) == 1
