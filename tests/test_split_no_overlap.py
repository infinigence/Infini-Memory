from infini_memory.memory import split_content_by_newlines_non_overlapping


def test_split_content_by_newlines_non_overlapping_trims_duplicate_prefix() -> None:
    # The trim algorithm scans with a 50-char step, so the duplicated
    # block must be substantially longer than 50 chars per line.
    dup_line = "- repeated fact item number with sufficient padding text"
    dup_block = "\n".join([dup_line] * 6)
    content = f"{dup_block}\n\n{dup_block}"

    part1, part2 = split_content_by_newlines_non_overlapping(content)

    assert part1

    # Ensure the duplicate prefix is trimmed from part2.
    assert not part2.startswith(dup_line)

