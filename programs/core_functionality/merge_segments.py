def merge_segments(segment_list, tolerance):
    # Make one big list of all blocks with scores
    all_blocks = []
    for tiny_pile in segment_list:
        for block in tiny_pile:
            # Keep score (default to 5 if missing)
            score = block[2] if len(block) > 2 else 5
            all_blocks.append([block[0], block[1], score])

    if not all_blocks:
        return []

    # Sort blocks by start time using simple function
    def get_start(block):
        return block[0]

    all_blocks.sort(key=get_start)

    # Start merged pile with first block
    merged_pile = [all_blocks[0]]

    # Go through each block
    for i in range(1, len(all_blocks)):
        start = all_blocks[i][0]
        end = all_blocks[i][1]
        score = all_blocks[i][2]
        last_end = merged_pile[-1][1]
        last_score = merged_pile[-1][2]

        # Merge if overlapping or within tolerance gap
        if start <= last_end + tolerance:
            merged_pile[-1][1] = max(last_end, end)
            merged_pile[-1][2] = max(last_score, score)  # Keep highest score
        else:
            # Otherwise, new block in pile
            merged_pile.append([start, end, score])

    return merged_pile
