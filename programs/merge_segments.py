def merge_segments(segment_list, tolerance):
    # Make one big list of all blocks
    all_blocks = []
    for tiny_pile in segment_list:
        for block in tiny_pile:
            all_blocks.append(block)

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
        last_start = merged_pile[-1][0]
        last_end = merged_pile[-1][1]

        # If blocks are close, hug them together
        if abs(last_end - start) <= tolerance:
            merged_pile[-1][1] = max(last_end, end)
        else:
            # Otherwise, new block in pile
            merged_pile.append([start, end])

    return merged_pile
