def flatten_list(nested_list: list):
    stack = list(reversed(nested_list))
    flattened_list = []

    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            flattened_list.append(item)

    return flattened_list


def char_count(s: str):
    set_s = dict()
    for char in s:
        if char in set_s:
            set_s[char] += 1
        else:
            set_s[char] = 1
    return set_s