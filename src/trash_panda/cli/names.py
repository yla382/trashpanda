"""How the commands write the names of blocks, so `plan` and `info` show them the same way."""


def group_pattern(name):
    """`blocks` -> `blocks.<N>.`: the blocks of a group are named with the group's name, a number and a dot."""
    return f"{name}.<N>."


def index_range(indices):
    """The numbers of a group's blocks: `0 to 27`, or `0` for a single block."""
    low, high = min(indices), max(indices)
    return str(low) if low == high else f"{low} to {high}"
