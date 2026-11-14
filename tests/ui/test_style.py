"""The style of the page: rules whose order matters."""

import re

from trash_panda.ui.server import STATIC


def rules():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    return [(m.group(1).strip(), m.group(2)) for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", css, flags=re.S))]


def test_a_tile_that_is_not_there_stays_hidden_whatever_else_its_tile_is():
    """A tile on its way of a model that is not selected is both `mover` and `other`: the rule that hides a tile must come last and
    be as specific as the rules about those states, or the hidden tiles of the other models show at the corner of the drawing."""
    found = [(i, sel) for i, (sel, body) in enumerate(rules()) if "opacity" in body and sel.startswith(".tile")]
    hide = [i for i, sel in found if ".hide" in sel and ".mover.hide" in sel]
    assert hide, "no rule hides a moving tile"
    assert hide[-1] == found[-1][0], "a rule about the opacity of a tile comes after the one that hides it"
    assert max(sel.count(".") for _i, sel in found if ".mover.hide" in sel) >= max(sel.count(".") for _i, sel in found)
