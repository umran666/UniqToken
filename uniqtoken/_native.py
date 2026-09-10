"""Availability checks for optional native operations; computation errors propagate."""

import logging


def native_text_supported(text):
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        logging.getLogger("uniqtoken.native").warning(
            "lone surrogate cannot cross the UTF-8 native boundary; using Python implementation"
        )
        return False
    return True


def native_function(module, name):
    function = getattr(module, name, None)
    if function is None:
        logging.getLogger("uniqtoken.native").warning("%s unavailable; using Python implementation", name)
    return function
