import typing
import sys

if not hasattr(typing, "io"):
    typing.io = typing
    sys.modules['typing.io'] = typing
