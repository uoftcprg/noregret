"""Module for utilities."""
from importlib import import_module


def import_object(object_path):
    """Import an object from a path.

    >>> import_object('math.inf')
    inf
    >>> import_object('math')
    <module 'math' (built-in)>

    :param object_path: Import path of the object.
    :return: Imported object.
    """
    if '.' in object_path:
        path, name = object_path.rsplit('.', 1)
        obj = getattr(import_module(path), name)
    else:
        obj = __import__(object_path)

    return obj


def tuple_or_none(values):
    """Create a tuple of values or return ``None``.

    >>> print(tuple_or_none(None))
    None
    >>> tuple_or_none(['j', 'a'])
    ('j', 'a')

    :param: Optional values.
    :return: Tuple or ``None``.
    """
    return None if values is None else tuple(values)
