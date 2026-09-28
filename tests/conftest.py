from rbc.types import Variable


def schema_n(n):
    return tuple(Variable(f"v{i}", f"item-{i}", f"item-{i} is damaged") for i in range(n))


def var(id):
    return {"op": "var", "id": id}


def neg(arg):
    return {"op": "not", "arg": arg}


def all_of(*args):
    return {"op": "and", "args": list(args)}


def any_of(*args):
    return {"op": "or", "args": list(args)}


def xor(a, b):
    return {"op": "xor", "args": [a, b]}


def count_eq(ids, k):
    return {"op": "count_eq", "ids": list(ids), "k": k}


def count_ge(ids, k):
    return {"op": "count_ge", "ids": list(ids), "k": k}
