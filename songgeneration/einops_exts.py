from einops import rearrange, reduce, repeat, pack, unpack


def rearrange_many(tensors, pattern, **axes_lengths):
    return tuple(rearrange(tensor, pattern, **axes_lengths) for tensor in tensors)


def repeat_many(tensors, pattern, **axes_lengths):
    return tuple(repeat(tensor, pattern, **axes_lengths) for tensor in tensors)


def reduce_many(tensors, pattern, reduction, **axes_lengths):
    return tuple(reduce(tensor, pattern, reduction, **axes_lengths) for tensor in tensors)


def pack_one(tensor, pattern):
    packed, ps = pack([tensor], pattern)
    return packed, ps


def unpack_one(tensor, ps, pattern):
    return unpack(tensor, ps, pattern)[0]
