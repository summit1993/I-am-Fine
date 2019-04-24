from functools import partial
def fn1(a, *args):
    print(a)
    fn2(*args)

def fn2(b):
    print(b)


def fn3(a, b, c='ss'):
    print(a)
    print(b)
    print(c)



if __name__ == '__main__':
    # fn1('poker', 'wuxuan')
    p_fn3 = partial(fn3, c='wuxuan', b='xuan')
    p_fn3('poker')
