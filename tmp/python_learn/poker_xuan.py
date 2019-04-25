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
    return [a, b, c]

class Xuan():
    def __init__(self):
        self.f = partial(fn3, c='wuxuan')

if __name__ == '__main__':
    # fn1('poker', 'wuxuan')
    model = Xuan()
    a = model.f('poker', 'xuan')
    print(a, type(a))
