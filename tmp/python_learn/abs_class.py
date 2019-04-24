from abc import ABC, abstractmethod

class A:
    @abstractmethod
    def fun1(self, *args):
        pass

    def fun2(self):
        print('fn2')
        self.fun1()
        print('fn1')