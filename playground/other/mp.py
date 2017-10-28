from x import X
import multiprocessing

class A:
    def __init__(self):
        self.x = 0

    def __call__(self):
        pass

    def cb(self, _):
        self.x += 1

pool = multiprocessing.Pool()

a = A()
for _ in range(10):
    pool.apply_async(a, callback=a.cb)
pool.close()
pool.join()
print(a.x)
