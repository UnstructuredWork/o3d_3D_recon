import multiprocessing as mt
from threading import Thread

def thread_method(fn):
    def run(*k, **kw):

        t = Thread(target=fn, args=k, kwargs=kw)
        # t.daemon = True
        t.start()
        return t
    return run

def process_method(fn):
    def run(*k, **kw):
        t = mt.Process(target=fn, args=k, kwargs=kw)
        t.daemon = True
        t.start()
        return t
    return run
