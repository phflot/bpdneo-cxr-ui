from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool


class TaskSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Task(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.error.emit(str(e))


_pool = QThreadPool.globalInstance()


def submit(fn, *args, **kwargs):
    t = Task(fn, *args, **kwargs)
    _pool.start(t)
    return t.signals
