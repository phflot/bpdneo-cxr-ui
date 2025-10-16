"""
Background Task Execution
==========================

This module provides Qt-based background task execution using QThreadPool.
It enables running long-running operations (e.g., model inference, ROI
computation) off the main GUI thread to prevent UI blocking.

Classes
-------
TaskSignals
    Qt signals for communicating task results/errors from worker threads
Task
    QRunnable wrapper for executing arbitrary functions in background

Functions
---------
submit
    Submit a function for background execution

Notes
-----
This module uses Qt's QThreadPool.globalInstance() to manage worker threads.
The thread pool automatically manages thread lifecycle and provides optimal
thread count based on CPU cores.

Examples
--------
>>> from bpd_ui.core.tasks import submit
>>>
>>> def expensive_computation(x, y):
...     import time
...     time.sleep(2)  # Simulate work
...     return x + y
>>>
>>> signals = submit(expensive_computation, 10, 20)
>>> signals.finished.connect(lambda result: print(f"Result: {result}"))
>>> signals.error.connect(lambda err: print(f"Error: {err}"))

See Also
--------
bpd_ui.ui.single_eval_tab : Uses tasks for model inference
bpd_ui.ui.preprocess_tab : Uses tasks for ROI computation
"""

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool


class TaskSignals(QObject):
    """
    Qt signals for communicating task status from worker threads.

    This class defines the signals that background tasks use to communicate
    results, errors, and progress updates back to the main GUI thread.

    Signals
    -------
    finished : Signal(object)
        Emitted when task completes successfully, carries return value
    error : Signal(str)
        Emitted when task raises exception, carries error message
    progress : Signal(int)
        Emitted for progress updates (0-100), currently unused

    Notes
    -----
    TaskSignals must inherit from QObject to use Qt's signal system. It is
    automatically created by the Task class and should not be instantiated
    directly.

    Examples
    --------
    >>> signals = task.signals
    >>> signals.finished.connect(on_success)
    >>> signals.error.connect(on_error)
    """

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)


class Task(QRunnable):
    """
    Background task wrapper for executing functions in Qt thread pool.

    This class wraps an arbitrary Python function for execution in a worker
    thread. It handles exception catching and emits appropriate signals on
    completion or error.

    Parameters
    ----------
    fn : callable
        Function to execute in background
    *args : tuple
        Positional arguments to pass to fn
    **kwargs : dict
        Keyword arguments to pass to fn

    Attributes
    ----------
    fn : callable
        The wrapped function
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments
    signals : TaskSignals
        Signal object for result communication

    Notes
    -----
    This class is designed for use with Qt's QThreadPool. The run() method
    is called automatically by the thread pool when a worker thread is
    available.

    Exceptions raised by fn are caught and converted to error signals with
    the exception message as a string.

    Examples
    --------
    >>> def compute(x, y):
    ...     return x * y
    >>>
    >>> task = Task(compute, 5, 10)
    >>> task.signals.finished.connect(lambda r: print(f"Result: {r}"))
    >>> QThreadPool.globalInstance().start(task)

    See Also
    --------
    submit : Convenience function for creating and starting tasks
    TaskSignals : Signal definitions
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self):
        """
        Execute the wrapped function and emit result or error signal.

        This method is called automatically by QThreadPool when a worker
        thread is available. It should not be called directly.

        Notes
        -----
        Execution flow:
        1. Call fn(*args, **kwargs)
        2. On success: emit signals.finished(result)
        3. On exception: emit signals.error(str(exception))

        The finished signal carries the function's return value. The error
        signal carries the exception message as a string.
        """
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(res)
        except Exception as e:
            self.signals.error.emit(str(e))


_pool = QThreadPool.globalInstance()


def submit(fn, *args, **kwargs):
    """
    Submit a function for background execution in the global thread pool.

    This is the main API for running functions off the GUI thread. It creates
    a Task, submits it to the thread pool, and returns the signals object for
    connecting callbacks.

    Parameters
    ----------
    fn : callable
        Function to execute in background
    *args : tuple
        Positional arguments for fn
    **kwargs : dict
        Keyword arguments for fn

    Returns
    -------
    TaskSignals
        Signal object with finished/error/progress signals

    Notes
    -----
    The function executes asynchronously. Connect to the returned signals to
    handle results:
    - signals.finished: Function completed, receives return value
    - signals.error: Function raised exception, receives error string
    - signals.progress: Progress updates (currently unused)

    The global thread pool manages worker threads automatically, typically
    creating one thread per CPU core.

    Examples
    --------
    >>> from bpd_ui.core.tasks import submit
    >>> import time
    >>>
    >>> def slow_function(duration):
    ...     time.sleep(duration)
    ...     return f"Slept for {duration}s"
    >>>
    >>> signals = submit(slow_function, 2)
    >>> signals.finished.connect(lambda r: print(r))
    >>> signals.error.connect(lambda e: print(f"Error: {e}"))
    >>>
    >>> # GUI remains responsive while task runs
    >>>
    >>> # With keyword arguments
    >>> def greet(name, greeting="Hello"):
    ...     return f"{greeting}, {name}!"
    >>>
    >>> signals = submit(greet, "Alice", greeting="Hi")

    See Also
    --------
    Task : Background task wrapper class
    TaskSignals : Signal definitions
    """
    t = Task(fn, *args, **kwargs)
    _pool.start(t)
    return t.signals
