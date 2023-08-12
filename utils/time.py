import time
from collections import defaultdict


class TimeContextManager:
    def __init__(self, verbose: bool = True) -> None:
        self.manager = defaultdict(list)
        self.verbose = verbose

    def __call__(self, context):
        self.context = context
        return self

    def measure(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            if self.verbose:
                print(f">>> | {self.context} : {round(elapsed_time, 3)} secs | <<<")

            self.manager[self.context].append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "elapsed_time": elapsed_time,
                }
            )
            return value

        return wrapper
