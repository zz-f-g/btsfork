import time


class Recorder:
    elapsed_time = {}

    def __init__(self, fn: str):
        self.fn = fn

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        Recorder.elapsed_time.update({self.fn: self.end - self.start})
        return True


class profiler:
    # def __init__(self, name: str):
    #     self.profile = {}

    @staticmethod
    def record_function(fn: str):
        recorder = Recorder(fn)
        # self.profile.update({fn: recorder.time_cost})
        return recorder
