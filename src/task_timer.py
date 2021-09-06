import os
import json
import time
from datetime import timedelta


class TaskTimer:
    def __init__(self):
        self.time_performance = {}

        self.start_times = {}

    def start(self, task):
        self.start_times[task] = time.time()
        print('--- [{}] Start "{}"'.format(time.ctime(self.start_times[task]), task))

    def end(self, task):
        saving_end = time.time()
        self.time_performance[task] = str(
            timedelta(seconds=(saving_end - self.start_times[task]))
        )
        print(
            '--- [{}] End "{}" in {} seconds'.format(
                time.ctime(saving_end), task, self.time_performance[task]
            )
        )

    def save(self, folder):
        with open(os.path.join(folder, "time_performance.json"), "w") as fp:
            json.dump(self.time_performance, fp)
