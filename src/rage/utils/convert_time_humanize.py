import humanize
from datetime import timedelta

def convert_time_humanize(milliseconds):
    return humanize.precisedelta(timedelta(milliseconds= milliseconds))