from functools import wraps
from mpi4py import MPI
import threading
import os
from time import time
import json
import logging

def get_rank():
    return MPI.COMM_WORLD.rank

def get_size():
    return MPI.COMM_WORLD.size

perftrace_logdir = "./"
perftrace_logfile = f"./.trace-{get_rank()}-of-{get_size()}"+".pfw"

class PerfTrace:
    __instance = None
    logger = None
    log_file = os.path.join(perftrace_logdir, perftrace_logfile)
    if os.path.isfile(log_file):
        os.remove(log_file)
    def __init___(self):
        if PerfTrace.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            PerfTrace.__instance = self
    @staticmethod
    def get_instance():
        """ Static access method. """
        if PerfTrace.__instance is None:
            PerfTrace()
        return PerfTrace.__instance
    def set_logdir(cls, logdir):
        global perftrace_logdir
        perftrace_logdir = logdir
        log_file = os.path.join(perftrace_logdir, perftrace_logfile)
        if os.path.isfile(log_file):
            os.remove(log_file)
    def event_logging(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            x = func(*args, **kwargs)
            end = time()
            event = cls.__create_dur_event(func.__qualname__, func.__module__, start, dur=end-start)
            cls.__flush_log(json.dumps(event))            
            return x
        return wrapper
    @staticmethod
    def __create_event(name, cat, ph):
        return {
            "name": name,
            "cat": cat,
            "pid": get_rank(),
            "tid": os.getpid(),
            "ts": time() * 1000000,
            "ph": ph
        }
    @staticmethod 
    def __create_dur_event(name, cat, ts, dur):
        return {
            "name": name,
            "cat": cat,
            "pid": get_rank(),
            "tid": os.getpid(),
            "ts": ts * 1000000, 
            "dur": dur * 1000000,
            "ph": "X"
        }
    def event_complete(cls, name, cat, ts, dur):
        event = cls.__create_dur_event(name, cat, ts, dur)
        cls.__flush_log(json.dumps(event))
    def event_start(cls, name, cat='default'):
        event = cls.__create_event(name, cat, 'B')
        cls.__flush_log(json.dumps(event))
    def event_stop(cls, name, cat='default'):
        event = cls.__create_event(name, cat, "E")
        cls.__flush_log(json.dumps(event))
    @staticmethod
    def __flush_log(s):
        if PerfTrace.logger == None:
            log_file = os.path.join(perftrace_logdir, perftrace_logfile)
            if os.path.isfile(log_file):
                started = True
            else:
                started = False
            PerfTrace.logger = logging.getLogger("perftrace")
            PerfTrace.logger.setLevel(logging.DEBUG)
            PerfTrace.logger.propagate = False
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(message)s")
            fh.setFormatter(formatter)
            PerfTrace.logger.addHandler(fh)
            if (not started):
                PerfTrace.logger.debug("[")
        PerfTrace.logger.debug(s)

perftrace = PerfTrace()
