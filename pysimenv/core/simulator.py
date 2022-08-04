import os
import h5py
import numpy as np
import ray
import time
from typing import List
from pytictoc import TicToc
from ray.remote_function import RemoteFunction
from pysimenv.core.util import SimClock, Timer, Logger
from pysimenv.core.system import DynObject


class Simulator(object):
    def __init__(self, model: DynObject, verbose: bool = True):
        self.sim_clock = SimClock()
        self.log_timer = Timer(np.inf)
        self.log_timer.attach_sim_clock(self.sim_clock)

        self.model = model
        self.model.attach_sim_clock(self.sim_clock)
        self.model.attach_log_timer(self.log_timer)
        self.model.initialize()

        self.verbose = verbose

    def reset(self):
        self.sim_clock.reset()
        self.log_timer.turn_off()
        self.model.reset()

    def begin_logging(self, log_interval: float):
        self.log_timer.turn_on(log_interval)

    def finish_logging(self):
        self.log_timer.turn_off()

    def step(self, dt: float, **kwargs):
        self.model.step(dt, **kwargs)

    def propagate(self, dt: float, time: float, save_history: bool = True, *args, **kwargs):
        if save_history:
            self.begin_logging(dt)

        if self.verbose:
            print("[simulator] Simulating...")
        tic_toc = TicToc()
        tic_toc.tic()
        self.model.propagate(dt, time, **kwargs)
        elapsed_time = tic_toc.tocvalue()

        if self.verbose:
            print("[simulator] Elapsed time: {:.4f} [s]".format(elapsed_time))
        self.finish_logging()


class ParallelSimulator(object):
    def __init__(self, simulation_fun: RemoteFunction):
        """
        :param simulation_fun: reference for a function object.
        """
        self.simulation_fun = simulation_fun
        self.logger = Logger()
        self.name = "par_simulator"

    def simulate(self, parameter_sets: List[dict], verbose: bool = False):
        print("[{:s}] Initializing the parallel simulation...".format(self.name))
        ray.init()

        num_sim = len(parameter_sets)
        remaining_ids = []
        parameters_mapping = {}

        for parameters in parameter_sets:
            result_id = self.simulation_fun.remote(**parameters)
            remaining_ids.append(result_id)
            parameters_mapping[result_id] = parameters

        num_done = 0
        start = time.time()
        print("[{:s}] Simulating...".format(self.name))
        while remaining_ids:
            done_ids, remaining_ids = ray.wait(remaining_ids)
            result_id = done_ids[0]
            num_done += 1

            parameters = parameters_mapping[result_id]
            result = ray.get(result_id)
            self.logger.append(**parameters, **result)
            print("[{:s}] [{:d}/{:d}] ".format(self.name, num_done, num_sim))
            if verbose:
                print("Parameters: " + str(parameters))
                print("Result: " + str(result) + "\n")

        duration = round(time.time() - start)
        print("[{:s}] Simulations done in {:d} seconds".format(self.name, duration))

    def get(self, *args):
        return self.logger.get(*args)

    def save(self, save_dir=None, data_group=''):
        if save_dir is None:
            save_dir = './data/'
        os.makedirs(save_dir, exist_ok=True)
        file = h5py.File(save_dir + 'par_sim.hdf5', 'w')
        self.logger.save(file, data_group)
