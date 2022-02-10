class NoSimClockError(Exception):
    def __str__(self):
        return "Attach a sim_clock first"
