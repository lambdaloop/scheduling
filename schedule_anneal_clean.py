#!/usr/bin/env python2

from __future__ import print_function
import math
import random
from simanneal import Annealer



n_slots = 3 # number of lab slots
n_people = 4 # number of lab assistants

slot_num = [1, 1, 2]
min_people = 1

l_slots = [1, 2, 1, 1]

l_possible = {
    0: [0, 1, 1],
    1: [1, 1, 1],
    2: [1, 0, 1],
    3: [0, 1, 1]
}

l_actual = {
    0: [1, 0, 0],
    1: [1, 1, 0],
    2: [1, 0, 0],
    3: [1, 0, 0]
}

def compute_gaps(state):
    c = 0
    
    for i in range(n_people):
        current = 0
        s = 0
        for j in range(n_slots):
            v = slot_num[j]
            if v != current:
                c += s * s
                s = 0
                current = v
            s += state[i][j]
        c += s*s
    return -1 * c

class ScheduleLA(Annealer):
    def move(self):
        # pick a lab assistant, swap two of his slots
        p = random.randint(0, n_people - 1)
        a = random.randint(0, n_slots - 1)
        b = random.randint(0, n_slots - 1)
        self.state[p][a], self.state[p][b] = self.state[p][b], self.state[p][a]
        
    def energy(self):
        energy = 0
        # each lab should have at least min_people
        for j in range(n_slots):
            s = sum(self.state[i][j] for i in range(n_people))
            if s < min_people:
                energy += 1000 * (min_people - s)

        # assistants should not be assigned slots they can't make
        for i in range(n_people):
            for j in range(n_slots):
                if self.state[i][j] == 1 and l_possible[i][j] == 0:
                    energy += 100000

        # spread out lab assistants
        m = max([sum(self.state[i][j] for i in range(n_people)) for j in range(n_slots)])
        energy += 100 * m
        
        # prefer schedules with minimum gaps
        energy += compute_gaps(self.state) * 10
        return energy
            
la = ScheduleLA(l_actual)

# la.Tmax = 25000.0  # Max (starting) temperature
# la.Tmin = 2.5      # Min (ending) temperature
# la.steps = 100000   # Number of iterations
# la.updates = 100   # Number of updates (by default an update prints to stdout)

# auto_schedule = la.auto(minutes=1)
# la.set_schedule(auto_schedule)
la.anneal()

print(la.state)
