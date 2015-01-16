#!/usr/bin/env python2

from __future__ import print_function
import random
from simanneal import Annealer
import csv
import numpy as np
from pprint import pprint

n_slots = 9
slot_num = [1, 1, 1,
            2, 2, 2,
            3, 3, 3]

min_people = 1

f = open('test.csv')
reader = csv.DictReader(f)

names = list()
n_people = 0
l_slots = list()

# l_possible = dict()
# l_actual = dict()

l_possible = np.zeros((1, n_slots))
l_actual = np.zeros((1, n_slots))

name_dict = dict()

for row in reader:

    slots = int(row['slots'])
    
    if row['preferences'] == '':
        print("warning: {0} did not list any available slots, ignoring...".format(
            row['name']))
        continue
    
    
    pref = [int(x) for x in row['preferences'].split(' ')]

    if len(pref) < slots:
        print("warning: {0} needs {1} slots, but only has {2} available slots: {3}, ignoring...".format(
            row['name'], slots, len(pref), row['preferences']))
        continue

    possible = np.zeros(n_slots)
    for i in pref:
        possible[i] = 1

    l_possible = np.vstack([l_possible, possible])
    # l_possible[n_people] = possible

    actual = np.zeros(n_slots)
    for i, p in enumerate(pref):
        if i >= slots:
            break
        actual[p] = 1

    # l_actual[n_people] = actual
    l_actual = np.vstack([l_actual, actual])

    names.append(row['name'])
    l_slots.append(slots)
    
    n_people += 1

f.close()

l_possible = l_possible[1:]
l_actual = l_actual[1:]

l_possible_slots = [np.where(l_possible[i])[0] for i in range(n_people)]
# l_possible = {
#     0: [0, 1, 1],
#     1: [1, 1, 1],
#     2: [1, 0, 1],
#     3: [0, 1, 1]
# }

# l_actual = {
#     0: [1, 0, 0],
#     1: [1, 1, 0],
#     2: [1, 0, 0],
#     3: [1, 0, 0]
# }

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
            s += state[i,j]
        c += s*s
    return -1 * c

class ScheduleLA(Annealer):
    def move(self):
        # pick a lab assistant, swap two of his possible slots
        p = random.randint(0, n_people - 1)
        slots = l_possible_slots[p]
        if len(slots) < 2:
            return
        
        a, b = random.sample(slots, 2)
        self.state[p, a], self.state[p, b] = self.state[p, b], self.state[p, a]

    def energy(self):
        energy = 0

        # number of total lab assistants in each lab
        lab_people = np.sum(self.state, axis=0)
        
        # each lab should have at least min_people
        energy += 5000 * np.sum((min_people - lab_people) * (lab_people < min_people))

        # assistants should not be assigned slots they can't make
        energy += 1000000 * np.sum(la.state > l_possible)

        # spread out lab assistants, each lab should have roughly same number of assistants
        # m = max(lab_people) - min(lab_people)
        # energy += 1000 * m
        energy += np.std(lab_people) * 500
        
        # prefer schedules with minimum gaps
        energy += compute_gaps(self.state) * 20
        return energy

    def copy_state(self, state):
        return np.copy(state)
    
la = ScheduleLA(l_actual)

la.updates = 200

la.Tmax = 1000.0  # Max (starting) temperature
la.Tmin = 10.0      # Min (ending) temperature
# la.steps = 100000   # Number of iterations
la.updates = 100   # Number of updates (by default an update prints to stdout)

# auto_schedule = la.auto(minutes=1)
# la.set_schedule(auto_schedule)
la.anneal()

pprint(la.state)
pprint(np.sum(la.state, axis=0))
print(la.energy())

f = open('test_out.csv', 'w')
fieldnames = ['name', 'slots']
writer = csv.DictWriter(f, fieldnames)
writer.writeheader()

for i in range(n_people):
    arr = la.state[i]
    slots = list()
    for j, s in enumerate(arr):
        if s == 1:
            slots.append(str(j))
    writer.writerow({
        'name': names[i],
        'slots': ' '.join(slots)
    })

f.close()


