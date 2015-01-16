#!/usr/bin/env python2

import csv
import random

f = open('test.csv', 'w')

fieldnames = ['name', 'preferences', 'num_slots']
writer = csv.DictWriter(f, fieldnames)

writer.writeheader()

n_people = 20
n_slots = 9

for i in range(n_people):
    #name = chr(65 + i)
    name = str(i)
    slots = random.randint(1, 3)
    pref_num = min(slots + random.randint(0, 3), n_slots)
    # if random.random() < 0.2:
    #     pref_num -= 1
    pref = sorted(random.sample(range(n_slots), pref_num))
    
    writer.writerow({
        'name': name,
        'num_slots': slots,
        'preferences': ' '.join(map(str, pref))
    })

f.close()
