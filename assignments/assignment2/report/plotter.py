import matplotlib.pyplot as plt
import sys
import csv

fnames = ['mpi2','mpi4','mpi8', 'openMp', 'singleThread']
handles= []
for fname in fnames:
    filename = fname+'.txt'
    with open(filename) as f:
        reader = csv.reader(f)
        t = []
        n = []
        for row in reader:
            n.append(int(row[0]))
            t.append(float(row[1]))
        handle,  = plt.plot(n,t, label=fname)
        handles.append(handle)
plt.legend(handles=handles, loc=2)
plt.xlabel('no of elements matrix')
plt.ylabel('time taken (seconds)')
plt.savefig('images/normal.png')

plt.clf()
handles= []
for fname in fnames:
    filename = fname+'.txt'
    with open(filename) as f:
        reader = csv.reader(f)
        t = []
        n = []
        for row in reader:
            n.append(int(row[0]))
            t.append(float(row[1]))
        handle,  = plt.loglog(n,t, label=fname)
        handles.append(handle)
plt.legend(handles=handles, loc=2)
plt.savefig('images/log.png')
