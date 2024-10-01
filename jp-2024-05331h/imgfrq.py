import os

for filename in os.listdir():

    if not filename.endswith('.log'):
        continue

    imgfrq = 0
    f = open(filename, 'rt')
    for line in f:
        if line.rstrip().endswith('ignored.'):
            imgfrq = int(line.split()[0])
            break
    f.close()

    if 'TS' in filename:
        if imgfrq != 1:
            print(filename, imgfrq)
    else:
        if imgfrq != 0:
            print(filename, imgfrq)

