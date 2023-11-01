import os

fifo_path = '/tmp/my_fifo'
fifo = os.open(fifo_path, os.O_RDONLY)

numbers_received = []
for _ in range(10):
    for _ in range(10):
        line = os.read(fifo, 16).decode('utf-8').strip()
        if line:
            numbers_received.append(int(line))
    print(numbers_received)

os.close(fifo)
