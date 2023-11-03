import os
import time

fifo_path = '/tmp/my_fifo'
fifo = open(fifo_path, 'r')  # 以文本模式打开管道

numbers_received = []
line = fifo.readline().strip()  # 使用readline()读取一行
if line:
    numbers_received.append(int(line))
start_time = time.time()
while time.time() - start_time < 10:
    line = fifo.readline().strip()  # 使用readline()读取一行
    if line:
        numbers_received.append(int(line))

total_data_received = len(numbers_received)
print(f'Total data received: {total_data_received}')

fifo.close()  # 关闭管道
