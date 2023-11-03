#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    std::string fifo_path = "/tmp/my_fifo";
    mkfifo(fifo_path.c_str(), 0666);

    std::ofstream fifo(fifo_path, std::ios::out);
    for (int i = 1; i <= 100000; ++i) {  // 更新循环次数以匹配10秒的运行时间
        fifo << i << std::endl;
   
//        usleep(100);  // 更新这里，以便每秒写入1000次
    }
    fifo.close();
    return 0;
}

