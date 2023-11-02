#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    std::string fifo_path = "/tmp/my_fifo";
    mkfifo(fifo_path.c_str(), 0666);

    std::ofstream fifo(fifo_path, std::ios::out);
    for (int i = 1; i <= 100; ++i) {
        fifo << i << std::endl;
        sleep(1);
    }
    fifo.close();
    return 0;
}
