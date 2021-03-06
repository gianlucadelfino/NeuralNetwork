#include <iostream>
#include <ctime>
#include <cstdlib>

#define DATASETSIZE 1500

int main()
{
    std::srand(time(NULL));

    for (unsigned i = 0; i < DATASETSIZE; ++i)
    {
        unsigned a = rand() & 1;
        unsigned b = rand() & 1;
        std::cout << "in:\t" << a << "\t" << b << std::endl;
        std::cout << "out:\t" << (a ^ b) << std::endl;
    }
    return 0;
}
