#include <iostream>
#include <vector>
#include <string>

#include "Network.h"
#include "Neuron.h"
#include "XorTrainingData.h"
#include "FileTrainingData.h"

#define DEBUG_LOG

namespace
{
    template <typename T>
    void PrintVector(const std::vector<T> vec, const std::string & label)
    {
        std::cout << label << ":\t";
        for (typename std::vector<T>::const_iterator cit = vec.begin();
            cit != vec.end(); ++cit)
        {
            std::cout << *cit << "\t";
        }
        std::cout << std::endl;
    }
}

int main()
{
    // Test training data.
    XorTrainingData testData(2000, 2, 1);

    std::vector<float> inputs;
    std::vector<float> targets;

    std::vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);

    Network net(topology);

    std::vector<float> outputValues(topology.back());

    unsigned epoch = 0;
    while (!testData.EndoOfTrainingData())
    {
        testData.GetNextInputAndTargetValues(inputs, targets);

        net.FeedForward(inputs, outputValues);

#ifdef DEBUG_LOG
        std::cout << "\nIteration: " << epoch << std::endl;
        PrintVector<float>(inputs, "in");
        PrintVector<float>(outputValues, "out");
        std::cout << "Current Sum of Squared Errors: " << net.GetLastError()
                  << std::endl;
#endif
        net.BackPropagate(targets);
        epoch++;
    }

    std::cout << "Done. Press Any key to exit." << std::endl;
    std::cin.get();

    return 0;
}
