#include "XorTrainingData.h"
#include <ctime>
#include <cstdlib>

XorTrainingData::XorTrainingData(unsigned datasetSize,
                                 unsigned numOfInputs,
                                 unsigned numOfTargets):
ITrainingData(numOfInputs, numOfTargets), m_datasetSize(datasetSize)
{
    std::srand(static_cast<unsigned>(time(NULL)));
}

void XorTrainingData::GetNextInputAndTargetValues(
    std::vector<float> & inputs,
    std::vector<float> & targets)
{
    inputs.clear();
    targets.clear();
    unsigned a = rand() % 2;
    inputs.push_back(a);

    unsigned b = rand() % 2;
    inputs.push_back(b);

    targets.push_back(a ^ b);

    if (m_datasetSize > 0)
    {
        m_datasetSize--;
    }
}

bool XorTrainingData::EndoOfTrainingData() const
{
    return m_datasetSize == 0;
}
