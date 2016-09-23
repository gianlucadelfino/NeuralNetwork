#pragma once

#include <vector>

#include "ITrainingData.h"

class XorTrainingData : public ITrainingData<float>
{
public:
    XorTrainingData(unsigned datasetSize,
                    unsigned numOfInputs,
                    unsigned numOfTargets);

    virtual void GetNextInputAndTargetValues(
        std::vector<float>* inputs, std::vector<float>* targets) override;

    virtual bool EndoOfTrainingData() const;

private:
    unsigned m_datasetSize;
};
