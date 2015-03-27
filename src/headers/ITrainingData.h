#ifndef ITRAININGDATA_H
#define ITRAININGDATA_H

#include <vector>

template <typename T>
class ITrainingData
{
public:
    ITrainingData(unsigned numOfInputs, unsigned numOfTargets):
        m_numOfInputs(numOfInputs), m_numOfTargets(numOfTargets)
    {}
    virtual bool EndoOfTrainingData() const = 0;
    virtual void GetNextInputAndTargetValues(
        std::vector<T> & inputs,
        std::vector<T> & targets) = 0;
    virtual unsigned GetNumberOfInputs() const { return m_numOfInputs; }
    virtual unsigned GetNumberOfTargets() const { return m_numOfTargets; }
    virtual ~ITrainingData(){}
private:
    // Disable copy and assignment
    ITrainingData(const ITrainingData &);
    ITrainingData& operator==(const ITrainingData &);

    unsigned m_numOfInputs;
    unsigned m_numOfTargets;
};

#endif
