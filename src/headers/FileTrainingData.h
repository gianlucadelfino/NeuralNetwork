#ifndef FILETRAININGDATA_H
#define FILETRAININGDATA_H

#include <vector>
#include <fstream>

#include "ITrainingData.h"

class FileTrainingData : public ITrainingData<float>
{
public:
    FileTrainingData(const std::string & filename,
                     unsigned numOfInputs,
                     unsigned numOfTargets);
    bool FileTrainingData::EndoOfTrainingData() const
    {
        return !m_fileStream.good();
    }

    virtual void GetNextInputAndTargetValues(
        std::vector<float> & inputs,
        std::vector<float> & targets);

private:
    template <typename T>
    void fillVectorWithNextLine(
        const std::string & label,
        std::vector<T> & vec);

    std::ifstream m_fileStream;
};

#endif
