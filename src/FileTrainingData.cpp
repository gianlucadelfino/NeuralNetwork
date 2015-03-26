#include <fstream>
#include <string>
#include <sstream>
#include <cassert>

#include "FileTrainingData.h"

FileTrainingData::FileTrainingData(const std::string & filename,
                                   unsigned numOfInputs,
                                   unsigned numOfTargets):
    ITrainingData(numOfInputs, numOfTargets), m_fileStream(filename.c_str())
{}

template <typename T>
void FileTrainingData::fillVectorWithNextLine(
    const std::string & label,
    std::vector<T> & vec)
{
    std::string line;
    getline(m_fileStream, line);

    std::stringstream ss(line);
    std::string lineLabel;

    // Make sure we are reading the inputs
    ss >> lineLabel;
    assert(lineLabel == label);

    T entry;
    while (ss >> entry)
    {
        vec.push_back(entry);
    }
}

void FileTrainingData::GetNextInputAndTargetValues(
    std::vector<float> & inputs,
    std::vector<float> & targets)
{
    inputs.clear();
    fillVectorWithNextLine<float>("in:", inputs);

    targets.clear();
    fillVectorWithNextLine<float>("out:", targets);
}
