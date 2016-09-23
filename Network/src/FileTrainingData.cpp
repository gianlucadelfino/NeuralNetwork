#include <fstream>
#include <string>
#include <sstream>
#include <cassert>

#include "FileTrainingData.h"

FileTrainingData::FileTrainingData(const std::string& filename,
                                   unsigned numOfInputs,
                                   unsigned numOfTargets)
    : ITrainingData(numOfInputs, numOfTargets), m_fileStream(filename.c_str())
{
}

template <typename T>
void FileTrainingData::fillVectorWithNextLine(const std::string& label,
                                              std::vector<T>* vec)
{
    std::string line;
    getline(m_fileStream, line);

    std::stringstream ss(line);
    std::string firstRow;

    // Make sure we are reading the inputs
    ss >> firstRow;
    if (firstRow != label)
    {
        throw std::runtime_error(
            "Could not find the right label in the file: expected " + label +
            " found " + firstRow);
    }

    T entry;
    while (ss >> entry)
    {
        vec->push_back(entry);
    }
}

void FileTrainingData::GetNextInputAndTargetValues(std::vector<float>* inputs,
                                                   std::vector<float>* targets)
{
    inputs->clear();
    fillVectorWithNextLine("in:", inputs);

    targets->clear();
    fillVectorWithNextLine("out:", targets);
}
