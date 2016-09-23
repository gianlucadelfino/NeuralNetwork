#pragma once
#include <vector>

#include "Neuron.h"


class Network
{
public:
    Network(const std::vector<unsigned>& topology);
    void FeedForward(const std::vector<float>& inputValues,
                     std::vector<float>& outputValues);
    void BackPropagate(const std::vector<float>& targets);

    float GetLastError() const
    {
        return m_lastError;
    }

private:
    typedef std::vector<Neuron> NeuronLayer;
    std::vector<NeuronLayer> m_NeuronLayers;
    float m_lastError;
};
