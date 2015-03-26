#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "Neuron.h"

typedef std::vector<Neuron> NeuronLayer;

class Network
{
public:
    Network(const std::vector<unsigned> & topology);
    void FeedForward(const std::vector<float> & inputValues,
                     std::vector<float> & outputValues);
    void BackPropagate(const std::vector<float> targets);

    float GetLastError() const { return m_lastError; }

private:
    std::vector<NeuronLayer> m_NeuronLayers;
    float m_lastError;
};

#endif
