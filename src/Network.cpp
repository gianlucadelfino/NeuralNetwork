#include <iostream>
#include <cassert>

#include "Network.h"
using namespace std;

#define DEBUG_LOG

Network::Network(const vector<unsigned> & topology):m_lastError(0.0f)
{
    m_NeuronLayers.reserve(topology.size());

    for(size_t i = 0; i < topology.size(); ++i)
    {
        m_NeuronLayers.push_back(NeuronLayer());
        NeuronLayer & curNeuronLayer = m_NeuronLayers.back();
        unsigned numOfNeurons = topology[i];

        if (i < topology.size() - 1)
        {
            // In the hidden layers add a bias neuron.
            numOfNeurons++;
        }

        curNeuronLayer.reserve(numOfNeurons);

        unsigned prevLayerNumOfNeurons = (i > 0)? topology[i-1] + 1 : 0;

        for (size_t j = 0; j < numOfNeurons; ++j)
        {
            curNeuronLayer.push_back(Neuron(j, prevLayerNumOfNeurons));
        }

        if (i < topology.size() - 1)
        {
            // Set the bias neuron output to 1.
            curNeuronLayer.back().SetIsBias(true);
            curNeuronLayer.back().SetLastOutput(1.0f);
        }
    }
}

void Network::FeedForward(const vector<float> & inputValues,
                          std::vector<float> & outputValues)
{
    // copy values input values into the first layer
    NeuronLayer & firstLayer = m_NeuronLayers[0];
    assert(inputValues.size() == firstLayer.size() - 1 &&
           "Network::FeedForward Error: "
           "the input size does not match the fist layer size!");
    for (size_t i = 0; i < firstLayer.size() - 1; ++i)
    {
        // Iterate all but the bias Neuron.
        firstLayer[i].SetLastOutput(inputValues[i]);
    }

    for (size_t i = 1; i < m_NeuronLayers.size(); ++i)
    {
        NeuronLayer & curNeuronLayer = m_NeuronLayers[i];
        const NeuronLayer & prevNeuronLayer = m_NeuronLayers[i-1];

        // We need to iterate over all the neurons but the bias, which is
        // only in the hidden layers.
        const size_t numOfNeurons = (i < m_NeuronLayers.size() - 1)?
            curNeuronLayer.size() - 1 : curNeuronLayer.size();
        for (size_t j = 0; j < numOfNeurons; ++j)
        {
            curNeuronLayer[j].UpdateNeuralOutput(prevNeuronLayer);
        }
    }

    // Fill the outputValues
    outputValues.clear();
    const NeuronLayer & outputLayer = m_NeuronLayers.back();
    for (NeuronLayer::const_iterator cit = outputLayer.begin();
        cit != outputLayer.end(); ++cit)
    {
        outputValues.push_back(cit->GetLastOutput());
    }
    assert(outputValues.size() == outputLayer.size() &&
           "Network::FeedForward Error: "
           "the output size does not match the fist layer size!");
}

void Network::BackPropagate(const std::vector<float> targets)
{
    // Calculate overall net error (sum of squared errors)
    NeuronLayer & outputLayer = m_NeuronLayers.back();
    float squaresSum = 0.0f;
    for (NeuronLayer::const_iterator cit = outputLayer.begin();
        cit != outputLayer.end(); ++cit)
    {
        size_t idx = cit - outputLayer.begin();
        float delta = (targets[idx] - cit->GetLastOutput());
        squaresSum += delta * delta;
    }
    m_lastError = 0.5f * squaresSum;

    // Calculate output layer gradients
    for (NeuronLayer::iterator it = outputLayer.begin();
        it != outputLayer.end(); ++it)
    {
        size_t idx = it - outputLayer.begin();
        it->UpdateLastGradientIfOutput(targets[idx]);
    }

    // Calculate hidden layer gradients
    for (size_t i = 1; i < m_NeuronLayers.size() - 1; ++i)
    {
        NeuronLayer & curLayer = m_NeuronLayers[i];
        NeuronLayer & nextLayer = m_NeuronLayers[i+1];
        for (NeuronLayer::iterator it = curLayer.begin();
            it != curLayer.end(); ++it)
        {
            it->UpdateLastGradientIfHidden(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (size_t i = 1; i < m_NeuronLayers.size(); ++i)
    {
        NeuronLayer & prevLayer = m_NeuronLayers[i-1];
        NeuronLayer & curLayer = m_NeuronLayers[i];

        // We need to iterate over all the neurons but the bias, which is
        // only in the hidden layers.
        const size_t numOfNeurons = (i < m_NeuronLayers.size() - 1)?
            curLayer.size() - 1 : curLayer.size();
        for (size_t j = 0; j < numOfNeurons; ++j)
        {
            // Loop through all Neurons but the bias:
            // the bias is not supposed to have input
            // connections
            curLayer[j].UpdateInputWeights(prevLayer);
        }
    }
}
