#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cassert>

#include "Neuron.h"

using namespace std;

//#define DEBUG_LOG

namespace //anonymous
{
    float ActivationFunction(float x)
    {
        return tanh(x);
    }

    float ActivationFunctionDerivative(float x)
    {
        float t = tanh(x);
        return 1 - t*t;
    }
}

Neuron::Neuron(size_t id, unsigned numOfInputs)
    :m_id(id), m_lastOutput(0.0f), m_lastGradient(0.0f), m_isBias(false)
{
    // Initialize all the weight randomly
    std::default_random_engine generator(4);
    std::uniform_real_distribution<float> distribution(0.0f, +1.0f);

    m_inputWeight.reserve(numOfInputs);
    m_lastInputWeightDelta.reserve(numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i)
    {
        m_inputWeight.push_back(distribution(generator));
        m_lastInputWeightDelta.push_back(0.0f);
    }

#ifdef DEBUG_LOG
    std::cout << "Neuron::Neuron Neuron with id " << m_id << std::endl;
#endif
}

void Neuron::UpdateNeuralOutput(const vector<Neuron> & upStreamLayer)
{
    assert(!m_isBias &&
        "Neuron Error: Operation not supported on bias Neuron");
    assert(upStreamLayer.size() == m_inputWeight.size() &&
        "Neuron::UpdateNeuralOutput Error: "
        "the size of the upStream layer does not match the size of the"
        "weight vector");
    float sum = 0.0f;
    for (size_t i = 0; i < upStreamLayer.size(); ++i)
    {
        sum += upStreamLayer[i].GetLastOutput() * m_inputWeight[i];
    }

    m_lastOutput = ActivationFunction(sum);
}

void Neuron::UpdateLastGradientIfOutput(float target)
{
    assert(!m_isBias &&
        "Neuron::UpdateLastGradientIfOutput Error: "
        "not supported on bias Neuron");
    float delta = (target - m_lastOutput);
    m_lastGradient = delta * ActivationFunctionDerivative(m_lastOutput);
}

void Neuron::UpdateLastGradientIfHidden(
    const std::vector<Neuron> & downStreamLayer)
{
    float delta = 0.0f;
    for (std::vector<Neuron>::const_iterator cit = downStreamLayer.begin();
        cit != downStreamLayer.end(); ++cit)
    {
        delta += cit->GetLastGradient() * cit->GetInputWeight(m_id);
    }
    m_lastGradient = ActivationFunctionDerivative(m_lastOutput) * delta;
}

void Neuron::UpdateInputWeights(
    const std::vector<Neuron> & upStreamLayer)
{
    assert(!m_isBias &&
        "Neuron::UpdateInputWeights Error: "
        "not supported on bias Neuron");

    for (vector<float>::iterator inWeightIter = m_inputWeight.begin();
        inWeightIter != m_inputWeight.end(); ++inWeightIter)
    {
        size_t idx = inWeightIter - m_inputWeight.begin();

        float weightDelta =
            eta * upStreamLayer[idx].GetLastOutput() * m_lastGradient
            + alpha * m_lastInputWeightDelta[idx];

        m_lastInputWeightDelta[idx] = weightDelta;
#ifdef DEBUG_LOG
        std::cout << "weightDelta m_id idx " << m_id << ", " << idx
            << " = " << weightDelta << std::endl;
#endif
        *inWeightIter += weightDelta;
    }
}
