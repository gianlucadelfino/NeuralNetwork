#ifndef NEURON_H
#define NEURON_H

#include <vector>

// Learning rate
#define eta 0.15f

// Momentum coefficient
#define alpha 0.5f

class Neuron
{
public:
    Neuron(size_t id, unsigned numOfInputs=0);
    void UpdateNeuralOutput(const std::vector<Neuron> & upStreamLayer);
    void UpdateLastGradientIfOutput(float target);
    void UpdateLastGradientIfHidden(
        const std::vector<Neuron> & downStreamLayer);
    void UpdateInputWeights(
        const std::vector<Neuron> & downStreamLayer);
    float GetInputWeight(size_t i) const { return m_inputWeight[i]; }
    float GetLastOutput() const { return m_lastOutput; }
    float GetLastGradient() const { return m_lastGradient; }
    void SetLastOutput(float value) { m_lastOutput = value; }
    void SetIsBias(bool value) { m_isBias = value; }
    bool IsBias() { return m_isBias; }
private:
    size_t m_id;
    std::vector<float> m_inputWeight;
    std::vector<float> m_lastInputWeightDelta;
    float m_lastOutput;
    float m_lastGradient;
    bool m_isBias;
};

#endif
