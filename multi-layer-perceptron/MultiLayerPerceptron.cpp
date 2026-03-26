#include "MultiLayerPerceptron.h"
#include "../../helpers/ActivateFunctions.h"
#include <vector>
using namespace std;

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layersDefinition, int entryNumber) {
    hiddenLayers.push_back(HidenLayer(entryNumber, layersDefinition[0], sigmoid, sigmoid_derivative, 1));
    for(int i = 1; i < layersDefinition.size(); i++) {
        hiddenLayers.push_back(HidenLayer(layersDefinition[i-1], layersDefinition[i], sigmoid, sigmoid_derivative, 1));
    }
}

void MultiLayerPerceptron::learn(vector<double> x, vector<double> d, double threshold) {
    // Propagate the data in each layer
    vector<double> layerOutput = hiddenLayers[0].propagate(x);
    for(int i = 1; i < hiddenLayers.size(); i++) {
        layerOutput = hiddenLayers[i].propagate(layerOutput);
    }
    // Error calculation
    double e = 0.0;
    for(int i = 0; i < layerOutput.size(); i++) {
        e += pow(d[i] - layerOutput[i], 2);
    }
    // Check if we continue
    if(e/2 < threshold) {
        return;
    }
    // Error signal calculation of output layer
    vector<double> outputLayerErrorSignals;
    vector<double> outputLayerPotentials = hiddenLayers.back().getLastPotentials();
    for(int i = 0; i < layerOutput.size(); i++) {
        double signal = (d[i] - layerOutput[i]) * (hiddenLayers[i].getDerivative()(1, outputLayerPotentials[i]));
        outputLayerErrorSignals.push_back(signal);
    }
    // Error signal calculation of hidden layers

    // Change ouput layer neuron weights
    HidenLayer outputLayer = hiddenLayers.back();
    for(int i = 0; i < outputLayer.getNeuronNumber(); i++) {
        vector<double> newWeights;
        double signalError = outputLayerErrorSignals[i];
        vector<double> lastEntry = outputLayer.getLastEntry(i);
        for(int j = 0; j < outputLayer.getEntryNumber(); j++) {
            double weight = outputLayer.getLearningRate() * signalError * lastEntry[j];
            newWeights.push_back(weight);
        }
        outputLayer.changeNeuronWeights(i, newWeights);
    }
    // Change hidden layer neuron weights

}