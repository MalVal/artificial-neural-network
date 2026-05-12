#include "MultiLayerPerceptron.h"
#include "../helpers/ActivateFunctions.h"
#include <iostream>
using namespace std;

MultiLayerPerceptron::MultiLayerPerceptron(int inputSize, vector<HidenLayerDefinition> layersDefinition) {
    hiddenLayers.push_back(HidenLayer(inputSize, layersDefinition[0].getHiddenNeuronNumber(), layersDefinition[0].getActivation(), layersDefinition[0].getDerivative(), layersDefinition[0].getLearningRate()));
    for(int i = 1; i < layersDefinition.size(); i++) {
        hiddenLayers.push_back(HidenLayer(layersDefinition[i-1].getHiddenNeuronNumber(), layersDefinition[i].getHiddenNeuronNumber(), layersDefinition[i].getActivation(), layersDefinition[i].getDerivative(), layersDefinition[i].getLearningRate()));
    }
}

void MultiLayerPerceptron::learn(vector<vector<double>> X, vector<vector<double>> D, double threshold, int maxIter) {
    if(X.size() != D.size()) {
        cerr << "Error: X and D must have the same number of samples" << endl;
        return;
    }
    if(D[0].size() != (size_t)hiddenLayers.back().getNeuronNumber()) {
        cerr << "Error: d size does not match output neuron number" << endl;
        return;
    }

    for(int epoch = 0; epoch < maxIter; epoch++) {
        double totalError = 0.0;

        for(int k = 0; k < (int)X.size(); k++) {
            vector<double> x = X[k];
            vector<double> d = D[k];

            // Propagation
            vector<double> layerOutput = hiddenLayers[0].propagate(x);
            for(int i = 1; i < (int)hiddenLayers.size(); i++) {
                layerOutput = hiddenLayers[i].propagate(layerOutput);
            }

            // Error calculation
            double e = 0.0;
            for(int i = 0; i < (int)layerOutput.size(); i++) {
                e += pow(d[i] - layerOutput[i], 2);
            }
            totalError += e;

            // Get the last layer (output layer)
            HidenLayer& outputLayer = hiddenLayers.back();

            // Error signal calculation of output layer
            for(int i = 0; i < outputLayer.getNeuronNumber(); i++) {
                double signal = (d[i] - outputLayer.getLastOutput(i))
                              * outputLayer.getDerivative()(1.0, outputLayer.getLastPotential(i));
                outputLayer.setNeuronLastSignal(i, signal);
            }

            // Error signal calculation of hidden layers
            for(int i = (int)hiddenLayers.size() - 2; i >= 0; i--) {
                HidenLayer& currentLayer = hiddenLayers[i];
                HidenLayer& nextLayer = hiddenLayers[i + 1];
                for(int j = 0; j < currentLayer.getNeuronNumber(); j++) {
                    double signal = currentLayer.getDerivative()(1.0, currentLayer.getLastPotential(j));
                    double sum = 0.0;
                    for(int s = 0; s < nextLayer.getNeuronNumber(); s++) {
                        sum += nextLayer.getNeuronLastSignal(s) * nextLayer.getNeuronWeights(s)[j + 1];
                    }
                    signal *= sum;
                    currentLayer.setNeuronLastSignal(j, signal);
                }
            }

            // Update weights of output layer
            for(int i = 0; i < outputLayer.getNeuronNumber(); i++) {
                double signal = outputLayer.getNeuronLastSignal(i);
                vector<double> lastEntry = outputLayer.getLastEntry(i);
                vector<double> newWeights;
                newWeights.push_back(outputLayer.getLearningRate() * signal * 1.0);
                for(int j = 0; j < outputLayer.getEntryNumber(); j++) {
                    newWeights.push_back(outputLayer.getLearningRate() * signal * lastEntry[j]);
                }
                outputLayer.changeNeuronWeights(i, newWeights);
            }

            // Update weights of hidden layers
            for(int i = (int)hiddenLayers.size() - 2; i >= 0; i--) {
                HidenLayer& currentLayer = hiddenLayers[i];
                for(int j = 0; j < currentLayer.getNeuronNumber(); j++) {
                    double signal = currentLayer.getNeuronLastSignal(j);
                    vector<double> lastEntry = currentLayer.getLastEntry(j);
                    vector<double> newWeights;
                    newWeights.push_back(currentLayer.getLearningRate() * signal * 1.0);
                    for(int s = 0; s < currentLayer.getEntryNumber(); s++) {
                        newWeights.push_back(currentLayer.getLearningRate() * signal * lastEntry[s]);
                    }
                    currentLayer.changeNeuronWeights(j, newWeights);
                }
            }
        }

        // Check global error
        if(totalError / 2.0 < threshold) {
            cout << "Converged at epoch " << epoch << endl;
            break;
        }
    }
}

vector<double> MultiLayerPerceptron::predict(vector<double> x) {
    vector<double> output = hiddenLayers[0].propagate(x);
    for(int i = 1; i < (int)hiddenLayers.size(); i++) {
        output = hiddenLayers[i].propagate(output);
    }
    return output;
}