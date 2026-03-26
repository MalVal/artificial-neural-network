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

void MultiLayerPerceptron::learn(vector<double> x, vector<double> d, double threshold, int maxIter) {
    int epoch = 0;
    while(epoch < maxIter) {
        // Propagate the data in each layer
        vector<double> layerOutput = hiddenLayers[0].propagate(x);
        for(int i = 1; i < (int)hiddenLayers.size(); i++) {
            layerOutput = hiddenLayers[i].propagate(layerOutput);
        }

        // Error calculation on the output layer output
        double e = 0.0;
        for(int i = 0; i < (int)layerOutput.size(); i++) {
            e += pow(d[i] - layerOutput[i], 2);
        }

        // Check if we continue
        if(e / 2.0 < threshold) {
            break;
        }

        // Get the last layer (output layer)
        HidenLayer& outputLayer = hiddenLayers.back();

        // Error signal calculation of output layer
        for(int i = 0; i < outputLayer.getNeuronNumber(); i++) {
            double signal = (d[i] - outputLayer.getLastOutput(i))
                          * outputLayer.getDerivative()(1.0, outputLayer.getLastPotential(i));
            outputLayer.setNeuronLastSignal(i, signal);
        }

        // Correction of weights for output layer
        for(int i = (int)hiddenLayers.size() - 2; i >= 0; i--) {
            HidenLayer& currentLayer = hiddenLayers[i];
            HidenLayer& nextLayer = hiddenLayers[i + 1];
            for(int j = 0; j < currentLayer.getNeuronNumber(); j++) {
                double signal = currentLayer.getDerivative()(1.0, currentLayer.getLastPotential(j));
                double sum = 0.0;
                for(int s = 0; s < nextLayer.getNeuronNumber(); s++) {
                    sum += nextLayer.getNeuronLastSignal(s) * nextLayer.getNeuronWeights(s)[j+1];
                }
                signal *= sum;
                currentLayer.setNeuronLastSignal(j, signal);
            }
        }

        // Error signal calculation of other layers
        for(int i = 0; i < outputLayer.getNeuronNumber(); i++) {
            double signal = outputLayer.getNeuronLastSignal(i);
            vector<double> lastEntry = outputLayer.getLastEntry(i);
            vector<double> newWeights;
            // Biais (weights[0]) → entrée fictive = 1.0
            newWeights.push_back(outputLayer.getLearningRate() * signal * 1.0);
            for(int j = 0; j < outputLayer.getEntryNumber(); j++) {
                newWeights.push_back(outputLayer.getLearningRate() * signal * lastEntry[j]);
            }
            outputLayer.changeNeuronWeights(i, newWeights);
        }

        // Correction of weights for other layers
        for(int i = (int)hiddenLayers.size() - 2; i >= 0; i--) {
            HidenLayer& currentLayer = hiddenLayers[i];
            for(int j = 0; j < currentLayer.getNeuronNumber(); j++) {
                double signal = currentLayer.getNeuronLastSignal(j);
                vector<double> lastEntry = currentLayer.getLastEntry(j);
                vector<double> newWeights;
                // Biais
                newWeights.push_back(currentLayer.getLearningRate() * signal * 1.0);
                for(int s = 0; s < currentLayer.getEntryNumber(); s++) {
                    newWeights.push_back(currentLayer.getLearningRate() * signal * lastEntry[s]);
                }
                currentLayer.changeNeuronWeights(j, newWeights);
            }
        }

        epoch++;
    }
}