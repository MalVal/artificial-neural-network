#include <iostream>
#include <vector>
#include "MultiLayerPerceptron.h"
using namespace std;

int main() {
    // Architecture : 2 entrées, 1 couche cachée de 2 neurones, 1 sortie
    MultiLayerPerceptron mlp({2, 1}, 2);

    // Dataset XOR
    vector<vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    vector<vector<double>> D = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Entraînement
    mlp.learn(X, D, 0.01, 100000);

    // Test
    cout << "=== Results ===" << endl;
    for(int k = 0; k < (int)X.size(); k++) {
        vector<double> layerOutput = mlp.predict(X[k]);
        cout << X[k][0] << " XOR " << X[k][1]
             << " --> " << layerOutput[0]
             << " (attendu: " << D[k][0] << ")" << endl;
    }

    return 0;
}