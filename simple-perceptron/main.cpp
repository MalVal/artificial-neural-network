#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

int main() {
    // Lecture du fichier CSV
    ifstream file("../../data/and.csv");

    if (!file.is_open()) {
        cerr << "Error when opening file..." << endl;
        return 1;
    }

    string line;
    vector<vector<int>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<int> row;

        while (getline(ss, cell, ',')) {
            row.push_back(stoi(cell));
        }

        data.push_back(row);
    }

    file.close();

    // ==== Perceptron simple ====

    // Fichiers résultats
    string dataCSV = "points.csv";
    string modelCSV = "model.csv";
    string errorCSV = "errors.csv";

    ofstream errorFile(errorCSV);

    // Initialisation des poids synaptiques
    double w0 = 0, w1 = 0, w2 = 0;
    // Nombre d'erreurs
    int nbError = 0;
    // Pas d'apprentissage
    double n = 1;
    // Numéro de l'itération
    int epoch = 1;

    while(true) {
        int i = 1;
        for(const auto &row : data) {
            cout << "================================" << endl;
            cout << "Exemple : " << i << endl;
            cout << "Poids synaptiques :" << endl;
            cout << "w0 = " << w0 << ", w1 = " << w1 << ", w2 = " << w2 << endl;

            // Initialisation des entrées
            double x0 = 1, x1 = row[0], x2 = row[1];
            cout << "Entrees :" << endl;
            cout << "x0 = " << x0 << ", x1 = " << x1 << ", x2 = " << x2 << endl;

            // Sortie voulue
            int d = row[2];
            cout << "Sortie voulue : " << d << endl;

            // Calcul du potentiel
            double p = (w0 * x0) + (w1 * x1) + (w2 * x2);
            cout << "Potentiel = " << p << endl;

            // Sortie calculée
            int y;
            if(p >= 0) {
                y = 1;
            }
            else {
                y = 0;
            }
            cout << "Sortie calculee : " << y << endl;

            // Calcul de l'erreur
            int e = d - y;
            cout << "Erreur : " << e << endl;

            // Si il y a une erreur
            if(e != 0) {
                nbError++;
                // Correction des poids synaptiques
                w0 = w0 + n*e*x0;
                w1 = w1 + n*e*x1;
                w2 = w2 + n*e*x2; 
            }

            // Passage à l'exemple suivant
            i++;
        }

        // Ecriture dans le fichier errors.csv
        errorFile << epoch << "," << nbError << endl;

        // Si pas d'erreur -> apprentissage terminé
        if(nbError == 0) {
            cout << "================================" << endl;
            cout << "Aucune erreur commise... Apprentissage fini" << endl;
            cout << "================================" << endl;
            break;
        }
        // Sinon reset du compteur
        cout << "================================" << endl;
        cout << "Nombre d'erreur(s) commise(s) : " << nbError << endl;
        cout << "================================" << endl;
        nbError = 0;

        // Passage à l'itération suivante
        epoch++;
    }

    errorFile.close();

    // Ecriture des poids synaptiques dans model.csv
    ofstream modelFile(modelCSV);
    modelFile << w0 << "," << w1 << "," << w2 << endl;
    modelFile.close();
    // Ecriture des points du dataset
    ofstream dataFile(dataCSV);
    for(const auto &row : data) {
        dataFile << row[0] << "," << row[1] << "," << row[2] << endl;
    }
    dataFile.close();

    // Affichage des graphiques
    system(("python ../model-point.py " + dataCSV + " " + modelCSV).c_str());
    system(("python ../error-epoch.py " + errorCSV).c_str());

    return 0;
}