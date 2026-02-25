#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h>
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

    // ==== Perceptron avec méthode de la descente du gradient ====

    // Fichiers résultats
    string dataCSV = "points.csv";
    string modelCSV = "model.csv";
    string errorCSV = "errors.csv";

    ofstream errorFile(errorCSV);

    // Initialisation des poids synaptiques
    double w0 = 0, w1 = 0, w2 = 0;
    // Seuil de tolérence
    double seuil = 0.125001;
    // Pas d'apprentissage
    double n = 0.2;
    // Nombre max d'itérations
    int nbIterMax = 100;
    // Numéro de l'itération
    int epoch = 0;

    while(epoch < nbIterMax) {
        // Initialisation des termes correcteurs (delta w)
        double dw0 = 0, dw1 = 0, dw2 = 0;
        cout << "==========================================================" << endl;
        cout << "Epoque : " << epoch + 1 << endl;
        cout << "Poids synaptiques :" << endl;
        cout << "w0 = " << w0 << ", w1 = " << w1 << ", w2 = " << w2 << endl;

        int exampleNumber = 1;
        double E = 0;

        for(const auto &row : data) {
            cout << "=============================" << endl;
            cout << "Exemple : " << exampleNumber << endl;
            cout << "Correcteurs des poids synaptiques :" << endl;
            cout << "dw0 = " << dw0 << ", dw1 = " << dw1 << ", dw2 = " << dw2 << endl;

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
            double y = p;
            cout << "Sortie calculee : " << y << endl;

            // Calcul de l'erreur locale
            double e = d - y;
            cout << "Erreur locale : " << e << endl;
            // Calcul de l'erreur quadriatique
            E = E + (0.5 * e * e);

            dw0 = dw0 + n*e*x0;
            dw1 = dw1 + n*e*x1;
            dw2 = dw2 + n*e*x2; 

            // Passage à l'exemple suivant
            exampleNumber++;
        }

        // Correction des poids synaptiques
        w0 = w0 + dw0;
        w1 = w1 + dw1;
        w2 = w2 + dw2;

        // Calcul de l'erreur quadriatique moyenne
        double Emoy = E / (exampleNumber - 1);

        // Ecriture dans le fichier errors.csv
        errorFile << epoch + 1 << "," << Emoy << endl;
        
        // Si l'erreur quadriatique moyenne est < au seuil -> apprentissage terminé
        if(Emoy < seuil) {
            cout << "================================" << endl;
            cout << "Apprentissage fini" << endl;
            cout << "================================" << endl;
            break;
        }

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