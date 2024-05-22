#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
using namespace std;

struct neuron {
	vector<float> w;	
	neuron() : w(785, 0) {}
};

struct perceptron {
	float x1;
	float x2;
	float bias;

	vector<neuron*> neurons;
	float n;

	perceptron(float n) : n(n), bias(1), x1(0), x2(0) {
		for (int i = 0; i < 10; i++) {
			neurons.push_back(new neuron);
		}
	}

	float func(float x) {
		float tmp = 1 / (1 + exp(-x));
		if (tmp > 0.5) return 1;
		else return 0;
	}

	bool check(vector<int>& entrada, vector<neuron*>& out) {
		float y = 0;
		float arr[10] = { 0 };
		arr[entrada[0]] = { 1 };
		for (int j = 0; j < 10; j++) {
			for (int i = 1; i < 785; i++) {
				y += entrada[i] * out[j]->w[i]; // Suma todos los pesos 
			}
			y += 1 * out[j]->w[0]; // suma el bias
			y = func(y);
			if (y != arr[j]) return false;
		}
		return true;
	}

	void sum_entrada(vector<int>& entrada,vector<neuron*>& out) {
		float y = 0;
		float arr[10] = { 0 };
		arr[entrada[0]] = { 1 };
        for (int j = 0; j < 10; j++) {
            for (int i = 1; i < 785; i++) {
                y += entrada[i] * out[j]->w[i]; // Suma todos los pesos 
            }
            y += 1 * out[j]->w[0]; // suma el bias
            y = func(y);
            if (y != arr[j]) { // Se modifican los pesos
                float error = arr[j] - y;
                for (int i = 1; i < 785; i++) {
                    out[j]->w[i] += n * entrada[i] * error;
                }
                out[j]->w[0] += n * 1 * error; // Modifica peso Bias
            }
        }
	}

	void training(vector<vector<int>>& data, vector<neuron*> salidas) {
		int epoch = 0;
		int cont = 0;
		while (epoch < 100) {
			for (int i = 0; i < 60000; i++)
				sum_entrada(data[i], salidas);
			epoch++;
			cont = 0;
			for (int i = 0; i < 60000; i++)
				if (check(data[i], salidas)) cont++;
				
			cout << "Epoch: " << epoch << endl;
			cout << "Accepted: " << cont << endl;
		}
	}
};

int main() {

	string datos = "mnist_train.csv";
	vector <vector<int>> entradas;
	ifstream archivo(datos);
	if (archivo.is_open()) {
		string linea;
		int i = 0;
		while (getline(archivo, linea)) {
			vector<int> fila;
			stringstream ss(linea);
			string valor;
			while (getline(ss, valor, ',')) {
				fila.push_back(stoi(valor));
			}
			entradas.push_back(fila);
		}
		archivo.close();
	}
	else cout << "error";
	cout << "Archivo leido" << endl;
	perceptron work(0.5);
	work.training(entradas, work.neurons);
	return 0;
}