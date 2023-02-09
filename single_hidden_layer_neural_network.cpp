// single_hidden_layer_neural_network.cpp
// Song Li
// 12/6/2021

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <string>

using std::cin; using std::cout; using std::endl;
using std::vector;
using std::make_pair;
using std::map;
using std::pair;

// Output Layer
class OutputNode {
public:
    OutputNode(int num): num_(num){};
    void setVal(int val) {val_ = val;}
    int getVal() const {return val_;}
    int getNum() const {return num_;}
private:
    int num_;
    int val_;
};

// Intermediate/Hidden Layer
class HiddenNode {
public: // methods - manipulate the objects
    HiddenNode(int num): num_(num){};
    void setVal(int val) {val_ = val;}
    int getVal() const {return val_;}
    int getNum() const {return num_;}

    void connectOutput(OutputNode* output_node, double weight) {
        auto ret = connections_.insert(make_pair(output_node, weight));
        if(!ret.second){
            cout << "This connection alrady exists: weight = " << connections_[output_node] << endl;
        }
    }
    
    // ->getWeight().second is 1, means there is a connection from 'this' to 'output_node', then access the weight with ->getWeight().first
    // ->getWeight().second is 0, means there is no connection from 'this' to 'output_node'
    pair<double, bool> getWeight(OutputNode* output_node) {
        auto it = connections_.find(output_node);
        if(it != connections_.end()){
            return make_pair(connections_[output_node], true);
        }
        return make_pair(NULL, false);
    }

    void setWeight(OutputNode* output_node, double weight){
        connections_[output_node] = weight;
    }

    // not needed
    map<OutputNode*, double> getMap(){
        return connections_;
    }

private: // member variables - holds data of entities
    int num_;
    int val_;
    map<OutputNode*, double> connections_; // connections from hidden layer to output layer
};

// Input Layer
class InputNode {
public: // methods - manipulate the objects
    InputNode(int num): num_(num){};
    void setVal(int val) {val_ = val;}
    int getVal() const {return val_;}
    int getNum() const {return num_;}

    void connectHidden(HiddenNode* hidden_nodes, double weight) {
        auto ret = connections_.insert(make_pair(hidden_nodes, weight));
        if(!ret.second)
            cout << "This connection alrady exists: weight = " << connections_[hidden_nodes] << endl;

    }

    pair<double, bool> getWeight(HiddenNode* hidden_nodes) {
        auto it = connections_.find(hidden_nodes);
        if(it != connections_.end()){
            return make_pair(connections_[hidden_nodes], true);
        }
        return make_pair(NULL, false);
    }

    // not needed
    map<HiddenNode*, double> getMap(){
        return connections_;
    }

private: // member variables - holds data of entities
    int num_;
    int val_;
    map<HiddenNode*, double> connections_;
};

double doubleRand() {
    return ((rand()%19) + 1) * 0.1 - 1.0; // generate a random double from -0.1 to 0.9
}

void connectLayers(vector<InputNode*> input_nodes, vector<HiddenNode*> hidden_nodes, vector<OutputNode*> output_nodes) {
    // InputNode -> HiddenNode
    for(int i = 0; i < input_nodes.size(); i++){
        for(int j = 0; j < hidden_nodes.size(); j++){
            // random integer decides whether to connect a node in the next layer or not - 50% probability
            int rand_int = rand()%2;
            if(rand_int){ 
                double rand_double = doubleRand();
                input_nodes[i]->connectHidden(hidden_nodes[j], rand_double);
            }
        }
    }

    // HiddenNode -> OutputNode
    for(int i = 0; i < hidden_nodes.size(); i++){
        for(int j = 0; j < output_nodes.size(); j++){
            int rand_int = rand()%2;
            if(rand_int){
                double rand_double = doubleRand();
                hidden_nodes[i]->connectOutput(output_nodes[j], rand_double);
            }
        }
    }

}

// limit to the range of (0, 1), not including 0 and 1
double activationFunction(double x) {
    return 1 / (1 + exp(-x)); // use sigmoid function as the activation function
}

void evaluateNodes(vector<InputNode*> input_nodes, vector<HiddenNode*> hidden_nodes, vector<OutputNode*> output_nodes, double bias, double threshold) {
    // evaluate hidden layer nodes
    for(const auto& h: hidden_nodes) {
        double sum = 0.0;
        for(const auto& i: input_nodes) {
            if(i->getWeight(h).second){
                sum += (i->getWeight(h).first * i->getVal());
            }
        }

        // determine whether the activated value is above threshold
        if(activationFunction(sum + bias) > threshold){
            h->setVal(1);
        }
        else{
            h->setVal(0);
        }
    }

    // evaluate output layer nodes after hidden layer nodes are evaluated
    for(const auto& o: output_nodes){
        double sum = 0.0;
        for(const auto& h: hidden_nodes) {
            if(h->getWeight(o).second){
                sum += (h->getWeight(o).first * h->getVal());
            }
        }
        if(activationFunction(sum + bias) > threshold)
            o->setVal(1);
        else
            o->setVal(0);
    }
}

double computerError(vector<OutputNode*> actual, vector<OutputNode*> target){
    double error = 0.0;
    // check if they have the same size
    if(actual.size() == target.size()){
        int sum = 0;
        for(int i = 0; i < actual.size(); i++) {
            sum += pow(target[i]->getVal() - actual[i]->getVal(), 2);
        }
        error = sqrt(sum);
    }
    else {
        cout << "Size does not match, ignore error." << endl;
    }
    return error;
}

// initialize input layer data entities
vector<vector<InputNode*>> generateInput(int data_size, int input_size){
    vector<vector<InputNode*>> data;
    for(int i = 0; i < data_size; i++) {
        vector<InputNode*> input;
        for(int i = 0; i < input_size; i++) {
            InputNode* node = new InputNode(i+1);
            input.push_back(node);
        }
        data.push_back(input);
    }
    return data;
}

// initialize hidden layer data entities
vector<vector<HiddenNode*>> generateHidden(int data_size, int hidden_size) {
    vector<vector<HiddenNode*>> data;
    for(int i = 0; i < data_size; i++) {
        vector<HiddenNode*> hidden;
        for(int i = 0; i < hidden_size; i++) {
            HiddenNode* node = new HiddenNode(i+1); //debug*//
            hidden.push_back(node);
        }
        data.push_back(hidden);
    }
    return data;
}

// initialize output layer data entities
vector<vector<OutputNode*>> generateOutput(int data_size, int output_size) {
    vector<vector<OutputNode*>> data;
    for(int i = 0; i < data_size; i++) {
        vector<OutputNode*> output;
        for(int i = 0; i < output_size; i++) {
            OutputNode* node = new OutputNode(i+1);
            output.push_back(node);
        }
        data.push_back(output);
    }
    return data;
}

// function adding spaces for formating
void spaces(int num){
    for(int i = 0; i < num; i++){
        cout << " ";
    }
}

void graphModeling(
    vector<InputNode*> input_nodes,
    vector<HiddenNode*> hidden_nodes,
    vector<OutputNode*> output_nodes){

    cout << "Input to Hidden Matrix: (Vertical - Input Nodes, Horizontal - Hidden Nodes)" << endl;
    spaces(6);

    for(const auto& node: hidden_nodes){
        cout << node->getNum();
        spaces(6);
    }
    cout << endl;

    for(const auto& node: input_nodes){
        cout << node->getNum();
        spaces(6 - std::to_string(node->getNum()).length());
        
        for(const auto& hn: hidden_nodes){
            // have connection
            if(node->getWeight(hn).second){
                double d = node->getWeight(hn).first;
                std::string str = std::to_string(d);
                while(str.back() == '0'){
                    str.pop_back();
                }
                cout << str;
                spaces(7 - str.length());
            }
            // have no connection, set as 0
            else{
                cout << "0"; // cout << "0."
                spaces(6);
            }
        }
        cout << endl;
    }

    cout << "Hidden to Output Matrix: (Vertical - Hidden Nodes, Horizontal - Output Nodes)" << endl;
    spaces(6);

    for(const auto& node: output_nodes){
        cout << node->getNum();
        spaces(6);
    }
    cout << endl;

    for(const auto& node: hidden_nodes){
        cout << node->getNum();
        spaces(6 - std::to_string(node->getNum()).length());
        for(const auto& on: output_nodes){
            // have connection
            if(node->getWeight(on).second){
                double d = node->getWeight(on).first;
                std::string str = std::to_string(d);
                while(str.back() == '0'){
                    str.pop_back();
                }

                cout << str;
                spaces(7 - str.length());
            }
            // have no connection, set as 0
            else{
                cout << "0"; // cout << "0."
                spaces(6);
            }
        }
        cout << endl;
    }
}

int main(){
    srand(time(nullptr)); // seed

    // adjustable parameters
    int data_size = 1;
    int input_output_size = 3;
    int hidden_size = 4;

    double bias = 0.5;
    double threshold = 0.5;

    // generate actual output and target output data structures
    vector<vector<OutputNode*>> actual_output = generateOutput(data_size, input_output_size);
    vector<vector<OutputNode*>> target_output = generateOutput(data_size, input_output_size);

    // generate input and hidden data structure
    vector<vector<InputNode*>> input = generateInput(data_size, input_output_size);
    vector<vector<HiddenNode*>> hidden = generateHidden(data_size, hidden_size);

    // assign random values to input and target output data
    for(const auto& nodes: input){
        for(const auto& node: nodes){
            int random = rand()%2;
            node->setVal(random);
        }
    }

    for(const auto& nodes: target_output){
        for(const auto& node: nodes){
            int random = rand()%2;
            node->setVal(random);
        }
    }

    cout << "Index, Input, Hidden, Actual, Target, Error" << endl;

    for(int i = 0; i < data_size; i++) {
        // randomly connect nodes (input nodes -> hidden nodes -> output nodes) with random weights
        connectLayers(input[i], hidden[i], actual_output[i]);
        
        // evaluate hidden nodes and then output nodes based on connected edges
        evaluateNodes(input[i], hidden[i], actual_output[i], bias, threshold);

        // computer error based on actual output and target output
        double error = computerError(actual_output[i], target_output[i]);
        
        // manipulate spaces for formating
        if(i < 9)
            cout << i+1 << "      ";
        else if(i >= 9 && i < 99)
            cout << i+1 << "     ";
        else if(i >= 99 && i < 999)
            cout << i+1 << "    ";
        else
            cout << "You need implement new format." << endl;

        for(const auto& in: input[i]) {
            cout << in->getVal();
        }
        cout << " -> ";
        for(const auto& hn: hidden[i]) {
            cout << hn->getVal();
        }
        cout << " -> ";
        for(const auto& on: actual_output[i]) {
            cout << on->getVal();
        }
        cout << "     ";
        for(const auto& on: target_output[i]) {
            cout << on->getVal();
        }
        cout << "     ";
        cout << error << endl;

        /*
        // Back Propagation - Can also be coded into a function
        */

        int error_threshold = 0;
        cout << "Input a error threshold value: ";
        cin >> error_threshold;

        while(error > error_threshold){
            for(int a = 0; a < hidden[i].size(); a++){
                for(int b = 0; b < actual_output[i].size(); b++){
                    if(actual_output[i][b]->getVal() != target_output[i][b]->getVal()){
                        if(hidden[i][a]->getWeight(actual_output[i][b]).second){
                            double weight_old = hidden[i][a]->getWeight(actual_output[i][b]).first;
                            double weight_new = 0.0;
                            hidden[i][a]->setWeight(actual_output[i][b], doubleRand());
                            weight_new = hidden[i][a]->getWeight(actual_output[i][b]).first;
                            cout << "h" << hidden[i][a]->getNum() << "-" << "o" << actual_output[i][b]->getNum() << ": " << weight_old << " -> " << weight_new << endl;
                        }
                    }
                }
            }
            evaluateNodes(input[i], hidden[i], actual_output[i], bias, threshold);
            error = computerError(actual_output[i], target_output[i]);
            cout << error << endl;
        }
    }
    /*
    int index = data_size+1;
    while(1){
        cout << endl;
        cout << "Enter the index of data you want to do graph modeling (Enter 0 to quit): ";
        cin >> index;
        if(index != 0)
            graphModeling(input[index-1], hidden[index-1], actual_output[index-1]);
        else
            break;
    }
    */
}