#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "utils.h"
#include <ctime>
#include <fstream>
#include <iostream>
using namespace std;
class Softmax
{
private:
    MatrixXd theta;
    int inputSize;
    int numClasses;
    MatrixXd randomInitialize(int lIn,int lOut);
    double computeCost(double lambda,MatrixXd &data,MatrixXi &labels,MatrixXd &thetaGrad);
    void miniBatchSGD(MatrixXd &trainData,MatrixXi &labels,double lambda,double alpha,int maxIter,int batchSize);
public:
    Softmax(int inputSize,int numClasses);
    MatrixXi predict(MatrixXd &data);
    void train(MatrixXd &data,MatrixXi &labels,double lambda,double alpha,int maxIter,int miniBatchSize);
    MatrixXd getTheta();
};


#endif // SOFTMAX_H
