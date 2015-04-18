#ifndef DAE_H
#define DAE_H
#include <Eigen/Dense>
#include <ctime>
#include <iostream>
#include <fstream>
#include "utils.h"
using namespace std;
using namespace Eigen;

class DAE{
public:
    MatrixXd theta1;
    MatrixXd theta2;
    MatrixXd b1;
    MatrixXd b2;
    int inputSize;
    int hiddenSize;
    DAE(int inputSize,int hiddenSize);
    void train(MatrixXd &trainData,double noiseRatio,double alpha,int maxIter,int miniBatchSize);
    MatrixXd getTheta();
    MatrixXd getBias();
private:
    MatrixXd noiseInput(MatrixXd &z,double noiseRatio);
    MatrixXd randomInitialize(int lIn,int lOut);
    void updateParameters(MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha);
    void miniBatchSGD(MatrixXd &trainData,MatrixXd &noiseData,double alpha,int maxIter,int batchSize);
    double computeCost(MatrixXd &data,MatrixXd &noiseData,MatrixXd &theta1Grad,MatrixXd &theta2Grad,MatrixXd &b1Grad,MatrixXd &b2Grad);
};

#endif // DAE_H
