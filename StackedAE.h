#ifndef STACKEDAE_H
#define STACKEDAE_H

#include "DAE.h"
#include "utils.h"
#include "SoftMax.h"
class StackedAE{
private:
    MatrixXd aeTheta1;
    MatrixXd aeTheta2;
    MatrixXd aeB1;
    MatrixXd aeB2;
    MatrixXd softMaxTheta;
    //momentum;
    MatrixXd V_theta1;
    MatrixXd V_theta2;
    MatrixXd V_B1;
    MatrixXd V_B2;
    MatrixXd V_softMaxtheta;
    //Historic Gradients
    MatrixXd G_theta1;
    MatrixXd G_theta2;
    MatrixXd G_B1;
    MatrixXd G_B2;
    MatrixXd G_softMaxtheta;
    //parameters
    int numClasses;
    int ae1HiddenSize;
    int ae2HiddenSize;
    int inputSize;
    MatrixXd softmaxGradient(MatrixXd &x);
    MatrixXd feedForward(MatrixXd &theta,MatrixXd &b,MatrixXd data);
    void SGD_updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,MatrixXd &b1Grad,MatrixXd &b2Grad,MatrixXd &softmaxTheta,double alpha);
    double computeCost(MatrixXd &theta1Grad,MatrixXd &b1Grad,MatrixXd &theta2Grad,MatrixXd &b2Grad,MatrixXd &softmaxThetaGrad,MatrixXd &data,MatrixXi &labels,double lambda);
public:
    StackedAE(int ae1HiddenSize,int ae2HiddenSize,int numClasses);
    MatrixXi predict(MatrixXd &data);
    void fineTune(MatrixXd &data,MatrixXi &labels,double lambda,double alpha,int maxIter,int batchSize);
    void preTrain(MatrixXd &data,MatrixXi &labels,double lambda[],double alpha[],int miniBatchSize[],int maxIter[],double noiseRatio[] = NULL,double beta[] = NULL);
    MatrixXd getAe1Theta();
    MatrixXd getAe2Theta();
};


#endif // STACKEDAE_H
