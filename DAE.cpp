#include "DAE.h"
using namespace std;
using namespace Eigen;
DAE::DAE(int inputSize,int hiddenSize)
{
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    theta1 = randomInitialize(hiddenSize,inputSize);
    theta2 = randomInitialize(inputSize,hiddenSize);
    b1 = MatrixXd::Zero(hiddenSize,1);
    b2 = MatrixXd::Zero(inputSize,1);
}

MatrixXd DAE::getTheta()
{
    return theta1;
}

MatrixXd DAE::getBias()
{
    return b1;
}

MatrixXd DAE::noiseInput(MatrixXd &z,double noiseRatio)
{
    MatrixXd result(z.rows(),z.cols());
    for(int i = 0;i < z.rows();i++)
    {
        for(int j = 0;j < z.cols();j++)
        {
            result(i,j) = z(i,j) * (rand() > (int)(noiseRatio * RAND_MAX));
        }
    }
    return result;
}


//random initialize the weights
MatrixXd DAE::randomInitialize(int lIn,int lOut)
{
    //random initialize the weight
    int i,j;
    double epsilon = sqrt(6.0/(lIn + lOut + 1));
    MatrixXd result(lIn,lOut);
    srand((unsigned int)time(NULL));
    for(i = 0;i < lOut;i++)
    {
        for(j = 0;j < lIn;j++)
        {
            result(j,i) = ((double)rand() / (double)RAND_MAX) * 2 * epsilon - epsilon;
        }
    }
    return result;
}

//gradient descent update rule
void DAE::updateParameters(
    MatrixXd &theta1Grad1,MatrixXd &theta2Grad2,
    MatrixXd &b1Grad,MatrixXd &b2Grad,double alpha)
{
    theta1 -= theta1Grad1*alpha;
    theta2 -= theta2Grad2*alpha;
    b1 -= b1Grad * alpha;
    b2 -= b2Grad * alpha;
}

//cost function
double DAE::computeCost(
    MatrixXd &data,MatrixXd &noiseData,
    MatrixXd &theta1Grad,MatrixXd &theta2Grad,
    MatrixXd &b1Grad,MatrixXd &b2Grad)
{

    double cost = 0;

    int numOfExamples = data.cols();

    MatrixXd a1 = noiseData;

    MatrixXd z2 = theta1 * noiseData + b1.replicate(1,numOfExamples);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd z3 = theta2 * a2 + b2.replicate(1,numOfExamples);
    MatrixXd a3 = sigmoid(z3);


    //compute delta
    MatrixXd delta3 = (a3 - data).cwiseProduct(sigmoidGradient(z3));

    MatrixXd delta2 = (theta2.transpose() * delta3).cwiseProduct(sigmoidGradient(z2));

    //compute gradients

    theta2Grad = delta3 * a2.transpose() * (1.0 / (double)numOfExamples);

    b2Grad = delta3.rowwise().sum() * (1.0 / (double)numOfExamples);

    theta1Grad = delta2 * a1.transpose() * ( 1.0 / (double)numOfExamples);

    b1Grad = delta2.rowwise().sum() * (1.0  / (double)numOfExamples);



    //compute cost

    cost = (a3 - data).array().square().sum() * (1.0 / 2.0 / numOfExamples);

    return cost;
}


//mini batch stochastic gradient descent
void DAE::miniBatchSGD(
    MatrixXd &trainData,MatrixXd &noiseData,double alpha,int maxIter,int batchSize)
{
    //get the binary code of labels
    MatrixXd theta1Grad(theta1.rows(),theta1.cols());
    MatrixXd theta2Grad(theta2.rows(),theta2.cols());
    MatrixXd b1Grad(b1.rows(),b1.cols());
    MatrixXd b2Grad(b2.rows(),b2.cols());
    MatrixXd miniTrainData(trainData.rows(),batchSize);
    MatrixXd miniNoiseData(trainData.rows(),batchSize);
    int iter = 1;
    int numBatches = trainData.cols() / batchSize;

    //mini batch stochastic gradient decent
    for(int i = 0; i < maxIter;i++)
    {
        double J = 0;
        // compute the cost
        for(int j = 0;j < numBatches; j++)
        {
            miniTrainData = trainData.middleCols(j * batchSize,batchSize);
            miniNoiseData = noiseData.middleCols(j * batchSize,batchSize);
            J += computeCost(miniTrainData,miniNoiseData,theta1Grad,theta2Grad,
                b1Grad,b2Grad);
            if(miniTrainData.cols() < 1 || miniTrainData.rows() < 1)
            {
                cout << "Too few training examples!"  << endl;
            }

            if(fabs(J) < 0.001)
            {
                break;
            }
            updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,alpha);
        }
        J = J / numBatches;
        cout << "iter: " << iter++ << "  cost: " << J << endl;

    }
}

//train the model
void DAE::train(MatrixXd &trainData,double noiseRatio,double alpha,int maxIter,int miniBatchSize)
{
    if(trainData.rows() != this->inputSize)
    {
        cout << "TrainData rows:" << trainData.rows() << endl;
        cout << "dimension mismatch!" << endl;
        return;
    }
    MatrixXd noiseData = noiseInput(trainData,noiseRatio);

    miniBatchSGD(trainData,noiseData,alpha,maxIter,miniBatchSize);
}


