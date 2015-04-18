#include "StackedAE.h"
using namespace std;
using namespace Eigen;
StackedAE::StackedAE(int ae1HiddenSize,int ae2HiddenSize,int numClasses)
{
    this->ae1HiddenSize = ae1HiddenSize;
    this->ae2HiddenSize = ae2HiddenSize;
    this->numClasses = numClasses;
}

MatrixXd StackedAE::getAe1Theta()
{
    return aeTheta1;
}

MatrixXd StackedAE::getAe2Theta()
{
    return aeTheta2;
}

//forward calculation
MatrixXd StackedAE::feedForward(MatrixXd &theta,MatrixXd &b,
                                     MatrixXd data)
{
    int m = data.cols();
    MatrixXd z2 = theta * data + b.replicate(1,m);
    MatrixXd a2 = sigmoid(z2);
    return a2;
}

//predict
MatrixXi StackedAE::predict(
        MatrixXd &data)
{
    //forward calculation
    char str[200] = {0};
    /*sprintf_s(str,"aetheta1 %d %d; %d %d",aeTheta1.rows(),aeTheta1.cols(),data.rows(),data.cols());
    MessageBoxA(NULL,str,"",MB_OK);*/
    MatrixXd term1 = aeTheta1 * data;
    MatrixXd z2 = bsxfunPlus(term1,aeB1);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd term2 = aeTheta2 * a2;
    /*sprintf_s(str,"aetheta2 %d %d; %d %d",aeTheta2.rows(),aeTheta2.cols(),a2.rows(),a2.cols());
    MessageBoxA(NULL,str,"",MB_OK);*/
    MatrixXd z3 = bsxfunPlus(term2,aeB2);
    MatrixXd a3 = sigmoid(z3);
    MatrixXd z4 = softMaxTheta * a3;
    /*sprintf_s(str,"softmaxTheta %d %d; %d %d",softMaxTheta.rows(),softMaxTheta.cols(),a3.rows(),a3.cols());
    MessageBoxA(NULL,str,"",MB_OK);*/
    //char str[200];
    /*sprintf_s(str,"%d %d",z4.rows(),z4.cols());
    MessageBoxA(NULL,str,"",MB_OK);*/
    MatrixXi pred(z4.cols(),1);
    for(int i = 0;i < z4.cols();i++)
    {
        double max = INT_MIN;
        int idx = 0;
        for(int j = 0;j < z4.rows();j++)
        {
            if(z4(j,i) > max)
            {
                idx = j;
                max = z4(j,i);
            }
        }
        pred(i,0) = idx;
    }
    return pred;
}

//component wise softmax gradient
MatrixXd StackedAE::softmaxGradient(MatrixXd &x)
{
    MatrixXd negX = x * (-1);
    MatrixXd expX = expMat(negX);
    MatrixXd term1 = (MatrixXd::Ones(expX.rows(),expX.cols())
        + expX).array().square();
    MatrixXd grad = expX.cwiseQuotient(term1);
    return grad;
}

//update all parameters
void StackedAE::SGD_updateParameters(MatrixXd &theta1Grad,MatrixXd &theta2Grad,
                           MatrixXd &b1Grad,MatrixXd &b2Grad,
                           MatrixXd &softmaxThetaGrad,double alpha)
{
    V_theta1=0.9*V_theta1-alpha*theta1Grad;
    V_theta2=0.9*V_theta2-alpha*theta2Grad;
    V_B1=0.9*V_B1-alpha*b1Grad;
    V_B2=0.9*V_B2-alpha*b2Grad;
    V_softMaxtheta=0.9*V_softMaxtheta-alpha*softmaxThetaGrad;
    aeTheta1 += V_theta1;
    aeTheta2 += V_theta2;
    aeB1 += V_B1;
    aeB2 += V_B2;
    softMaxTheta += V_softMaxtheta;
}

//cost function
double StackedAE::computeCost(MatrixXd &theta1Grad,
        MatrixXd &b1Grad,MatrixXd &theta2Grad,
        MatrixXd &b2Grad,MatrixXd &softmaxThetaGrad,
        MatrixXd &data,MatrixXi &labels,double lambda)
{
    MatrixXd groundTruth = binaryCols(labels,numClasses);
    int M = labels.rows();
    //forward calculate
    MatrixXd term1 = aeTheta1 * data;
    MatrixXd z2 = bsxfunPlus(term1,aeB1);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd term2 = aeTheta2 * a2;
    MatrixXd z3 = bsxfunPlus(term2,aeB2);
    MatrixXd a3 = sigmoid(z3);
    MatrixXd z4 = softMaxTheta * a3;
    MatrixXd a4 = expMat(z4);
    MatrixXd a4ColSum = a4.colwise().sum();
    a4 = bsxfunRDivide(a4,a4ColSum);
    //calculate delta
    MatrixXd delta4 = a4 - groundTruth;
    MatrixXd delta3 = (softMaxTheta.transpose() * delta4).cwiseProduct(sigmoidGradient(z3));
    MatrixXd delta2 = (aeTheta2.transpose() * delta3).cwiseProduct(sigmoidGradient(z2));

    //calculate delta
    softmaxThetaGrad = (groundTruth - a4) * a3.transpose() * (-1.0 / M) + softMaxTheta * lambda;

    theta2Grad = delta3 * a2.transpose() * (1.0 / M) + aeTheta2 * lambda;
    b2Grad = delta3.rowwise().sum() * (1.0 / M);
    theta1Grad = delta2 * data.transpose() * (1.0 / M) + aeTheta1 * lambda;
    b1Grad = delta2.rowwise().sum() * (1.0 / M);

    //compute cost
    double cost = (-1.0 / M) * (groundTruth.cwiseProduct(logMat(a4))).array().sum()
        + lambda / 2.0 * softMaxTheta.array().square().sum()
        + lambda / 2.0 * aeTheta1.array().square().sum()
        + lambda / 2.0 * aeTheta2.array().square().sum();

    return cost;
}

//fine tune the model
void StackedAE::fineTune(MatrixXd &data,MatrixXi &labels,
                   double lambda,double alpha,int maxIter,int batchSize)
{
    MatrixXd theta1Grad(aeTheta1.rows(),aeTheta1.cols());
    MatrixXd theta2Grad(aeTheta2.rows(),aeTheta2.cols());
    MatrixXd b1Grad(aeB1.rows(),aeB1.cols());
    MatrixXd b2Grad(aeB2.rows(),aeB2.cols());
    MatrixXd softmaxThetaGrad(softMaxTheta.rows(),softMaxTheta.cols());
    MatrixXd miniTrainData(data.rows(),batchSize);
    MatrixXi miniLabels(batchSize,1);
    int iter = 1;
    int numBatches = data.cols() / batchSize;

    //mini batch stochastic gradient decent
    for(int i = 0; i < maxIter;i++)
    {
        double J = 0;
        // compute the cost
        for(int j = 0;j < numBatches; j++)
        {
            miniTrainData = data.middleCols(j * batchSize,batchSize);
            miniLabels = labels.middleRows(j * batchSize,batchSize);
            J += computeCost(theta1Grad,b1Grad,theta2Grad,
                b2Grad,softmaxThetaGrad,miniTrainData,miniLabels,lambda);
            if(miniTrainData.cols() < 1 || miniTrainData.rows() < 1)
            {
                cout << "Too few training examples!"  << endl;
            }


            if(fabs(J) < 0.001)
            {

            }
            if(j==0){
                V_theta1.setZero(theta1Grad.rows(),theta1Grad.cols());
                V_theta2.setZero(theta2Grad.rows(),theta2Grad.cols());
                V_B1.setZero(b1Grad.rows(),b1Grad.cols());
                V_B2.setZero(b2Grad.rows(),b2Grad.cols());
                V_softMaxtheta.setZero(softmaxThetaGrad.rows(),softmaxThetaGrad.cols());
            }
            SGD_updateParameters(theta1Grad,theta2Grad,b1Grad,b2Grad,softmaxThetaGrad,alpha);
        }
        J = J / numBatches;
        cout << "iter: " << iter++ << "  cost: " << J << endl;
    }
}

//pretrain the model
void StackedAE::preTrain(MatrixXd &data,MatrixXi &labels,
        double lambda[],double alpha[],int miniBatchSize[],
        int maxIter[],double noiseRatio[],
        double beta[])
{
        int numOfExamples = data.cols();
        int ndim = data.rows();
        inputSize = ndim;
        //stacked denoising autoencoders
        cout << "PreTraining with denoising autoencoder ..." << endl;
        //train the first denoising autoencoder
        DAE ae1(ndim,ae1HiddenSize);
        cout << "PreTraining ae1 ..." << endl;
        ae1.train(data,noiseRatio[0],alpha[0],maxIter[0],miniBatchSize[0]);

        MatrixXd theta1 = ae1.getTheta();
        aeTheta1.resize(theta1.rows(),theta1.cols());
        aeTheta1 = theta1;
        MatrixXd b1 = ae1.getBias();
        aeB1.resize(b1.rows(),b1.cols());
        aeB1 = b1;

        //train the second denoising autoencoder
        MatrixXd ae1Features = feedForward(aeTheta1,aeB1,data);
        DAE ae2(ae1HiddenSize,ae2HiddenSize);
        cout << "PreTraining ae2 ..." << endl;
        ae2.train(ae1Features,noiseRatio[1],alpha[1],maxIter[1],miniBatchSize[1]);

        MatrixXd theta2 = ae2.getTheta();
        aeTheta2.resize(theta2.rows(),theta2.cols());
        aeTheta2 = theta2;
        MatrixXd b2 = ae2.getBias();
        aeB2.resize(b2.rows(),b2.cols());
        aeB2 = b2;
        //train the softmax regression
        MatrixXd ae2Features = feedForward(aeTheta2,aeB2,ae1Features);
        cout << "PreTraining softmax ..." << endl;
        Softmax softmax(ae2HiddenSize,numClasses);
        softmax.train(ae2Features,labels,lambda[2],alpha[2],maxIter[2],miniBatchSize[2]);
        MatrixXd smTheta = softmax.getTheta();
        softMaxTheta.resize(smTheta.rows(),smTheta.cols());
        softMaxTheta = smTheta;
}




