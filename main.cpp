#include <iostream>
#include <fstream>
#include <string>
#include "Read_Data.h"
#include "StackedAE.h"
#include "Cifar_Loader.h"
#include <Eigen/Core>
using namespace std;
using namespace Eigen;
int main(int argc, char ** argv){
    int ae1HiddenSize = 100;
    int ae2HiddenSize = 100;
    int numClasses = 10;
    double noiseRatio[2] = {0.3,0.3};
    double lambda[4] = {3e-3,3e-3,3e-3,1e-4};
    double alpha[4] = {0.2,0.2,0.2,0.2};
    double beta[2] = {3,3};
    int maxIter[4] = {100,100,100,200};
    int miniBatchSize[4] = {1000,1000,1000,1000};
    MatrixXd trainingData(1,1);
    MatrixXi trainingLabels(1,1);
    MatrixXd testData(1,1);
    MatrixXi testLabels(1,1);
    clock_t start = clock();
    bool mnist=true;
    if(mnist){
    bool ret;
    ret = loadMnistData(trainingData,"/Users/ruizhang/Desktop/hw5/denosing_autoencoder/mnist/train-images-idx3-ubyte");
    cout << trainingData.rows() << " " << trainingData.cols() << endl;
    cout << "Loading training data..." << endl;
    ret = loadMnistLabels(trainingLabels,"/Users/ruizhang/Desktop/hw5/denosing_autoencoder/mnist/train-labels-idx1-ubyte");
    ret = loadMnistData(testData,"/Users/ruizhang/Desktop/hw5/denosing_autoencoder/mnist/t10k-images-idx3-ubyte");
    ret = loadMnistLabels(testLabels,"/Users/ruizhang/Desktop/hw5/denosing_autoencoder/mnist/t10k-labels-idx1-ubyte");
        std::ofstream ofs ("test.txt", std::ofstream::out);
        ofs<<testLabels<<endl;
        ofs.close();
    }
    else{
        CIFARLoader loader("/Users/ruizhang/Desktop/hw5/test/Cifar");
        trainingData=loader.trainingInput.transpose();
        cout<<trainingData.rows()<<" "<<trainingData.cols()<<endl;
        trainingLabels=loader.trainingOutput;
        testData=loader.testInput.transpose();
         cout<<testData.rows()<<" "<<testData.cols()<<endl;
        testLabels=loader.testOutput;
        std::ofstream ofs ("test.txt", std::ofstream::out);
        ofs<<testLabels<<endl;
        ofs.close();
    }

    StackedAE stackedAE(ae1HiddenSize,ae2HiddenSize,numClasses);
    stackedAE.preTrain(trainingData,trainingLabels,lambda,alpha,miniBatchSize,
        maxIter,noiseRatio,beta);
    cout << "Loading test data..." << endl;
    MatrixXi pred1 = stackedAE.predict(testData);
    cout << pred1.rows() << " " << pred1.cols() << endl;
    cout << testLabels.rows() << " " << testLabels.cols() << endl;
    double acc1 = calcAccurancy(testLabels,pred1);
    cout << "Accurancy before fine tuning: " << acc1 * 100 << "%" << endl;
    MatrixXd aeTheta1 = stackedAE.getAe1Theta();
    MatrixXd aeTheta2 = stackedAE.getAe2Theta();
    MatrixXd filter = aeTheta2 * aeTheta1;
    cout << "Fine Tuning..." << endl;
    stackedAE.fineTune(trainingData,trainingLabels,lambda[3],
        alpha[3],maxIter[3],miniBatchSize[3]);
    MatrixXi pred2 = stackedAE.predict(testData);
    double acc2 = calcAccurancy(testLabels,pred2);
    cout << "Accurancy: " << acc2 * 100 << "%" << endl;
    aeTheta1 = stackedAE.getAe1Theta();
    aeTheta2 = stackedAE.getAe2Theta();
    filter = aeTheta2 * aeTheta1;
    clock_t end = clock();
    cout << "The code ran for " <<
        (end - start)/(double)(CLOCKS_PER_SEC*60) <<
        " minutes on " << Eigen::nbThreads() << " thread(s)." << endl;
    return 0;
}

