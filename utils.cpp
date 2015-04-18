#include "utils.h"
using namespace std;
using namespace Eigen;
MatrixXd bsxfunMinus(MatrixXd &m,MatrixXd &x)
{
    MatrixXd r = m;
    if(x.rows() == 1)
    {
        r = x.replicate(m.rows(),1);
    }
    if(x.cols() == 1)
    {
        r = x.replicate(1,m.cols());
    }
    return m - r;
}

MatrixXd bsxfunRDivide(MatrixXd &m,MatrixXd &x)
{
    MatrixXd r = m;
    if(x.rows() == 1)
    {
        r = x.replicate(m.rows(),1);
    }
    if(x.cols() == 1)
    {
        r = x.replicate(1,m.cols());
    }
    return m.cwiseQuotient(r);
}



MatrixXd bsxfunPlus(MatrixXd &m,MatrixXd &x)
{
    MatrixXd r = m;
    if(x.rows() == 1)
    {
        r = x.replicate(m.rows(),1);
    }
    if(x.cols() == 1)
    {
        r = x.replicate(1,m.cols());
    }
    return m + r;
}

MatrixXd sigmoidGradient(MatrixXd &z)
{
    //return sigmoid(z) .* (1 - sigmoid(z))
    MatrixXd result;
    MatrixXd sigm = sigmoid(z);
    MatrixXd item = MatrixXd::Ones(z.rows(),z.cols()) - sigm;
    result = sigm.cwiseProduct(item);
    return result;
}

//component wise sigmoid function
MatrixXd sigmoid(MatrixXd &z)
{
    return z.unaryExpr(ptr_fun(sigmoidScalar));
}

MatrixXd sqrtMat(MatrixXd &z)
{
    return z.unaryExpr(ptr_fun(sqrtScalar));
}

MatrixXd binaryCols(MatrixXi &labels,int numOfClasses)
{
    // return binary code of labels
    //eye function
    MatrixXd e = MatrixXd::Identity(numOfClasses,numOfClasses);
    int numOfExamples = labels.rows();
    int inputSize = e.cols();
    MatrixXd result(inputSize,numOfExamples);
    for(int i = 0; i < numOfExamples; i++)
    {
        int idx = labels(i,0);
        result.col(i) = e.col(idx);
    }
    return result;
}

//component wise exp function
MatrixXd expMat(MatrixXd &z)
{
    return z.unaryExpr(ptr_fun(expScalar));
}

//component wise log function
MatrixXd logMat(MatrixXd &z)
{
    return z.unaryExpr(ptr_fun(logScalar));
}

double calcAccurancy(MatrixXi &pred,MatrixXi &labels)
{
    int numOfExamples = pred.rows();
    double sum = 0;
    for(int i = 0; i < numOfExamples; i++)
    {
        if(pred(i,0) == labels(i,0))
        {
            sum += 1;
        }
    }
    return sum / numOfExamples;
}

//return 1.0 ./ z
MatrixXd reciprocal(MatrixXd &z)
{
    return z.unaryExpr(ptr_fun(reciprocalScalar));
}

double reciprocalScalar(double x)
{
    return 1.0/x;
}

//scalar sigmoid function
double sigmoidScalar(double x)
{
    return 1.0 / (1 + exp(-x));
}

//scalar log function
double logScalar(double x)
{
    return log(x);
}

//scalar exp function
double expScalar(double x)
{
    return exp(x);
}

//scalar sqrt function
double sqrtScalar(double x)
{
    return sqrt(x);
}

