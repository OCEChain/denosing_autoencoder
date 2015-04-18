#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
double reciprocalScalar(double x);
double sigmoidScalar(double x);
double logScalar(double x);
double expScalar(double x);
double sqrtScalar(double x);
MatrixXd bsxfunMinus(MatrixXd &m,MatrixXd &x);
MatrixXd bsxfunRDivide(MatrixXd &m,MatrixXd &x);
MatrixXd bsxfunPlus(MatrixXd &m,MatrixXd &x);
MatrixXd sigmoid(MatrixXd &z);
MatrixXd sigmoidGradient(MatrixXd &z);
MatrixXd binaryCols(MatrixXi &labels,int numOfClasses);
MatrixXd expMat(MatrixXd &z);
MatrixXd logMat(MatrixXd &z);
MatrixXd sqrtMat(MatrixXd &z);
MatrixXd powMat(MatrixXd &z,int power);
MatrixXd reciprocal(MatrixXd &z);
double calcAccurancy(MatrixXi &pred,MatrixXi &labels);

#endif // UTILS_H
