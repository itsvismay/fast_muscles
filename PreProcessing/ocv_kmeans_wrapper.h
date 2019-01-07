#ifndef OPENCV_KMEANS_WRAPPER
#define OPENCV_KMEANS_WRAPPER

#include <opencv2/opencv.hpp>
// #include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
void ocv_kmeans(const Eigen::MatrixXd& F, const int num_labels, const int num_iter, Eigen::MatrixXd& D, Eigen::VectorXi& labels);


#endif