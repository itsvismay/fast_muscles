#include "ocv_kmeans_wrapper.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
void ocv_kmeans(const Eigen::MatrixXd& F, const int num_labels, const int num_iter, Eigen::MatrixXd& D, Eigen::VectorXi& labels){ 
	// const Eigen::MatrixXd& F, //data. Every column is a feature
	// const int num_labels, // number of clusters
	// const int num_iter, // number of iterations
	// Eigen::MatrixXd& D, // dictionary of clusters (every column is a cluster)
	// Eigen::VectorXi& labels){ // map D to F.

	assert(sizeof(float) == 4);
	cv::Mat cv_F(F.rows(), F.cols(), CV_32F);
	for (int i = 0; i < F.rows(); ++i){
	    for (int j = 0; j < F.cols(); ++j){
	        cv_F.at<float>(i,j) = F(i,j);
	    }
	}

	cv::Mat cv_labels;
	cv::Mat cv_centers;
	cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.01);
	cv::kmeans(cv_F, num_labels, cv_labels, criteria, num_iter, cv::KMEANS_PP_CENTERS, cv_centers);

	int num_points = F.rows();
	int num_features = F.cols();
	// D.resize(num_features, num_labels);

	// for (int i=0; i<cv_centers.rows; ++i)
	//     for (int j=0; j<cv_centers.cols; ++j)
	//         D(j,i) = cv_centers.at<float>(i,j);

	labels.resize(num_points);
	for (int i=0; i<labels.rows(); ++i){
	    labels(i) = cv_labels.at<int>(i,0);
	}
}