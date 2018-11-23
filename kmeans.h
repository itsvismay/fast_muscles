// #pragma once
// #include <Eigen/Dense>

// // namespace tyro
// // {
// //    void kmeans(const Eigen::MatrixXd& F, //data. Every column is a feature
// //                const int num_labels, // number of clusters
// //                const int num_iter, // number of iterations
// //                Eigen::MatrixXd& D, // dictionary of clusters (every column is a cluster)
// //                Eigen::VectorXi& labels); // map D to F.
// // }

// // #include "kmeans.h"
// #include <opencv2/opencv.hpp>

// namespace tyro
// {

// void kmeans(const Eigen::MatrixXd& F, //data. Every column is a feature
//            const int num_labels, // number of clusters
//            const int num_iter, // number of iterations
//            Eigen::MatrixXd& D, // dictionary of clusters (every column is a cluster)
//            Eigen::VectorXi& labels) // map D to F.
// {
//    assert(sizeof(float) == 4);
//     cv::Mat cv_F(F.cols(), F.rows(), CV_32F);
//     for (int i = 0; i < F.cols(); ++i)
//         for (int j = 0; j < F.rows(); ++j)
//             cv_F.at<float>(i,j) = F(j,i);

//     cv::Mat cv_labels;
//     cv::Mat cv_centers;
//     cv::TermCriteria criteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
//     cv::kmeans(cv_F, num_labels, cv_labels, criteria, num_iter, cv::KMEANS_RANDOM_CENTERS, cv_centers);

//    int num_features = F.rows();
//    int num_points = F.cols();
//     D.resize(num_features, num_labels);

//    for (int i=0; i<cv_centers.rows; ++i)
//        for (int j=0; j<cv_centers.cols; ++j)
//            D(j,i) = cv_centers.at<float>(i,j);

//    labels.resize(num_points);
//    for (int i=0; i<labels.rows(); ++i)
//    {
//        labels(i) = cv_labels.at<int>(i,0);
//    }
// }

// }