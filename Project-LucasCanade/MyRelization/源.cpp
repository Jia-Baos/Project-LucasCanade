#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "AffineEstimate.h"

int main(int argc, char** argv)
{

    /*if (argc != 2)
    {

        cout << "Usage: ./execute path_to_image" << endl;
        return -1;
    }*/

    //cv::Mat image = cv::imread(argv[1], 0);
    std::string path = "D:/Code-VS/Lucas-Canade/Lucas-Canade/data/graffiti_crop.png";
    cv::Mat image = cv::imread(path, 0);

    // Estimate the optical flow
    cv::Mat template_image = image(cv::Rect(32, 52, 100, 100)).clone();

    AffineEstimator::AffineParameter affine_param;

    affine_param.p1 = 0.05;
    affine_param.p2 = 0.1;
    affine_param.p3 = 0.1;
    affine_param.p4 = 0.1;
    affine_param.p5 = 30;
    affine_param.p6 = 30;

   /* affine_param.p1 = 0.1;
    affine_param.p2 = 0;
    affine_param.p3 = 0;
    affine_param.p4 = 0;
    affine_param.p5 = 30;
    affine_param.p6 = 50;*/

    AffineEstimator estimator;
    estimator.compute(image, template_image, affine_param, Method::kForwardCompositional);

    cv::waitKey();
    return 0;
}

/*double array[6] = { 1,2,3,4,5,6 };
    double* pointer = array;

    cv::Mat test = cv::Mat(6, 1, CV_64FC1, pointer);

    std::cout << pointer << std::endl;
    std::cout << pointer + 1 << std::endl;
    std::cout << pointer + 2 << std::endl;


    for (int i = 0; i < test.rows; i++)
    {
        test.at<double>(i, 0) = i * i;
    }

    for (int i = 0; i < test.rows; i++)
    {
        std::cout << array[i] << std::endl;
    }*/