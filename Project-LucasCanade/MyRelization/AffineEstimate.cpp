#include "AffineEstimate.h"

/*******************Class of ImageProcessor*******************/

ImageProcessor::ImageProcessor()
{
	// define the mask to compute gradient
	kx_ = (cv::Mat_<double>(3, 3) << 0, 0, 0, -0.5, 0, 0.5, 0, 0, 0);
	ky_ = (cv::Mat_<double>(3, 3) << 0, -0.5, 0, 0, 0, 0, 0, 0.5, 0);
}

bool ImageProcessor::setInput(const cv::Mat& image)
{
	if (image.empty()) return false;

	// change the BGR to Gray
	cv::Mat gray;

	// asset image is BGR or Gray
	if (image.type() == CV_8UC3)
	{
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else
	{
		gray = image;
	}

	// change the type of data
	gray.convertTo(image_, CV_64FC1);

	return true;
}

void ImageProcessor::getGradient(cv::Mat& gx, cv::Mat& gy) const
{
	if (image_.empty())
	{

		std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
		return;
	}

	// compute the gradient of image_
	cv::filter2D(image_, gx, -1, kx_);
	cv::filter2D(image_, gy, -1, ky_);

}

double ImageProcessor::getBilinearInterpolation(const cv::Mat& image, double x, double y)
{
	if (image.empty() || image.type() != CV_64FC1)
	{

		std::cerr << "Input image is empty, please use setInput() before call function getGradient()" << std::endl;
		return -1;
	}

	int row = (int)y;
	int col = (int)x;

	double rr = y - row;
	double cc = x - col;

	// Bilinear Interpolation and get the value of pixel
	return (1 - rr) * (1 - cc) * image.at<double>(row, col) +
		(1 - rr) * cc * image.at<double>(row, col + 1) +
		rr * (1 - cc) * image.at<double>(row + 1, col) +
		rr * cc * image.at<double>(row + 1, col + 1);
}

double ImageProcessor::getBilinearInterpolation(double x, double y) const
{
	return getBilinearInterpolation(image_, x, y);
}


/*******************Class of AffineEstimator*******************/

AffineEstimator::AffineEstimator() : max_iter_(80), debug_show_(true)
{
	image_processor_ = new ImageProcessor;
}

AffineEstimator::~AffineEstimator()
{
	if (image_processor_ != nullptr)
		delete image_processor_;
}

// compute the points of region which has been affined
std::array<cv::Point2d, 4> affinedRectangle(const AffineEstimator::AffineParameter& affine, const cv::Rect2d& rect)
{
	std::array<cv::Point2d, 4> result;

	result[0] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * rect.y + affine.p5,
		affine.p2 * rect.x + (1 + affine.p4) * rect.y + affine.p6);

	result[1] = cv::Point2d((1 + affine.p1) * rect.x + affine.p3 * (rect.y + rect.height) + affine.p5,
		affine.p2 * rect.x + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

	result[2] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * (rect.y + rect.height) + affine.p5,
		affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * (rect.y + rect.height) + affine.p6);

	result[3] = cv::Point2d((1 + affine.p1) * (rect.x + rect.width) + affine.p3 * rect.y + affine.p5,
		affine.p2 * (rect.x + rect.width) + (1 + affine.p4) * rect.y + affine.p6);

	return result;
}

cv::Mat AffineEstimator::debugShow()
{
	auto points = affinedRectangle(affine_, cv::Rect2d(0, 0, tx_.cols, tx_.rows));

	cv::Mat imshow = imshow_.clone();

	// the region has been affined
	cv::line(imshow, points[0], points[1], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[1], points[2], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[2], points[3], cv::Scalar(0, 0, 255));
	cv::line(imshow, points[3], points[0], cv::Scalar(0, 0, 255));

	// the orginal region
	cv::rectangle(imshow, cv::Rect(32, 52, 100, 100), cv::Scalar(0, 255, 0));

	return imshow;
}

// the entry of main function
void AffineEstimator::compute(const cv::Mat& source_image, const cv::Mat& template_image,
	const AffineParameter& affine_init, const Method& method)
{

	// copy(deep) the affine matrix to another memory
	memcpy(affine_.data, affine_init.data, sizeof(double) * 6);

	// change the bgr to gray
	image_processor_->setInput(source_image);
	template_image.convertTo(tx_, CV_64FC1);

	if (debug_show_)
		cv::cvtColor(source_image, imshow_, cv::COLOR_GRAY2BGR);

	switch (method)
	{
	case Method::kForwardAdditive:
		computeFA();
		break;

	case Method::kForwardCompositional:
		computeFC();
		break;

	case Method::kBackwardAdditive:
		computeBA();
		break;

	case Method::kBackwardCompositional:
		computeBC();
		break;

	default:
		std::cerr << "Invalid method type, please check." << std::endl;
		break;
	}
}

void AffineEstimator::computeFA()
{
	// affine matrix
	// the p is the reference of affine_.data
	// so that the cv::Mat maybe a pointer to the data's first address
	cv::Mat p = cv::Mat(6, 1, CV_64FC1, affine_.data);

	// compute the gradient of source image
	cv::Mat gx, gy;
	image_processor_->getGradient(gx, gy);

	if (debug_show_)
	{
		cv::Mat initImage = debugShow();
		cv::namedWindow("initImage", cv::WINDOW_NORMAL);
		cv::imshow("initImage", initImage);
	}

	int i = 0;
	for (; i < max_iter_; ++i)
	{

		std::string flex = ".png";
		if (debug_show_)
		{
			cv::Mat imshow = debugShow();
			cv::namedWindow("imshow", cv::WINDOW_NORMAL);
			cv::imshow("imshow", imshow);
			std::string save_path = std::to_string(i) + flex;
			cv::imwrite(save_path, imshow);
		}

		cv::Mat hessian = cv::Mat::zeros(6, 6, CV_64FC1);
		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);

		double cost = 0.;
		// Traverse the pixels in template image
		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{
				// new coordinate of every pixel of template
				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
					continue;

				double i_warped = image_processor_->getBilinearInterpolation(wx, wy);

				// the err of value of pixel
				double err = tx_.at<double>(y, x) - i_warped;

				double gx_warped = image_processor_->getBilinearInterpolation(gx, wx, wy);
				double gy_warped = image_processor_->getBilinearInterpolation(gy, wx, wy);

				cv::Mat jacobian = (cv::Mat_<double>(1, 6) << x * gx_warped, x * gy_warped,
					y * gx_warped, y * gy_warped, gx_warped, gy_warped);

				cv::Mat jacobian_transpose;
				cv::transpose(jacobian, jacobian_transpose);
				hessian += jacobian_transpose * jacobian;
				residual += jacobian_transpose * err;

				cost += err * err;
			}
		}

		cv::Mat hessian_inverse;
		cv::invert(hessian, hessian_inverse, cv::DECOMP_CHOLESKY);
		cv::Mat delta_p = hessian_inverse * residual;

		// this code influences the affine_
		p += delta_p;

		std::cout << "Iteration " << i << " cost = " << cost
			<< " squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;

		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
			break;

		std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
			<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
			<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
	}

	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}

void AffineEstimator::computeFC()
{
	// affine matrix
	cv::Mat p = cv::Mat(6, 1, CV_64FC1, affine_.data);

	if (debug_show_)
	{
		cv::Mat initImage = debugShow();
		cv::namedWindow("initImage", cv::WINDOW_NORMAL);
		cv::imshow("initImage", initImage);
	}

	int i = 0;
	for (; i < max_iter_; ++i)
	{
		std::string flex = ".png";
		if (debug_show_)
		{
			cv::Mat imshow = debugShow();
			cv::namedWindow("imshow", cv::WINDOW_NORMAL);
			cv::imshow("imshow", imshow);
			std::string save_path = std::to_string(i) + flex;
			cv::imwrite(save_path, imshow);
		}

		std::cout << "******************************" << std::endl;
		cv::Mat hessian = cv::Mat::zeros(6, 6, CV_64FC1);
		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);

		// compute the value of pixel whcih has affined and store tempeorarily
		double cost = 0.;
		cv::Mat warped_i(tx_.size(), CV_64FC1);
		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{

				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
				{
					warped_i.at<double>(y, x) = 0;
					continue;
				}

				// remap the value of pixel in affied in image to original image
				warped_i.at<double>(y, x) = image_processor_->getBilinearInterpolation(wx, wy);
			}
		}

		ImageProcessor temp_proc;
		temp_proc.setInput(warped_i);

		// compute the gradient of pixels which affined to original image
		cv::Mat warped_gx, warped_gy;
		temp_proc.getGradient(warped_gx, warped_gy);

		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{
				double err = tx_.at<double>(y, x) - warped_i.at<double>(y, x);

				double gx_warped = warped_gx.at<double>(y, x);
				double gy_warped = warped_gy.at<double>(y, x);

				/*cv::Mat jacobian = (cv::Mat_<double>(1, 6) << x * gx_warped, x * gy_warped,
					y * gx_warped, y * gy_warped, gx_warped, gy_warped);*/

				// revised the expression of jacobian
				// 这里是不是有问题，gx_warped 和 gy_warped 应该使用原图像 I 的梯度
				/*cv::Mat jacobian_left_temp = (cv::Mat_<double>(1, 2)
					<< gx_warped * (1 + affine_.p1) + gy_warped * (affine_.p2),
					gx_warped * (affine_.p3) + gy_warped * (1 + affine_.p4));*/
				cv::Mat jacobian_left_temp = (cv::Mat_<double>(1,2)
					<< gx_warped * (1 + affine_.p1) + gy_warped * (affine_.p2),
					gx_warped * (affine_.p3) + gy_warped * (1 + affine_.p4));
				cv::Mat jacobian_right_temp = (cv::Mat_<double>(2, 6)
					<< x, 0, y, 0, 1, 0, 0, x, 0, y, 0, 1);
				cv::Mat jacobian = jacobian_left_temp * jacobian_right_temp;

				cv::Mat jacobian_transpose;
				cv::transpose(jacobian, jacobian_transpose);
				hessian += jacobian_transpose * jacobian;
				residual += jacobian_transpose * err;

				cost += err * err;
			}
		}

		cv::Mat hessian_inverse = cv::Mat::zeros(6, 1, CV_64FC1);
		cv::invert(hessian, hessian_inverse, cv::DECOMP_CHOLESKY);
		cv::Mat delta_p = hessian_inverse * residual;

		double inc[6] = { 0. };
		memcpy(inc, delta_p.data, sizeof(double) * 6);
		inc[0] += (affine_.p1 * delta_p.at<double>(0, 0) + affine_.p3 * delta_p.at<double>(1, 0));
		inc[1] += (affine_.p2 * delta_p.at<double>(0, 0) + affine_.p4 * delta_p.at<double>(1, 0));
		inc[2] += (affine_.p1 * delta_p.at<double>(2, 0) + affine_.p3 * delta_p.at<double>(3, 0));
		inc[3] += (affine_.p2 * delta_p.at<double>(2, 0) + affine_.p4 * delta_p.at<double>(3, 0));
		inc[4] += (affine_.p1 * delta_p.at<double>(4, 0) + affine_.p3 * delta_p.at<double>(5, 0));
		inc[5] += (affine_.p2 * delta_p.at<double>(4, 0) + affine_.p4 * delta_p.at<double>(5, 0));

		cv::Mat increment = cv::Mat(6, 1, CV_64FC1, inc);
		p += increment;

		std::cout << "Iteration " << i << " cost = " << cost
			<< " squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;

		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
			break;

		std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
			<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
			<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
	}

	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}
 
void AffineEstimator::computeBA()
{
	// affine matrix
	cv::Mat p = cv::Mat(6, 1, CV_64FC1, affine_.data);

	if (debug_show_)
	{
		cv::Mat initImage = debugShow();
		cv::namedWindow("initImage", cv::WINDOW_NORMAL);
		cv::imshow("initImage", initImage);
	}

	ImageProcessor temp_proc;
	temp_proc.setInput(tx_);

	// Pre-compute
	cv::Mat gx, gy;
	temp_proc.getGradient(gx, gy);

	std::cout << "******************************" << std::endl;
	int i = 0;
	for (; i < max_iter_; ++i)
	{
		std::string flex = ".png";
		if (debug_show_)
		{
			cv::Mat imshow = debugShow();
			cv::namedWindow("imshow", cv::WINDOW_NORMAL);
			cv::imshow("imshow", imshow);
			std::string save_path = std::to_string(i) + flex;
			cv::imwrite(save_path, imshow);
		}

		double cost = 0.;
		cv::Mat hessian = cv::Mat::zeros(6, 6, CV_64FC1);
		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);
		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{

				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
					continue;

				double err = image_processor_->getBilinearInterpolation(wx, wy) - tx_.at<double>(y, x);

				cv::Mat x_ka = (cv::Mat_<double>(2, 2) << 1 + affine_.p1, affine_.p3, affine_.p2, 1 + affine_.p4);
				cv::Mat x_ka_inverse;
				cv::invert(x_ka, x_ka_inverse, cv::DECOMP_CHOLESKY);

				cv::Mat jacobian_left_temp = (cv::Mat_<double>(1, 2)
					<< gx.at<double>(y, x) * x_ka_inverse.at<double>(0, 0) + gy.at<double>(y, x) * x_ka_inverse.at<double>(1, 0),
					gx.at<double>(y, x) * x_ka_inverse.at<double>(0, 1) + gy.at<double>(y, x) * x_ka_inverse.at<double>(1, 1));

				cv::Mat jacobian_right_temp = (cv::Mat_<double>(2, 6)
					<< x, 0, y, 0, 1, 0, 0, x, 0, y, 0, 1);

				cv::Mat jacobian = jacobian_left_temp * jacobian_right_temp;

				cv::Mat jacobian_transpose;
				cv::transpose(jacobian, jacobian_transpose);
				hessian += jacobian_transpose * jacobian;
				residual += jacobian_transpose * err;
				cost += err * err;
			}
		}

		cv::Mat hessian_inverse = cv::Mat(6, 6, CV_64FC1);
		cv::invert(hessian, hessian_inverse, cv::DECOMP_CHOLESKY);
		cv::Mat delta_p = hessian_inverse * residual;

		p -= delta_p;

		std::cout << "Iteration " << i << " cost = " << cost <<
			" squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;

		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
			break;

		std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
			<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
			<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
	}

	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}

void AffineEstimator::computeBC()
{
	if (debug_show_)
	{
		cv::Mat initImage = debugShow();
		cv::namedWindow("initImage", cv::WINDOW_NORMAL);
		cv::imshow("initImage", initImage);
	}

	ImageProcessor temp_proc;
	temp_proc.setInput(tx_);

	// Pre-compute
	cv::Mat gx, gy;
	temp_proc.getGradient(gx, gy);

	cv::Mat xgx(gx.size(), gx.type());
	cv::Mat xgy(gx.size(), gx.type());
	cv::Mat ygx(gx.size(), gx.type());
	cv::Mat ygy(gx.size(), gx.type());

	cv::Mat hessian = cv::Mat::zeros(6, 6, CV_64FC1);

	for (int y = 0; y < tx_.rows; y++)
	{
		for (int x = 0; x < tx_.cols; x++)
		{
			xgx.at<double>(y, x) = x * gx.at<double>(y, x);
			xgy.at<double>(y, x) = x * gy.at<double>(y, x);
			ygx.at<double>(y, x) = y * gx.at<double>(y, x);
			ygy.at<double>(y, x) = y * gy.at<double>(y, x);


			cv::Mat jacobian = (cv::Mat_<double>(1, 6) << xgx.at<double>(y, x), xgy.at<double>(y, x),
				ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x));

			cv::Mat jacobian_transpose;
			cv::transpose(jacobian, jacobian_transpose);
			hessian += jacobian_transpose * jacobian;
		}
	}

	cv::Mat hessian_inverse = cv::Mat(6, 6, CV_64FC1);
	cv::invert(hessian, hessian_inverse, cv::DECOMP_CHOLESKY);

	int i = 0;
	for (; i < max_iter_; ++i)
	{
		std::string flex = ".png";
		if (debug_show_)
		{
			cv::Mat imshow = debugShow();
			cv::namedWindow("imshow", cv::WINDOW_NORMAL);
			cv::imshow("imshow", imshow);
			std::string save_path = std::to_string(i) + flex;
			cv::imwrite(save_path, imshow);
		}

		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);

		double cost = 0.;
		for (int y = 0; y < tx_.rows; y++)
		{
			for (int x = 0; x < tx_.cols; x++)
			{

				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;

				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
					continue;

				double err = image_processor_->getBilinearInterpolation(wx, wy) - tx_.at<double>(y, x);

				cv::Mat jacobian = (cv::Mat_<double>(1, 6) << xgx.at<double>(y, x), xgy.at<double>(y, x),
					ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x));

				cv::Mat jacobian_transpose;
				cv::transpose(jacobian, jacobian_transpose);
				residual += jacobian_transpose * err;
				cost += err * err;
			}
		}

		cv::Mat delta_p = hessian_inverse * residual;

		cv::Mat warp_m = (cv::Mat_<double>(3, 3) << 1 + affine_.p1, affine_.p3, affine_.p5,
			affine_.p2, 1 + affine_.p4, affine_.p6, 0, 0, 1);
		cv::Mat	delta_m = (cv::Mat_<double>(3, 3) << 1 + delta_p.at<double>(0, 0), delta_p.at<double>(2, 0),delta_p.at<double>(4, 0),
			delta_p.at<double>(1, 0), 1 + delta_p.at<double>(3, 0), delta_p.at<double>(5, 0), 0, 0, 1);

		cv::Mat delta_m_inverse = cv::Mat(3, 3, CV_64FC1);
		cv::invert(delta_m, delta_m_inverse, cv::DECOMP_CHOLESKY);

		cv::Mat new_warp = warp_m * delta_m_inverse;
		affine_.p1 = new_warp.at<double>(0, 0) - 1.;
		affine_.p2 = new_warp.at<double>(1, 0);
		affine_.p3 = new_warp.at<double>(0, 1);
		affine_.p4 = new_warp.at<double>(1, 1) - 1.;
		affine_.p5 = new_warp.at<double>(0, 2);
		affine_.p6 = new_warp.at<double>(1, 2);

		std::cout << "Iteration " << i << " cost = " << cost
			<< " squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;

		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
			break;

		std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
			<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
			<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
	}

	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
}

//void AffineEstimator::computeBA()
//{
//	// affine matrix
//	cv::Mat p = cv::Mat(6, 1, CV_64FC1, affine_.data);
//
//	if (debug_show_)
//	{
//		cv::Mat initImage = debugShow();
//		cv::namedWindow("initImage", cv::WINDOW_NORMAL);
//		cv::imshow("initImage", initImage);
//	}
//
//	ImageProcessor temp_proc;
//	temp_proc.setInput(tx_);
//
//	// Pre-compute
//	cv::Mat gx, gy;
//	temp_proc.getGradient(gx, gy);
//
//	cv::Mat xgx(gx.size(), gx.type());
//	cv::Mat xgy(gx.size(), gx.type());
//	cv::Mat ygx(gx.size(), gx.type());
//	cv::Mat ygy(gx.size(), gx.type());
//
//	std::cout << "******************************" << std::endl;
//	cv::Mat hessian_star = cv::Mat::zeros(6, 6, CV_64FC1);
//
//	for (int y = 0; y < tx_.rows; y++)
//	{
//		for (int x = 0; x < tx_.cols; x++)
//		{
//			xgx.at<double>(y, x) = x * gx.at<double>(y, x);
//			xgy.at<double>(y, x) = x * gy.at<double>(y, x);
//			ygx.at<double>(y, x) = y * gx.at<double>(y, x);
//			ygy.at<double>(y, x) = y * gy.at<double>(y, x);
//
//			cv::Mat jacobian = (cv::Mat_<double>(1, 6) << xgx.at<double>(y, x), xgy.at<double>(y, x),
//				ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x));
//
//			cv::Mat jacobian_transpose;
//			cv::transpose(jacobian, jacobian_transpose);
//			hessian_star += jacobian_transpose * jacobian;
//		}
//	}
//
//	cv::Mat hessian_star_inverse = cv::Mat(6, 6, CV_64FC1);
//	cv::invert(hessian_star, hessian_star_inverse, cv::DECOMP_CHOLESKY);
//
//	int i = 0;
//	for (; i < max_iter_; ++i)
//	{
//		std::string flex = ".png";
//		if (debug_show_)
//		{
//			cv::Mat imshow = debugShow();
//			cv::namedWindow("imshow", cv::WINDOW_NORMAL);
//			cv::imshow("imshow", imshow);
//			/*std::string save_path = std::to_string(i) + flex;
//			cv::imwrite(save_path, imshow);*/
//		}
//
//		cv::Mat residual = cv::Mat::zeros(6, 1, CV_64FC1);
//
//		double cost = 0.;
//		for (int y = 0; y < tx_.rows; y++)
//		{
//			for (int x = 0; x < tx_.cols; x++)
//			{
//
//				double wx = (double)x * (1. + affine_.p1) + (double)y * affine_.p3 + affine_.p5;
//				double wy = (double)x * affine_.p2 + (double)y * (1. + affine_.p4) + affine_.p6;
//
//				if (wx < 1 || wx > image_processor_->width() - 2 || wy < 1 || wy > image_processor_->height() - 2)
//					continue;
//
//				double err = image_processor_->getBilinearInterpolation(wx, wy) - tx_.at<double>(y, x);
//
//				cv::Mat jacobian = (cv::Mat_<double>(1, 6) << xgx.at<double>(y, x), xgy.at<double>(y, x),
//					ygx.at<double>(y, x), ygy.at<double>(y, x), gx.at<double>(y, x), gy.at<double>(y, x));
//
//				cv::Mat jacobian_transpose;
//				cv::transpose(jacobian, jacobian_transpose);
//				residual += jacobian_transpose * err;
//				cost += err * err;
//			}
//		}
//
//		cv::Mat sigma = cv::Mat::zeros(6, 6, CV_64FC1);
//
//		sigma.at<double>(0, 0) = 1 + affine_.p1;
//		sigma.at<double>(0, 1) = affine_.p3;
//		sigma.at<double>(1, 0) = affine_.p2;
//		sigma.at<double>(1, 1) = 1 + affine_.p4;
//
//		sigma.at<double>(2, 2) = 1 + affine_.p1;
//		sigma.at<double>(2, 3) = affine_.p3;
//		sigma.at<double>(3, 2) = affine_.p2;
//		sigma.at<double>(3, 3) = 1 + affine_.p4;
//
//		sigma.at<double>(4, 4) = 1 + affine_.p1;
//		sigma.at<double>(4, 5) = affine_.p3;
//		sigma.at<double>(5, 4) = affine_.p2;
//		sigma.at<double>(5, 5) = 1 + affine_.p4;
//
//		cv::Mat delta_p = sigma * hessian_star_inverse * residual;
//
//		p += delta_p;
//
//		std::cout << "Iteration " << i << " cost = " << cost <<
//			" squared delta p L2 norm = " << cv::norm(delta_p, cv::NORM_L2) << std::endl;
//
//		if (cv::norm(delta_p, cv::NORM_L2) < 1e-12)
//			break;
//
//		std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
//			<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
//			<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
//	}
//
//	std::cout << "After " << i + 1 << " iteration, the final estimate affine matrix is: \n"
//		<< affine_.p1 + 1 << " " << affine_.p3 << " " << affine_.p5 << " \n"
//		<< affine_.p2 << " " << affine_.p4 + 1 << " " << affine_.p6 << std::endl;
//}