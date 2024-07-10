#include "deconv.h"
#include "deconv_blind.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

// linear motion blur distortion
// code from: https://docs.opencv.org/3.4/d1/dfd/tutorial_motion_deblur_filter.html
void genPSF(cv::Mat& outputImg, cv::Size filterSize, int len /*length of a motion*/,
            double theta /*angle of a motion in degrees*/)
{
    cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
    cv::Point point(filterSize.width / 2, filterSize.height / 2);
    cv::ellipse(h, point, cv::Size(0, cvRound(float(len) / 2.0)), 90.0 - theta, 0, 360, cv::Scalar(255), cv::FILLED);
    cv::Scalar summa = sum(h);
    outputImg = h / summa[0];
}
// circular point spread function
// code from: https://docs.opencv.org/3.4/de/d3c/tutorial_out_of_focus_deblur_filter.html
void genPSF(cv::Mat& outputImg, cv::Size filterSize, int R)
{
    cv::Mat h(filterSize, CV_32F, cv::Scalar(0));
    cv::Point point(filterSize.width / 2, filterSize.height / 2);
    cv::circle(h, point, R, 255, -1, 8);
    cv::Scalar summa = sum(h);
    outputImg = h / summa[0];
}

void wnrFilter(cv::Mat const& input_h_PSF, cv::Mat& output_G, double nsr)
{
    cv::Mat h_PSF_shifted;
    Rearrange(input_h_PSF, h_PSF_shifted);
    cv::Mat planes[2] = {cv::Mat_<float>(h_PSF_shifted.clone()), cv::Mat::zeros(h_PSF_shifted.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    cv::split(complexI, planes);
    cv::Mat denom;
    cv::pow(abs(planes[0]), 2, denom);
    denom += nsr;
    cv::divide(planes[0], denom, output_G);
}

void filter2DFreq(cv::Mat const& inputImg, cv::Mat& outputImg, cv::Mat const& H)
{
    cv::Mat planes[2] = {cv::Mat_<float>(inputImg.clone()), cv::Mat::zeros(inputImg.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI, cv::DFT_SCALE);
    cv::Mat planesH[2] = {cv::Mat_<float>(H.clone()), cv::Mat::zeros(H.size(), CV_32F)};
    cv::Mat complexH;
    cv::merge(planesH, 2, complexH);
    cv::Mat complexIH;
    cv::mulSpectrums(complexI, complexH, complexIH, 0);
    cv::idft(complexIH, complexIH);
    cv::split(complexIH, planes);
    outputImg = planes[0];
}

void edgetaper(cv::Mat const& inputImg, cv::Mat& outputImg, double gamma = 5.0, double beta = 0.2)
{
    int Nx = inputImg.cols;
    int Ny = inputImg.rows;
    cv::Mat w1(1, Nx, CV_32F, cv::Scalar(0));
    cv::Mat w2(Ny, 1, CV_32F, cv::Scalar(0));
    float* p1 = w1.ptr<float>(0);
    float* p2 = w2.ptr<float>(0);
    float dx = float(2.0 * CV_PI / Nx);
    float x = float(-CV_PI);
    for (int i = 0; i < Nx; i++)
    {
        p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
        x += dx;
    }
    float dy = float(2.0 * CV_PI / Ny);
    float y = float(-CV_PI);
    for (int i = 0; i < Ny; i++)
    {
        p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
        y += dy;
    }
    cv::Mat w = w2 * w1;
    cv::multiply(inputImg, w, outputImg);
}

int main(int argc, char* argv[])
{
    cv::Mat I = cv::imread(cv::samples::findFile(argv[1]), cv::IMREAD_GRAYSCALE);
    if (I.empty())
    {
        std::cout << "Error opening input image" << std::endl;
        return EXIT_FAILURE;
    }
    cv::imshow("Input Image", I);

    //cv::Mat PSF = cv::imread(cv::samples::findFile(argv[2]), cv::IMREAD_GRAYSCALE);
    //if (PSF.empty())
    //{
    //    std::cout << "Error opening PSF image" << std::endl;
    //    return EXIT_FAILURE;
    //}
    cv::Rect roi = cv::Rect(0, 0, I.cols & -2, I.rows & -2); // it needs to process even image only
    cv::Mat PSF;
    //genPSF(PSF, roi.size(), 125, 0);
    genPSF(PSF, roi.size(), 20);
    cv::imshow("PSF Image", PSF);

    I.convertTo(I, CV_32FC1, 1.0 / 255.0);
    PSF.convertTo(PSF, CV_32FC1, 1.0 / 255.0);
    PSF /= cv::sum(PSF)[0];

    cv::Mat conv;
    Convolution(I, PSF, conv);
    cv::imshow("Convolved Image", conv);

    PSF.convertTo(PSF, CV_64FC1);
    cv::Mat deconv;

    WienerFilter(conv, PSF, deconv);

    //cv::Mat Hw;
    //wnrFilter(PSF, Hw, 0.01);
    ////I.convertTo(I, CV_32F);
    ////edgetaper(I, I);
    ////cv::imshow("Edge Tapered Image", I);
    //filter2DFreq(I(roi), deconv, Hw);
    ////deconv.convertTo(deconv, CV_8U);
    ////cv::normalize(deconv, deconv, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Wiener Filter Deconvolved Image", deconv);

    LucyRichardsonFilter(conv, PSF, deconv, 100);
    cv::imshow("LR Filter Deconvolved Image", deconv);

    DECONV_BLIND::uk_t uk;
    DECONV_BLIND::params_t params;

    params.MK = 20 * 2; // row
    params.NK = 20 * 2; // col
    params.niters = 200;

    conv.convertTo(conv, CV_8UC1, 1. * 255.);
    cv::Mat conv_v[3] = {conv, conv, conv};
    cv::Mat conv_vv;
    cv::merge(conv_v, 3, conv_vv);
    std::cout << std::boolalpha << DECONV_BLIND::isGrayImage(conv_vv) << '\n';
    DECONV_BLIND::blind_deconv(conv_vv, 0.0006, params, uk, 1);
    cv::Mat tmpu;
    uk.u.convertTo(tmpu, CV_8U, 1. * 255.);
    cv::imshow("BlindDeconv Image", tmpu);

    cv::waitKey();

    return 0;
}