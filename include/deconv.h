#pragma once

#include "fft.h"
#include <opencv2/imgproc.hpp>

// code from: https://stackoverflow.com/questions/40713929/weiner-deconvolution-using-opencv
void WienerFilter(cv::Mat const& src, cv::Mat const& kernel, cv::Mat& dst, double nsr = 0.01)
{
    const double eps = 1E-8;

    int ImgW = src.size().width;
    int ImgH = src.size().height;

    cv::Mat YfComplex, Yf[2];
    ForwardFFT(src, YfComplex);
    cv::split(YfComplex, Yf);

    cv::Mat _h;
    if (kernel.type() != CV_64FC1)
        kernel.convertTo(_h, CV_64FC1);
    else
        _h = kernel;
    cv::Mat h = cv::Mat::zeros(ImgH, ImgW, CV_64FC1);

    int padx = h.cols - _h.cols;
    int pady = h.rows - _h.rows;

    cv::copyMakeBorder(_h, h, pady / 2, pady - pady / 2, padx / 2, padx - padx / 2, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    cv::Mat HfComplex, Hf[2];
    ForwardFFT(h, HfComplex);
    cv::split(HfComplex, Hf);

    cv::Mat FuComplex, Fu[2];
    Fu[0] = cv::Mat::zeros(ImgH, ImgW, CV_64FC1);
    Fu[1] = cv::Mat::zeros(ImgH, ImgW, CV_64FC1);

    std::complex<double> a;
    std::complex<double> b;
    std::complex<double> c;

    double Hf_Re;
    double Hf_Im;
    double Phf;
    double hfz;
    double hz;
    double A;

    for (int i = 0; i < h.rows; i++)
        for (int j = 0; j < h.cols; j++)
        {
            Hf_Re = Hf[0].at<double>(i, j);
            Hf_Im = Hf[1].at<double>(i, j);
            Phf = Hf_Re * Hf_Re + Hf_Im * Hf_Im;
            hfz = (Phf < eps) * eps;
            hz = (h.at<double>(i, j) > 0);
            A = Phf / (Phf + hz + nsr);
            a = std::complex<double>(Yf[0].at<double>(i, j), Yf[1].at<double>(i, j));
            b = std::complex<double>(Hf_Re + hfz, Hf_Im + hfz);
            c = a / b; // Deconvolution :) other work to avoid division by zero
            Fu[0].at<double>(i, j) = (c.real() * A);
            Fu[1].at<double>(i, j) = (c.imag() * A);
        }
    cv::merge(Fu, 2, FuComplex);
    InverseFFT(FuComplex, dst);
    Rearrange(dst, dst);
}

// code from: https://github.com/chrrrisw/RL_deconv
void LucyRichardsonFilter(cv::Mat const& src, cv::Mat const& psf, cv::Mat& dst, int iterations)
{
    // Uniform grey starting estimation
    dst = cv::Mat(src.size(), src.type(), cv::Scalar(0.5));

    // Flip the point spread function (NOT the inverse)
    cv::Mat psf_hat = cv::Mat(psf.size(), CV_64FC1);
    //int psf_row_max = psf.rows - 1;
    //int psf_col_max = psf.cols - 1;
    //for (int row = 0; row <= psf_row_max; row++)
    //    for (int col = 0; col <= psf_col_max; col++)
    //        psf_hat.at<double>(psf_row_max - row, psf_col_max - col) = psf.at<double>(row, col);
    cv::flip(psf, psf_hat, -1);

    cv::Mat est_conv;
    cv::Mat relative_blur;
    cv::Mat error_est;

    // Iterate
    for (int i = 0; i < iterations; i++)
    {

        cv::filter2D(dst, est_conv, -1, psf);

        // Element-wise division
        relative_blur = src.mul(1.0 / est_conv);

        cv::filter2D(relative_blur, error_est, -1, psf_hat);

        // Element-wise multiplication
        dst = dst.mul(error_est);
    }
}
