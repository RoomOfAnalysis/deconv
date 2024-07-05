#include "psf/psf.h"

#include <opencv2/highgui.hpp>

using namespace psf;

std::vector<double> linspace(double start, double end, double delta)
{
    if (start + delta > end) return {start};

    std::vector<double> ls;
    size_t num = static_cast<size_t>(floor((end - start) / delta));
    for (size_t i = 0; i < num - 1; ++i)
        ls.push_back(start + delta * i);
    ls.push_back(end);
    return ls;
}

void Show(double pz, int nx, double* xp, std::vector<double> z, Parameters const& p)
{
    // um -> m
    std::for_each(z.begin(), z.end(), [](double& z) { z *= 1e-6; });

    auto nz = static_cast<int>(z.size());

    ScalarPSF psf(xp, z.data(), nz, nx, p);
    //VectorialPSF psf(xp, z.data(), nz, nx, p);
    psf.calculatePSF();
    //psf.calculatePSFdxp();
    printf("Done.\n");

    // nx * ny * nz
    // i = (z * nx * ny) + (y * nx) + x
    double* pixels = new double[nx * nz];
    for (auto x = 0; x < nx; x++)
        for (auto y = 0; y < nz; y++)
            pixels[x * nz + y] = psf.pixels_[y * nx * nx + nx / 2 * nx + x]; // y = nx / 2
    cv::Mat I(nx, nz, CV_64FC1, pixels);
    I = I.t();
    cv::imshow("psf", I);

    // power norm
    // https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.PowerNorm.html
    cv::normalize(I, I, 1, 0, cv::NormTypes::NORM_MINMAX);
    cv::imshow("psf_n", I);

    cv::pow(I, 0.4, I);
    cv::imshow("psf_np", I);

    cv::waitKey(0);

    delete[] pixels;
}

int main(void)
{
    double ti0 = 150.0;
    double ni0 = 1.515;
    double ni = 1.515;
    double tg0 = 170.0;
    double tg = 170.0;
    double ng0 = 1.515;
    double ng = 1.515;
    double ns = 1.47;
    double wvl = 0.6;
    double NA = 1.42;
    double dxy = 0.02;

    auto p = GetParameters(ti0, ni0, ni, tg0, tg, ng0, ng, ns, wvl, NA, dxy);

    double pz = 0.0;
    int nx = 101;
    double xp[] = {0.0, 0.0, 0.0};
    auto z = linspace(-2 + pz, 2 + dxy + pz, dxy);
    Show(pz, nx, xp, z, p);

    pz = 2.0;
    p.ns = 1.38;
    p.ns_2 = p.ns * p.ns;
    z = linspace(-2 + pz, 2 + dxy + pz, dxy);
    Show(pz, nx, xp, z, p);

    return 0;
}