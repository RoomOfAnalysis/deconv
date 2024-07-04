#include "psf.h"

/* scalar computes the scalar PSF model described by Gibson and Lanni
 * [1]. For more information and implementation details, see [2].
 *
 * [1] F. Aguet et al., Opt. Express 17(8), pp. 6829-6848, 2009
 * [2] F. Aguet, Ph.D Thesis, Swiss Federal Institute of Technology, Lausanne
 * (EPFL), 2009
 *
 * Copyright (C) 2005-2013 Francois Aguet
 */

// Constants for polynomial Bessel function approximation from [Abramowitz (p. 369)]
constexpr double j0c[7] = {1, -2.2499997, 1.2656208, -0.3163866, 0.0444479, -0.0039444, 0.0002100};
constexpr double t0c[7] = {-.78539816, -.04166397, -.00003954, 0.00262573, -.00054125, -.00029333, .00013558};
constexpr double f0c[7] = {.79788456, -0.00000077, -.00552740, -.00009512, 0.00137237, -0.00072805, 0.00014476};
constexpr double j1c[7] = {0.5, -0.56249985, 0.21093573, -0.03954289, 0.00443319, -0.00031761, 0.00001109};
constexpr double f1c[7] = {0.79788456, 0.00000156, 0.01659667, 0.00017105, -0.00249511, 0.00113653, -0.00020033};
constexpr double t1c[7] = {-2.35619449, 0.12499612, 0.00005650, -0.00637897, 0.00074348, 0.00079824, -0.00029166};

// Bessel functions J0(x) and J1(x)
// Uses the polynomial approximations on p. 369-70 of Abramowitz & Stegun (1972).
// The error in J0 is supposed to be less than or equal to 5 x 10^-8.
double J0(double x)
{
    double r{};

    if (x < 0.0) x *= -1.0;

    if (x <= 3.0)
    {
        double y = x * x / 9.0;
        r = j0c[0] + y * (j0c[1] + y * (j0c[2] + y * (j0c[3] + y * (j0c[4] + y * (j0c[5] + y * j0c[6])))));
    }
    else
    {
        double y = 3.0 / x;
        double theta0 =
            x + t0c[0] + y * (t0c[1] + y * (t0c[2] + y * (t0c[3] + y * (t0c[4] + y * (t0c[5] + y * t0c[6])))));
        double f0 = f0c[0] + y * (f0c[1] + y * (f0c[2] + y * (f0c[3] + y * (f0c[4] + y * (f0c[5] + y * f0c[6])))));
        r = sqrt(1.0 / x) * f0 * cos(theta0);
    }
    return r;
}

double J1(double x)
{
    double r{};
    double sign = 1.0;
    if (x < 0.0)
    {
        x *= -1.0;
        sign *= -1.0;
    }
    if (x <= 3.0)
    {
        double y = x * x / 9.0;
        r = x * (j1c[0] + y * (j1c[1] + y * (j1c[2] + y * (j1c[3] + y * (j1c[4] + y * (j1c[5] + y * j1c[6]))))));
    }
    else
    {
        double y = 3.0 / x;
        double theta1 =
            x + t1c[0] + y * (t1c[1] + y * (t1c[2] + y * (t1c[3] + y * (t1c[4] + y * (t1c[5] + y * t1c[6])))));
        double f1 = f1c[0] + y * (f1c[1] + y * (f1c[2] + y * (f1c[3] + y * (f1c[4] + y * (f1c[5] + y * f1c[6])))));
        r = sqrt(1.0 / x) * f1 * cos(theta1);
    }
    return sign * r;
}

// Evaluates the optical path difference, with derivative d/d_theta in theta, the angle between 0 and alpha
void L_theta(std::complex<double>* L, double theta, psf::Parameters const& p, double ci, double z, double z_p)
{
    double ni2sin2theta = p.ni_2 * sin(theta) * sin(theta);
    std::complex<double> sroot = sqrt(std::complex<double>(p.ns_2 - ni2sin2theta));
    std::complex<double> groot = sqrt(std::complex<double>(p.ng_2 - ni2sin2theta));
    std::complex<double> g0root = sqrt(std::complex<double>(p.ng0_2 - ni2sin2theta));
    std::complex<double> i0root = sqrt(std::complex<double>(p.ni0_2 - ni2sin2theta));
    L[0] = p.ni * (ci - z) * cos(theta) + z_p * sroot + p.tg * groot - p.tg0 * g0root - p.ti0 * i0root;
    L[1] = p.ni * sin(theta) *
           (z - ci + p.ni * cos(theta) * (p.tg0 / g0root + p.ti0 / i0root - p.tg / groot - z_p / sroot));
}

// Evaluates the optical path difference, together with its partial derivative d/d_rho in rho
void L_rho(std::complex<double>* L, double rho, psf::Parameters const& p, double ci, double z, double z_p)
{
    double NA2rho2 = p.NA * p.NA * rho * rho;
    std::complex<double> iroot = sqrt(std::complex<double>(p.ni_2 - NA2rho2));
    std::complex<double> sroot = sqrt(std::complex<double>(p.ns_2 - NA2rho2));
    std::complex<double> groot = sqrt(std::complex<double>(p.ng_2 - NA2rho2));
    std::complex<double> g0root = sqrt(std::complex<double>(p.ng0_2 - NA2rho2));
    std::complex<double> i0root = sqrt(std::complex<double>(p.ni0_2 - NA2rho2));
    L[0] = (ci - z) * iroot + z_p * sroot + p.tg * groot - p.tg0 * g0root - p.ti0 * i0root;
    L[1] = 2.0 * p.NA * p.NA * rho * ((z - ci) / iroot - z_p / sroot - p.tg / groot + p.tg0 / g0root + p.ti0 / i0root);
}

psf::Parameters psf::GetParameters(double ti0 /*um*/, double ni0, double ni, double tg0 /*um*/, double tg /*um*/,
                                   double ng0, double ng, double ns, double wvl /*um*/, double NA, double dxy /*um*/,
                                   int sf, int mode)
{
    psf::Parameters p;

    p.ti0 = ti0 * 1e-6;
    p.ni0 = ni0;
    p.ni = ni;
    p.tg0 = tg0 * 1e-6;
    p.tg = tg * 1e-6;
    p.ng0 = ng0;
    p.ng = ng;
    p.ns = ns;
    p.lambda = wvl * 1e-6;
    p.NA = NA;
    p.dxy = dxy * 1e-6;

    p.k0 = 2 * psf::PI / p.lambda;
    p.ni0_2 = p.ni0 * p.ni0;
    p.ni_2 = p.ni * p.ni;
    p.ng0_2 = p.ng0 * p.ng0;
    p.ng_2 = p.ng * p.ng;
    p.ns_2 = p.ns * p.ns;
    p.alpha = asin(p.NA / p.ni);
    p.NA_2 = p.NA * p.NA;

    p.sf = sf;
    p.mode = mode;

    return p;
}

std::complex<double> const psf::ScalarPSF::i = std::complex<double>(0.0, 1.0);

psf::ScalarPSF::ScalarPSF(double const xp[], double const z[], int const nz, int const nx, Parameters const& p)
{
    xp_ = xp[0];
    yp_ = xp[1];
    zp_ = xp[2];

    z_ = z;
    nz_ = nz;
    nx_ = nx;
    p_ = p;

    xystep_ = p.dxy;

    xymax_ = ((nx_)*p.sf - 1) / 2; // always fine scale
    if (!p_.mode) nx_ *= p_.sf;    // oversampling factor

    N_ = nx_ * nx_ * nz_;

    // position in pixels
    xp_ *= p.sf / xystep_;
    yp_ *= p.sf / xystep_;

    int rn = 1 + (int)sqrt(xp_ * xp_ + yp_ * yp_);

    rmax_ = ceil(sqrt(2.0) * xymax_) + rn + 1; // +1 for interpolation, dx, dy
    npx_ = (2 * xymax_ + 1) * (2 * xymax_ + 1);

    pixels_ = new double[N_];
    pixelsDxp_ = new double[N_];
    pixelsDyp_ = new double[N_];
    pixelsDzp_ = new double[N_];

    integral_ = new double*[nz_];
    for (int k = 0; k < nz_; ++k)
        integral_[k] = new double[rmax_];
    // initialize since loops add to these arrays
    memset(pixels_, 0, sizeof(double) * N_);
    memset(pixelsDxp_, 0, sizeof(double) * N_);
    memset(pixelsDyp_, 0, sizeof(double) * N_);
    memset(pixelsDzp_, 0, sizeof(double) * N_);

    // pre-calculate radial coordinates
    R = new double[npx_];
    int idx = 0;
    double xi, yi;
    for (int y = -xymax_; y <= xymax_; ++y)
    {
        for (int x = -xymax_; x <= xymax_; ++x)
        {
            xi = (double)x - xp_;
            yi = (double)y - yp_;
            R[idx] = sqrt(xi * xi + yi * yi);
            ++idx;
        }
    }
}

psf::ScalarPSF::~ScalarPSF()
{
    delete[] R;
    for (int k = 0; k < nz_; ++k)
        delete[] integral_[k];
    delete[] integral_;
    delete[] pixelsDzp_;
    delete[] pixelsDyp_;
    delete[] pixelsDxp_;
    delete[] pixels_;
}

void psf::ScalarPSF::calculatePSF()
{

    double r;
    int n;

    std::complex<double> sum_I0, expW;

    // constant component of OPD
    double ci = zp_ * (1.0 - p_.ni / p_.ns) + p_.ni * (p_.tg0 / p_.ng0 + p_.ti0 / p_.ni0 - p_.tg / p_.ng);

    double theta, sintheta, costheta, ni2sin2theta;
    double bessel_0;

    double A0 = p_.ni_2 * p_.ni_2 / (p_.NA_2 * p_.NA_2);

    // Integration parameters
    double constJ;
    int nSamples;
    double step;

    double w_exp, cst, iconst;
    double ud = 3.0 * p_.sf;

    std::complex<double> L_th[2];
    for (int k = 0; k < nz_; ++k)
    {

        L_theta(L_th, p_.alpha, p_, ci, z_[k], zp_);
        w_exp = abs(L_th[1]); // missing p.k0, multiply below

        cst = 0.975;
        while (cst >= 0.9)
        {
            L_theta(L_th, cst * p_.alpha, p_, ci, z_[k], zp_);
            if (abs(L_th[1]) > w_exp) w_exp = abs(L_th[1]);
            cst -= 0.025;
        }
        w_exp *= p_.k0;

        for (int ri = 0; ri < rmax_; ++ri)
        {
            r = xystep_ / p_.sf * (double)(ri);
            constJ = p_.k0 * r * p_.ni; // samples required for bessel term

            if (w_exp > constJ)
                nSamples = 4 * (int)(1.0 + p_.alpha * w_exp / PI);
            else
                nSamples = 4 * (int)(1.0 + p_.alpha * constJ / PI);
            if (nSamples < 20) nSamples = 20;
            step = p_.alpha / (double)nSamples;
            iconst = step / ud;
            iconst *= iconst;

            // Simpson's rule
            sum_I0 = 0.0;
            for (n = 1; n < nSamples / 2; n++)
            {
                theta = 2.0 * n * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                bessel_0 = 2.0 * J0(constJ * sintheta) * sintheta * costheta; // 2.0 factor : Simpson's rule
                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta)) +
                            p_.tg * sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta)) -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                sum_I0 += expW * bessel_0;
            }
            for (n = 1; n <= nSamples / 2; n++)
            {
                theta = (2.0 * n - 1.0) * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                bessel_0 = 4.0 * J0(constJ * sintheta) * sintheta * costheta;
                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta)) +
                            p_.tg * sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta)) -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                sum_I0 += expW * bessel_0;
            }
            // theta = alpha;
            bessel_0 = J0(p_.k0 * r * p_.NA) * cos(p_.alpha) * sin(p_.alpha);
            expW = exp(i * p_.k0 *
                       ((ci - z_[k]) * sqrt(std::complex<double>(p_.ni_2 - p_.NA_2)) +
                        zp_ * sqrt(std::complex<double>(p_.ns_2 - p_.NA_2)) +
                        p_.tg * sqrt(std::complex<double>(p_.ng_2 - p_.NA_2)) -
                        p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - p_.NA_2)) -
                        p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - p_.NA_2))));
            sum_I0 += expW * bessel_0;

            integral_[k][ri] = A0 * abs(sum_I0) * abs(sum_I0) * iconst;
        }
    } // z loop

    int k;
    double dr;

    // Interpolate (linear)
    int r0;
    int index = 0;
    if (p_.mode == 1)
    { // average if sf>1
        div_t divRes;
        for (k = 0; k < nz_; ++k)
        {
            for (int i = 0; i < npx_; ++i)
            {
                r0 = (int)R[i];
                if (r0 + 1 < rmax_)
                {
                    dr = R[i] - r0;
                    divRes = div(i, 2 * xymax_ + 1);
                    index = divRes.rem / p_.sf + (divRes.quot / p_.sf) * nx_; // integer operations!
                    pixels_[index + k * nx_ * nx_] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                } // else '0'
            }
        }
    }
    else
    { // oversample if sf>1
        for (k = 0; k < nz_; ++k)
        {
            for (int i = 0; i < npx_; ++i)
            {
                r0 = (int)R[i];
                if (r0 + 1 < rmax_)
                {
                    dr = R[i] - r0;
                    pixels_[i + k * npx_] = dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                } // else '0'
            }
        }
    }
}

void psf::ScalarPSF::calculatePSFdxp()
{

    double r;
    int n;

    double constJ;
    int nSamples;
    double step;

    double theta, sintheta, costheta, ni2sin2theta;
    std::complex<double> bessel_0, bessel_1, expW, dW, nsroot;
    std::complex<double> sum_I0, sum_dxI0, sum_dzI0;
    std::complex<double> tmp;

    // allocate dynamic structures
    double** integralD;
    double** integralDz;
    integralD = new double*[nz_];
    integralDz = new double*[nz_];
    for (int k = 0; k < nz_; k++)
    {
        integralD[k] = new double[rmax_];
        integralDz[k] = new double[rmax_];
    }

    // constant component of OPD
    double ci = zp_ * (1.0 - p_.ni / p_.ns) + p_.ni * (p_.tg0 / p_.ng0 + p_.ti0 / p_.ni0 - p_.tg / p_.ng);

    double A0 = p_.ni_2 * p_.ni_2 / (p_.NA_2 * p_.NA_2);

    int ri;
    double ud = 3.0 * p_.sf;

    double w_exp, cst, iconst;

    std::complex<double> L_th[2];

    for (int k = 0; k < nz_; ++k)
    {

        L_theta(L_th, p_.alpha, p_, ci, z_[k], zp_);
        w_exp = abs(L_th[1]);

        cst = 0.975;
        while (cst >= 0.9)
        {
            L_theta(L_th, cst * p_.alpha, p_, ci, z_[k], zp_);
            if (abs(L_th[1]) > w_exp) w_exp = abs(L_th[1]);
            cst -= 0.025;
        }
        w_exp *= p_.k0;

        for (ri = 0; ri < rmax_; ++ri)
        {

            r = xystep_ / p_.sf * (double)(ri);
            constJ = p_.k0 * r * p_.ni;
            if (w_exp > constJ)
                nSamples = 4 * (int)(1.0 + p_.alpha * w_exp / PI);
            else
                nSamples = 4 * (int)(1.0 + p_.alpha * constJ / PI);
            if (nSamples < 20) nSamples = 20;
            step = p_.alpha / (double)nSamples;
            iconst = step / ud;
            iconst *= iconst;

            // Simpson's rule
            sum_I0 = 0.0;
            sum_dxI0 = 0.0;
            sum_dzI0 = 0.0;

            for (n = 1; n < nSamples / 2; n++)
            {
                theta = 2.0 * n * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));

                bessel_0 = 2.0 * J0(constJ * sintheta) * sintheta * costheta; // 2.0 factor : Simpson's rule
                bessel_1 = 2.0 * J1(constJ * sintheta) * sintheta * costheta;

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot +
                            p_.tg * sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta)) -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

                tmp = expW * bessel_0;
                sum_I0 += tmp;
                tmp *= dW;
                sum_dzI0 += tmp;
                sum_dxI0 += expW * bessel_1 * sintheta;
            }
            for (n = 1; n <= nSamples / 2; n++)
            {
                theta = (2.0 * n - 1.0) * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));

                bessel_0 = 4.0 * J0(constJ * sintheta) * sintheta * costheta; // 4.0 factor : Simpson's rule
                bessel_1 = 4.0 * J1(constJ * sintheta) * sintheta * costheta;

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot +
                            p_.tg * sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta)) -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

                tmp = expW * bessel_0;
                sum_I0 += tmp;
                tmp *= dW;
                sum_dzI0 += tmp;
                sum_dxI0 += expW * bessel_1 * sintheta;
            }
            // theta = alpha;
            sintheta = sin(p_.alpha);
            costheta = cos(p_.alpha);
            nsroot = sqrt(std::complex<double>(p_.ns_2 - p_.NA_2));

            bessel_0 = J0(constJ * sintheta) * sintheta * costheta;
            bessel_1 = J1(constJ * sintheta) * sintheta * costheta;

            expW = exp(i * p_.k0 *
                       ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot +
                        p_.tg * sqrt(std::complex<double>(p_.ng_2 - p_.NA_2)) -
                        p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - p_.NA_2)) -
                        p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - p_.NA_2))));
            dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

            tmp = expW * bessel_0;
            sum_I0 += tmp;
            tmp *= dW;
            sum_dzI0 += tmp;
            sum_dxI0 += expW * bessel_1 * sintheta;

            integral_[k][ri] = A0 * abs(sum_I0) * abs(sum_I0) * iconst;
            integralD[k][ri] =
                p_.k0 * p_.ni * A0 / r * 2.0 * real(conj(sum_I0) * sum_dxI0) * iconst; // multiply with (x-xp)
            integralDz[k][ri] = p_.k0 * A0 * 2.0 * real(conj(sum_I0) * sum_dzI0) * iconst;
        }
        integralD[k][0] = 0.0; // overwrite because of singularity
    }                          // z loop

    // Interpolate (linear)
    int r0;
    double dr, rx;
    double xi, yi, tmp2;
    int index = 0;
    int k, x, y;
    if (p_.mode == 1)
    {
        for (k = 0; k < nz_; k++)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {

                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;

                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        index = (x + xymax_) / p_.sf + ((y + xymax_) / p_.sf) * nx_ + k * nx_ * nx_;

                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                        pixelsDzp_[index] += dr * integralDz[k][r0 + 1] + (1.0 - dr) * integralDz[k][r0];

                        xi *= xystep_ / p_.sf;
                        yi *= xystep_ / p_.sf;

                        tmp2 = dr * integralD[k][r0 + 1] + (1.0 - dr) * integralD[k][r0];
                        pixelsDxp_[index] += xi * tmp2;
                        pixelsDyp_[index] += yi * tmp2;
                    } // else '0'
                }
            }
        }
    }
    else
    {
        for (k = 0; k < nz_; k++)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {

                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;

                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                        pixelsDzp_[index] += dr * integralDz[k][r0 + 1] + (1.0 - dr) * integralDz[k][r0];

                        xi *= xystep_ / p_.sf;
                        yi *= xystep_ / p_.sf;

                        tmp2 = dr * integralD[k][r0 + 1] + (1.0 - dr) * integralD[k][r0];
                        pixelsDxp_[index] += xi * tmp2;
                        pixelsDyp_[index] += yi * tmp2;
                    } // else '0'
                    index++;
                }
            }
        }
    }

    delete[] integralDz;
    delete[] integralD;
}

// Intensity PSF for an isotropically emitting point source (average of all
// dipole orientations)
void psf::VectorialPSF::calculatePSF()
{

    double r;
    int n;

    // Integration parameters
    double constJ;
    int nSamples;
    double step;

    double theta, sintheta, costheta, sqrtcostheta, ni2sin2theta;
    std::complex<double> bessel_0, bessel_1, bessel_2, expW;
    std::complex<double> ngroot, nsroot;
    std::complex<double> ts1ts2, tp1tp2;
    std::complex<double> sum_I0, sum_I1, sum_I2;

    // constant component of OPD
    double ci = zp_ * (1.0 - p_.ni / p_.ns) + p_.ni * (p_.tg0 / p_.ng0 + p_.ti0 / p_.ni0 - p_.tg / p_.ng);

    int x, y, index, ri;
    double iconst;
    double ud = 3.0 * p_.sf;

    double w_exp;

    std::complex<double> L_th[2];
    double cst;

    for (int k = 0; k < nz_; k++)
    {

        L_theta(L_th, p_.alpha, p_, ci, z_[k], zp_);
        w_exp = abs(L_th[1]); // missing p.k0, multiply below

        cst = 0.975;
        while (cst >= 0.9)
        {
            L_theta(L_th, cst * p_.alpha, p_, ci, z_[k], zp_);
            if (abs(L_th[1]) > w_exp) w_exp = abs(L_th[1]);
            cst -= 0.025;
        }
        w_exp *= p_.k0;

        for (ri = 0; ri < rmax_; ++ri)
        {

            r = xystep_ / p_.sf * (double)(ri);
            constJ = p_.k0 * r * p_.ni; // = w_J;

            if (w_exp > constJ)
                nSamples = 4 * (int)(1.0 + p_.alpha * w_exp / PI);
            else
                nSamples = 4 * (int)(1.0 + p_.alpha * constJ / PI);
            if (nSamples < 20) nSamples = 20;

            step = p_.alpha / (double)nSamples;
            iconst = step / ud;

            // Simpson's rule
            sum_I0 = 0.0;
            sum_I1 = 0.0;
            sum_I2 = 0.0;

            for (n = 1; n < nSamples / 2; n++)
            {
                theta = 2.0 * n * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                sqrtcostheta = sqrt(costheta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));
                ngroot = sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta));

                ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
                tp1tp2 = ts1ts2;
                tp1tp2 /=
                    (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
                ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

                bessel_0 = 2.0 * J0(constJ * sintheta) * sintheta * sqrtcostheta; // 2.0 factor : Simpson's rule
                bessel_1 = 2.0 * J1(constJ * sintheta) * sintheta * sqrtcostheta;
                if (constJ != 0.0)
                    bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
                else
                    bessel_2 = 0.0;
                bessel_0 *= (ts1ts2 + tp1tp2 / p_.ns * nsroot);
                bessel_1 *= (tp1tp2 * p_.ni / p_.ns * sintheta);
                bessel_2 *= (ts1ts2 - tp1tp2 / p_.ns * nsroot);

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot + p_.tg * ngroot -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                sum_I0 += expW * bessel_0;
                sum_I1 += expW * bessel_1;
                sum_I2 += expW * bessel_2;
            }
            for (n = 1; n <= nSamples / 2; n++)
            {
                theta = (2.0 * n - 1) * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                sqrtcostheta = sqrt(costheta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));
                ngroot = sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta));

                ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
                tp1tp2 = ts1ts2;
                tp1tp2 /=
                    (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
                ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

                bessel_0 = 4.0 * J0(constJ * sintheta) * sintheta * sqrtcostheta; // 4.0 factor : Simpson's rule
                bessel_1 = 4.0 * J1(constJ * sintheta) * sintheta * sqrtcostheta;
                if (constJ != 0.0)
                    bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
                else
                    bessel_2 = 0.0;
                bessel_0 *= (ts1ts2 + tp1tp2 / p_.ns * nsroot);
                bessel_1 *= (tp1tp2 * p_.ni / p_.ns * sintheta);
                bessel_2 *= (ts1ts2 - tp1tp2 / p_.ns * nsroot);

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot + p_.tg * ngroot -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                sum_I0 += expW * bessel_0;
                sum_I1 += expW * bessel_1;
                sum_I2 += expW * bessel_2;
            }
            // theta = alpha;
            sintheta = sin(p_.alpha);
            costheta = cos(p_.alpha);
            sqrtcostheta = sqrt(costheta);
            nsroot = sqrt(std::complex<double>(p_.ns_2 - p_.NA_2));
            ngroot = sqrt(std::complex<double>(p_.ng_2 - p_.NA_2));

            ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
            tp1tp2 = ts1ts2;
            tp1tp2 /= (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
            ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

            bessel_0 = J0(constJ * sintheta) * sintheta * sqrtcostheta;
            bessel_1 = J1(constJ * sintheta) * sintheta * sqrtcostheta;
            if (constJ != 0.0)
                bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
            else
                bessel_2 = 0.0;
            bessel_0 *= (ts1ts2 + tp1tp2 / p_.ns * nsroot);
            bessel_1 *= (tp1tp2 * p_.ni / p_.ns * sintheta);
            bessel_2 *= (ts1ts2 - tp1tp2 / p_.ns * nsroot);

            expW = exp(i * p_.k0 *
                       ((ci - z_[k]) * sqrt(std::complex<double>(p_.ni_2 - p_.NA_2)) + zp_ * nsroot + p_.tg * ngroot -
                        p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - p_.NA_2)) -
                        p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - p_.NA_2))));
            sum_I0 += expW * bessel_0;
            sum_I1 += expW * bessel_1;
            sum_I2 += expW * bessel_2;

            sum_I0 = abs(sum_I0);
            sum_I1 = abs(sum_I1);
            sum_I2 = abs(sum_I2);

            integral_[k][ri] =
                8.0 * PI / 3.0 * real(sum_I0 * sum_I0 + 2.0 * sum_I1 * sum_I1 + sum_I2 * sum_I2) * iconst * iconst;
        }
    } // z loop

    // Interpolate (linear)
    int r0;
    double dr, rx, xi, yi;
    index = 0;
    if (p_.mode == 1)
    {
        for (int k = 0; k < nz_; ++k)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {
                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;
                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        index = (x + xymax_) / p_.sf + ((y + xymax_) / p_.sf) * nx_ + k * nx_ * nx_;
                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                    } // else '0'
                }
            }
        }
    }
    else
    {
        for (int k = 0; k < nz_; ++k)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {
                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;
                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                    } // else '0'
                    index++;
                }
            }
        }
    }
} // psf

// Same PSF calculation as above, but including partial derivatives relative to
// source pos. xp
void psf::VectorialPSF::calculatePSFdxp()
{

    double r;
    int n;

    // Integration parameters
    double constJ;
    int nSamples;
    double step;

    double theta, sintheta, costheta, sqrtcostheta, ni2sin2theta;
    std::complex<double> bessel_0, bessel_1, bessel_2, bessel_3;
    std::complex<double> ngroot, nsroot;
    std::complex<double> ts1ts2, tp1tp2;
    std::complex<double> sum_I0, sum_I1, sum_I2, sum_dxI0, sum_dxI1, sum_dxI2, sum_dzI0, sum_dzI1, sum_dzI2;
    std::complex<double> t0, t1, t2;
    std::complex<double> expW, dW, tmp;

    double xystep = p_.dxy;

    // constant component of OPD
    double ci = zp_ * (1.0 - p_.ni / p_.ns) + p_.ni * (p_.tg0 / p_.ng0 + p_.ti0 / p_.ni0 - p_.tg / p_.ng);

    // allocate dynamic structures
    double** integralDx;
    double** integralDz;
    integralDx = new double*[nz_];
    integralDz = new double*[nz_];
    for (int k = 0; k < nz_; ++k)
    {
        integralDx[k] = new double[rmax_];
        integralDz[k] = new double[rmax_];
    }

    int x, y, index, ri;
    double iconst;
    double ud = 3.0 * p_.sf;

    double w_exp;

    std::complex<double> L_th[2];
    double cst;

    for (int k = 0; k < nz_; ++k)
    {

        L_theta(L_th, p_.alpha, p_, ci, z_[k], zp_);
        w_exp = abs(L_th[1]); // missing p.k0 !

        cst = 0.975;
        while (cst >= 0.9)
        {
            L_theta(L_th, cst * p_.alpha, p_, ci, z_[k], zp_);
            if (abs(L_th[1]) > w_exp) w_exp = abs(L_th[1]);
            cst -= 0.025;
        }
        w_exp *= p_.k0;

        for (ri = 0; ri < rmax_; ++ri)
        {

            r = xystep / p_.sf * (double)(ri);
            constJ = p_.k0 * r * p_.ni; // = w_J;

            if (w_exp > constJ)
                nSamples = 4 * (int)(1.0 + p_.alpha * w_exp / PI);
            else
                nSamples = 4 * (int)(1.0 + p_.alpha * constJ / PI);
            if (nSamples < 20) nSamples = 20;
            step = p_.alpha / (double)nSamples;
            iconst = step / ud;

            // Simpson's rule
            sum_I0 = 0.0;
            sum_I1 = 0.0;
            sum_I2 = 0.0;
            sum_dxI0 = 0.0;
            sum_dxI1 = 0.0;
            sum_dxI2 = 0.0;
            sum_dzI0 = 0.0;
            sum_dzI1 = 0.0;
            sum_dzI2 = 0.0;

            for (n = 1; n < nSamples / 2; n++)
            {
                theta = 2.0 * n * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                sqrtcostheta = sqrt(costheta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));
                ngroot = sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta));

                ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
                tp1tp2 = ts1ts2;
                tp1tp2 /=
                    (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
                ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

                bessel_0 = 2.0 * J0(constJ * sintheta) * sintheta * sqrtcostheta; // 2.0 factor : Simpson's rule
                bessel_1 = 2.0 * J1(constJ * sintheta) * sintheta * sqrtcostheta;
                if (constJ != 0.0)
                {
                    bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
                    bessel_3 = 4.0 * bessel_2 / (constJ * sintheta) - bessel_1;
                }
                else
                {
                    bessel_2 = 0.0;
                    bessel_3 = 0.0;
                }

                t0 = ts1ts2 + tp1tp2 / p_.ns * nsroot;
                t1 = tp1tp2 * p_.ni / p_.ns * sintheta;
                t2 = ts1ts2 - tp1tp2 / p_.ns * nsroot;

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot + p_.tg * ngroot -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

                tmp = expW * bessel_0 * t0;
                sum_I0 += tmp;
                sum_dzI0 += tmp * dW;
                tmp = expW * bessel_1 * t1;
                sum_I1 += tmp;
                sum_dzI1 += tmp * dW;
                tmp = expW * bessel_2 * t2;
                sum_I2 += tmp;
                sum_dzI2 += tmp * dW;

                sum_dxI0 += expW * bessel_1 * t0 * sintheta;
                sum_dxI1 += expW * (bessel_0 - bessel_2) * t1 * sintheta;
                sum_dxI2 += expW * (bessel_1 - bessel_3) * t2 * sintheta;
            }
            for (n = 1; n <= nSamples / 2; n++)
            {
                theta = (2.0 * n - 1) * step;
                sintheta = sin(theta);
                costheta = cos(theta);
                sqrtcostheta = sqrt(costheta);
                ni2sin2theta = p_.ni_2 * sintheta * sintheta;
                nsroot = sqrt(std::complex<double>(p_.ns_2 - ni2sin2theta));
                ngroot = sqrt(std::complex<double>(p_.ng_2 - ni2sin2theta));

                ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
                tp1tp2 = ts1ts2;
                tp1tp2 /=
                    (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
                ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

                bessel_0 = 4.0 * J0(constJ * sintheta) * sintheta * sqrtcostheta; // 4.0 factor : Simpson's rule
                bessel_1 = 4.0 * J1(constJ * sintheta) * sintheta * sqrtcostheta;
                if (constJ != 0.0)
                {
                    bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
                    bessel_3 = 4.0 * bessel_2 / (constJ * sintheta) - bessel_1;
                }
                else
                {
                    bessel_2 = 0.0;
                    bessel_3 = 0.0;
                }
                t0 = ts1ts2 + tp1tp2 / p_.ns * nsroot;
                t1 = tp1tp2 * p_.ni / p_.ns * sintheta;
                t2 = ts1ts2 - tp1tp2 / p_.ns * nsroot;

                expW = exp(i * p_.k0 *
                           ((ci - z_[k]) * p_.ni * costheta + zp_ * nsroot + p_.tg * ngroot -
                            p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - ni2sin2theta)) -
                            p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - ni2sin2theta))));
                dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

                tmp = expW * bessel_0 * t0;
                sum_I0 += tmp;
                sum_dzI0 += tmp * dW;
                tmp = expW * bessel_1 * t1;
                sum_I1 += tmp;
                sum_dzI1 += tmp * dW;
                tmp = expW * bessel_2 * t2;
                sum_I2 += tmp;
                sum_dzI2 += tmp * dW;

                sum_dxI0 += expW * bessel_1 * t0 * sintheta;
                sum_dxI1 += expW * (bessel_0 - bessel_2) * t1 * sintheta;
                sum_dxI2 += expW * (bessel_1 - bessel_3) * t2 * sintheta;
            }
            // theta = alpha;
            sintheta = sin(p_.alpha);
            costheta = cos(p_.alpha);
            sqrtcostheta = sqrt(costheta);
            nsroot = sqrt(std::complex<double>(p_.ns_2 - p_.NA_2));
            ngroot = sqrt(std::complex<double>(p_.ng_2 - p_.NA_2));

            ts1ts2 = 4.0 * p_.ni * costheta * ngroot;
            tp1tp2 = ts1ts2;
            tp1tp2 /= (p_.ng * costheta + p_.ni / p_.ng * ngroot) * (p_.ns / p_.ng * ngroot + p_.ng / p_.ns * nsroot);
            ts1ts2 /= (p_.ni * costheta + ngroot) * (ngroot + nsroot);

            bessel_0 = J0(constJ * sintheta) * sintheta * sqrtcostheta;
            bessel_1 = J1(constJ * sintheta) * sintheta * sqrtcostheta;
            if (constJ != 0.0)
            {
                bessel_2 = 2.0 * bessel_1 / (constJ * sintheta) - bessel_0;
                bessel_3 = 4.0 * bessel_2 / (constJ * sintheta) - bessel_1;
            }
            else
            {
                bessel_2 = 0.0;
                bessel_3 = 0.0;
            }
            t0 = ts1ts2 + tp1tp2 / p_.ns * nsroot;
            t1 = tp1tp2 * p_.ni / p_.ns * sintheta;
            t2 = ts1ts2 - tp1tp2 / p_.ns * nsroot;

            expW = exp(i * p_.k0 *
                       ((ci - z_[k]) * sqrt(std::complex<double>(p_.ni_2 - p_.NA_2)) + zp_ * nsroot + p_.tg * ngroot -
                        p_.tg0 * sqrt(std::complex<double>(p_.ng0_2 - p_.NA_2)) -
                        p_.ti0 * sqrt(std::complex<double>(p_.ni0_2 - p_.NA_2))));
            dW = i * ((1.0 - p_.ni / p_.ns) * p_.ni * costheta + nsroot);

            tmp = expW * bessel_0 * t0;
            sum_I0 += tmp;
            sum_dzI0 += tmp * dW;
            tmp = expW * bessel_1 * t1;
            sum_I1 += tmp;
            sum_dzI1 += tmp * dW;
            tmp = expW * bessel_2 * t2;
            sum_I2 += tmp;
            sum_dzI2 += tmp * dW;

            sum_dxI0 += expW * bessel_1 * t0 * sintheta;
            sum_dxI1 += expW * (bessel_0 - bessel_2) * t1 * sintheta;
            sum_dxI2 += expW * (bessel_1 - bessel_3) * t2 * sintheta;

            if (ri > 0)
            {
                integral_[k][ri] =
                    8.0 * PI / 3.0 *
                    (abs(sum_I0) * abs(sum_I0) + 2.0 * abs(sum_I1) * abs(sum_I1) + abs(sum_I2) * abs(sum_I2)) * iconst *
                    iconst;
                integralDx[k][ri] =
                    16.0 * PI / 3.0 * p_.k0 * p_.ni *
                    real(-sum_dxI0 * conj(sum_I0) + sum_dxI1 * conj(sum_I1) + sum_dxI2 * conj(sum_I2) / 2.0) / r *
                    iconst * iconst;
                integralDz[k][ri] =
                    16.0 * PI / 3.0 * p_.k0 *
                    real(conj(sum_dzI0) * sum_I0 + 2.0 * conj(sum_dzI1) * sum_I1 + conj(sum_dzI2) * sum_I2) * iconst *
                    iconst;
            }
            else
            {
                integral_[k][0] = 8.0 * PI / 3.0 * (abs(sum_I0) * abs(sum_I0)) * iconst * iconst;
                integralDx[k][0] = 0.0;
                integralDz[k][0] = 16.0 * PI / 3.0 * p_.k0 * real(sum_I0 * conj(sum_dzI0)) * iconst * iconst;
            }
        }
    } // z loop

    // Interpolate (linear)
    int r0;
    double dr, rx, xi, yi, xd, yd;
    index = 0;
    if (p_.mode == 1)
    {
        for (int k = 0; k < nz_; ++k)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {
                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    xd = xp_ - x * xystep / p_.sf;
                    yd = yp_ - y * xystep / p_.sf;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;
                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        index = (x + xymax_) / p_.sf + ((y + xymax_) / p_.sf) * nx_ + k * nx_ * nx_;
                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                        pixelsDxp_[index] += xd * (dr * integralDx[k][r0 + 1] + (1.0 - dr) * integralDx[k][r0]);
                        pixelsDyp_[index] += yd * (dr * integralDx[k][r0 + 1] + (1.0 - dr) * integralDx[k][r0]);
                        pixelsDzp_[index] += dr * integralDz[k][r0 + 1] + (1.0 - dr) * integralDz[k][r0];
                    } // else '0'
                }
            }
        }
    }
    else
    {
        for (int k = 0; k < nz_; ++k)
        {
            for (y = -xymax_; y <= xymax_; y++)
            {
                for (x = -xymax_; x <= xymax_; x++)
                {
                    xi = (double)x - xp_;
                    yi = (double)y - yp_;
                    xd = xp_ - x * xystep_ / p_.sf;
                    yd = yp_ - y * xystep_ / p_.sf;
                    rx = sqrt(xi * xi + yi * yi);
                    r0 = (int)rx;
                    if (r0 + 1 < rmax_)
                    {
                        dr = rx - r0;
                        pixels_[index] += dr * integral_[k][r0 + 1] + (1.0 - dr) * integral_[k][r0];
                        pixelsDxp_[index] += xd * (dr * integralDx[k][r0 + 1] + (1.0 - dr) * integralDx[k][r0]);
                        pixelsDyp_[index] += yd * (dr * integralDx[k][r0 + 1] + (1.0 - dr) * integralDx[k][r0]);
                        pixelsDzp_[index] += dr * integralDz[k][r0 + 1] + (1.0 - dr) * integralDz[k][r0];
                    } // else '0'
                    index++;
                }
            }
        }
    }
    // free dynamic structures
    for (int k = 0; k < nz_; ++k)
    {
        delete[] integralDx[k];
        delete[] integralDz[k];
    }
    delete[] integralDx;
    delete[] integralDz;
}