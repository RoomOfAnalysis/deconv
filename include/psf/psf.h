/* psf.h contains functions required for the calculation of point spread
 * function models.
 *
 * Copyright (C) 2005-2013 Francois Aguet
 *
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <complex>

namespace psf
{
    constexpr double PI = 3.14159265358979311599796346854;

    struct Parameters
    {
        double ti0;
        double ni0;
        double ni0_2; // ni0 ^ 2
        double ni;
        double ni_2; // ni ^ 2
        double tg0;
        double tg;
        double ng0;
        double ng0_2; // ng0 ^ 2
        double ng;
        double ng_2; // ng ^ 2
        double ns;
        double ns_2; // ns ^ 2
        double lambda;
        double k0;
        double dxy;
        double NA;
        double NA_2; // NA ^ 2
        double alpha;
        int sf;
        int mode;
    };

    Parameters GetParameters(double ti0 /*um*/, double ni0, double ni, double tg0 /*um*/, double tg /*um*/, double ng0,
                             double ng, double ns, double wvl /*um*/, double NA, double dxy /*um*/, int sf = 3,
                             int mode = 1);

    class ScalarPSF
    {
    public:
        ScalarPSF(double const xp[], double const z[], int const nz, int const nx, Parameters const& p);
        ~ScalarPSF();

        void calculatePSF();
        void calculatePSFdxp();

        double* pixels_;
        double* pixelsDxp_;
        double* pixelsDyp_;
        double* pixelsDzp_;

        // non-copyable and non-movable
        ScalarPSF(ScalarPSF const&) = delete;
        ScalarPSF(ScalarPSF&&) = delete;
        ScalarPSF& operator=(ScalarPSF const&) = delete;
        ScalarPSF& operator=(ScalarPSF&&) = delete;

    protected:
        double xp_;
        double yp_;
        double zp_;
        const double* z_;
        int nz_;
        int nx_;
        Parameters p_;

        int N_;
        double xystep_;

        double** integral_;
        double* R;

        int xymax_;
        int rmax_;
        int npx_;

        static const std::complex<double> i;
    };
    class VectorialPSF: public ScalarPSF
    {
    public:
        VectorialPSF(double const xp[], double const z[], int const nz, int const nx, Parameters const& p)
            : ScalarPSF(xp, z, nz, nx, p)
        {
        }
        ~VectorialPSF() = default;

        void calculatePSF();
        void calculatePSFdxp();

        // non-copyable and non-movable
        VectorialPSF(VectorialPSF const&) = delete;
        VectorialPSF(VectorialPSF&&) = delete;
        VectorialPSF& operator=(VectorialPSF const&) = delete;
        VectorialPSF& operator=(VectorialPSF&&) = delete;
    };
} // namespace psf
