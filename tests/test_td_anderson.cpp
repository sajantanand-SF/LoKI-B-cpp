/** \file
 *
 *  Test of AndersonMixer (time-dependent mode, periodic accelerator anderson) on linear maps g(x) = A x + b with a
 *  known fixed point x* = (I - A)^-1 b, as the one-period map near the periodic state:
 *
 *  1. Three slow modes (contraction 0.995, 0.99, 0.98) among fast ones (<= 0.3): plain iteration needs ~5500 maps to
 *     reach 1e-12; Anderson (depth 8) converges to 1e-12 in a few tens, and the fixed point is exact to 1e-10.
 *  2. The same with residual weights spanning 1e-4 to 1e4 (as the relative weights of a tail EEDF): same.
 *  3. depth >= 3 (the number of slow modes): fewer than 1/20 of the plain maps; depth 1 and 2 are printed only.
 */

#include "LoKI-B/TimeDependent.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <iostream>
#include <random>

namespace
{

struct LinearMap
{
    loki::Matrix A;
    loki::Vector b, xStar;
};

LinearMap slowModes(int n, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const loki::Matrix G = loki::Matrix::NullaryExpr(n, n, [&]() { return u(rng) - 0.5; });
    const loki::Matrix Q = G.householderQr().householderQ();
    loki::Vector lambda(n);
    for (int k = 0; k < n; ++k)
        lambda[k] = 0.3 * u(rng);
    lambda[0] = 0.995;
    lambda[1] = 0.99;
    lambda[2] = 0.98;
    LinearMap m;
    m.A = Q * lambda.asDiagonal() * Q.transpose();
    m.b = loki::Vector::NullaryExpr(n, [&]() { return u(rng); });
    m.xStar = (loki::Matrix::Identity(n, n) - m.A).partialPivLu().solve(m.b);
    return m;
}

// iterate with the mixer from x0 = 0 until max |w (g - x)| < tol; returns the number of maps
unsigned iterate(const LinearMap &m, unsigned depth, const loki::Vector &w, double tol, loki::Vector &x)
{
    loki::AndersonMixer mixer(depth);
    mixer.setWeights(w);
    x = loki::Vector::Zero(m.b.size());
    for (unsigned k = 1; k <= 2000; ++k)
    {
        const loki::Vector g = m.A * x + m.b;
        loki::Vector next;
        if (mixer.update(x, g, next) < tol)
        {
            x = g;
            return k;
        }
        x = next;
    }
    return 0;
}

} // namespace

int main()
{
    std::mt19937 rng(2026);
    const int n = 300;
    const LinearMap m = slowModes(n, rng);
    const double plain = std::log(1e-12) / std::log(0.995);
    loki::Vector x;
    {
        const unsigned k = iterate(m, 8, loki::Vector::Ones(n), 1e-12, x);
        const double err = (x - m.xStar).cwiseAbs().maxCoeff() / m.xStar.cwiseAbs().maxCoeff();
        std::clog << "depth 8: " << k << " maps (plain ~" << plain << "), error " << err << std::endl;
        test_expr(k > 0 && k <= 60);
        test_expr(err < 1e-10);
    }
    {
        std::uniform_real_distribution<double> e(-4.0, 4.0);
        const loki::Vector w = loki::Vector::NullaryExpr(n, [&]() { return std::pow(10.0, e(rng)); });
        const unsigned k = iterate(m, 8, w, 1e-12 * w.maxCoeff(), x);
        const double err = (x - m.xStar).cwiseAbs().maxCoeff() / m.xStar.cwiseAbs().maxCoeff();
        std::clog << "depth 8, weights 1e-4..1e4: " << k << " maps, error " << err << std::endl;
        test_expr(k > 0 && k <= 60);
        test_expr(err < 1e-10);
    }
    {
        // depth vs maps; depth 1 (a secant step) cannot capture three slow modes and is not required to converge
        for (unsigned depth : {1u, 2u, 3u, 5u, 8u, 12u})
        {
            const unsigned k = iterate(m, depth, loki::Vector::Ones(n), 1e-12, x);
            std::clog << "depth " << depth << ": " << (k ? std::to_string(k) : std::string("> 2000")) << " maps"
                      << std::endl;
            if (depth >= 3)
                test_expr(k > 0 && k < plain / 20);
        }
    }
    test_report;
    return nerrors;
}
