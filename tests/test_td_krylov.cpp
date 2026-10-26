/** \file
 *
 *  Test of gmresHessenberg (time-dependent mode, linearSolver auto): GMRES preconditioned with the upper-Hessenberg
 *  part, for stage matrices with entries below the first sub-diagonal (superelastic collisions).
 *
 *  1. Upper-Hessenberg matrix plus two far sub-diagonals (offsets -50 and -120, entries ~1e-3 of the diagonal, as
 *     superelastic terms with small metastable populations): the solution equals dense LU to 1e-11.
 *  2. No entries below the first sub-diagonal: the preconditioner is exact, one iteration.
 *  3. Far sub-diagonals as large as the diagonal and 3 iterations allowed: not converged (returns 0, b unchanged),
 *     which makes the caller fall back to LU.
 */

#include "LoKI-B/TimeDependent.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <iostream>
#include <random>

namespace
{

loki::Matrix hessenbergPlusFar(int n, double farScale, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    loki::Matrix M = loki::Matrix::Zero(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= std::min(j + 1, n - 1); ++i)
            M(i, j) = (i == j) ? 0.0 : 0.5 * u(rng) / n;
    for (int k = 0; k < n; ++k)
        M(k, k) = 2.0 + u(rng);  // diagonally dominant, as W + a N (losses)
    for (int off : {50, 120})
        for (int k = off; k < n; ++k)
            M(k, k - off) = farScale * u(rng);
    return M;
}

} // namespace

int main()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 400;
    loki::LinAlg::HessenbergWorkspace hws(n);
    loki::Matrix V, Z;
    loki::Vector b0(n);
    for (int k = 0; k < n; ++k)
        b0[k] = u(rng);

    {
        const loki::Matrix M = hessenbergPlusFar(n, 1e-3, rng);
        const loki::Vector ref = M.partialPivLu().solve(b0);
        loki::Vector x = b0;
        const unsigned it = loki::gmresHessenberg(M, x, 1e-14, 30, hws, V, Z);
        const double err = (x - ref).norm() / ref.norm();
        std::clog << "far sub-diagonals 1e-3: " << it << " iterations, |x - x_LU|/|x_LU| = " << err << std::endl;
        test_expr(it > 0 && it <= 6);
        test_expr(err < 1e-11);
    }
    {
        const loki::Matrix M = hessenbergPlusFar(n, 0.0, rng);
        const loki::Vector ref = M.partialPivLu().solve(b0);
        loki::Vector x = b0;
        const unsigned it = loki::gmresHessenberg(M, x, 1e-14, 30, hws, V, Z);
        const double err = (x - ref).norm() / ref.norm();
        std::clog << "upper Hessenberg only: " << it << " iterations, error " << err << std::endl;
        test_expr(it == 1);
        test_expr(err < 1e-12);
    }
    {
        const loki::Matrix M = hessenbergPlusFar(n, 3.0, rng);
        loki::Vector x = b0;
        const unsigned it = loki::gmresHessenberg(M, x, 1e-14, 3, hws, V, Z);
        std::clog << "far sub-diagonals 3, 3 iterations allowed: returned " << it << std::endl;
        test_expr(it == 0);
        test_expr(x == b0);
    }
    test_report;
    return nerrors;
}
