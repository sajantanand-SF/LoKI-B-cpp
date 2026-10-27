/** \file
 *
 *  Test of TridiagonalPlusSparse and gmresTridiagonal (time-dependent mode, linearSolver sparse): GMRES
 *  preconditioned with the tridiagonal part, for stage matrices kept as tridiagonal + sparse remainder.
 *
 *  1. Tridiagonal plus far bands above (offsets +40, +150, as inelastic gains) and below (-60, as superelastic
 *     gains), entries ~0.1 of the diagonal: the solution equals dense LU of toDense() to 1e-11, and apply()
 *     equals the dense product.
 *  2. Empty remainder: the preconditioner is exact, one iteration.
 *  3. Far bands as large as the diagonal and 3 iterations allowed: not converged (returns 0, b unchanged),
 *     which makes the caller fall back to the dense solve.
 */

#include "LoKI-B/TimeDependent.h"
#include "tests/TestUtilities.h"

#include <cmath>
#include <iostream>
#include <random>

namespace
{

loki::TridiagonalPlusSparse tridiagonalPlusFar(int n, double farScale, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    loki::TridiagonalPlusSparse M;
    loki::Vector lo(n), d(n), up(n);
    for (int k = 0; k < n; ++k)
    {
        d[k] = 2.0 + u(rng);  // diagonally dominant, as W + a N (losses)
        lo[k] = k > 0 ? 0.3 * u(rng) : 0.0;
        up[k] = k < n - 1 ? 0.3 * u(rng) : 0.0;
    }
    M.setTridiagonal(-0.5, lo, d, up);  // scale as s = -a N; R is scaled by the same s
    std::vector<Eigen::Triplet<double>> t;
    if (farScale > 0.0)
        for (int off : {40, 150, -60})
            for (int i = 0; i < n; ++i)
                if (i + off >= 0 && i + off < n)
                    t.emplace_back(i, i + off, farScale * u(rng));
    M.R.resize(n, n);
    M.R.setFromTriplets(t.begin(), t.end());
    M.factor();
    return M;
}

} // namespace

int main()
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 400;
    loki::Matrix V, Z, dense;
    loki::Vector b0(n);
    for (int k = 0; k < n; ++k)
        b0[k] = u(rng);

    {
        const loki::TridiagonalPlusSparse M = tridiagonalPlusFar(n, 0.1, rng);
        M.toDense(dense);
        const double applyErr = (M.apply(b0) - dense * b0).norm() / (dense * b0).norm();
        const loki::Vector ref = dense.partialPivLu().solve(b0);
        loki::Vector x = b0;
        const unsigned it = loki::gmresTridiagonal(M, x, 1e-14, 30, V, Z);
        const double err = (x - ref).norm() / ref.norm();
        std::clog << "far bands 0.1: apply error " << applyErr << ", " << it
                  << " iterations, |x - x_LU|/|x_LU| = " << err << std::endl;
        test_expr(applyErr < 1e-15);
        test_expr(it > 0 && it <= 15);
        test_expr(err < 1e-11);
    }
    {
        const loki::TridiagonalPlusSparse M = tridiagonalPlusFar(n, 0.0, rng);
        M.toDense(dense);
        const loki::Vector ref = dense.partialPivLu().solve(b0);
        loki::Vector x = b0;
        const unsigned it = loki::gmresTridiagonal(M, x, 1e-14, 30, V, Z);
        const double err = (x - ref).norm() / ref.norm();
        std::clog << "tridiagonal only: " << it << " iterations, error " << err << std::endl;
        test_expr(it == 1);
        test_expr(err < 1e-12);
    }
    {
        const loki::TridiagonalPlusSparse M = tridiagonalPlusFar(n, 3.0, rng);
        loki::Vector x = b0;
        const unsigned it = loki::gmresTridiagonal(M, x, 1e-14, 3, V, Z);
        std::clog << "far bands 3, 3 iterations allowed: returned " << it << std::endl;
        test_expr(it == 0);
        test_expr(x == b0);
    }
    test_report;
    return nerrors;
}
