/** \file
 *
 *  Implementation of LoKI-B's linear algebra facilities.
 *
 *  LoKI-B solves a time and space independent form of the two-term
 *  electron Boltzmann equation (EBE), for non-magnetised non-equilibrium
 *  low-temperature plasmas excited by DC/HF electric fields from
 *  different gases or gas mixtures.
 *  Copyright (C) 2018-2025 A. Tejero-del-Caz, V. Guerra, D. Goncalves,
 *  M. Lino da Silva, L. Marques, N. Pinhao, C. D. Pintassilgo and
 *  L. L. Alves
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 *  \author Daan Boer and Jan van Dijk (C++ version)
 *  \date   15 May 2019
 */

#include "LoKI-B/LinearAlgebra.h"
#include "LoKI-B/Log.h"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace loki {

double maxRelDiff(const Vector& v1, const Vector& v2)
{
//#define LOKIB_USE_OLD_MAXRELDIFF
#ifdef LOKIB_USE_OLD_MAXRELDIFF
    const double res = ((v2 - v1).cwiseAbs().array() / v2.array()).maxCoeff();
#else
    double res=-1.0;
    assert(v1.size()==v2.size());
    for (Vector::Index i=0; i!=v1.size(); ++i)
    {
        const double ref = (std::abs(v1(i)) + std::abs(v2(i)))/2;
        if (ref!=0)
        {
            const double dif = std::abs(v1(i)-v2(i));
            res = std::max(res,dif/ref);
        }
    }
    if (res<0.0)
    {
        throw std::runtime_error("Maximum relative difference: vectors empty or zero.");
    }
#endif
    // std::cout << "maxRelDiff: " << res << std::endl;
    return res;
}

Bandwidth calculateBandwidth(const Matrix& m)
{
    /** \todo In principle, this can be optimized, since for each row the
     *  columns that are within bands that are in [min,max] do not need to
     *  be checked another time.
     */
    static_assert(std::is_signed_v<Matrix::Index>);
    /* 'illegal' starting values. For a zero-valued matrix
     * these will also be the final values.
     */
    Matrix::Index min=+std::max(m.rows(),m.cols());
    Matrix::Index max=-std::max(m.rows(),m.cols());

    for (Matrix::Index r=0; r!=m.rows(); ++r)
    for (Matrix::Index c=0; c!=m.cols(); ++c)
    {
        if (m(r,c)!=0.0)
        {
            const Matrix::Index del = c-r;
            min = std::min(min,del);
            max = std::max(max,del);
        }
    }
    return { min, max };
}

} // namespace loki

namespace loki {
namespace LinAlg {

namespace impl {

void givens(double a, double b, double &c, double &s)
{
    const double root = std::hypot(a,b);

    c =  a / root;
    s = -b / root;
}

} // namespace impl

double *hessenberg(const double *A, double *b, uint32_t n, HessenbergWorkspace& hws)
{
    if (hws.size()<n)
    {
        Log<Message>::Error("hessenberg: workspace is too small.");
    }
    auto *c = hws.c.data();
    auto *s = hws.s.data();
    auto *v = hws.v.data();
    double *x = b;

    // Matrix A is column major
    std::memcpy(v, A + (n - 1) * n, n * sizeof(double));

    for (uint32_t k = n - 1; k > 0; --k)
    {
        impl::givens(v[k], A[(k - 1) * n + k], c[k], s[k]);

        x[k] /= c[k] * v[k] - s[k] * A[(k - 1) * n + k];

        for (int64_t i = 0; i < k - 1; ++i)
        {
            x[i] += x[k] * s[k] * A[(k - 1) * n + i] - x[k] * c[k] * v[i];
            v[i] = c[k] * A[(k - 1) * n + i] + s[k] * v[i];
        }

        x[k - 1] += x[k] * s[k] * A[(k - 1) * n + k - 1] - x[k] * c[k] * v[k - 1];
        v[k - 1] = c[k] * A[(k - 1) * n + k - 1] + s[k] * v[k - 1];
    }

    double t1 = x[0] / v[0];

    for (uint32_t k = 1; k < n; ++k)
    {
        const double t2 = x[k];
        x[k - 1] = c[k] * t1 - s[k] * t2;
        t1 = c[k] * t2 + s[k] * t1;
    }

    x[n - 1] = t1;

    return x;
}

double *hessenberg(const double *A, double *b, uint32_t n)
{
    HessenbergWorkspace hws(n);
    return hessenberg(A,b,n,hws);
}

void solveTDMA(const Matrix& A, const Vector& b, Vector& x)
{
    static_assert(std::is_signed_v<Vector::Index>);
    Vector P( x.size() );
    Vector Q( x.size() );

    const Vector::Index last=x.size()-1;

    // 1. forward sweep. Calculares auxiliary vectors P and Q from A and b.

    if (A(0,0)==0.0)
    {
            throw std::runtime_error("Solver error: ap==0.");
    }
    P[0] = A(0,1)/A(0,0);
    Q[0] =   b[0]/A(0,0);
    for (Vector::Index p=1; p <= last ; ++p)
    {
            const typename Matrix::Scalar denom = A(p,p)-A(p,p-1)*P[p-1];
            if (denom==0.0)
            {
                     throw std::runtime_error("TDMA solver error: zero denominator.");
            }
            const typename Matrix::Scalar _1_denom = 1.0/denom;
            P[p] = (p==last) ? 0.0 : A(p,p+1)*_1_denom;
            Q[p] = (b[p]-A(p,p-1)*Q[p-1])*_1_denom;
    }

    // 2. backward sweep. Calculates x from P and Q.

    x[last]=Q[last];
    for (Vector::Index p=last-1; p>=0; --p)
    {
            x[p] = Q[p] - P[p]*x[p+1];
    }
}

void solveTDMA(const Matrix& A, Vector& bx)
{
    solveTDMA(A,bx,bx);
}

} // namespace LinAlg
} // namespace loki

namespace loki {

namespace
{

/// GMRES from x0 = 0 for M x = b (x overwrites b), right-preconditioned: apply(z) returns M z, precondition(z) sets
/// z <- P^-1 z. Returns the number of iterations, or 0 if not converged after maxIter (b is then unchanged).
template <class Apply, class Precondition>
unsigned gmres(Eigen::Index n, const Apply &apply, const Precondition &precondition, Vector &b, double tol,
               unsigned maxIter, Matrix &V, Matrix &Z)
{
    const double bnorm = b.norm();
    if (bnorm == 0.0)
        return 1;
    if (!std::isfinite(bnorm))
        return 0;
    // right preconditioning: M P^-1 v = b, x = P^-1 v; Givens rotations on the small Hessenberg matrix H
    const unsigned m = std::max(1u, maxIter);
    V.resize(n, m + 1);
    Z.resize(n, m);
    Matrix H = Matrix::Zero(m + 1, m);
    Vector cs(m), sn(m), g = Vector::Zero(m + 1);
    g[0] = bnorm;
    V.col(0) = b / bnorm;
    unsigned j = 0;
    while (j < m)
    {
        Vector z = V.col(j);
        precondition(z);
        Z.col(j) = z;
        Vector w = apply(z);
        for (unsigned i = 0; i <= j; ++i)  // modified Gram-Schmidt
        {
            H(i, j) = V.col(i).dot(w);
            w -= H(i, j) * V.col(i);
        }
        H(j + 1, j) = w.norm();
        if (H(j + 1, j) > 0.0)
            V.col(j + 1) = w / H(j + 1, j);
        for (unsigned i = 0; i < j; ++i)
        {
            const double t = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
            H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
            H(i, j) = t;
        }
        const double r = std::hypot(H(j, j), H(j + 1, j));
        cs[j] = H(j, j) / r;
        sn[j] = H(j + 1, j) / r;
        H(j, j) = r;
        H(j + 1, j) = 0.0;
        g[j + 1] = -sn[j] * g[j];
        g[j] *= cs[j];
        ++j;
        if (!std::isfinite(g[j]))
            return 0;
        if (std::abs(g[j]) <= tol * bnorm)
        {
            const Vector y = H.topLeftCorner(j, j).triangularView<Eigen::Upper>().solve(g.head(j));
            b = Z.leftCols(j) * y;
            return j;
        }
    }
    return 0;
}

} // namespace

unsigned gmresHessenberg(const Matrix &M, Vector &b, double tol, unsigned maxIter, LinAlg::HessenbergWorkspace &hws,
                         Matrix &V, Matrix &Z)
{
    // P is the upper-Hessenberg part of M (LinAlg::hessenberg reads only the entries i <= j + 1)
    const auto n = M.rows();
    return gmres(
        n, [&M](const Vector &z) -> Vector { return M * z; },
        [&](Vector &z) { LinAlg::hessenberg(M.data(), z.data(), static_cast<uint32_t>(n), hws); }, b, tol, maxIter, V,
        Z);
}

void TridiagonalPlusSparse::setTridiagonal(double scale, const Vector &lo0, const Vector &d0, const Vector &up0)
{
    s = scale;
    lo = scale * lo0;
    d = scale * d0;
    up = scale * up0;
}

void TridiagonalPlusSparse::add(Eigen::Index i, Eigen::Index j, double v)
{
    if (j == i)
        d[i] += v;
    else if (j + 1 == i)
        lo[i] += v;
    else if (j == i + 1)
        up[i] += v;
    else
        throw std::logic_error("TridiagonalPlusSparse::add: entry off the tridiagonal.");
}

Vector TridiagonalPlusSparse::apply(const Vector &x) const
{
    Vector y = R * x;
    y *= s;
    const auto n = x.size();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        y[i] += d[i] * x[i];
        if (i > 0)
            y[i] += lo[i] * x[i - 1];
        if (i < n - 1)
            y[i] += up[i] * x[i + 1];
    }
    return y;
}

void TridiagonalPlusSparse::factor()
{
    const auto n = d.size();
    m_c.resize(n);
    m_inv.resize(n);
    m_inv[0] = 1.0 / d[0];
    m_c[0] = up[0] * m_inv[0];
    for (Eigen::Index i = 1; i < n; ++i)
    {
        m_inv[i] = 1.0 / (d[i] - lo[i] * m_c[i - 1]);
        m_c[i] = i < n - 1 ? up[i] * m_inv[i] : 0.0;
    }
}

void TridiagonalPlusSparse::solveTridiagonal(Vector &x) const
{
    const auto n = x.size();
    x[0] *= m_inv[0];
    for (Eigen::Index i = 1; i < n; ++i)
        x[i] = (x[i] - lo[i] * x[i - 1]) * m_inv[i];
    for (Eigen::Index i = n - 2; i >= 0; --i)
        x[i] -= m_c[i] * x[i + 1];
}

void TridiagonalPlusSparse::toDense(Matrix &M) const
{
    M = s * Matrix(R);
    const auto n = d.size();
    for (Eigen::Index i = 0; i < n; ++i)
    {
        M(i, i) = d[i];
        if (i > 0)
            M(i, i - 1) = lo[i];
        if (i < n - 1)
            M(i, i + 1) = up[i];
    }
}

unsigned gmresTridiagonal(const TridiagonalPlusSparse &M, Vector &b, double tol, unsigned maxIter, Matrix &V, Matrix &Z)
{
    return gmres(
        b.size(), [&M](const Vector &z) { return M.apply(z); }, [&M](Vector &z) { M.solveTridiagonal(z); }, b, tol,
        maxIter, V, Z);
}

} // namespace loki
