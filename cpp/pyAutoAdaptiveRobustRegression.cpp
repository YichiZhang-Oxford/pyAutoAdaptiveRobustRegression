#include <armadillo>
#include <cstring>
#include <iostream>

using namespace std;
using namespace arma;

/*  ----------------- useful help functions --------------  */

struct MyVec
{
    double *data;
    size_t n;
};

struct MyMat
{
    double *data;
    size_t num_rows, num_cols;
};

extern "C" void freeMyVec(MyVec &vec)
{
    delete[] vec.data;
    memset((void *)&vec, 0, sizeof(vec));
}

extern "C" void freeMyMat(MyMat &mat)
{
    delete[] mat.data;
    memset((void *)&mat, 0, sizeof(mat));
}

struct HuberCovResult
{
    MyVec means;
    MyMat cov;
};

extern "C" void freeHuberCovResult(const HuberCovResult &result)
{
    delete[] result.means.data;
    delete[] result.cov.data;
    memset((void *)&result, 0, sizeof(result));
}

              /* --------------------------------- */    
/*------------ Tuning-Free Huber Regression Funtions ---------------*/
              /* --------------------------------- */ 

int sgn(const double x)
{
    return (x > 0) - (x < 0);
}

double f1(const double x, const arma::vec &resSq, const int n, const double rhs)
{
    return arma::mean(arma::min(resSq / x, arma::ones(n))) - rhs;
}

double rootf1(const arma::vec &resSq, const int n, const double rhs, double low, double up, const double tol = 0.001, const int maxIte = 500)
{
    int ite = 1;
    while (ite <= maxIte && up - low > tol)
    {
        double mid = 0.5 * (up + low);
        double val = f1(mid, resSq, n, rhs);
        if (val < 0)
        {
            up = mid;
        }
        else
        {
            low = mid;
        }
        ite++;
    }
    return 0.5 * (low + up);
}

double f2(const double x, const arma::vec &resSq, const int N, const double rhs)
{
    return arma::mean(arma::min(resSq / x, arma::ones(N))) - rhs;
}

double rootf2(const arma::vec &resSq, const int n, const int d, const int N, const double rhs, double low, double up, const double tol = 0.001,
              const int maxIte = 500)
{
    int ite = 0;
    while (ite <= maxIte && up - low > tol)
    {
        double mid = 0.5 * (up + low);
        double val = f2(mid, resSq, N, rhs);
        if (val < 0)
        {
            up = mid;
        }
        else
        {
            low = mid;
        }
        ite++;
    }
    return 0.5 * (low + up);
}

double g1(const double x, const arma::vec &resSq, const int n, const double rhs)
{
    return arma::mean(arma::min(resSq / x, arma::ones(n))) - rhs;
}

double rootg1(const arma::vec &resSq, const int n, const double rhs, double low, double up, const double tol = 0.001, const int maxIte = 500)
{
    int ite = 0;
    while (ite <= maxIte && up - low > tol)
    {
        double mid = 0.5 * (up + low);
        double val = g1(mid, resSq, n, rhs);
        if (val < 0)
        {
            up = mid;
        }
        else
        {
            low = mid;
        }
        ite++;
    }
    return 0.5 * (low + up);
}

double huberDer(const arma::vec &res, const double tau, const int n)
{
    double rst = 0.0;
    for (int i = 0; i < n; i++)
    {
        double cur = res(i);
        rst -= std::abs(cur) <= tau ? cur : tau * sgn(cur);
    }
    return rst / n;
}

extern "C" double huberMean(MyVec _X, const double tol = 0.001, const int iteMax = 500)
{
    arma::vec X(_X.data, _X.n);
    int n = X.n_rows;
    double rhs = std::log(n) / n;
    double mx = arma::mean(X);
    X -= mx;
    double tau = arma::stddev(X) * std::sqrt((long double)n / std::log(n));
    double derOld = huberDer(X, tau, n);
    double mu = -derOld, muDiff = -derOld;
    arma::vec res = X - mu;
    arma::vec resSq = arma::square(res);
    tau = std::sqrt((long double)rootf1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
    double derNew = huberDer(res, tau, n);
    double derDiff = derNew - derOld;
    int ite = 1;
    while (std::abs(derNew) > tol && ite <= iteMax)
    {
        double alpha = 1.0;
        double cross = muDiff * derDiff;
        if (cross > 0)
        {
            double a1 = cross / derDiff * derDiff;
            double a2 = muDiff * muDiff / cross;
            alpha = std::min(std::min(a1, a2), 100.0);
        }
        derOld = derNew;
        muDiff = -alpha * derNew;
        mu += muDiff;
        res = X - mu;
        resSq = arma::square(res);
        tau = std::sqrt((long double)rootf1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
        derNew = huberDer(res, tau, n);
        derDiff = derNew - derOld;
        ite++;
    }
    return mu + mx;
}

double _huberMean(arma::vec X, const double tol = 0.001, const int iteMax = 500)
{
    int n = X.n_rows;
    double rhs = std::log(n) / n;
    double mx = arma::mean(X);
    X -= mx;
    double tau = arma::stddev(X) * std::sqrt((long double)n / std::log(n));
    double derOld = huberDer(X, tau, n);
    double mu = -derOld, muDiff = -derOld;
    arma::vec res = X - mu;
    arma::vec resSq = arma::square(res);
    tau = std::sqrt((long double)rootf1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
    double derNew = huberDer(res, tau, n);
    double derDiff = derNew - derOld;
    int ite = 1;
    while (std::abs(derNew) > tol && ite <= iteMax)
    {
        double alpha = 1.0;
        double cross = muDiff * derDiff;
        if (cross > 0)
        {
            double a1 = cross / derDiff * derDiff;
            double a2 = muDiff * muDiff / cross;
            alpha = std::min(std::min(a1, a2), 100.0);
        }
        derOld = derNew;
        muDiff = -alpha * derNew;
        mu += muDiff;
        res = X - mu;
        resSq = arma::square(res);
        tau = std::sqrt((long double)rootf1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
        derNew = huberDer(res, tau, n);
        derDiff = derNew - derOld;
        ite++;
    }
    return mu + mx;
}

double hMeanCov(const arma::vec &Z, const int n, const int d, const int N, double rhs, const double epsilon = 0.0001, const int iteMax = 500)
{
    double muOld = 0;
    double muNew = arma::mean(Z);
    double tau = arma::stddev(Z) * std::sqrt((long double)n / (2 * std::log(d) + std::log(n)));
    int iteNum = 0;
    arma::vec res(n), resSq(n), w(n);
    while ((std::abs(muNew - muOld) > epsilon) && iteNum < iteMax)
    {
        muOld = muNew;
        res = Z - muOld;
        resSq = arma::square(res);
        tau = std::sqrt((long double)rootf2(resSq, n, d, N, rhs, arma::min(resSq), arma::accu(resSq)));
        w = arma::min(tau / arma::abs(res), arma::ones(N));
        muNew = arma::as_scalar(Z.t() * w) / arma::accu(w);
        iteNum++;
    }
    return muNew;
}

extern "C" HuberCovResult huberCov(const MyMat &_X)
{
    const arma::mat X(_X.data, _X.num_rows, _X.num_cols);
    const int n = X.n_rows;
    const int p = X.n_cols;

    double rhs2 = (2 * std::log(p) + std::log(n)) / n;
    arma::vec mu(p);
    arma::mat sigmaHat(p, p);
    for (int j = 0; j < p; j++)
    {
        mu(j) = _huberMean(X.col(j), n);
        double theta = _huberMean(arma::square(X.col(j)), n);
        double temp = mu(j) * mu(j);
        if (theta > temp)
        {
            theta -= temp;
        }
        sigmaHat(j, j) = theta;
    }
    int N = n * (n - 1) >> 1;
    arma::mat Y(N, p);
    for (int i = 0, k = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            Y.row(k++) = X.row(i) - X.row(j);
        }
    }
    for (int i = 0; i < p - 1; i++)
    {
        for (int j = i + 1; j < p; j++)
        {
            sigmaHat(i, j) = sigmaHat(j, i) = hMeanCov(0.5 * Y.col(i) % Y.col(j), n, p, N, rhs2);
        }
    }

    HuberCovResult result;

    result.means.data = new double[mu.size()];
    result.means.n = mu.size();
    memcpy(result.means.data, mu.memptr(), mu.size() * sizeof(double));

    result.cov.data = new double[sigmaHat.n_rows * sigmaHat.n_cols];
    result.cov.num_rows = sigmaHat.n_rows;
    result.cov.num_cols = sigmaHat.n_cols;
    memcpy(result.cov.data, sigmaHat.memptr(), sigmaHat.n_rows * sigmaHat.n_cols * sizeof(double));

    return result;
}

double mad(const arma::vec &x)
{
    return 1.482602 * arma::median(arma::abs(x - arma::median(x)));
}

arma::mat standardize(arma::mat X, const arma::rowvec &mx, const arma::vec &sx, const int p)
{
    for (int i = 0; i < p; i++)
    {
        X.col(i) = (X.col(i) - mx(i)) / sx(i);
    }
    return X;
}

void updateHuber(const arma::mat &Z, const arma::vec &res, arma::vec &der, arma::vec &grad, const int n, const double tau, const double n1)
{
    for (int i = 0; i < n; i++)
    {
        double cur = res(i);
        if (std::abs(cur) <= tau)
        {
            der(i) = -cur;
        }
        else
        {
            der(i) = -tau * sgn(cur);
        }
    }
    grad = n1 * Z.t() * der;
}

extern "C" MyVec adaHuberReg(MyMat _X, MyVec _Y, const double tol = 0.0001, const int iteMax = 5000)
{
    arma::mat X(_X.data, _X.num_rows, _X.num_cols);
    arma::vec Y(_Y.data, _Y.n);
    int n = X.n_rows;
    int p = X.n_cols;
    const double n1 = 1.0 / n;
    double rhs = n1 * (p + std::log(n * p));
    arma::rowvec mx = arma::mean(X, 0);
    arma::vec sx = arma::stddev(X, 0, 0).t();
    double my = arma::mean(Y);
    arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
    Y -= my;
    double tau = 1.345 * mad(Y);
    arma::vec der(n);
    arma::vec gradOld(p + 1), gradNew(p + 1);
    updateHuber(Z, Y, der, gradOld, n, tau, n1);
    arma::vec beta = -gradOld, betaDiff = -gradOld;
    arma::vec res = Y - Z * beta;
    arma::vec resSq = arma::square(res);
    tau = std::sqrt((long double)rootg1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
    updateHuber(Z, res, der, gradNew, n, tau, n1);
    arma::vec gradDiff = gradNew - gradOld;
    int ite = 1;
    while (arma::norm(gradNew, "inf") > tol && ite <= iteMax)
    {
        double alpha = 1.0;
        double cross = arma::as_scalar(betaDiff.t() * gradDiff);
        if (cross > 0)
        {
            double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
            double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
            alpha = std::min(std::min(a1, a2), 100.0);
        }
        gradOld = gradNew;
        betaDiff = -alpha * gradNew;
        beta += betaDiff;
        res -= Z * betaDiff;
        resSq = arma::square(res);
        tau = std::sqrt((long double)rootg1(resSq, n, rhs, arma::min(resSq), arma::accu(resSq)));
        updateHuber(Z, res, der, gradNew, n, tau, n1);
        gradDiff = gradNew - gradOld;
        ite++;
    }
    beta.rows(1, p) /= sx;
    beta(0) = _huberMean(Y + my - X * beta.rows(1, p));

    MyVec result;
    result.data = new double[n];
    result.n = n;
    memcpy(result.data, beta.memptr(), n * sizeof(double));
    return result;
}

extern "C" MyVec huberReg(MyMat _X, MyVec _Y, const double tol = 0.0001, const double constTau = 1.345, const int iteMax = 5000)
{
    arma::mat X(_X.data, _X.num_rows, _X.num_cols);
    arma::vec Y(_Y.data, _Y.n);
    int n = X.n_rows;
    int p = X.n_cols;
    const double n1 = 1.0 / n;
    arma::rowvec mx = arma::mean(X, 0);
    arma::vec sx = arma::stddev(X, 0, 0).t();
    double my = arma::mean(Y);
    arma::mat Z = arma::join_rows(arma::ones(n), standardize(X, mx, sx, p));
    Y -= my;
    double tau = constTau * mad(Y);
    arma::vec der(n);
    arma::vec gradOld(p + 1), gradNew(p + 1);
    updateHuber(Z, Y, der, gradOld, n, tau, n1);
    arma::vec beta = -gradOld, betaDiff = -gradOld;
    arma::vec res = Y - Z * beta;
    tau = constTau * mad(res);
    updateHuber(Z, res, der, gradNew, n, tau, n1);
    arma::vec gradDiff = gradNew - gradOld;
    int ite = 1;
    while (arma::norm(gradNew, "inf") > tol && ite <= iteMax)
    {
        double alpha = 1.0;
        double cross = arma::as_scalar(betaDiff.t() * gradDiff);
        if (cross > 0)
        {
            double a1 = cross / arma::as_scalar(gradDiff.t() * gradDiff);
            double a2 = arma::as_scalar(betaDiff.t() * betaDiff) / cross;
            alpha = std::min(std::min(a1, a2), 100.0);
        }
        gradOld = gradNew;
        betaDiff = -alpha * gradNew;
        beta += betaDiff;
        res -= Z * betaDiff;
        tau = constTau * mad(res);
        updateHuber(Z, res, der, gradNew, n, tau, n1);
        gradDiff = gradNew - gradOld;
        ite++;
    }
    beta.rows(1, p) /= sx;
    beta(0) = _huberMean(Y + my - X * beta.rows(1, p), n);

    MyVec result;
    result.data = new double[n];
    result.n = n;
    memcpy(result.data, beta.memptr(), n * sizeof(double));
    return result;
}

              /* - */    
/*------------ LASSO ------------*/
              /* - */ 

arma::vec softThresh(const arma::vec& x, const arma::vec& lambda)
{
  return arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros(x.size()));
}

arma::vec cmptLambda(const arma::vec& beta, const double lambda)
{
  arma::vec rst = lambda * arma::ones(beta.size());
  rst(0) = 0;
  return rst;
}

double loss(const arma::vec& Y, const arma::vec& Ynew, const std::string lossType, const double tau)
{
  double rst = 0;
  if (lossType == "l2") {
    rst = arma::mean(arma::square(Y - Ynew)) / 2;
  } else if (lossType == "Huber") {
    arma::vec res = Y - Ynew;
    for (int i = 0; i < Y.size(); i++) {
      if (std::abs(res(i)) <= tau) {
        rst += res(i) * res(i) / 2;
      } else {
        rst += tau * std::abs(res(i)) - tau * tau / 2;
      }
    }
    rst /= Y.size();
  }
  return rst;
}

arma::vec gradLoss(const arma::mat& X, const arma::vec& Y, const arma::vec& beta,
                   const std::string lossType, const double tau)
{
  arma::vec res = Y - X * beta;
  arma::vec rst = arma::zeros(beta.size());
  if (lossType == "l2") {
    rst = -1 * (res.t() * X).t();
  } else if (lossType == "Huber") {
    for (int i = 0; i < Y.size(); i++) {
      if (std::abs(res(i)) <= tau) {
        rst -= res(i) * X.row(i).t();
      } else {
        rst -= tau * sgn(res(i)) * X.row(i).t();
      }
    }
  }
  return rst / Y.size();
}

arma::vec updateBeta(const arma::mat& X, const arma::vec& Y, arma::vec beta, const double phi,
                     const arma::vec& Lambda, const std::string lossType, const double tau)
{
  arma::vec first = beta - gradLoss(X, Y, beta, lossType, tau) / phi;
  arma::vec second = Lambda / phi;
  return softThresh(first, second);
}

double cmptPsi(const arma::mat& X, const arma::vec& Y, const arma::vec& betaNew,
               const arma::vec& beta, const double phi, const std::string lossType,
               const double tau) 
{
  arma::vec diff = betaNew - beta;
  double rst = loss(Y, X * beta, lossType, tau)
    + arma::as_scalar((gradLoss(X, Y, beta, lossType, tau)).t() * diff)
    + phi * arma::as_scalar(diff.t() * diff) / 2;
    return rst;
}

Rcpp::List LAMM(const arma::mat& X, const arma::vec& Y, const arma::vec& Lambda, arma::vec beta,
                const double phi, const std::string lossType, const double tau, 
                const double gamma) 
{
  double phiNew = phi;
  arma::vec betaNew = arma::vec();
  double FVal, PsiVal;
  while (true) {
    betaNew = updateBeta(X, Y, beta, phiNew, Lambda, lossType, tau);
    FVal = loss(Y, X * betaNew, lossType, tau);
    PsiVal = cmptPsi(X, Y, betaNew, beta, phiNew, lossType, tau);
    if (FVal <= PsiVal) {
      break;
    }
    phiNew *= gamma;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("phi") = phiNew);
}

arma::vec lasso(const arma::mat& X, const arma::vec& Y, const double lambda,
                const double phi0 = 0.001, const double gamma = 1.5, 
                const double epsilon_c = 0.001, const int iteMax = 500) 
{
  int d = X.n_cols - 1;
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  arma::vec Lambda = cmptLambda(beta, lambda);
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  while (ite < iteMax) {
    ite++;
    listLAMM = LAMM(X, Y, Lambda, beta, phi, "l2", 1, gamma);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
  }
  return betaNew;
}

Rcpp::List huberLasso(const arma::mat& X, const arma::vec& Y, const double lambda,
                      double tau = -1, const double constTau = 1.345, const double phi0 = 0.001, 
                      const double gamma = 1.5, const double epsilon_c = 0.001, 
                      const int iteMax = 500) 
{
  int n = X.n_rows;
  int d = X.n_cols - 1;
  arma::vec beta = arma::zeros(d + 1);
  arma::vec betaNew = arma::zeros(d + 1);
  arma::vec Lambda = cmptLambda(beta, lambda);
  double mad;
  if (tau <= 0) {
    arma::vec betaLasso = lasso(X, Y, lambda, phi0, gamma, epsilon_c, iteMax);
    arma::vec res = Y - X * betaLasso;
    mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
    tau = constTau * mad;
  }
  double phi = phi0;
  int ite = 0;
  Rcpp::List listLAMM;
  arma::vec res(n);
  while (ite < iteMax) {
    ite++;
    listLAMM = LAMM(X, Y, Lambda, beta, phi, "Huber", tau, gamma);
    betaNew = Rcpp::as<arma::vec>(listLAMM["beta"]);
    phi = listLAMM["phi"];
    phi = std::max(phi0, phi / gamma);
    if (arma::norm(betaNew - beta, "inf") <= epsilon_c) {
      break;
    }
    beta = betaNew;
    res = Y - X * beta;
    mad = arma::median(arma::abs(res - arma::median(res))) / 0.6744898;
    tau = constTau * mad;
  }
  return Rcpp::List::create(Rcpp::Named("beta") = betaNew, Rcpp::Named("tau") = tau,
                            Rcpp::Named("iteration") = ite);
}

arma::uvec getIndex(const int n, const int low, const int up) 
{
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq >= low && seq <= up);
}

arma::uvec getIndexComp(const int n, const int low, const int up)
{
  arma::vec seq = arma::regspace(0, n - 1);
  return arma::find(seq < low || seq > up);
}

double pairPred(const arma::mat& X, const arma::vec& Y, const arma::vec& beta)
{
  int n = X.n_rows;
  int d = X.n_cols - 1;
  int m = n * (n - 1) >> 1;
  arma::mat pairX(m, d + 1);
  arma::vec pairY(m);
  int k = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      pairX.row(k) = X.row(i) - X.row(j);
      pairY(k++) = Y(i) - Y(j);
    }
  }
  arma::vec predY = pairX * beta;
  return arma::accu(arma::square(pairY - predY));
}

Rcpp::List cvHuberLasso(const arma::mat& X, const arma::vec& Y,
                        Rcpp::Nullable<Rcpp::NumericVector> lSeq = R_NilValue, int nlambda = 30,
                        const double constTau = 2.5, const double phi0 = 0.001, 
                        const double gamma = 1.5, const double epsilon_c = 0.001, 
                        const int iteMax = 500, int nfolds = 3) 
{
  int n = X.n_rows;
  int d = X.n_cols;
  arma::mat Z(n, d + 1);
  Z.cols(1, d) = X;
  Z.col(0) = arma::ones(n);
  arma::vec lambdaSeq = arma::vec();
  if (lSeq.isNotNull()) {
    lambdaSeq = Rcpp::as<arma::vec>(lSeq);
    nlambda = lambdaSeq.size();
  } else {
    double lambdaMax = arma::max(arma::abs(Y.t() * Z)) / n;
    double lambdaMin = 0.01 * lambdaMax;
    lambdaSeq = arma::exp(arma::linspace(std::log((long double)lambdaMin),
                                   std::log((long double)lambdaMax), nlambda));
  }
  if (nfolds > 10 || nfolds > n) {
    nfolds = n < 10 ? n : 10;
  }
  int size = n / nfolds;
  arma::vec mse = arma::zeros(nlambda);
  int low, up;
  arma::uvec idx, idxComp;
  Rcpp::List hLassoList;
  arma::vec thetaHat(d + 1);
  for (int i = 0; i < nlambda; i++) {
    for (int j = 0; j < nfolds; j++) {
      low = j * size;
      up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
      idx = getIndex(n, low, up);
      idxComp = getIndexComp(n, low, up);
      hLassoList = huberLasso(Z.rows(idxComp), Y.rows(idxComp), lambdaSeq(i), -1, constTau, phi0, 
                              gamma, epsilon_c, iteMax);
      thetaHat = Rcpp::as<arma::vec>(hLassoList["beta"]);
      mse(i) += pairPred(Z.rows(idx), Y.rows(idx), thetaHat);
    }
  }
  arma::uword cvIdx = mse.index_min();
  hLassoList = huberLasso(Z, Y, lambdaSeq(cvIdx), -1, constTau, phi0, gamma, epsilon_c, iteMax);
  arma::vec theta = Rcpp::as<arma::vec>(hLassoList["beta"]);
  Rcpp::List listMean = huberMean(Y - Z.cols(1, d) * theta.rows(1, d));
  theta(0) = listMean["mu"];
  return Rcpp::List::create(Rcpp::Named("theta") = theta, Rcpp::Named("lambdaSeq") = lambdaSeq,
                            Rcpp::Named("lambdaMin") = lambdaSeq(cvIdx), 
                            Rcpp::Named("tauCoef") = hLassoList["tau"], 
                            Rcpp::Named("tauItcp") = listMean["tau"], 
                            Rcpp::Named("iteCoef") = hLassoList["iteration"],
                            Rcpp::Named("iteItcp") = listMean["iteration"]);
}

              /* ------------------------------------ */    
/*------------ Alternating Gradient Descent Functions ------------*/
              /* ------------------------------------ */ 

double LnVal(const arma::vec &Y, double mu, double tau, const int n, double z)
{
    return (arma::accu(arma::sqrt(tau * tau + arma::square(Y - mu)) - tau) / (z * std::sqrt(n)) + z * tau / std::sqrt(n)) / n;
}

double gradientMuVal(const arma::vec &Y, double mu, double tau, const int n, double z)
{
    return -(1 / std::sqrt(n)) * arma::accu((Y - mu) / (z * arma::sqrt(tau * tau + arma::square(Y - mu)))) / n;
}

double gradientTauVal(const arma::vec &Y, double mu, double tau, const int n, double z)
{
    return (1 / std::sqrt(n)) * arma::accu(tau / (z * arma::sqrt(tau * tau + arma::square(Y - mu)))) / n - (std::sqrt(n) / z - z / std::sqrt(n)) / n;
}

extern "C" double agd(MyVec _Y, double epsilon = 1e-5, int iteMax = 5000)
{
    arma::vec Y(_Y.data, _Y.n);
    int n = Y.n_elem;
    double eta1 = 1.0;
    double eta2 = std::sqrt(n);
    double muOld = std::numeric_limits<double>::infinity();
    double muNew = arma::mean(Y);
    double tauOld = std::numeric_limits<double>::infinity();
    double delta = 0.05;
    double z = std::min(std::sqrt(std::log(n)), std::sqrt(std::log(1 / delta)));
    double tauNew = arma::stddev(Y) * std::sqrt(n) / (std::sqrt(2) * z);
    int iteNum = 0;
    double LMuOld = std::numeric_limits<double>::infinity();
    double LMuNew = LnVal(Y, muNew, tauNew, n, z);

    while (((LMuOld - LMuNew > 1e-10) && std::abs(muOld - muNew) > epsilon) && std::abs(tauOld - tauNew) > epsilon && iteNum < iteMax)
    {
        muOld = muNew;
        tauOld = tauNew;
        LMuOld = LMuNew;

        double gradientLnTauVal = gradientTauVal(Y, muOld, tauOld, n, z);
        tauNew = tauOld - eta2 * gradientLnTauVal;

        double gradientLnMuVal = gradientMuVal(Y, muOld, tauNew, n, z);
        muNew = muOld - eta1 * gradientLnMuVal;
        LMuNew = LnVal(Y, muNew, tauNew, n, z);
        iteNum += 1;
    }
    return muNew;
}

extern "C" double agdBB(MyVec _Y, double epsilon = 1e-5, int iteMax = 5000)
{
    arma::vec Y(_Y.data, _Y.n);
    int n = Y.n_elem;
    double eta1 = 1.0;
    double eta2 = std::sqrt(n);
    double muOld = std::numeric_limits<double>::infinity();
    double muNew = arma::mean(Y);
    double tauOld = std::numeric_limits<double>::infinity();
    double delta = 0.05;
    double z = std::min(std::sqrt(std::log(n)), std::sqrt(std::log(1 / delta)));
    double tauNew = arma::stddev(Y) * std::sqrt(n) / (std::sqrt(2) * z);
    int iteNum = 0;

    double stepSizeTau = eta2;
    double stepSizeMu = eta1;

    double LMuOld = std::numeric_limits<double>::infinity();
    double LMuNew = LnVal(Y, muNew, tauNew, n, z);

    double lastTauTilde;
    double lastGradTau;
    double lastMuTilde;
    double lastGradMu;

    while (((LMuOld - LMuNew) > epsilon * epsilon && std::abs(muOld - muNew) > epsilon) && std::abs(tauOld - tauNew) > epsilon && iteNum < iteMax)
    {
        LMuOld = LMuNew;
        muOld = muNew;
        tauOld = tauNew;

        double gradTau = gradientTauVal(Y, muOld, tauOld, n, z);
        if (iteNum > 0)
        {
            double sTau = tauOld - lastTauTilde;
            double yTau = gradTau - lastGradTau;
            stepSizeTau = std::max(eta2, std::min(std::min((sTau * yTau) / (yTau * yTau), (sTau * sTau) / std::max(0.000001, sTau * yTau)), 1e3));
        }
        lastGradTau = gradTau;
        lastTauTilde = tauOld;

        tauNew = std::max(1.35, tauOld - stepSizeTau * (gradientTauVal(Y, muOld, tauOld, n, z)));

        double gradMu = gradientMuVal(Y, muOld, tauNew, n, z);
        if (iteNum > 0)
        {
            double sMu = muOld - lastMuTilde;
            double yMu = gradMu - lastGradMu;
            stepSizeMu = std::max(eta1, std::min(std::min((sMu * yMu) / (yMu * yMu), (sMu * sMu) / std::max(0.000001, sMu * yMu)), 1e3));
        }
        lastGradMu = gradMu;
        lastMuTilde = muOld;

        muNew = muOld - stepSizeMu * (gradientMuVal(Y, muOld, tauNew, n, z));
        LMuNew = LnVal(Y, muNew, tauNew, n, z);
        iteNum += 1;
    }
    return muNew;
}


extern "C" double agdBacktracking(MyVec _Y, double s1 = 1.0, double gamma1 = 0.5, double beta1 = 0.8, double s2 = 1.0, 
                                  double gamma2 = 0.5, double beta2 = 0.8, double epsilon = 1e-5, int iteMax = 5000)
{    
    arma::vec Y(_Y.data, _Y.n);
    int n = Y.n_elem;
    double muOld = std::numeric_limits<double>::infinity();
    double muNew = arma::mean(Y);
    double tauOld = std::numeric_limits<double>::infinity();
    double delta = 0.05;
    double z = std::min(std::sqrt(std::log(n)), std::sqrt(std::log(1 / delta)));
    double tauNew = arma::stddev(Y) * std::sqrt(n) / (std::sqrt(2) * z);
    int iteNum = 0;

    while (std::abs(muOld - muNew) > epsilon && std::abs(tauOld - tauNew) > epsilon && iteNum < iteMax)
    {
        muOld = muNew;
        tauOld = tauNew;
        double LTauOld = LnVal(Y, muOld, tauOld, n, z);
        double gradientLnTauVal = gradientTauVal(Y, muOld, tauOld, n, z);

        /*backtracking for eta2*/
        double eta2 = s2;
        tauNew = tauOld - eta2*gradientLnTauVal;
        double LTauNew = LnVal(Y, muOld, tauNew, n ,z);
        
        while (LTauNew > LTauOld + gamma2*eta2*gradientLnTauVal)
        {
            eta2 = beta2*eta2;
            tauNew = tauOld - eta2*gradientLnTauVal;
            LTauNew = LnVal(Y, muOld, tauNew, n ,z);
            gradientLnTauVal = gradientTauVal(Y, muOld, tauNew, n, z);
        }
        
        double LMuOld = LTauNew;
        double gradientLnMuVal = gradientMuVal(Y, muOld, tauNew, n, z);
        
        /*backtracking for eta1*/
        double eta1 = s1;
        muNew = muOld - eta1*gradientLnMuVal;
        double LMuNew = LnVal(Y, muNew, tauNew, n ,z);
        
        while (LMuNew > LMuOld + gamma1*eta1*gradientLnMuVal)
        {
            eta1 = beta1*eta1;
            muNew = muOld - eta1*gradientLnMuVal;
            LMuNew = LnVal(Y, muNew, tauNew, n ,z);
            gradientLnMuVal = gradientMuVal(Y, muNew, tauNew, n, z);
        }
        iteNum += 1;
    }
    return muNew;
}
