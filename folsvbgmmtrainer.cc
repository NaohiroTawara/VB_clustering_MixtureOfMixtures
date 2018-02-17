//
//  folsvbgmmtrainer.cc
//  VB
//
//  Created by 俵 直弘 on 13/01/08.
//  Copyright (c) 2013年 俵 直弘. All rights reserved.
//

#include "folsvbgmmtrainer.h"

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

void CFoLSVBGmmTrainer::update(const double& _X, const int _u)
{
  CVBGmm *gmm         = getGmm();
  const int dimension = getDimension();
  const int covtype   = getCovType();
  CSegment* features  = getFeatures();

  CvMat stub;
  CvMat* matrix  = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* rowvec  = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec  = cvCreateMat(dimension, 1, CV_32F);
  
  CvMat* B  = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* nu = cvCreateMat(1, dimension, CV_32F);
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Ztp;
  
  const int numframes = features->getNumFrames(_u);
  double gamma_t      = _X;
  
  iter_Z = m_Z.begin();
  for (int j = 0; iter_Z != m_Z.end(); ++iter_Z, ++j)
  { /* iterate for # of mixtures */
    
    const CvMat* nu_not_u   = gmm->getNu(j);
    const CvMat* B_not_u    = gmm->getB(j);
    const double eta_not_u  = gmm->getEta(j);
    const double xi_not_u   = gmm->getXi(j);
    const double beta_not_u = gmm->getBeta(j);
    
    // _u 番目の発話に起因する統計量を追加してハイパーパラメタを再計算する（eta, xi, beta）
    const double eta  = eta_not_u  + gamma_t * numframes;
    const double xi   = xi_not_u   + gamma_t * numframes;
    const double beta = beta_not_u + gamma_t * numframes;

    // _u 番目の発話に起因する統計量を追加してハイパーパラメタを再計算する（nu）
    cvReduce(features->getSegment(_u), rowvec);
    cvAddWeighted(nu_not_u, xi_not_u, rowvec, gamma_t, 0.0, nu);
    cvConvertScale(nu, nu, 1.0 / xi);

    // _u 番目の発話に起因する統計量を追加してハイパーパラメタを再計算する（B）
#if 0
    cvSub(rowvec, nu_not_u, rowvec);
    if (covtype == FULLC)
    {
      cvmTranspose(rowvec, colvec);
      cvMatMul(colvec, rowvec, matrix);
    }
    else
      cvMul(rowvec, rowvec, matrix);
    cvAddWeighted(B_not_u, 1.0,  matrix,  xi_not_u * gamma_t / xi, 0.0, B);
#else
    if (covtype == FULLC)
    {
      cvmTranspose(nu_not_u, colvec);
      cvMatMul(colvec, nu_not_u, matrix);
    }
    else
      cvMul(nu_not_u, nu_not_u, matrix);
    if (covtype == FULLC)
    {
      cvmTranspose(nu, colvec);
      cvMatMul(colvec, nu, matrix2);
    }
    else
      cvMul(nu, nu, matrix2);
    cvAddWeighted(matrix, xi_not_u, matrix2, - xi, 0.0, matrix);
    cvAdd(B_not_u, matrix, B);
    
    cvSetZero(matrix);
    for (int t = 0; t < numframes; ++t)
    {
      const CvMat* f_frame = features->getFrame(_u, t, stub); // f_frame <- [1 x g]
      if (covtype == FULLC)
      {
        cvmTranspose(f_frame, colvec);
        cvMatMul(colvec, f_frame, matrix2);
      }
      else
        cvMul(f_frame, f_frame, matrix2);
      cvAdd(matrix, matrix2,  matrix);
    }
    cvAddWeighted(B, 1.0,  matrix, gamma_t, 0.0, B);
#endif
    
    //    cvSetZero(matrix);
//    for (int t = 0; t < numframes; ++t)
//    {
//      const CvMat* f_frame = features->getFrame(_u, t, stub); // f_frame <- [1 x g]
//      cvSub(f_frame, nu_not_u, rowvec);
//      if (covtype == FULLC)
//      {
//       cvmTranspose(rowvec, colvec);
//       cvMatMul(colvec, rowvec, matrix2);
//      }
//      else
//        cvMul(rowvec, rowvec, matrix2);
//      cvAdd(matrix, matrix2, matrix);
//    }

    gmm->setNu  (j, nu);
    gmm->setB   (j, B);
    gmm->setEta (j, eta);
    gmm->setXi  (j, xi);
    gmm->setBeta(j, beta);
 
  }
  
  if (matrix)  cvReleaseMat(&matrix);
  if (matrix2) cvReleaseMat(&matrix2);
  if (rowvec)  cvReleaseMat(&rowvec);
  if (colvec)  cvReleaseMat(&colvec);
  if (B)       cvReleaseMat(&B);
  if (nu)      cvReleaseMat(&nu);
}

//----------------------------------------------------------------------


void CFoLSVBGmmTrainer::updateZ(const double& _X, const int _u)
{
  CVBGmm *gmm         = getGmm();
  CSegment* features  = getFeatures();
  const int covtype   = getCovType();
  const int dimension = getDimension();
  
  CvMat stub;
  CvMat* matrix  = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* rowvec  = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec  = cvCreateMat(dimension, 1, CV_32F);
  
  CvMat* B_not_u  = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* nu_not_u = cvCreateMat(1, dimension, CV_32F);
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Zut;
  
  const int numframes = features->getNumFrames(_u);
  
  iter_Z = m_Z.begin();
  for (int j = 0; iter_Z != m_Z.end(); ++iter_Z, ++j)
  { /* iterate for # of mixtures */
    CvMat* nu   = gmm->getNu(j);
    CvMat* B    = gmm->getB(j);
    double eta  = gmm->getEta(j);
    double xi   = gmm->getXi(j);
    double beta = gmm->getBeta(j);
    
    double gamma_t = _X;
    
    // _u 番目の発話に起因する統計量を取り除いてハイパーパラメタを再計算する (eta_{/u}, xi_{/u}, beta_{/u})
    const double eta_not_u  = eta  - gamma_t * numframes;
    const double xi_not_u   = xi   - gamma_t * numframes;
    const double beta_not_u = beta - gamma_t * numframes;
    
    // _u 番目の発話に起因する統計量を取り除いてハイパーパラメタを再計算する（nu_{/u}）
    cvReduce(features->getSegment(_u), rowvec);
    cvAddWeighted(nu, xi, rowvec, - gamma_t, 0.0, nu_not_u);
    cvConvertScale(nu_not_u, nu_not_u, 1.0 / xi_not_u);
    
#if 0
    cvSub(rowvec, nu, rowvec);
    if (covtype == FULLC)
    {
      cvmTranspose(rowvec, colvec);
      cvMatMul(colvec, rowvec, matrix);
    }
    else
      cvMul(rowvec, rowvec, matrix);
    cvAddWeighted(B, 1.0,  matrix, - xi * gamma_t / xi_not_u, 0.0, B_not_u);
#else
    // _u 番目の発話に起因する統計量を取り除いてハイパーパラメタを再計算する（B_{/u}）
    if (covtype == FULLC)
    {
      cvmTranspose(nu, colvec);
      cvMatMul(colvec, nu, matrix);
    }
    else
      cvMul(nu, nu, matrix);
    if (covtype == FULLC)
    {
      cvmTranspose(nu_not_u, colvec);
      cvMatMul(colvec, nu_not_u, matrix2);
    }
    else
      cvMul(nu_not_u, nu_not_u, matrix2);
    cvAddWeighted(matrix, xi, matrix2, - xi_not_u, 0.0, matrix);
    cvAdd(B, matrix, B_not_u);

    cvSetZero(matrix);
    for (int t = 0; t < numframes; ++t)
    {
      const CvMat* f_frame = features->getFrame(_u, t, stub); // f_frame <- [1 x g]
      if (covtype == FULLC)
      {
        cvmTranspose(f_frame, colvec);
        cvMatMul(colvec, f_frame, matrix2);
      }
      else
        cvMul(f_frame, f_frame, matrix2);
      cvAdd(matrix, matrix2,  matrix);
    }
    cvAddWeighted(B_not_u, 1.0,  matrix, - gamma_t, 0.0, B_not_u);
#endif
    
    // ----------------------------------
    // Student-t 分布の確率尤度を計算
    if (covtype == FULLC)
      cvConvertScale(B_not_u, matrix, (eta_not_u + 1) / ((eta_not_u - (dimension - 1) * 0.5) * eta_not_u));
    else
      cvConvertScale(B_not_u, matrix, (xi_not_u + 1)  / (eta_not_u * xi_not_u));
    
    double logqGamma = 0;
    for (int t = 0; t < numframes; ++t)
    {
      const CvMat* f_frame = features->getFrame(_u, t, stub); // f_frame <- [1 x g]
      logqGamma += _distStudent(f_frame, nu_not_u, matrix, eta_not_u);
    }
    // _u 番目の発話のこのクラスタに対する1事後平均値を更新する
    m_logsumZ[_u]    = logqGamma;
    m_lowerlimit[_u] = 0;
    
    gmm->setNu  (j, nu_not_u);
    gmm->setB   (j, B_not_u);
    gmm->setEta (j, eta_not_u);
    gmm->setXi  (j, xi_not_u);
    gmm->setBeta(j, beta_not_u);
    /*
     if (isnan(logqGamma))
     {
     _print_matrix(B_not_u);
     exit(3);
     }
     */
    // ----------------------------------
    // Gamma の対数変分事後期待値を算出
    //    iter_Zut     = (iter_Z->at(_u)).begin();
    //      iter_logsumZtp = logsumZtp.begin();
  } /* iterate # of mixtures */
  
  // ----------------------------------------------
  /* 正規化 と free energy の算出*/
#if 0
  (*iter_fe)        = 0.0;
  (*iter_mlogsumZt) = 0.0;
  iter_Z = m_Z.begin();
  for (int i = 0; iter_Z != m_Z.end(); ++iter_Z, ++i)
  { /* iterate for # of mixtures */
    iter_Ztp       = (iter_Z->at(t)).begin();
    iter_logsumZtp = logsumZtp.begin();
    for (; iter_Ztp != (iter_Z->at(t)).end(); ++iter_Ztp, ++iter_logsumZtp)
    { /* iterate # of frames times */
      if (i==0) (*iter_mlogsumZt) += (*iter_logsumZtp);
      
      double n_gamma = exp((*iter_Ztp) - (*iter_logsumZtp));
      if (n_gamma < 1.0E-3) n_gamma = 0.0;
      if (n_gamma > 0.999)  n_gamma = 1.0;
      
      (*iter_fe) += (n_gamma == 0.0) ? 0.0: ( n_gamma * ((*iter_Ztp) + log(n_gamma)));
      (*iter_Ztp) = n_gamma;
    }
  }
#endif
  if (matrix)   cvReleaseMat(&matrix);
  if (matrix2)  cvReleaseMat(&matrix2);
  if (rowvec)   cvReleaseMat(&rowvec);
  if (colvec)   cvReleaseMat(&colvec);
  if (B_not_u)  cvReleaseMat(&B_not_u);
  if (nu_not_u) cvReleaseMat(&nu_not_u);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------


void CFoLSVBGmmTrainer::update(DoubleVect _X, const int _u)
{
  CVBGmm *gmm        = getGmm();
  CSegment* features = getFeatures();
  CvMat stub;
  const int covtype      = getCovType();
  const int dimension    = getDimension();
  
  CvMat* matrix  = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* rowvec  = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec  = cvCreateMat(dimension, 1, CV_32F);
  
  CvMat* B_not_u  = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* nu_not_u = cvCreateMat(1, dimension, CV_32F);
  
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Zut;
  
  DoubleVect::iterator iter_X = _X.begin(); // 1 ~ # of segments
  
  cvCopy(gmm->getNu0(0), nu_not_u);
  cvCopy(gmm->getB0(0),  B_not_u);
  cvMul(nu_not_u, nu_not_u, matrix);
  cvAddWeighted(B_not_u, 1.0, matrix, gmm->getXi0(0), 0.0, B_not_u);
  
  double eta_not_u = 0.0;
  double xi_not_u  = 0.0;
  for (int u = 0; iter_X != _X.end(); ++iter_X, ++u)
  {
    // _u 番目の発話に起因する統計量を取り除いてハイパーパラメタを再計算する（nu_{/u}）
    if (u==_u) continue;
    const int numframes = features->getNumFrames(u);
    double gamma_t = (*iter_X);
    cvReduce(features->getSegment(u), rowvec);
    cvAddWeighted(nu_not_u, 1.0, rowvec, gamma_t , 0.0, nu_not_u);
    
    cvSetZero(matrix);
    for (int t = 0; t < numframes; ++t){
      const CvMat* f_frame = features->getFrame(u, t, stub); // f_frame <- [1 x g]
      cvMul(f_frame, f_frame, matrix2);
      cvAdd(matrix, matrix2,  matrix);
    }
    cvAddWeighted(B_not_u, 1.0,  matrix, gamma_t, 0.0, B_not_u);
    
    xi_not_u  += gamma_t * numframes;
    eta_not_u += gamma_t * numframes;
  }
  cvConvertScale(nu_not_u, nu_not_u, 1.0 / xi_not_u);
  cvMul(nu_not_u, nu_not_u, matrix);
  cvAddWeighted(B_not_u, 1.0, matrix, - xi_not_u, 0.0, B_not_u);
  
  
  
  gmm->setNu  (0, nu_not_u);
  gmm->setB   (0, B_not_u);
  gmm->setEta (0, eta_not_u);
  gmm->setXi  (0, xi_not_u);
  //gmm->setBeta(j, beta_not_u);
  /*
   if (isnan(logqGamma))
   {
   _print_matrix(B_not_u);
   exit(3);
   }
   */
  // ----------------------------------
  // Gamma の対数変分事後期待値を算出
  //    iter_Zut     = (iter_Z->at(_u)).begin();
  //      iter_logsumZtp = logsumZtp.begin();
  
  // ----------------------------------------------
  
  if (matrix)   cvReleaseMat(&matrix);
  if (matrix2)  cvReleaseMat(&matrix2);
  if (rowvec)   cvReleaseMat(&rowvec);
  if (colvec)   cvReleaseMat(&colvec);
  if (B_not_u)  cvReleaseMat(&B_not_u);
  if (nu_not_u) cvReleaseMat(&nu_not_u);
}



void CFoLSVBGmmTrainer::updateZ(DoubleVect _X, const int _u)
{
  CVBGmm *gmm        = getGmm();
  CSegment* features = getFeatures();
  CvMat stub;
  
  const int covtype      = getCovType();
  const int dimension    = getDimension();
  
  CvMat* matrix  = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* rowvec  = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec  = cvCreateMat(dimension, 1, CV_32F);
  
  CvMat* B_not_u  = (covtype == FULLC) ?
  cvCreateMat(dimension, dimension, CV_32F) : /* full */
  cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* nu_not_u = cvCreateMat(1, dimension, CV_32F);
  
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Zut;
  
  
  
  cvCopy(gmm->getNu0(0), nu_not_u);
  cvConvertScale(nu_not_u, nu_not_u, gmm->getXi0(0));
  cvCopy(gmm->getB0(0),  B_not_u);
  cvMul(nu_not_u, nu_not_u, matrix);
  cvAddWeighted(B_not_u, 1.0, matrix, gmm->getXi0(0), 0.0, B_not_u);
  
  double eta_not_u = 0.0;
  double xi_not_u  = 0.0;
  DoubleVect::iterator iter_X = _X.begin(); // 1 ~ # of segments
  for (int u = 0; iter_X != _X.end(); ++iter_X, ++u)
  {
    // _u 番目の発話に起因する統計量を取り除いてハイパーパラメタを再計算する（nu_{/u}）
    if (u==_u) continue;
    const int numframes = features->getNumFrames(u);
    double gamma_t = (*iter_X);
    cvReduce(features->getSegment(u), rowvec);
    cvAddWeighted(nu_not_u, 1.0, rowvec, gamma_t , 0.0, nu_not_u);
    
    cvSetZero(matrix);
    for (int t = 0; t < numframes; ++t){
      const CvMat* f_frame = features->getFrame(u, t, stub); // f_frame <- [1 x g]
      cvMul(f_frame, f_frame, matrix2);
      cvAdd(matrix, matrix2,  matrix);
    }
    cvAddWeighted(B_not_u, 1.0,  matrix, gamma_t, 0.0, B_not_u);
    
    xi_not_u  += gamma_t * numframes;
    eta_not_u += gamma_t * numframes;
  }
  cvConvertScale(nu_not_u, nu_not_u, 1.0 / xi_not_u);
  cvMul(nu_not_u, nu_not_u, matrix);
  cvAddWeighted(B_not_u, 1.0, matrix, - xi_not_u, 0.0, B_not_u);
  
  // ----------------------------------
  // Student-t 分布の確率尤度を計算
  if (covtype == FULLC)
    cvConvertScale(B_not_u, matrix, (eta_not_u + 1) / ((eta_not_u - (dimension - 1) * 0.5) * eta_not_u), 0 );
  else
    cvConvertScale(B_not_u, matrix, (xi_not_u + 1)  / (eta_not_u * xi_not_u), 0 );
  
  const int numframes = features->getNumFrames(_u);
  double logqGamma = 0;
  for (int t = 0; t < numframes; ++t)
  {
    const CvMat* f_frame = features->getFrame(_u, t, stub); // f_frame <- [1 x g]
    logqGamma += _distStudent(f_frame, nu_not_u, matrix, eta_not_u);
  }
  
  
  m_logsumZ[_u] = logqGamma;
  
  
  gmm->setNu  (0, nu_not_u);
  gmm->setB   (0, B_not_u);
  gmm->setEta (0, eta_not_u);
  gmm->setXi  (0, xi_not_u);
  //gmm->setBeta(j, beta_not_u);
  /*
   if (isnan(logqGamma))
   {
   _print_matrix(B_not_u);
   exit(3);
   }
   */
  // ----------------------------------
  // Gamma の対数変分事後期待値を算出
  //    iter_Zut     = (iter_Z->at(_u)).begin();
  //      iter_logsumZtp = logsumZtp.begin();
  
  // ----------------------------------------------
  
  if (matrix)   cvReleaseMat(&matrix);
  if (matrix2)  cvReleaseMat(&matrix2);
  if (rowvec)   cvReleaseMat(&rowvec);
  if (colvec)   cvReleaseMat(&colvec);
  if (B_not_u)  cvReleaseMat(&B_not_u);
  if (nu_not_u) cvReleaseMat(&nu_not_u);
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

