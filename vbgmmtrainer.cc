#include <vector>
#include <list>
#include <algorithm>

#include "vbgmmtrainer.h"

using namespace std;

const int CVBGmmTrainer::INIT_Z_RANDOM    = 16;
const int CVBGmmTrainer::INIT_Z_KMEANS    = 32;
const int CVBGmmTrainer::INIT_Z_MANUAL    = 64;
const int CVBGmmTrainer::INIT_Z_PARAM     = 128;

//----------------------------------------------------------------------

void CVBGmmTrainer::malloc(const int _dimension, const int _num_segments, IntVect& _num_frames)
{
  m_dimension = _dimension;
  //     m_X.resize(_num_segments);
  
  m_lowerlimit.resize(_num_segments, 0.0);
  m_KLDivergence.resize(m_num_mixtures, 0.0);
  
  CvMatVect::iterator iter;
  /* 一次統計量 */
  for (iter = m_acc_O.begin(); iter != m_acc_O.end(); iter++)
    (*iter) = cvCreateMat(1, _dimension, CV_32F);
    
  /* ニ次統計量 */
  for (iter = m_acc_C.begin(); iter != m_acc_C.end(); ++iter)
    (*iter) = cvCreateMat(1, _dimension, CV_32F);
      
  /* フレームレベル潜在変数 */
  vector< vector< DoubleVect > >::iterator iter_z = m_Z.begin();
  for (; iter_z != m_Z.end(); ++iter_z)
  { /* num of mixtures */
    (*iter_z).resize(_num_segments);
        
     IntVect::iterator iter_nf = _num_frames.begin();
     vector <DoubleVect>::iterator iter_zp = (*iter_z).begin();
    for (; iter_zp != (*iter_z).end(); ++iter_zp, ++iter_nf)
     /* num of segments */
       (*iter_zp).resize((*iter_nf), 0);
  }
  m_logsumZ.resize(_num_segments);
#if !DEBUG
  vector <DoubleVect>::iterator iter_logsumZ = m_logsumZ.begin();
  IntVect::iterator iter_nf               = _num_frames.begin();
  for (; iter_logsumZ != m_logsumZ.end(); ++iter_logsumZ, ++iter_nf)
  /* num of segments */
    (*iter_logsumZ).resize((*iter_nf), 0);
#endif
}

//----------------------------------------------------------------------

void CVBGmmTrainer::free(void)
{
  delete m_gmm;
  
  CvMatVect::iterator iter;
  
  iter = m_acc_O.begin();
  for (; iter != m_acc_O.end(); ++iter)
    if (*iter) cvReleaseMat(&(*iter));
  
  iter = m_acc_C.begin();
  for (; iter != m_acc_C.end(); ++iter)
    if (*iter) cvReleaseMat(&(*iter));
}

//----------------------------------------------------------------------

void CVBGmmTrainer::init(const int _dimension,
                         const int _num_segments,
                         IntVect& _num_frames)
{
  malloc(_dimension, _num_segments, _num_frames);
}

//----------------------------------------------------------------------

void CVBGmmTrainer::setFeatures(CSegment* _features)
{ 
  m_features = _features;
}

//----------------------------------------------------------------------

void CVBGmmTrainer::initGlobalParam(const char* _model_filename,
                                    const double& _beta0,
                                    const double& _xi0,
                                    const double& _eta0)
{
  // 平均と共分散行列の超パラメタはモデルファイルから与え，スカラー値の超パラメタのみ明示的に与える
//  if (m_gmm) delete m_gmm;
//  m_gmm = new CVBGmm();
  
  m_gmm->setCovType(m_covtype);
  m_gmm->setDimension(m_dimension);
  m_gmm->readFromHtk(_model_filename, _beta0, _xi0, _eta0);

}

//----------------------------------------------------------------------

void CVBGmmTrainer::initGlobalParam(const double& _beta0,
                                    const double& _xi0,
                                    CvMat* _nu0,
                                    const double& _eta0,
                                    CvMat* _B0)
{
  // 全ての超パラメタを明示的に与える
  m_gmm->setCovType(m_covtype);
  m_gmm->setDimension(m_dimension);
  m_gmm->setGlobalParam(_beta0, _xi0, _nu0, _eta0, _B0);
}

//----------------------------------------------------------------------

void CVBGmmTrainer::initGlobalParam(CSegment* _features,
                                    const double& _beta0, const double& _xi0,
                                    const double& _eta0,  const double& _B0)
{
  // 平均と共分散行列の超パラメタはデータから推定し，スカラー値の超パラメタのみ明示的に与える
  const int num_segments  = _features->getNumSegments();
  const int num_allframes = _features->getNumAllFrames();
  const int dimension     = m_dimension;

  CvMat* rowvec = cvCreateMat(1, dimension, CV_32F);
  CvMat* nu0 = cvCreateMat(1, dimension, CV_32F);
  CvMat* B0  = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) :   /* full */
    cvCreateMat(1, dimension, CV_32F);            /* diagonal */

  /* Nu0 の初期化 : 初期セグメントに含まれる全フレームの平均ベクトル */
  cvSetZero(nu0);
  for (int i = 0; i < num_segments; ++i)
  {
    cvReduce(_features->getSegment(i), rowvec, CV_REDUCE_SUM);
    cvAdd(nu0, rowvec, nu0);
  }
  cvConvertScale(nu0, nu0, 1.0 / static_cast<double>(num_allframes));

  /* B0 の初期化 : 初期セグメントに含まれる全フレームの分散行列 */
  cvSetZero(B0);
  cvSet( B0, cvScalar(_B0));
  initGlobalParam(_beta0, _xi0, nu0, _eta0, B0);  

  cvReleaseMat(&rowvec);
  cvReleaseMat(&nu0);
  cvReleaseMat(&B0);
}

//----------------------------------------------------------------------

void CVBGmmTrainer::initGlobalParam(CSegment* _features,
                                    const double& _beta0, const double& _xi0,
                                    const double& _eta0)
{
  // 平均と共分散行列の超パラメタはデータから推定し，スカラー値の超パラメタのみ明示的に与える
  const int num_segments  = _features->getNumSegments();
  const int num_allframes = _features->getNumAllFrames();
  const int dimension     = m_dimension;

  double eta0;
  double xi0;
  int ave_numframes = 0;

  CvMat* rowvec = cvCreateMat(1, dimension, CV_32F);
  CvMat* nu0 = cvCreateMat(1, dimension, CV_32F);
  CvMat* B0  = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) :   /* full */
    cvCreateMat(1, dimension, CV_32F);            /* diagonal */
  cvSetZero(nu0);
  cvSetZero(B0);
  
#if 0
  /* Nu0 の初期化 : 初期セグメントに含まれる全フレームの平均ベクトル */

  for (int i = 0; i < num_segments; ++i)
  {
    cvReduce(_features->getSegment(i), rowvec, CV_REDUCE_SUM);
    cvAdd(nu0, rowvec, nu0);
  }
  cvConvertScale(nu0, nu0, 1.0 / static_cast<double>(num_allframes));
  
  /* B0 の初期化 : 初期セグメントに含まれる全フレームの分散行列 */


  CvMat* allframe_mat = cvCreateMat(num_allframes, dimension, CV_32F);
  for (int t = 0; t < num_segments; ++t)
  {
    int num_frames = _features->getNumFrames(t);
    for (int f = 0; f < num_frames; ++f)
    {
      for (int d = 0; d < dimension; ++d)
        cvmSet(allframe_mat, f, d, cvmGet(_features->getFrame(t, f, m_stub), 0, d));
      ave_numframes += num_frames;
    }
  }
  ave_numframes /= num_segments;  
  // 共分散行列の算出
  _getCovMat(allframe_mat, nu0, B0, m_covtype);
//  _getCovMat(_features[0]->getSegment(), nu0, B0, m_covtype);

  eta0 = _eta0;// * num_allframes;
  xi0  = _xi0;// * num_allframes;

  cvConvertScale(B0, B0, eta0);
  
  cvReleaseMat(&allframe_mat);
#else
  
  CvMat* rowvec2 = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec = (m_covtype == FULLC) ?
    cvCreateMat(dimension, 1, CV_32F) :   /* full */
    NULL;                                 /* diagonal */
  CvMat* matrix = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */

  /* 全 frame データの一次統計量を算出 */
  for (int u = 0; u < num_segments; ++u)
  {
    int numframes   = _features->getNumFrames(u);
    /* 発話内平均ベクトルの算出 */
    cvReduce(_features->getSegment(u), rowvec, CV_REDUCE_SUM);
    cvAdd(nu0, rowvec, nu0);
    cvConvertScale(rowvec, rowvec, 1.0 / static_cast<double>(numframes));
    
    /* x 発話内共分散行列の算出 */
    /* o 発話内二次統計量の算出 */
    cvSetZero(matrix2);
    for (int t = 0; t < numframes; ++t)
    { /* iterate # of frames times */
      const CvMat* feature_f = _features->getFrame(u, t, m_stub);
      cvSub(feature_f, rowvec, rowvec2);
      if (m_covtype == FULLC)
      {
        cvmTranspose(rowvec2, colvec);
        cvMatMul(colvec, rowvec2, matrix);
      }
      else
        cvMul(rowvec2, rowvec2, matrix);
      cvAdd(matrix2, matrix, matrix2);
    }
    cvConvertScale(matrix2, matrix2, 1.0 / static_cast<double>(numframes)); //発話内共分散行列
    cvAdd(matrix2, B0, B0);
    ave_numframes += numframes;
  }

  /* 発話毎に算出した平均ベクトルについて，全発話の平均値を算出し，これを平均ベクトルの事前分布平均値とする */
  cvConvertScale(nu0, (nu0), 1.0 / static_cast<double>(num_allframes));
  

  /* 発話毎に算出した共分散行列について，全発話の平均値を算出し，これを共分散行列の事前分布平均値とする */
  //  cvConvertScale(B0, B0, 1.0 / static_cast<double>(num_segments));



  ave_numframes /= num_segments;  
  eta0 = _eta0;
  xi0  = _xi0;
//  eta0 = _eta0 * ave_numframes; // 共分散行列の事前平均値は，発話内平均フレーム数に依存して設定する．
//  xi0  = _xi0  * ave_numframes;          // 平均ベクトルの事前平均値の分散は，事前平均値が各発話の平均値とれるように出来るだけ大きく設定する

  cvConvertScale(B0, B0,  eta0);
  
  
  if (colvec) cvReleaseMat(&colvec);
  if (rowvec2) cvReleaseMat(&rowvec2);
  if (matrix)  cvReleaseMat(&matrix);
  if (matrix2) cvReleaseMat(&matrix2);
#endif

  _print_matrix(B0);//exit(3);
  initGlobalParam(_beta0, xi0, nu0, eta0, B0);

  cvReleaseMat(&rowvec);
  cvReleaseMat(&nu0);
  cvReleaseMat(&B0);

}

void CVBGmmTrainer::initZ(const int _initOpt)
{
  // フレームレベル潜在変数の対数変分事後期待値として全混合要素で一律の重みを与える
  //double weight = 1.0 / static_cast<double>(m_num_mixtures);

  //  ランダムに初期値を決める (s.t. [0, 1], sum(Z) = 1.0 )
  if (_initOpt & INIT_Z_PARAM)
  {
    updateZ();
  }
  else if (_initOpt & INIT_Z_RANDOM)
  {
    vector<DoubleVect> sumrnd;
    vector<vector <DoubleVect> >::iterator iter_z;
    vector<DoubleVect>::iterator iter_zt;
    DoubleVect::iterator iter_ztp;
    
    vector<DoubleVect>::iterator iter_sumrnd;
    DoubleVect::iterator         iter_sumrndtp;
    
    iter_zt =  m_Z.begin()->begin();
    int num_segments = m_Z[0].size();
    
    for (int t = 0; t < num_segments; ++t)
    { /* num of segments */
      int num_frames = m_Z[0][t].size();
      sumrnd.push_back(DoubleVect(num_frames, 0.0));
    }
    
    iter_z = m_Z.begin();
    for (; iter_z != m_Z.end(); ++iter_z)
    { /* num of mixtures */
      iter_zt = iter_z->begin();
      iter_sumrnd = sumrnd.begin();
      for (; iter_zt != iter_z->end(); ++iter_zt, ++iter_sumrnd)
      { /* num of segments */
        iter_ztp      = iter_zt->begin();
        iter_sumrndtp = iter_sumrnd->begin();
        for (; iter_ztp != iter_zt->end(); ++iter_ztp, ++iter_sumrndtp)
        { /* num of frames */
          double weight = rand() / static_cast<double>(RAND_MAX);
          (*iter_ztp) = weight;
          (*iter_sumrndtp) += weight;
        }
      }
    }
    
    /* 正規化 */
    DoubleVect::iterator iter_fe = m_lowerlimit.begin();
    for (; iter_fe != m_lowerlimit.end(); ++iter_fe)
      (*iter_fe) = 0.0;
    
    iter_z = m_Z.begin();
    for (; iter_z != m_Z.end(); ++iter_z)
    { /* num of mixtures */
      iter_zt     = iter_z->begin();
      iter_sumrnd = sumrnd.begin();
      iter_fe     = m_lowerlimit.begin();
      for (; iter_zt != iter_z->end(); ++iter_zt, ++iter_sumrnd, ++iter_fe)
      { /* num of segments */
        double value = 0.0;
        iter_ztp      = iter_zt->begin();
        iter_sumrndtp = iter_sumrnd->begin();
        for (; iter_ztp != iter_zt->end(); ++iter_ztp, ++iter_sumrndtp)
        {  /* num of frames */
          double n_gamma = (*iter_ztp) / (*iter_sumrndtp);
          
          value += (n_gamma==0.0)? 0.0:
          n_gamma * ((*iter_ztp) - log(n_gamma));
          (*iter_ztp) /= (*iter_sumrndtp);
        }
        (*iter_fe) += value;
      }
    }
    
    /* logsumZ の初期化 */
#if !DEBUG
    vector<DoubleVect>::iterator iter_logsumZ;
    DoubleVect::iterator iter_logsumZtp;
    
    iter_logsumZ = m_logsumZ.begin();
    iter_sumrnd  = sumrnd.begin();
    for (; iter_logsumZ != m_logsumZ.end(); ++iter_logsumZ)
    { /* num of segments */
      iter_logsumZtp = iter_logsumZ->begin();
      iter_sumrndtp  = iter_sumrnd->begin();
      for (; iter_logsumZtp != iter_logsumZ->end(); ++iter_logsumZtp)
      {      /* num of frames */
        (*iter_logsumZtp) = (*iter_sumrndtp);
      }
    }
#endif
  }
}

//----------------------------------------------------------------------

void CVBGmmTrainer::initParam(DoubleVect& __X)
{
  update(__X);
}

//----------------------------------------------------------------------

void CVBGmmTrainer::initParam()
{
  for (int i = 0; i < m_num_mixtures; ++i)
  {
    m_gmm->setBeta(i, m_gmm->getBeta0(i));
    m_gmm->setEta(i,  m_gmm->getEta0(i));
    m_gmm->setXi(i,   m_gmm->getXi0(i));
    m_gmm->setNu(i,   m_gmm->getNu0(i));
    m_gmm->setB(i,    m_gmm->getB0(i));
  }
}

//----------------------------------------------------------------------

void CVBGmmTrainer::updateZ(void)
{
#if !DEBUG
  const int covtype      = m_covtype;
  const int dimension    = m_dimension;
  
  CvMat* matrix1 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    NULL;                                           /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    NULL;                                           /* diagonal */
  CvMat* rowvec = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec = (covtype == FULLC) ?
    cvCreateMat(dimension, 1, CV_32F) : /* full */
    NULL;                                 /* diagonal */
  CvMat* scalar = (covtype == FULLC) ?
    cvCreateMat(1, 1, CV_32F) : /* full */
    NULL;                       /* diagonal */
  
  CvMat* Gamma = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  
  double sumBeta = m_gmm->addBeta(); /* 全混合要素に対する beta の和 */
  double glog2 = dimension * log(2);
  
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Ztp;
  vector<DoubleVect>::iterator iter_logsumZt;
  DoubleVect::iterator iter_logsumZtp;
  
  bool first_loop = true;
  
  iter_Z = m_Z.begin();
  
  for (int i = 0; iter_Z != m_Z.end(); ++iter_Z, ++i)
  { /* iterate # of mixtures times */
    //   cout <<"i: "<< i <<endl;
    double logqGamma = 0.0; // Gamma の対数変分事後期待値
    double mahala     = 0.0; // マハラノビス距離項
    
    cvConvertScale(m_gmm->getInvertB(i), Gamma, m_gmm->getEta(i)); // B の期待値
    
    CvMat* nu      = m_gmm->getNu(i);
    double eta     = m_gmm->getEta(i);
    double xi      = m_gmm->getXi(i);
    double logdetB = m_gmm->getLogDetB(i);
    double beta    = m_gmm->getBeta(i);
    
    double logqBeta = digamma(beta) - digamma(sumBeta);
    // ----------------------------------
    // Gamma の対数変分事後期待値を算出
    double Psi = 0.0; // digamma 項
    if (m_covtype == FULLC)
      for (int d = 1; d <= m_dimension; ++d)
        Psi +=
        digamma((eta + 1.0 -
                 static_cast<double>(d)) / 2.0);
    else
      Psi = dimension * digamma(eta / 2.0);
    
    if (m_covtype == FULLC)
      logqGamma = Psi - logdetB + glog2;
    else
      logqGamma = Psi - logdetB;
    
    iter_Zt       = iter_Z->begin();
    iter_logsumZt = m_logsumZ.begin();
    
    for (int t = 0; iter_Zt != iter_Z->end(); ++t, ++iter_Zt, ++iter_logsumZt)
    { /* iterate # of segments times */
      iter_Ztp       = iter_Zt->begin();
      iter_logsumZtp = iter_logsumZt->begin();
      for (int p = 0; iter_Ztp != iter_Zt->end(); ++iter_Ztp, ++iter_logsumZtp, ++p)
      { /* iterate # of frames times */
        // ----------------------------------
        // マハラノビス距離項を算出
        const CvMat* f_frame = m_features->getFrame(t, p, m_stub); // f_frame <- [1 x g]
        cvSub(f_frame, nu, rowvec);
        if (m_covtype == FULLC)
        {
          cvmTranspose(rowvec, colvec);
          cvMatMul(rowvec, Gamma, rowvec);
          cvMatMul(rowvec, colvec, scalar);
          mahala = cvmGet(scalar, 0, 0);
        }
        else
        {
          cvMul(rowvec, rowvec, rowvec);
          cvMul(rowvec, Gamma, rowvec);
          mahala = cvSum(rowvec).val[0];
        }
        
        // --------------------------------------------
        // フレームレベル潜在変数の対数変分事後期待値を算出
        double gamma = logqBeta +
          0.5 * logqGamma -
          0.5 * mahala -
          0.5 * dimension / xi;
        (*iter_Ztp) = gamma;
        
        (*iter_logsumZtp) = first_loop ? gamma: LAddS(gamma, (*iter_logsumZtp));
      }  /* iterate # of frames times */
    } /* iterate # of segments times */
    
    /*
     cout << "meta: "<<eta <<endl;
     cout << "logdetB: "<<logdetB <<endl;
     cout << "beta: "<<beta <<endl;
     cout << "logqBeta: "<<logqBeta <<endl;
     cout << "logqLambda: "<<logqLambda <<endl;
     cout << "mahala: " <<mahala <<endl;
     */
    if (first_loop) first_loop = false;
  } /* iterate # of mixtures times */
  
  
  // ----------------------------------------------
  /* 正規化 と free energy の算出*/
  DoubleVect::iterator iter_fe = m_lowerlimit.begin();
  for (; iter_fe != m_lowerlimit.end(); ++iter_fe)
    (*iter_fe) = 0.0;
  
  iter_Z    = m_Z.begin();
  for (; iter_Z != m_Z.end(); ++iter_Z)
  { /* iterate # of mixtures times */
    iter_Zt       = iter_Z->begin();
    iter_logsumZt = m_logsumZ.begin();
    iter_fe       = m_lowerlimit.begin();
    for (; iter_Zt != iter_Z->end(); ++iter_Zt, ++iter_logsumZt, ++iter_fe)
    { /* iterate # of segments times */
      double value = 0.0;
      iter_Ztp       = iter_Zt->begin();
      iter_logsumZtp = iter_logsumZt->begin();
      for (; iter_Ztp != (*iter_Zt).end(); ++iter_Ztp, ++iter_logsumZtp)
      { /* iterate # of frames times */
        double n_gamma = exp((*iter_Ztp) - (*iter_logsumZtp));
        if (n_gamma < 1.0E-3) n_gamma = 0.0;
        if (n_gamma > 0.999)  n_gamma = 1.0;
        
        value += (n_gamma == 0.0) ? 0.0:
        ( n_gamma * ((*iter_Ztp) + log(n_gamma)));
        (*iter_Ztp) = n_gamma;
        
      }
      (*iter_fe) += value;
    }
  }
  
  if (matrix1) cvReleaseMat(&matrix1);
  if (matrix2) cvReleaseMat(&matrix2);
  if (rowvec)  cvReleaseMat(&rowvec);
  if (colvec)  cvReleaseMat(&colvec);
  if (scalar)  cvReleaseMat(&scalar);
  if (Gamma) cvReleaseMat(&Gamma);
  
#else
  const int covtype      = m_covtype;
  const int dimension    = m_dimension;
  const int numsegments  = m_features->getNumSegments();
  
  CvMat* matrix1 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    NULL;                                       /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    NULL;                                       /* diagonal */
  CvMat* rowvec = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec = (covtype == FULLC) ?
    cvCreateMat(dimension, 1, CV_32F) : /* full */
    NULL;                               /* diagonal */
  CvMat* scalar = (covtype == FULLC) ?
    cvCreateMat(1, 1, CV_32F) : /* full */
    NULL;                       /* diagonal */
  
  CvMat* Gamma = (m_covtype == FULLC) ?
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  
  double sumBeta = m_gmm->addBeta(); /* 全混合要素に対する beta の和 */
  double glog2 = dimension * log(2);
  
  
  vector<vector <DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Ztp;
  
  m_logsumZ.resize(numsegments, 0);

  DoubleVect::iterator iter_mlogsumZt = m_logsumZ.begin();
  DoubleVect::iterator iter_fe       = m_lowerlimit.begin();
  
  for (int t = 0; iter_mlogsumZt != m_logsumZ.end(); ++iter_mlogsumZt, ++iter_fe, ++t)
  { /* iterate # of segments times */

    int numframes = m_features->getNumFrames(t);
    DoubleVect logsumZtp(numframes, -DBL_MAX); // # of frames
    DoubleVect::iterator iter_logsumZtp;
    
    iter_Z = m_Z.begin();
    for (int i = 0; iter_Z != m_Z.end(); ++iter_Z, ++i)
    { /* iterate for # of mixtures */
      double logqGamma = 0.0; // Gamma の対数変分事後期待値
      double mahala    = 0.0; // マハラノビス距離項
      
      cvConvertScale(m_gmm->getInvertB(i), Gamma, m_gmm->getEta(i)); // B の期待値の算出
      
      CvMat* nu      = m_gmm->getNu(i);
      double eta     = m_gmm->getEta(i);
      double xi      = m_gmm->getXi(i);
      double logdetB = m_gmm->getLogDetB(i);
      double beta    = m_gmm->getBeta(i);
      
      double logqBeta = digamma(beta) - digamma(sumBeta);
      // ----------------------------------
      // Gamma の対数変分事後期待値を算出
      double Psi = 0.0; // digamma 項
      if (m_covtype == FULLC)
        for (int d = 1; d <= m_dimension; ++d)
          Psi += digamma((eta + 1.0 - static_cast<double>(d)) / 2.0);
      else
        Psi = dimension * digamma(eta / 2.0);
      
      if (m_covtype == FULLC)
        logqGamma = Psi - logdetB + glog2;
      else
        logqGamma = Psi - logdetB;
      
      iter_Ztp      = (iter_Z->at(t)).begin();
      iter_logsumZtp = logsumZtp.begin();
      
      for (int p = 0; iter_Ztp != (iter_Z->at(t)).end(); ++iter_Ztp, ++iter_logsumZtp, ++p)
      { /* iterate # of frames times */
        // ----------------------------------
        // マハラノビス距離項を算出
        const CvMat* f_frame = m_features->getFrame(t, p, m_stub); // f_frame <- [1 x g]
        cvSub(f_frame, nu, rowvec);
        if (m_covtype == FULLC)
        {
          cvmTranspose(rowvec, colvec);
          cvMatMul(rowvec, Gamma, rowvec);
          cvMatMul(rowvec, colvec, scalar);
          mahala = cvmGet(scalar, 0, 0);
        }
        else
        {
          cvMul(rowvec, rowvec, rowvec);
          cvMul(rowvec, Gamma, rowvec);
          mahala = cvSum(rowvec).val[0];
        }
        
        // --------------------------------------------
        // フレームレベル潜在変数の対数変分事後期待値を算出
        double gamma = logqBeta +
                        0.5 * logqGamma -
                        0.5 * mahala -
                        0.5 * dimension / xi;
        (*iter_Ztp) = gamma;
        
        (*iter_logsumZtp) = LAddS(gamma, *iter_logsumZtp);
        
      }  /* iterate # of frames times */
    } /* iterate # of mixtures */
    
    // ----------------------------------------------
    /* 正規化 と free energy の算出*/
    (*iter_fe)        = 0.0;
    (*iter_mlogsumZt) = 0.0;
    iter_Z = m_Z.begin();
    for (int i = 0; iter_Z != m_Z.end(); ++iter_Z, ++i)
    { /* iterate for # of mixtures */
      iter_Ztp      = (iter_Z->at(t)).begin();
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

/*
        cout << "meta: "<<eta <<endl;
    cout << "logdetB: "<<logdetB <<endl;
    cout << "beta: "<<beta <<endl;
    cout << "logqBeta: "<<logqBeta <<endl;
    cout << "logqLambda: "<<logqLambda <<endl;
    cout << "mahala: " <<mahala <<endl;
*/
    
  } /* iterate # of segments */
  
  if (matrix1) cvReleaseMat(&matrix1);
  if (matrix2) cvReleaseMat(&matrix2);
  if (rowvec)  cvReleaseMat(&rowvec);
  if (colvec)  cvReleaseMat(&colvec);
  if (scalar)  cvReleaseMat(&scalar);
  if (Gamma)   cvReleaseMat(&Gamma);
#endif

}

//----------------------------------------------------------------------

double CVBGmmTrainer::update(DoubleVect& __X)
{
  /* 統計量の更新 */
  const int covtype   = m_covtype;
  const int dimension = m_dimension;

  CvMat* rowvec = cvCreateMat(1, dimension, CV_32F);
  CvMat* colvec = (m_covtype == FULLC) ? 
    cvCreateMat(dimension, 1, CV_32F) :   /* full */
    NULL;                                 /* diagonal */
  CvMat* matrix1 = (m_covtype == FULLC) ? 
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (m_covtype == FULLC) ? 
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  
  vector<vector<DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zt;
  DoubleVect::iterator iter_Ztp;
  DoubleVect::iterator iter_X;
  DoubleVect::iterator iter_N;
  CvMatVect::iterator  iter_O;
  CvMatVect::iterator  iter_C;
  
  double dif = 0.0;

  iter_Z = m_Z.begin();
  
  iter_N = m_acc_N.begin();
  iter_O = m_acc_O.begin();
  iter_C = m_acc_C.begin();
//  cout << "======="<<endl;
//  for (int i=0;i<m_num_mixtures;++i)
//    cout << m_KLDivergence[i]<<endl;
//  cout << "->"<<endl;
  for (int i = 0; iter_Z != m_Z.end(); ++iter_Z, ++iter_N, ++iter_O, ++iter_C, i++)
  { /* iterate # of mixtures times */
    /* EM カウントと一次統計量の更新 */
    double Nij = 0.0;
    double Nj  = 0.0;
    
    cvSetZero(*iter_O);
    // ---------------------------------------------
    iter_X  = __X.begin();
    iter_Zt = iter_Z->begin();

    double past_acc_N = (*iter_N);
    for (int t = 0; iter_Zt != iter_Z->end(); ++iter_X, ++iter_Zt, ++t)
    { /* iterate for # of segments: t = 1, ..., T*/
      cvSetZero(rowvec);
      Nj += (*iter_X) * m_features->getNumFrames(t);
      
      // ---------------------------------------------
      iter_Ztp = iter_Zt->begin();
      double sumZt = 0.0;
      for (int p = 0; iter_Ztp != iter_Zt->end(); ++iter_Ztp, ++p)
      { /* iterate # of frames times: p = 1, ..., D*/
        const CvMat* feature_f = m_features->getFrame(t, p, m_stub);
        cvAddWeighted(feature_f, (*iter_Ztp), rowvec, 1.0, 0.0, rowvec);
        sumZt += (*iter_Ztp);
      }
      // ---------------------------------------------
      cvAddWeighted(rowvec, (*iter_X), (*iter_O), 1.0, 0.0, (*iter_O));
      Nij += (*iter_X) * sumZt;
      
    } /* end of iteration for # of segments */
    // ---------------------------------------------
#if 0
    if (Nij!=0)
      cvConvertScale((*iter_O), (*iter_O), 1.0 / Nij); // 一次統計量の初期化
#endif
    /* 二次統計量の更新 */
    /* 相関行列として定義 */
    cvSetZero(*iter_C);
    // ---------------------------------------------
    iter_X  = __X.begin();
    iter_Zt = iter_Z->begin();
    for (int t = 0; iter_Zt != iter_Z->end(); ++iter_X, ++iter_Zt, ++t)
    { /* iterate # of segments times */
      cvSetZero(matrix1);
      if(Nij == 0) continue;
      // ---------------------------------------------
      iter_Ztp = iter_Zt->begin();
      for (int p = 0; iter_Ztp != iter_Zt->end(); ++iter_Ztp, ++p)
      { /* iterate # of frames times */
        const CvMat* feature_f = m_features->getFrame(t, p, m_stub);
//        cvSub(feature_f, (*iter_O), rowvec);
        if (covtype == FULLC)
        {
          cvmTranspose(feature_f, colvec);
          cvMatMul(colvec, feature_f, matrix2);
        }
        else
          cvMul(feature_f, feature_f, matrix2);
        cvAddWeighted(matrix2, (*iter_Ztp), matrix1, 1.0, 0.0, matrix1);
      }
      // ---------------------------------------------
      cvAddWeighted(matrix1, (*iter_X), (*iter_C), 1.0, 0.0, (*iter_C));
    } /* iterate # of segments times */
    // ---------------------------------------------
    
    if (Nj != 0)
      (*iter_N) = Nij / Nj; // フレームレベル潜在変数の EM カウントの初期化
#if 0
    if (Nij != 0)
      cvConvertScale((*iter_C), (*iter_C), 1.0 / Nij); // 二次統計量の初期化
#endif
    m_gmm->calcBeta(i, Nij);
    m_gmm->calcEta(i, Nij);
    m_gmm->calcXi(i, Nij);
    m_gmm->calcNu(i, Nij, (*iter_O));
    m_gmm->calcB(i, Nij, (*iter_O), (*iter_C));
    dif += fabs(past_acc_N - (*iter_N));
    
  } /* iterate # of mixtures times */

  /* KL Divergence の更新 */
  updateKLDivergence();

//  for (int i=0;i<m_num_mixtures;++i)
//    cout << m_KLDivergence[i]<<endl;

  dif /= static_cast<double>(m_num_mixtures);
  
  if (rowvec) cvReleaseMat(&rowvec);
  if (colvec) cvReleaseMat(&colvec);
  if (matrix1) cvReleaseMat(&matrix1);
  if (matrix2) cvReleaseMat(&matrix2);

  return dif;
}

//----------------------------------------------------------------------

/* 超パラメタの KL Divergence を更新 */
void CVBGmmTrainer::updateKLDivergence(void)
{
  const int covtype      = m_covtype;
  const int dimension    = m_dimension;
  
  CvMat* matrix1 = (covtype == FULLC) ? 
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */
  CvMat* matrix2 = (covtype == FULLC) ? 
    cvCreateMat(dimension, dimension, CV_32F) : /* full */
    cvCreateMat(1, dimension, CV_32F);          /* diagonal */  

  double KLDir       = 0.0;
  double KLGaussian  = 0.0;
  double KLWishGamma = 0.0;
  KLDir = _KLDir(*(m_gmm->getBeta()), *(m_gmm->getBeta0()));

  DoubleVect::iterator iter_KL = m_KLDivergence.begin();
  for (int i = 0; iter_KL != m_KLDivergence.end(); ++iter_KL, ++i)
  {
    cvConvertScale(m_gmm->getB(i), matrix1,
                    1.0 / (m_gmm->getXi(i) * m_gmm->getEta(i)));
    cvConvertScale(m_gmm->getB(i), matrix2, 
                    1.0 / (m_gmm->getXi0(i) * m_gmm->getEta(i)));
    KLGaussian = _KLGaussian(m_gmm->getNu(i),  matrix1, 
			     m_gmm->getNu0(i), matrix2, 
			     covtype);
  
    if (covtype ==FULLC)
    {
    }
    else
    {
#if 1
      for (int d = 0; d < dimension; ++d)
      {
	//	cout << ", "<< KLWishGamma<<endl;
	KLWishGamma += 
	  _KLGamma(m_gmm->getEta(i) , cvmGet(m_gmm->getB(i),  0, d),
		   m_gmm->getEta0(i), cvmGet(m_gmm->getB0(i), 0, d));
      }
#else     
    KLWishGamma = _KLWishart(m_gmm->getEta(i),  m_gmm->getB(i),
			     m_gmm->getEta0(i), m_gmm->getB0(i), 
			     covtype);
#endif
    }
    (*iter_KL) = KLDir + KLGaussian + KLWishGamma;
  }
//  cout << KLDir<<endl;
	//   cout << KLGaussian<<endl;

  cvReleaseMat(&matrix1);
  cvReleaseMat(&matrix2);
}


//----------------------------------------------------------------------

void CVBGmmTrainer::getMAP_Z(vector<IntVect>* _segment_fcc)
{
  int num_segments = m_Z[0].size();
  //  int speaker_id = 0;
  

  vector<vector<DoubleVect> >::iterator iter_Z;
  vector<DoubleVect>::iterator iter_Zu;
  DoubleVect::iterator iter_Zut;

  _segment_fcc->resize(num_segments);
  vector<DoubleVect> maxZu(num_segments);
  
  vector<DoubleVect>::iterator iter_maxZu = maxZu.begin();
  vector<IntVect>::iterator iter_fccu     = _segment_fcc->begin();

  IntVect::iterator iter_fccut;
  DoubleVect::iterator iter_maxZut;
  
  iter_Zu = m_Z[0].begin();
  for (;iter_maxZu != maxZu.end(); ++iter_Zu, ++iter_fccu, ++iter_maxZu)
  {
    int numframes = iter_Zu->size();
    iter_fccu->resize(numframes);
    iter_maxZu->resize(numframes, -DBL_MAX);
    
  }
  
  iter_Z     = m_Z.begin();
  for (int j = 0; iter_Z != m_Z.end(); ++iter_Z, ++j)
  { /* iterate num of clusters: j = 0, ..., M */
    iter_Zu    = iter_Z->begin();
    iter_maxZu = maxZu.begin();
    iter_fccu  = _segment_fcc->begin();
    
    for (;iter_Zu != iter_Z->end(); ++iter_Zu, ++iter_maxZu, ++iter_fccu)
    {/* iterate num of segments times: t = 0, ..., T */
      iter_Zut    = iter_Zu->begin();
      iter_maxZut = iter_maxZu->begin();
      iter_fccut  = iter_fccu->begin();
      for (;iter_Zut != iter_Zu->end(); ++iter_Zut, ++iter_maxZut, ++iter_fccut)
        if ((*iter_Zut) > (*iter_maxZut))
        {
          (*iter_maxZut) = (*iter_Zut);
          (*iter_fccut)    = j;
        }
    }
  }
}

//----------------------------------------------------------------------


void CVBGmmTrainer::printZ(void)
{
  vector<vector <DoubleVect> >::iterator iter_Z = m_Z.begin();
  for (int i=0; iter_Z != m_Z.end(); ++iter_Z,i++)
  {
    cout <<"========="<<i<<"==========="<<endl;
    vector<DoubleVect>::iterator iter_Zt = iter_Z->begin();
    for (; iter_Zt != iter_Z->end(); ++iter_Zt)
    {
      _print_matrix(*iter_Zt);
    }
    cout <<endl;
  }
}

//----------------------------------------------------------------------

string CVBGmmTrainer::info(void)
{
  ostringstream os;
  os << "covtype:      " << m_covtype << endl << 
        "dimension:    " << m_dimension << endl <<
        "num_mixtures: " << m_num_mixtures<<endl;

  return os.str();
}

//----------------------------------------------------------------------

string CVBGmmTrainer::outputGMM(void)
{
  ostringstream os;
  
  os << (*m_gmm);
  
  return os.str();
}

//----------------------------------------------------------------------

string CVBGmmTrainer::outputPrior(void)
{
  ostringstream os;
  
  os << m_gmm->outputPrior();
  
  return os.str();
}

//----------------------------------------------------------------------
//                    End: vbgmmgtrainer.cc
//----------------------------------------------------------------------
