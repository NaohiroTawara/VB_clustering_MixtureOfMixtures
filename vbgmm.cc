#include "vbgmm.h"
#include "util_mat.h"
using namespace std;

//----------------------------------------------------------------------

void CVBGmm::setGlobalParam
  (DoubleVect& _beta0, DoubleVect& _xi0,
   CvMatVect& _nu0,    DoubleVect& _eta0,
   CvMatVect& _B0)
{
  DoubleVect::iterator _iter_beta0 = _beta0.begin();
  DoubleVect::iterator _iter_xi0   = _xi0.begin();
  CvMatVect::iterator  _iter_nu0   = _nu0.begin();
  DoubleVect::iterator _iter_eta0  = _eta0.begin();
  CvMatVect::iterator  _iter_B0    = _B0.begin();

  DoubleVect::iterator iter_beta0 = m_beta0.begin();
  DoubleVect::iterator iter_xi0   = m_xi0.begin();
  CvMatVect::iterator  iter_nu0   = m_nu0.begin();
  DoubleVect::iterator iter_eta0  = m_eta0.begin();
  CvMatVect::iterator  iter_B0    = m_B0.begin();

  for (;iter_beta0 != _beta0.end();
       ++iter_beta0, ++iter_xi0, ++iter_nu0, ++iter_eta0, ++iter_B0,
       ++_iter_beta0, ++_iter_xi0, ++_iter_nu0, ++_iter_eta0, ++_iter_B0)
  {
    (*iter_beta0) = (*_iter_beta0);
    (*iter_xi0)   = (*_iter_xi0);
    (*iter_nu0)   = cvCloneMat(*_iter_nu0);
    (*iter_eta0)  = (*_iter_eta0);
    (*iter_B0)    = cvCloneMat(*_iter_B0);
  }
}

//----------------------------------------------------------------------

void CVBGmm::setGlobalParam(const double& _beta0, const double& _xi0,
                            CvMat* _nu0,          const double& _eta0,
                            CvMat* _B0)
{
  DoubleVect::iterator iter_beta0 = m_beta0.begin();
  DoubleVect::iterator iter_xi0   = m_xi0.begin();
  CvMatVect::iterator  iter_nu0   = m_nu0.begin();
  DoubleVect::iterator iter_eta0  = m_eta0.begin();
  CvMatVect::iterator  iter_B0    = m_B0.begin();

  for (;iter_beta0 != m_beta0.end();
       ++iter_beta0, ++iter_xi0, ++iter_nu0, ++iter_eta0, ++iter_B0)
  {
    (*iter_beta0) = _beta0;
    (*iter_xi0)   = _xi0;
    (*iter_nu0)   = cvCloneMat(_nu0);
    (*iter_eta0)  = _eta0;
    (*iter_B0)    = cvCloneMat(_B0);
  }
}

//----------------------------------------------------------------------

void CVBGmm::setBeta0(const int _m, const double& _beta)
{
  m_beta0[_m] = _beta;
}

void CVBGmm::setBeta0(const double& _beta)
{
  DoubleVect::iterator iter = m_beta0.begin();
  for (;iter != m_beta0.end(); ++iter)
    (*iter) = _beta;
}

//----------------------------------------------------------------------
  
void CVBGmm::setXi0(const int _m, const double& _xi)
{
  m_xi0[_m] = _xi;
}

void CVBGmm::setXi0(const double& _xi)
{
  DoubleVect::iterator iter = m_xi0.begin();
  for (;iter != m_xi0.end(); ++iter)
    (*iter) = _xi;
}

//----------------------------------------------------------------------
  
void CVBGmm::setNu0(const int _m, CvMat* _nu)
{
  if (m_nu0[_m] != NULL) cvReleaseMat(&(m_nu0[_m]));
  m_nu0[_m] = cvCloneMat(_nu);
}

void CVBGmm::setNu0(CvMat* _nu)
{
  CvMatVect::iterator iter = m_nu0.begin();
  for (;iter != m_nu0.end(); ++iter)
  {
    if (*iter) cvReleaseMat(&(*iter));
    (*iter) = cvCloneMat(_nu);
  }
}

//----------------------------------------------------------------------

void CVBGmm::setEta0(const int _m, const double& _eta)
{
  m_eta0[_m] = _eta;
}
void CVBGmm::setEta0(const double& _eta)
{
  DoubleVect::iterator iter = m_eta0.begin();
  for (;iter != m_eta0.end(); ++iter)
    (*iter) = _eta;
}
//----------------------------------------------------------------------

void CVBGmm::setB0(const int _m, CvMat* __B)
{
  if (m_B0[_m] != NULL) cvReleaseMat(&(m_B0[_m]));
  m_B0[_m] = cvCloneMat(__B);
}

void CVBGmm::setB0(CvMat* __B)
{
  CvMatVect::iterator iter = m_B0.begin();
  for (;iter != m_B0.end(); ++iter)
  {
    if (*iter) cvReleaseMat(&(*iter));
    (*iter) = cvCloneMat(__B);
  }
}

//----------------------------------------------------------------------

void CVBGmm::setBeta(const int _m, const double& _beta)
{
  m_beta[_m] = _beta;
}

//----------------------------------------------------------------------
  
void CVBGmm::setXi(const int _m, const double& _xi)
{
  m_xi[_m] = _xi;
}

//----------------------------------------------------------------------
  
void CVBGmm::setNu(const int _m, CvMat* _nu)
{
  if (m_nu[_m] != NULL) cvReleaseMat(&(m_nu[_m]));
  m_nu[_m] = cvCloneMat(_nu);
}

//----------------------------------------------------------------------

void CVBGmm::setEta(const int _m, const double& _eta)
{
  m_eta[_m] = _eta;
}

//----------------------------------------------------------------------

void CVBGmm::setB(const int _m, CvMat* __B)
{
  if (m_B[_m] != NULL) cvReleaseMat(&(m_B[_m]));
  if (m_B[_m] != NULL) cvReleaseMat(&(m_B[_m]));
  m_B[_m] = cvCloneMat(__B);
}

//----------------------------------------------------------------------

void CVBGmm::calcBeta(const int _m, const double& _acc_N)
{
  m_beta[_m] = m_beta0[_m] + _acc_N;
}

//----------------------------------------------------------------------
  
void CVBGmm::calcXi(const int _m, const double& _acc_N)
{
  m_xi[_m] = m_xi0[_m] + _acc_N;
}

//----------------------------------------------------------------------
  
void CVBGmm::calcNu(const int _m, const double& _acc_N, CvMat* _acc_O)
{
  CvMat* rowvec = cvCreateMat(1, m_dimension, CV_32F);

  cvAddWeighted(m_nu0[_m], m_xi0[_m], _acc_O, 1.0, 0.0, rowvec);
  cvConvertScale(rowvec, rowvec, 1.0 / (_acc_N + m_xi0[_m]));
  
  if (m_nu[_m] != NULL) cvReleaseMat(&(m_nu[_m]));
  m_nu[_m] = rowvec;
}

//----------------------------------------------------------------------

void CVBGmm::calcEta(const int _m, const double& _acc_N)
{
  m_eta[_m] = m_eta0[_m] + _acc_N;
}

//----------------------------------------------------------------------

void CVBGmm::calcB(const int _m, 
		   const double& _acc_N, 
		   CvMat* _acc_O, 
		   CvMat* _acc_C)
{
  CvMat* rowvec = cvCreateMat(1, m_dimension, CV_32F);
  CvMat* colvec = (m_covtype == FULLC) ? 
    cvCreateMat(m_dimension, 1, CV_32F) : /* full */
    NULL;                                 /* diagonal */
  CvMat* matrix1 = (m_covtype == FULLC) ?
    cvCreateMat(m_dimension, m_dimension, CV_32F) : /* full */
    cvCreateMat(1, m_dimension, CV_32F);            /* diagonal */
  CvMat* matrix2 = (m_covtype == FULLC) ?
    cvCreateMat(m_dimension, m_dimension, CV_32F) : /* full */
    cvCreateMat(1, m_dimension, CV_32F);            /* diagonal */
  
#if 0 // _acc_O が共分散行列の場合
  cvSub(_acc_O, m_nu0[_m], rowvec);
  if (m_covtype == FULLC)
  {
    cvmTranspose(rowvec, colvec);
    cvMatMul(colvec, rowvec, matrix1);
  }
  else
    cvMul(rowvec, rowvec, matrix1);

  cvConvertScale(matrix1, matrix1, _acc_N * m_xi0[_m] / (_acc_N + m_xi0[_m]));
  cvAddWeighted(_acc_C, _acc_N, matrix1, 1.0, 0.0, matrix1);

  cvAdd(m_B0[_m], matrix1, matrix1);

//  cvSub(matrix1, _acc_O, matrix1);
#else // _acc_O が相関行列の場合
  if (m_covtype == FULLC)
  {
    cvmTranspose(m_nu0[_m], colvec);
    cvMatMul(colvec, m_nu0[_m], matrix1);
  }
  else
    cvMul(m_nu0[_m], m_nu0[_m], matrix1);
  if (m_covtype == FULLC)
  {
    cvmTranspose(m_nu[_m], colvec);
    cvMatMul(colvec, m_nu[_m], matrix2);
  }
  else
    cvMul(m_nu[_m], m_nu[_m], matrix2);
  
  cvAddWeighted(matrix1,  m_xi0[_m], matrix2, -m_xi[_m], 0.0, matrix1);
  cvAddWeighted(m_B0[_m], 1.0,      _acc_C,         1.0, 0.0, matrix2);
  cvAddWeighted(matrix1,  1.0,       matrix2,       1.0, 0.0, matrix1);

#endif
  if (m_B[_m]) cvReleaseMat(&(m_B[_m]));
  m_B[_m] = matrix1;
  
  if (rowvec)  cvReleaseMat(&rowvec);
  if (colvec)  cvReleaseMat(&colvec);
  if (matrix2) cvReleaseMat(&matrix2);
}

//----------------------------------------------------------------------

double CVBGmm::addBeta(void)
{
  double sum = 0.0;
  
  DoubleVect::iterator it = m_beta.begin();
  for (; it != m_beta.end(); ++it)
    sum += *it;
  
  return sum;
}

//----------------------------------------------------------------------
// htk 形式のモデルファイルから基底を生成する

void CVBGmm::readFromHtk(const char* _filename,
                        const double& _beta0,
                        const double& _xi0,
                        const double& _eta0)
{
  htktools::CHTKModel* htkmodel
    = new htktools::CHTKModel(_filename);
  
  int dimension    = htkmodel->getNumDimension();
  int num_mixtures = htkmodel->getNumMixtures();
  int cov_type     = htkmodel->getCovType();
  string name      = htkmodel->getName();
  

  if (dimension != getDimension())
    Error(1111, "[vbgmm.cc]: Invalid number of dimension: %d", dimension);
  if (cov_type != getCovType())
    Error(1111, "[vbgmm.cc]: Invalid type of covariation: %d", cov_type);
  if (num_mixtures != getNumMixtures())
    Error(1111, "[vbgmm.cc]: Invalid number of mixtures: %d", num_mixtures);
  
  for (int i = 0; i < num_mixtures; ++i)
  {
    double beta0  = 1.0;
    double xi0    = 1.0;
    double eta0   = 1.0;
    CvMat* nu0    = cvCreateMat(1, dimension, CV_32F);
    CvMat* B0     = (cov_type==FULLC) ?
      cvCreateMat(dimension, dimension, CV_32F):
      cvCreateMat(1, dimension, CV_32F);
    
    if (_beta0 == null)
      htkmodel->readGaussian(&beta0, nu0, B0);
    else
      htkmodel->readGaussian(null, nu0, B0);
    if (_xi0    == null)
      xi0 = beta0;
    if (_eta0     == null)
      eta0  = beta0;
    
    setBeta0(i, beta0);
    setXi0(i, xi0);
    setEta0(i, eta0);
    setNu0(i,   nu0);
    setB0(i,    B0);
    
    cvReleaseMat(&nu0);
    cvReleaseMat(&B0);
  }
  delete htkmodel;
}

//----------------------------------------------------------------------
// モデルパラメタを外部ファイルへ書き込む

ostream& operator << (ostream& _os, const CVBGmm& _vbgmm)
{
  _os << "~h \"" << _vbgmm.m_name << "\"\n";
  _os << "<VECSIZE> " << _vbgmm.m_dimension;
  if (_vbgmm.m_covtype == FULLC) _os << " <FULLC>\n";
  else if (_vbgmm.m_covtype == DIAGC) _os << " <DIAGC>\n";
  else
  {
    cerr << "Invalid covariance type"<<endl;
    exit(-1);
  }
  
  _os << "<BEGIN_GMM>\n";
  
  
  _os << "<NUMMIXES> " << _vbgmm.m_num_mixtures << "\n";
  for (int m = 0; m < _vbgmm.m_num_mixtures; ++m)
  {
    _os << "<MIXTURE> " << m + 1              << "\n";
    
    _os << "<BETA> "    << _vbgmm.m_beta[m]   << "\n";
    
    _os << "<XI> "      << _vbgmm.m_xi[m]     << "\n";
    
    _os << "<NU> " << _vbgmm.m_dimension << "\n";
    for (int d = 0; d < _vbgmm.m_dimension; ++d)
      _os << cvmGet(_vbgmm.m_nu[m], 0, d) << " ";
    _os << "\n";

    _os << "<ETA> "  << _vbgmm.m_eta[m]  << "\n";

    _os << "<B> ";
    
    if (_vbgmm.m_covtype == FULLC)
      _os << _vbgmm.m_dimension << " " << _vbgmm.m_dimension << "\n";
    else if (_vbgmm.m_covtype == DIAGC)
      _os << _vbgmm.m_dimension << "\n";
    else
    {
      cerr << "Invalid covariance type"<<endl;
      exit(-1);
    }
    
    for (int d = 0; d < _vbgmm.m_dimension; ++d)
    {
      if (_vbgmm.m_covtype == FULLC)
      {
        for (int i = 0; i < _vbgmm.m_dimension; ++i)
          _os << cvmGet(_vbgmm.m_B[m], d, i) << " ";
      }
      else if (_vbgmm.m_covtype == DIAGC)
      {
        _os << cvmGet(_vbgmm.m_B[m], 0, d) << " ";
      }
      else
      {
        cerr << "Invalid covariance type"<<endl;
        exit(-1);
      }
    }
    _os << "\n";
  }
  _os << "<END_GMM>\n";
  
  return _os;
}

//----------------------------------------------------------------------

string CVBGmm::outputPrior(void)
{
  ostringstream os;
  
  os << "~h \""      << "prior" << "\"\n";
  os << "<VECSIZE> " << m_dimension;
  if (m_covtype == FULLC) os << " <FULLC>\n";
  else if (m_covtype == DIAGC) os << " <DIAGC>\n";
  else
  {
    cerr << "Invalid covariance type"<<endl;
    exit(-1);
  }
  
  os << "<BEGIN_GMM>\n";
  
  
  os << "<NUMMIXES> " << m_num_mixtures << "\n";
  for (int m = 0; m < m_num_mixtures; ++m)
  {
    os << "<MIXTURE> " << m + 1              << "\n";
    
    os << "<BETA> "    << m_beta0[m]   << "\n";
    
    os << "<XI> "      << m_xi0[m]     << "\n";
    
    os << "<NU> " << m_dimension << "\n";
    for (int d = 0; d < m_dimension; ++d)
      os << cvmGet(m_nu0[m], 0, d) << " ";
    os << "\n";
    os << "<ETA> "  << m_eta0[m]  << "\n";
    
    os << "<B> ";
    if (m_covtype == FULLC)
      os << m_dimension << " " << m_dimension << "\n";
    else if (m_covtype == DIAGC)
      os << m_dimension << "\n";
    else
    {
      cerr << "Invalid covariance type"<<endl;
      exit(-1);
    }
    
    for (int d = 0; d < m_dimension; ++d)
    {
      if (m_covtype == FULLC)
      {
        for (int i = 0; i < m_dimension; ++i)
          os << cvmGet(m_B0[m], d, i) << " ";
      }
      else if (m_covtype == DIAGC)
      {
        os << cvmGet(m_B0[m], 0, d) << " ";
      }
      else
      {
        cerr << "Invalid covariance type"<<endl;
        exit(-1);
      }
    }
    os << "\n";
  }
  os << "<END_GMM>\n";
  
  return os.str();
}

//----------------------------------------------------------------------
//               End: vbgmm.cc
//----------------------------------------------------------------------
