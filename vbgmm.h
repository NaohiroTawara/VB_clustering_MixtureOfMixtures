#ifndef __VB_GMM_H__
#define __VB_GMM_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cv.h>

#include "util.h"
#include "htkmodel.h"
using namespace std;

typedef vector<CvMat*> CvMatVect;
typedef vector<double> DoubleVect;
typedef vector<int> IntVect;

/**
 * @class CVBGmm vbgmm.h
 * @brief GMM ����ʬ�٥����ѥ�᡼����������륯�饹
 * @author Naohiro Tawara
 * @date 2010-06-16
 */
class CVBGmm
{
private:
  
  string m_name;  ///< ���饹̾��
  
  int m_covtype; ///< ʬ������μ���
  
  int m_num_mixtures; ///< �����
  int m_dimension;    ///< ������
  
  DoubleVect m_beta0;
  DoubleVect m_xi0;
  CvMatVect  m_nu0;
  DoubleVect m_eta0;
  CvMatVect  m_B0;
  CvMatVect  m_invB0;
  
  DoubleVect m_beta; ///< ����Ťߤ�Ķ�ѥ�᡼��
  DoubleVect m_xi;   ///< ʿ�ѥ٥��ȥ��Ķ�ѥ�᡼��
  CvMatVect  m_nu;   ///< ʿ�ѥ٥��ȥ��Ķ�ѥ�᡼��
  DoubleVect m_eta;  ///< ʬ�������Ķ�ѥ�᡼��
  CvMatVect  m_B;    ///< ʬ�������Ķ�ѥ�᡼��
  CvMatVect  m_invB; ///< B �εչ���
  
public: /* ���󥹥ȥ饯�����ǥ��ȥ饯�� */
  /*
  CVBGmm(void)
    : m_name(""), m_covtype(0), m_num_mixtures(0), m_dimension(0) 
    {}
  
  CVBGmm(const string& _name)
    : m_name(_name), m_covtype(0), m_num_mixtures(0), m_dimension(0)
    {}
  */
  CVBGmm(const string& _name, const int _num_mixtures)
    : m_name(_name), m_covtype(0), 
      m_num_mixtures(_num_mixtures), m_dimension(0)
    {
      malloc(_num_mixtures);
    }
  
  ~CVBGmm(void)
    {
      free();
    }
  
private: /* ������� */
  
  void malloc(const int _num_mixtures)
    {
      m_num_mixtures = _num_mixtures;
      
      m_beta0.resize(_num_mixtures);
      m_xi0.resize(_num_mixtures);
      m_nu0.resize(_num_mixtures, NULL);
      m_eta0.resize(_num_mixtures);
      m_B0.resize(_num_mixtures, NULL);
      m_invB0.resize(_num_mixtures, NULL);
      m_beta.resize(_num_mixtures);
      m_xi.resize(_num_mixtures);
      m_nu.resize(_num_mixtures, NULL);
      m_eta.resize(_num_mixtures);
      m_B.resize(_num_mixtures, NULL);
      m_invB.resize(_num_mixtures, NULL);
    }
  
  void free(void)
    {
      CvMatVect::iterator iter_nu = m_nu.begin();
      CvMatVect::iterator iter_B = m_B.begin();
      CvMatVect::iterator iter_invB = m_invB.begin();
      
      CvMatVect::iterator iter_B0 = m_B0.begin();
      CvMatVect::iterator iter_invB0 = m_invB0.begin();
      CvMatVect::iterator iter_nu0 = m_nu0.begin();
      for (;iter_nu != m_nu.end(); ++iter_nu, ++iter_B, ++iter_invB, ++iter_B0, ++iter_nu0)
      {
        if (*iter_nu)    cvReleaseMat(&(*iter_nu));
        if (*iter_B)     cvReleaseMat(&(*iter_B));
        if (*iter_invB)  cvReleaseMat(&(*iter_invB));
        if (*iter_nu0)   cvReleaseMat(&(*iter_nu0));
        if (*iter_B0)    cvReleaseMat(&(*iter_B0));
        if (*iter_invB0) cvReleaseMat(&(*iter_invB0));
      }
    }
  
public: /* private���Ф��������� */
  
  /**
   * @brief ��٥�̾�Τ����ꤹ��ؿ�
   * @param _name: ��٥�̾��
   */
  void setName(const string& _name)
    { m_name = _name; }
  
  /**
   * @brief ʬ������μ�������ꤹ��ؿ�
   * @param _covtype: ʬ������μ��� (DIAGC, FULLC)
   */
  void setCovType(const int _covtype)
    { m_covtype = _covtype; }
  
  /**
   * @brief �����������ꤹ��ؿ�
   * @param _dimension: �ǡ����μ�����
   */
  void setDimension(const int _dimension)
    { m_dimension = _dimension; }
  
  /**
   * @brief �����Х��Ķ�ѥ�᡼�������ꤹ��ؿ�
   * @param _beta: Dirichlet ʬ�ۤ�Ķ�ѥ�᡼�� [1 x 1]
   * @param _xi:  Gauss ʬ�ۤ�Ķ�ѥ�᡼��     [1 x 1]
   * @param _nu:  Gauss ʬ�ۤ�Ķ�ѥ�᡼��     [1 x D]
   * @param _eta: Wishart ʬ�ۤ�Ķ�ѥ�᡼��   [1 x 1]
   * @param _B:   Wishart ʬ�ۤ�Ķ�ѥ�᡼��   [D x D]
   */
  void setGlobalParam(DoubleVect& _beta0, DoubleVect& _xi0,
                      CvMatVect& _nu0,    DoubleVect& _eta0,
                      CvMatVect& _B0);

  void setGlobalParam(const double& _beta0, const double& _xi0,
                      CvMat* _nu0,          const double& _eta0,
                      CvMat* _B0);

  /// Ķ�ѥ�᡼�� Beta �����ꤹ��ؿ�
  void setBeta0(const int _m, const double& _beta);
  void setBeta0(const double& _beta);
  
  /// Ķ�ѥ�᡼�� Xi �����ꤹ��ؿ�
  void setXi0(const int _m, const double& _xi);
  void setXi0(const double& _xi);
  
  /// Ķ�ѥ�᡼�� Nu �����ꤹ��ؿ�
  void setNu0(const int _m, CvMat* _nu);
  void setNu0(CvMat* _nu);
  
  /// Ķ�ѥ�᡼�� Eta �����ꤹ��ؿ�
  void setEta0(const int _m, const double& _eta);
  void setEta0(const double& _eta);
  
  /// Ķ�ѥ�᡼�� B �����ꤹ��ؿ�
  void setB0(const int _m, CvMat* __B);
  void setB0(CvMat* __B);
  
  /// Ķ�ѥ�᡼�� Beta �����ꤹ��ؿ�
  void setBeta(const int _m, const double& _beta);
  
  /// Ķ�ѥ�᡼�� Xi �����ꤹ��ؿ�
  void setXi(const int _m, const double& _xi);
  
  /// Ķ�ѥ�᡼�� Nu �����ꤹ��ؿ�
  void setNu(const int _m, CvMat* _nu);
  
  /// Ķ�ѥ�᡼�� Eta �����ꤹ��ؿ�
  void setEta(const int _m, const double& _eta);
  
  /// Ķ�ѥ�᡼�� B �����ꤹ��ؿ�
  void setB(const int _m, CvMat* __B);
  
  /// Ķ�ѥ�᡼�� Beta ��׻�����ؿ�
  void calcBeta(const int _m, const double& _acc_N);
  
  /// Ķ�ѥ�᡼�� Xi ��׻�����ؿ�
  void calcXi(const int _m, const double& _acc_N);
  
  /// Ķ�ѥ�᡼�� Nu ��׻�����ؿ�
  void calcNu(const int _m, const double& _acc_N, CvMat* _acc_X);
  
  /// Ķ�ѥ�᡼�� Eta ��׻�����ؿ�
  void calcEta(const int _m, const double& _acc_N);
  
  /// Ķ�ѥ�᡼�� B ��׻�����ؿ�
  void calcB(const int _m,  const double& _acc_N, 
             CvMat* _acc_X, CvMat* _acc_C);
  
  /// ���饹̾�Τ��֤��ؿ�
  string getName(void) const
    { return m_name; }
  
  /// ��������֤��ؿ�
  int getNumMixtures(void) const
    { return m_num_mixtures; }
  
  int getCovType(void) const
    { return m_covtype; }
  /// ���������֤��ؿ�
  int getDimension(void) const
    { return m_dimension; }

  /// Ķ�ѥ��᡼�� beta ��Ķ�ѥ�᥿���֤��ؿ�
  double getBeta0(const int _m) const
    { return m_beta0[_m]; }

  DoubleVect* getBeta0()
    { return &m_beta0; }
  
  /// Ķ�ѥ��᡼�� xi ��Ķ�ѥ�᥿���֤��ؿ�
  double getXi0(const int _m) const
    { return m_xi0[_m]; }
  
  /// Ķ�ѥ��᡼�� eta ��Ķ�ѥ�᥿���֤��ؿ�
  double getEta0(const int _m) const
    { return m_eta0[_m]; }

  /// Ķ�ѥ��᡼�� nu ��Ķ�ѥ�᥿���֤��ؿ�
  CvMat* getNu0(const int _m)
    { return m_nu0[_m]; }

  /// Ķ�ѥ��᡼�� B ��Ķ�ѥ�᥿���֤��ؿ�
  CvMat* getB0(const int _m)
    { return m_B0[_m]; }

  CvMat* getInvertB0(const int _i)
    {
      if (m_invB0[_i]) cvReleaseMat(&m_invB0[_i]);
      m_invB0[_i] = (m_covtype == FULLC) ?
        cvCreateMat(m_dimension, m_dimension, CV_32F) : /* full */
        cvCreateMat(1, m_dimension, CV_32F);            /* diagonal */
      if (m_covtype ==FULLC)
        cvInvert(m_B0[_i], m_invB0[_i]);
      else
        for (int d = 0; d < m_dimension; ++d)
          cvmSet(m_invB0[_i], 0, d, 1.0 / cvmGet(m_B0[_i], 0, d));
      return m_invB0[_i];
    }
  
    /// Ķ�ѥ�᡼�� beta ���֤��ؿ�
  double getBeta(const int _m) const
    { return m_beta[_m]; }

  DoubleVect* getBeta()
    { return &m_beta;}
  
  /// Ķ�ѥ�᡼�� xi ���֤��ؿ�
  double getXi(const int _m) const
    { return m_xi[_m]; }
  
  /// Ķ�ѥ�᡼�� nu ���֤��ؿ�
  CvMat* getNu(const int _m) const
    { return m_nu[_m]; }
  
  /// Ķ�ѥ�᡼�� eta ���֤��ؿ�
  double getEta(const int _m) const
    { return m_eta[_m]; }
  
  /// Ķ�ѥ�᡼�� B ���֤��ؿ�
  CvMat* getB(const int _m) const
    { return m_B[_m]; }

  /// Ķ�ѥ�᡼�� B �εչ��� ���֤��ؿ�
  CvMat* getInvertB(const int _i)
    {
      if (m_invB[_i]) cvReleaseMat(&m_invB[_i]);
      m_invB[_i] = (m_covtype == FULLC) ?
        cvCreateMat(m_dimension, m_dimension, CV_32F) : /* full */
        cvCreateMat(1, m_dimension, CV_32F);            /* diagonal */
      if (m_covtype ==FULLC)
        cvInvert(m_B[_i], m_invB[_i]);
      else
        for (int d = 0; d < m_dimension; ++d)
          cvmSet(m_invB[_i], 0, d, 1.0 / cvmGet(m_B[_i], 0, d));
      return m_invB[_i];
    }
  
  /// Ķ�ѥ�᡼�� B ���п������ͤ��֤��ؿ�
  double getLogDetB(const int _i) const
    {
      CvMat* colvec = cvCreateMat(1, m_dimension, CV_32F);
      if (m_covtype == FULLC)
      {
        cvSVD(m_B[_i], colvec);
        cvLog(colvec, colvec);
      }
      else
        cvLog(m_B[_i], colvec);
      double logdet = cvSum(colvec).val[0];
      cvReleaseMat(&colvec);
      
      return logdet;
    }
  
  /**
   * @brief �ѥ�᡼�� Phi �ˤĤ������Ƥκ������Ǥ��Ф����¤���ؿ�
   * @return Phi �κ������Ǥ��Ф�����
   */
  double addBeta(void);
  
public: /* �ե����������� */
  
  /**
   * @brief prior �Υϥ��ѡ��ѥ�᥿�����
   */
  string outputPrior();
  
  
  /**
   * @brief htk �����Υ�ǥ�ե����뤫�� prior �Υϥ��ѡ��ѥ�᥿���ɤ߹���
   */
  void readFromHtk(const char* _filename,
                   const double& _beta0 = null,
                   const double& _xi0   = null,
                   const double& _eta0  = null);
    
  /**
   * @brief ���ϱ黻��
   * @param _os: ���ϥ��ȥ꡼��
   * @param _vbgmm: CVBGmm ���֥�������
   * @return ���ϥ��ȥ꡼��
   */
  friend ostream& operator << (ostream& _ifs, const CVBGmm& _vbgmm);
  
};

#endif // __VB_GMM_H__
