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
 * @brief GMM の変分ベイズパラメータを管理するクラス
 * @author Naohiro Tawara
 * @date 2010-06-16
 */
class CVBGmm
{
private:
  
  string m_name;  ///< クラス名称
  
  int m_covtype; ///< 分散行列の種類
  
  int m_num_mixtures; ///< 混合数
  int m_dimension;    ///< 次元数
  
  DoubleVect m_beta0;
  DoubleVect m_xi0;
  CvMatVect  m_nu0;
  DoubleVect m_eta0;
  CvMatVect  m_B0;
  CvMatVect  m_invB0;
  
  DoubleVect m_beta; ///< 混合重みの超パラメータ
  DoubleVect m_xi;   ///< 平均ベクトルの超パラメータ
  CvMatVect  m_nu;   ///< 平均ベクトルの超パラメータ
  DoubleVect m_eta;  ///< 分散行列の超パラメータ
  CvMatVect  m_B;    ///< 分散行列の超パラメータ
  CvMatVect  m_invB; ///< B の逆行列
  
public: /* コンストラクタ・デストラクタ */
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
  
private: /* メモリ管理 */
  
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
  
public: /* privateメンバの操作・参照用 */
  
  /**
   * @brief ラベル名称を設定する関数
   * @param _name: ラベル名称
   */
  void setName(const string& _name)
    { m_name = _name; }
  
  /**
   * @brief 分散行列の種類を設定する関数
   * @param _covtype: 分散行列の種類 (DIAGC, FULLC)
   */
  void setCovType(const int _covtype)
    { m_covtype = _covtype; }
  
  /**
   * @brief 次元数を設定する関数
   * @param _dimension: データの次元数
   */
  void setDimension(const int _dimension)
    { m_dimension = _dimension; }
  
  /**
   * @brief グローバルな超パラメータを設定する関数
   * @param _beta: Dirichlet 分布の超パラメータ [1 x 1]
   * @param _xi:  Gauss 分布の超パラメータ     [1 x 1]
   * @param _nu:  Gauss 分布の超パラメータ     [1 x D]
   * @param _eta: Wishart 分布の超パラメータ   [1 x 1]
   * @param _B:   Wishart 分布の超パラメータ   [D x D]
   */
  void setGlobalParam(DoubleVect& _beta0, DoubleVect& _xi0,
                      CvMatVect& _nu0,    DoubleVect& _eta0,
                      CvMatVect& _B0);

  void setGlobalParam(const double& _beta0, const double& _xi0,
                      CvMat* _nu0,          const double& _eta0,
                      CvMat* _B0);

  /// 超パラメータ Beta を設定する関数
  void setBeta0(const int _m, const double& _beta);
  void setBeta0(const double& _beta);
  
  /// 超パラメータ Xi を設定する関数
  void setXi0(const int _m, const double& _xi);
  void setXi0(const double& _xi);
  
  /// 超パラメータ Nu を設定する関数
  void setNu0(const int _m, CvMat* _nu);
  void setNu0(CvMat* _nu);
  
  /// 超パラメータ Eta を設定する関数
  void setEta0(const int _m, const double& _eta);
  void setEta0(const double& _eta);
  
  /// 超パラメータ B を設定する関数
  void setB0(const int _m, CvMat* __B);
  void setB0(CvMat* __B);
  
  /// 超パラメータ Beta を設定する関数
  void setBeta(const int _m, const double& _beta);
  
  /// 超パラメータ Xi を設定する関数
  void setXi(const int _m, const double& _xi);
  
  /// 超パラメータ Nu を設定する関数
  void setNu(const int _m, CvMat* _nu);
  
  /// 超パラメータ Eta を設定する関数
  void setEta(const int _m, const double& _eta);
  
  /// 超パラメータ B を設定する関数
  void setB(const int _m, CvMat* __B);
  
  /// 超パラメータ Beta を計算する関数
  void calcBeta(const int _m, const double& _acc_N);
  
  /// 超パラメータ Xi を計算する関数
  void calcXi(const int _m, const double& _acc_N);
  
  /// 超パラメータ Nu を計算する関数
  void calcNu(const int _m, const double& _acc_N, CvMat* _acc_X);
  
  /// 超パラメータ Eta を計算する関数
  void calcEta(const int _m, const double& _acc_N);
  
  /// 超パラメータ B を計算する関数
  void calcB(const int _m,  const double& _acc_N, 
             CvMat* _acc_X, CvMat* _acc_C);
  
  /// クラス名称を返す関数
  string getName(void) const
    { return m_name; }
  
  /// 混合数を返す関数
  int getNumMixtures(void) const
    { return m_num_mixtures; }
  
  int getCovType(void) const
    { return m_covtype; }
  /// 次元数を返す関数
  int getDimension(void) const
    { return m_dimension; }

  /// 超パラメメータ beta の超パラメタを返す関数
  double getBeta0(const int _m) const
    { return m_beta0[_m]; }

  DoubleVect* getBeta0()
    { return &m_beta0; }
  
  /// 超パラメメータ xi の超パラメタを返す関数
  double getXi0(const int _m) const
    { return m_xi0[_m]; }
  
  /// 超パラメメータ eta の超パラメタを返す関数
  double getEta0(const int _m) const
    { return m_eta0[_m]; }

  /// 超パラメメータ nu の超パラメタを返す関数
  CvMat* getNu0(const int _m)
    { return m_nu0[_m]; }

  /// 超パラメメータ B の超パラメタを返す関数
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
  
    /// 超パラメータ beta を返す関数
  double getBeta(const int _m) const
    { return m_beta[_m]; }

  DoubleVect* getBeta()
    { return &m_beta;}
  
  /// 超パラメータ xi を返す関数
  double getXi(const int _m) const
    { return m_xi[_m]; }
  
  /// 超パラメータ nu を返す関数
  CvMat* getNu(const int _m) const
    { return m_nu[_m]; }
  
  /// 超パラメータ eta を返す関数
  double getEta(const int _m) const
    { return m_eta[_m]; }
  
  /// 超パラメータ B を返す関数
  CvMat* getB(const int _m) const
    { return m_B[_m]; }

  /// 超パラメータ B の逆行列 を返す関数
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
  
  /// 超パラメータ B の対数行列式値を返す関数
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
   * @brief パラメータ Phi について全ての混合要素に対して和を取る関数
   * @return Phi の混合要素に対する和
   */
  double addBeta(void);
  
public: /* ファイル入出力 */
  
  /**
   * @brief prior のハイパーパラメタを出力
   */
  string outputPrior();
  
  
  /**
   * @brief htk 形式のモデルファイルから prior のハイパーパラメタを読み込む
   */
  void readFromHtk(const char* _filename,
                   const double& _beta0 = null,
                   const double& _xi0   = null,
                   const double& _eta0  = null);
    
  /**
   * @brief 出力演算子
   * @param _os: 出力ストリーム
   * @param _vbgmm: CVBGmm オブジェクト
   * @return 出力ストリーム
   */
  friend ostream& operator << (ostream& _ifs, const CVBGmm& _vbgmm);
  
};

#endif // __VB_GMM_H__
