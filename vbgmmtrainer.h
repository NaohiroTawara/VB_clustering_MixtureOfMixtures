#ifndef __VBGMMTRAINER_H__
#define __VBGMMTRAINER_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include "vbgmm.h"
#include "segment.h"
#include "util_mat.h"
#include "util.h"
#include "util_stat_densities.h"

using namespace std;


/**
 * @class CVBTrainer vbtrainer.h
 * @brief 話者クラスタ GMM の変分ベイズ学習を行うクラス
 * @memo  データ集合全体を管理
 * @author Naohiro TAWARA
 * @date 2011-05-11
 */

#define DEBUG 1

class CVBGmmTrainer
{
public:
  static const int INIT_Z_RANDOM;
  static const int INIT_Z_KMEANS;
  static const int INIT_Z_MANUAL;
  static const int INIT_Z_PARAM;
  
protected:
  vector<vector <DoubleVect> > m_Z;  //// フレームクラスタレベル潜在変数の変分事後期待値（負担率）行列:
                                     ////      M x (# of segments x # of frames in each segment)
#if !DEBUG
  vector <DoubleVect> m_logsumZ;     //// m_Z の対数値の混合要素に関する和:  # of segments x # of frames in each segment
#else
  DoubleVect m_logsumZ;              //// m_Z の対数値の混合要素に関する和:  1 x # of segments
#endif
  DoubleVect m_lowerlimit;           //// free Energy 算出用の値を保持: 1 x # of segments
  DoubleVect m_KLDivergence;         //// 超パラメタの KL Divergence を保持: 1 x M
  

  CSegment*   getFeatures() { return m_features;}
//  vector <DoubleVect>* getLogSumZ() { return &m_logsumZ;}
  
  vector<vector <DoubleVect> >* getZ() { return &m_Z;}
  
private:

  int m_covtype;        //// 共分散行列のタイプ:
  int m_num_mixtures;   //// フレームクラスタ数: M
  int m_dimension;      //// 特徴量の次元数:     g
  
  CvMat m_stub; //// 部分行列取得用一時変数

  CSegment* m_features;              //// セグメント特徴量構造体へのポインタ配列: 1 * # of segments
  CVBGmm* m_gmm;                     //// GMM のモデルパラメータへのポインタ

  
  /* 各フレームクラスタに含まれるデータの統計量 */
  // update() 関数以外からは呼び出されないため，本当はクラスメンバにする必要はないが，
  // 関数呼び出し時に毎回再確保するのは無駄なため最初に全て確保してそれを使いまわす
  DoubleVect m_acc_N; //// データ数の統計量（EM カウント）  1 x M
  CvMatVect  m_acc_O; //// 一次統計量（モーメント（EMカウントで除算）ではない） M x [1 x g]
  CvMatVect  m_acc_C; //// 二次統計量（モーメント（EMカウントで除算）ではない） M x [g x g]

  
public: /* コンストラクタ・デストラクタ */

  CVBGmmTrainer(const string& _label_name,
                 const int _num_mixtures)
     : m_num_mixtures(_num_mixtures), 
       m_dimension(0),
       m_covtype(FULLC),
       m_gmm(NULL)
    {
      m_gmm = new CVBGmm(_label_name, _num_mixtures);
      
      m_acc_N.resize(_num_mixtures);
      m_acc_O.resize(_num_mixtures, NULL);
      m_acc_C.resize(_num_mixtures, NULL);
      m_Z.resize(_num_mixtures);
    }
  
  
  ~CVBGmmTrainer(void)
    {
      free();
    }
  
private: /* メモリ管理 */
  
  void malloc(const int _dimension, const int _num_segments, IntVect& _num_frames);

  void free(void);

  ///  KL Divergence を更新
  void updateKLDivergence(void);

public: /* privateメンバの操作・参照用 */

    CVBGmm*     getGmm()      { return m_gmm;}
  /// 共分散行列のタイプを返す関数
  int getCovType(void) const
    { return m_covtype; }

  /// 共分散行列のタイプを設定する関数
  void setCovType(const int _covtype)
    { m_covtype = _covtype; }
  
  /// 特徴量の次元数を返す関数
  int getDimension(void) const
    { return m_dimension; }
  
  /// 特徴量の次元数を設定する関数
  void setDimension(const int _value)
    { m_dimension = _value; }
  
  /// 混合数を返す関数
  int getNumMixtures(void) const
    { return m_num_mixtures; }
  
  /// 混合数を設定する関数
  void setNumMixtures(const int _value)
    { m_num_mixtures = _value; }

  /*
   * @brief  free energy を返す関数
   * @param _t: セグメントのインデックス
   *  @return free energy
   */
  double getLowerLimit(const int _t) const
    { return m_lowerlimit[_t]; }
  /*
   * @brief  超パラメタの KL Divergence を返す関数
   * @param _i: フレームレベル混合要素のインデックス
   *  @return free energy
   */
  double getKLDivergence(const int _i) const
    { return m_KLDivergence[_i]; }
  double getKLDivergence(void) const
    { return _sum(m_KLDivergence); }
  /*
   * @brief フレームレベル潜在変数のセグメント内同時分布の EM カウントの混合数に関する和を返す関数
   * @param _t: セグメントのインデックス
   * @return Prod_{p}(Sum_{i}(log(m_Z[i][_t][p])))
   *          i: mixtures, p: frames
   */
#if !DEBUG
  double getProdSumLogZ(const int _t) const
    { return _sum(m_logsumZ[_t]); }
#else
  double getProdSumLogZ(const int _t) const
  { return m_logsumZ[_t]; }
#endif

public: /* トップレベル制御 */
  
  void getMAP_Z(vector<IntVect>* _segment_fcc);


  /// セグメント特徴量構造体へのポインタ配列を設定する関数
  void setFeatures(CSegment* _features);
  
  /// メモリ領域を初期化する関数
  void init(const int _dimension, const int _num_segments,
            IntVect& _num_frames);
  
  /// フレームレベル潜在変数を更新する関数
  void updateZ(void);

  /*
   *  @brief GMM のモデルパラメータを更新する関数
   *  @param: _X : 話者レベル潜在変数の変分事後確率期待値
   *  @return 全混合要素における EM カウントの更新量の平均
   */
  double update(DoubleVect& __X);

  /// 潜在変数を初期化する関数
  void initZ(const int _initOpt);

  /*
   * @brief 超パラメータの超パラメタ(分布共通)をファイルから初期化する関数
   * @param _model_filename: GMM model filename
   */
  void initGlobalParam(const char* _model_filename,
                       const double& _beta0 = null,
                       const double& _xi0   = null,
                       const double& _eta0  = null);
  /*
   * @brief 超パラメータの超パラメタ(分布共通)を初期化する関数
   * @param _beta0: beta の事前分布超パラメタ
   * @param _xi0:  xi  の事前分布超パラメタ
   * @param _nu0:  nu  の事前分布の超パラメタ
   * @param _eta0: eta の事前分布の超パラメタ
   * @param _B0:   B   の事前分布の超パラメタ
   */
  void initGlobalParam(const double& _beta0,
                       const double& _xi0,
                       CvMat* _nu0,
                       const double& _eta0,
                       CvMat* _B0);
  /*
   * @brief 超パラメータの超パラメタ(分布共通)を初期化する関数
   * @param _beta0: beta の事前分布超パラメタ
   * @param _xi0:  xi  の事前分布超パラメタ
   * @param _nu0:  nu  の事前分布の超パラメタ
   * @param _eta0: eta の事前分布の超パラメタ
   * @param _B0:   B   の事前分布の超パラメタ
   */
  void initGlobalParam(CSegment* _features,
                       const double& _beta0, const double& _xi0,
                       const double& _eta0,  const double& _B0);
    /*
   * @brief 超パラメータの超パラメタ(分布共通)を初期化する関数
   * @param _features: nu0, B0, を決定するために用いる特徴データ
   * @param _beta0: beta の事前分布超パラメタ
   * @param _xi0:  xi  の事前分布超パラメタ
   * @param _eta0: eta の事前分布の超パラメタ
  */
  void initGlobalParam(CSegment* _features,
                       const double& _beta0, const double& _xi0,
                       const double& _eta0);
  /**
   * @brief: 話者・フレームレベル潜在変数の初期値を用いて，統計量と超パラメータを初期化する関数
   * @param: _X : 話者レベル潜在変数の変分事後確率期待値
   */
  void initParam(DoubleVect& __X);
  
  /**
   * @brief: モデルの超パラメタを先に初期化する．具体的には事前分布の超パラメタをコピーして初期化する
   */
  void initParam();
  
public: /* 情報表示系関数 */
  string info(void);

  void printZ(void);

public: /* ファイル入出力 */

  /// CVBGmm オブジェクトの情報を返す関数
  string outputGMM(void);
  
  string outputPrior(void);
  
};

#endif // __VBTRAINER_H__
