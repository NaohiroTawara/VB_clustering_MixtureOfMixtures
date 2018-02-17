#ifndef __VBMMMTRAINER_H__
#define __VBMMMTRAINER_H__

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cv.h>
#include <highgui.h>
#include <time.h>

#include <boost/random.hpp>

#include "util_stat.h"

#include "segment.h"
#include "spkrClustering.h"

#include "vbgmmtrainer.h"
#include "folsvbgmmtrainer.h"

#include "evalK.h"

using namespace std;
using namespace boost;



enum {STOP, CONTINUE};

/**
 * @class CVBTrainer vbtrainer.h
 * @brief 変分ベイズ学習を行うクラス
 * @memo  データ集合全体を管理
 * @author Naohiro TAWARA
 * @date 2011-06-03
 */
class CVBMmmTrainer : public ISpkrClustering
{
public:
  static const int INIT_X_RANDOM;
  static const int INIT_X_KMEANS;
  static const int INIT_X_MANUAL;
  static const int INIT_X_PARAM;
  static const int INIT_Z_RANDOM;
  static const int INIT_Z_KMEANS;
  static const int INIT_Z_MANUAL;
  static const int INIT_Z_PARAM;
private:
  
  Rand *m_rand; // defined in util_stat.h
  string m_name;
  
  vector <CVBGmmTrainer*> m_cluster; // 各話者クラスタに関連付けられた VBTrainer へのポインタ: 1 x S
  vector <DoubleVect> m_X;           // 話者クラスタレベル潜在変数の変分事後期待値 # of clusters x # of segments
  
  double m_alpha0;     // 話者クラスタレベルのクラスタ重み事前分布の超パラメタ
  DoubleVect m_alpha;
  
  int m_thresh_N;      //// イタレーション終了基準閾値
  int m_max_iteration; //// 最大イタレーション数
  
  string m_latent_filename; // 発話レベル潜在変数の初期設定用ファイル名
  
  int m_initOpt;
public: /* コンストラクタ・デストラクタ */
  
  CVBMmmTrainer(const int _num_clusters,
                const int _num_mixtures,
                const int _covtype,
                ostream* _ros)
    : m_alpha0(0.0),
      m_thresh_N(0),
      m_max_iteration(0),
      m_rand(NULL),
      ISpkrClustering(_num_clusters, _covtype, _ros)
    {
      malloc(_num_clusters, _num_mixtures);
    }
  
  ~CVBMmmTrainer(void)
    {
      free();
    }
  
private: /* メモリ管理 */
  
  void malloc(const int _num_clusters,
              const int _num_mixtures);
  
  void free(void);

private:
  
  void updateX(void);
  
  void initX(const int _initOpt, const char* _latent_filename);
  void initCluster(const int _initOpt);
  void initParam(const int _initOpt);

  void updateAlpha(void);
  double addAlpha(void);

  double calcBIC(void);
  double calcFreeEnergy(void);
  double calcLowerBound(void);
  double calcKLDivergence(void);
  
  void kMeansClustering(DoubleVect* _index);

  void getMAP_X(IntVect* _segment_cc);
  
  /// クラスタを初期化する関数
  
  void mallocCluster(CSegment* _segment);
  
public: /* privateメンバの操作・参照用 */

  /// BIC value を返す関数
  double getBIC()
    { return calcBIC(); }

  /// Free Energy を返す関数
  double getFreeEnergy()
    { return calcFreeEnergy(); }

  /// ハイパーパラメタの変分事後分布と事前分布の KL Divergence を返す関数
  double getKLDivergence()
    { return calcKLDivergence(); }

  /// 下限 を返す関数
  double getLowerBound()
    { return calcLowerBound(); }
  

public: /* トップレベル制御 */
  
  void initRandomSeed(int _n);
  
  /// (1) 初期化を行う関数
  void init();
  
  void init(const int _initOpt, const char *_latent_filename = "");
  
  /// (2) クラスタリング実行関数
  int run(void);

  /// (3) クラスタリング結果を返す関数
  int getClusteringResultSeg(IntVect* _segment);
  int getClusteringResultCl(vector<IntVect>* _cluster);
  
  /*
   * @brief 基底を初期化する関数
   * @param _filename 基底モデルファイル名
   */
  void setBasisFromFile(const char* _filename,
                        const float& _w0,
                        const float& _xi0,
                        const float& _eta0);
  
  /*
   * @brief 基底をデータから初期化する関数
   * @param _filename 基底モデルファイル名
   */
  void setBasisFromData(CSegment* _feature,
                        const float& _w0,
                        const float& _xi0,
                        const float& _eta0);

  
  void setRunParameter(const float _thresh_N,
                       const int   _max_iteration);
  string printInfo(void);

  void setFeature(CSegment* _segment);
  
  string outputModel();
  
  void getModelAllignment(vector<IntVect>* _segment_fcc, const int _cl);

  int getNumClusters() { return m_cluster.size();}
  
};

#endif // __VBMMMTRAINER_H__

