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
 * @brief ��ʬ�٥����ؽ���Ԥ����饹
 * @memo  �ǡ����������Τ����
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
  
  vector <CVBGmmTrainer*> m_cluster; // ���üԥ��饹���˴�Ϣ�դ���줿 VBTrainer �ؤΥݥ���: 1 x S
  vector <DoubleVect> m_X;           // �üԥ��饹����٥������ѿ�����ʬ��������� # of clusters x # of segments
  
  double m_alpha0;     // �üԥ��饹����٥�Υ��饹���Ť߻���ʬ�ۤ�Ķ�ѥ�᥿
  DoubleVect m_alpha;
  
  int m_thresh_N;      //// �����졼�����λ�������
  int m_max_iteration; //// ���祤���졼������
  
  string m_latent_filename; // ȯ�å�٥������ѿ��ν�������ѥե�����̾
  
  int m_initOpt;
public: /* ���󥹥ȥ饯�����ǥ��ȥ饯�� */
  
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
  
private: /* ������� */
  
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
  
  /// ���饹������������ؿ�
  
  void mallocCluster(CSegment* _segment);
  
public: /* private���Ф��������� */

  /// BIC value ���֤��ؿ�
  double getBIC()
    { return calcBIC(); }

  /// Free Energy ���֤��ؿ�
  double getFreeEnergy()
    { return calcFreeEnergy(); }

  /// �ϥ��ѡ��ѥ�᥿����ʬ����ʬ�ۤȻ���ʬ�ۤ� KL Divergence ���֤��ؿ�
  double getKLDivergence()
    { return calcKLDivergence(); }

  /// ���� ���֤��ؿ�
  double getLowerBound()
    { return calcLowerBound(); }
  

public: /* �ȥåץ�٥����� */
  
  void initRandomSeed(int _n);
  
  /// (1) �������Ԥ��ؿ�
  void init();
  
  void init(const int _initOpt, const char *_latent_filename = "");
  
  /// (2) ���饹����󥰼¹Դؿ�
  int run(void);

  /// (3) ���饹����󥰷�̤��֤��ؿ�
  int getClusteringResultSeg(IntVect* _segment);
  int getClusteringResultCl(vector<IntVect>* _cluster);
  
  /*
   * @brief �������������ؿ�
   * @param _filename �����ǥ�ե�����̾
   */
  void setBasisFromFile(const char* _filename,
                        const float& _w0,
                        const float& _xi0,
                        const float& _eta0);
  
  /*
   * @brief �����ǡ���������������ؿ�
   * @param _filename �����ǥ�ե�����̾
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

