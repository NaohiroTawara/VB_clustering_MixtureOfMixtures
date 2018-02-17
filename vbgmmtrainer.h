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
 * @brief �üԥ��饹�� GMM ����ʬ�٥����ؽ���Ԥ����饹
 * @memo  �ǡ����������Τ����
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
  vector<vector <DoubleVect> > m_Z;  //// �ե졼�९�饹����٥������ѿ�����ʬ��������͡���ôΨ�˹���:
                                     ////      M x (# of segments x # of frames in each segment)
#if !DEBUG
  vector <DoubleVect> m_logsumZ;     //// m_Z ���п��ͤκ������Ǥ˴ؤ�����:  # of segments x # of frames in each segment
#else
  DoubleVect m_logsumZ;              //// m_Z ���п��ͤκ������Ǥ˴ؤ�����:  1 x # of segments
#endif
  DoubleVect m_lowerlimit;           //// free Energy �����Ѥ��ͤ��ݻ�: 1 x # of segments
  DoubleVect m_KLDivergence;         //// Ķ�ѥ�᥿�� KL Divergence ���ݻ�: 1 x M
  

  CSegment*   getFeatures() { return m_features;}
//  vector <DoubleVect>* getLogSumZ() { return &m_logsumZ;}
  
  vector<vector <DoubleVect> >* getZ() { return &m_Z;}
  
private:

  int m_covtype;        //// ��ʬ������Υ�����:
  int m_num_mixtures;   //// �ե졼�९�饹����: M
  int m_dimension;      //// ��ħ�̤μ�����:     g
  
  CvMat m_stub; //// ��ʬ��������Ѱ���ѿ�

  CSegment* m_features;              //// ����������ħ�̹�¤�ΤؤΥݥ�������: 1 * # of segments
  CVBGmm* m_gmm;                     //// GMM �Υ�ǥ�ѥ�᡼���ؤΥݥ���

  
  /* �ƥե졼�९�饹���˴ޤޤ��ǡ����������� */
  // update() �ؿ��ʳ�����ϸƤӽФ���ʤ����ᡤ�����ϥ��饹���Фˤ���ɬ�פϤʤ�����
  // �ؿ��ƤӽФ��������Ƴ��ݤ���Τ�̵�̤ʤ���ǽ�����Ƴ��ݤ��Ƥ����Ȥ��ޤ魯
  DoubleVect m_acc_N; //// �ǡ������������̡�EM ������ȡ�  1 x M
  CvMatVect  m_acc_O; //// �켡�����̡ʥ⡼���ȡ�EM������Ȥǽ����ˤǤϤʤ��� M x [1 x g]
  CvMatVect  m_acc_C; //// �������̡ʥ⡼���ȡ�EM������Ȥǽ����ˤǤϤʤ��� M x [g x g]

  
public: /* ���󥹥ȥ饯�����ǥ��ȥ饯�� */

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
  
private: /* ������� */
  
  void malloc(const int _dimension, const int _num_segments, IntVect& _num_frames);

  void free(void);

  ///  KL Divergence �򹹿�
  void updateKLDivergence(void);

public: /* private���Ф��������� */

    CVBGmm*     getGmm()      { return m_gmm;}
  /// ��ʬ������Υ����פ��֤��ؿ�
  int getCovType(void) const
    { return m_covtype; }

  /// ��ʬ������Υ����פ����ꤹ��ؿ�
  void setCovType(const int _covtype)
    { m_covtype = _covtype; }
  
  /// ��ħ�̤μ��������֤��ؿ�
  int getDimension(void) const
    { return m_dimension; }
  
  /// ��ħ�̤μ����������ꤹ��ؿ�
  void setDimension(const int _value)
    { m_dimension = _value; }
  
  /// ��������֤��ؿ�
  int getNumMixtures(void) const
    { return m_num_mixtures; }
  
  /// ����������ꤹ��ؿ�
  void setNumMixtures(const int _value)
    { m_num_mixtures = _value; }

  /*
   * @brief  free energy ���֤��ؿ�
   * @param _t: �������ȤΥ���ǥå���
   *  @return free energy
   */
  double getLowerLimit(const int _t) const
    { return m_lowerlimit[_t]; }
  /*
   * @brief  Ķ�ѥ�᥿�� KL Divergence ���֤��ؿ�
   * @param _i: �ե졼���٥뺮�����ǤΥ���ǥå���
   *  @return free energy
   */
  double getKLDivergence(const int _i) const
    { return m_KLDivergence[_i]; }
  double getKLDivergence(void) const
    { return _sum(m_KLDivergence); }
  /*
   * @brief �ե졼���٥������ѿ��Υ���������Ʊ��ʬ�ۤ� EM ������Ȥκ�����˴ؤ����¤��֤��ؿ�
   * @param _t: �������ȤΥ���ǥå���
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

public: /* �ȥåץ�٥����� */
  
  void getMAP_Z(vector<IntVect>* _segment_fcc);


  /// ����������ħ�̹�¤�ΤؤΥݥ�����������ꤹ��ؿ�
  void setFeatures(CSegment* _features);
  
  /// �����ΰ����������ؿ�
  void init(const int _dimension, const int _num_segments,
            IntVect& _num_frames);
  
  /// �ե졼���٥������ѿ��򹹿�����ؿ�
  void updateZ(void);

  /*
   *  @brief GMM �Υ�ǥ�ѥ�᡼���򹹿�����ؿ�
   *  @param: _X : �üԥ�٥������ѿ�����ʬ�����Ψ������
   *  @return ���������Ǥˤ����� EM ������Ȥι����̤�ʿ��
   */
  double update(DoubleVect& __X);

  /// �����ѿ�����������ؿ�
  void initZ(const int _initOpt);

  /*
   * @brief Ķ�ѥ�᡼����Ķ�ѥ�᥿(ʬ�۶���)��ե����뤫����������ؿ�
   * @param _model_filename: GMM model filename
   */
  void initGlobalParam(const char* _model_filename,
                       const double& _beta0 = null,
                       const double& _xi0   = null,
                       const double& _eta0  = null);
  /*
   * @brief Ķ�ѥ�᡼����Ķ�ѥ�᥿(ʬ�۶���)����������ؿ�
   * @param _beta0: beta �λ���ʬ��Ķ�ѥ�᥿
   * @param _xi0:  xi  �λ���ʬ��Ķ�ѥ�᥿
   * @param _nu0:  nu  �λ���ʬ�ۤ�Ķ�ѥ�᥿
   * @param _eta0: eta �λ���ʬ�ۤ�Ķ�ѥ�᥿
   * @param _B0:   B   �λ���ʬ�ۤ�Ķ�ѥ�᥿
   */
  void initGlobalParam(const double& _beta0,
                       const double& _xi0,
                       CvMat* _nu0,
                       const double& _eta0,
                       CvMat* _B0);
  /*
   * @brief Ķ�ѥ�᡼����Ķ�ѥ�᥿(ʬ�۶���)����������ؿ�
   * @param _beta0: beta �λ���ʬ��Ķ�ѥ�᥿
   * @param _xi0:  xi  �λ���ʬ��Ķ�ѥ�᥿
   * @param _nu0:  nu  �λ���ʬ�ۤ�Ķ�ѥ�᥿
   * @param _eta0: eta �λ���ʬ�ۤ�Ķ�ѥ�᥿
   * @param _B0:   B   �λ���ʬ�ۤ�Ķ�ѥ�᥿
   */
  void initGlobalParam(CSegment* _features,
                       const double& _beta0, const double& _xi0,
                       const double& _eta0,  const double& _B0);
    /*
   * @brief Ķ�ѥ�᡼����Ķ�ѥ�᥿(ʬ�۶���)����������ؿ�
   * @param _features: nu0, B0, ����ꤹ�뤿����Ѥ�����ħ�ǡ���
   * @param _beta0: beta �λ���ʬ��Ķ�ѥ�᥿
   * @param _xi0:  xi  �λ���ʬ��Ķ�ѥ�᥿
   * @param _eta0: eta �λ���ʬ�ۤ�Ķ�ѥ�᥿
  */
  void initGlobalParam(CSegment* _features,
                       const double& _beta0, const double& _xi0,
                       const double& _eta0);
  /**
   * @brief: �üԡ��ե졼���٥������ѿ��ν���ͤ��Ѥ��ơ������̤�Ķ�ѥ�᡼������������ؿ�
   * @param: _X : �üԥ�٥������ѿ�����ʬ�����Ψ������
   */
  void initParam(DoubleVect& __X);
  
  /**
   * @brief: ��ǥ��Ķ�ѥ�᥿����˽�������롥����Ū�ˤϻ���ʬ�ۤ�Ķ�ѥ�᥿�򥳥ԡ����ƽ��������
   */
  void initParam();
  
public: /* ����ɽ���ϴؿ� */
  string info(void);

  void printZ(void);

public: /* �ե����������� */

  /// CVBGmm ���֥������Ȥξ�����֤��ؿ�
  string outputGMM(void);
  
  string outputPrior(void);
  
};

#endif // __VBTRAINER_H__
