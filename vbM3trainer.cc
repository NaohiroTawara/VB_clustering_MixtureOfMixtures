#include "vbM3trainer.h"

#define ISLOG 0
#define FoLS  0 // First order latent space modeling (Collapsed VB)

const int CVBMmmTrainer::INIT_X_RANDOM    = 1;
const int CVBMmmTrainer::INIT_X_KMEANS    = 2;
const int CVBMmmTrainer::INIT_X_MANUAL    = 4;
const int CVBMmmTrainer::INIT_X_PARAM     = 8;
const int CVBMmmTrainer::INIT_Z_RANDOM    = CVBGmmTrainer::INIT_Z_RANDOM;
const int CVBMmmTrainer::INIT_Z_KMEANS    = CVBGmmTrainer::INIT_Z_KMEANS;
const int CVBMmmTrainer::INIT_Z_MANUAL    = CVBGmmTrainer::INIT_Z_MANUAL;
const int CVBMmmTrainer::INIT_Z_PARAM     = CVBGmmTrainer::INIT_Z_PARAM;

//----------------------------------------------------------------------

void CVBMmmTrainer::initRandomSeed(int _n)
{
  m_rand = new Rand(_n);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::mallocCluster(CSegment* _segment)
{
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  vector<DoubleVect>::iterator iter_X       = m_X.begin();
  
  int num_segments     = getNumSegments();
  int dimension        = getDimension();
  int covtype          = getCovType();
  IntVect num_frames   = getNumFramesVector();
  for (; iter_cl != m_cluster.end(); ++iter_cl, ++iter_X)
  {
    /* (0-1-1) 特徴量へのポインタを設定し，メモリ領域の確保 */
    (*iter_cl)->setFeatures(_segment);
    
    /* (0-1-2) 分散のタイプを設定 */
    (*iter_cl)->setCovType(covtype);
    
    /* (0-1-3) メモリ領域の確保 */
    (*iter_cl)->init(dimension, num_segments, num_frames);
    
  }
}

//----------------------------------------------------------------------

void CVBMmmTrainer::setFeature(CSegment* _segment)
{
  ISpkrClustering::setFeature(_segment);
  mallocCluster(_segment);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::malloc(const int _num_clusters,
                           const int _num_mixtures)
{
  m_cluster.resize(_num_clusters);
  m_alpha.resize(_num_clusters);
  m_X.resize(_num_clusters);

  vector<CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (int i = 0; iter_cl != m_cluster.end(); ++iter_cl, ++i)
  {
    stringstream ss;
    ss << i;
    (*iter_cl) = new CVBGmmTrainer(ss.str(), _num_mixtures);
  }

}

//----------------------------------------------------------------------

void CVBMmmTrainer::free(void)
{
  vector<CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (; iter_cl != m_cluster.end(); ++iter_cl){
    if (*iter_cl) delete (*iter_cl);
  }
}

//----------------------------------------------------------------------

void CVBMmmTrainer::kMeansClustering(DoubleVect* _index)
{
  const int num_segments  = getNumSegments();
  const int dimension     = getDimension();
  const int num_clusters  = getNumClusters();
  const int num_mixtures  = num_clusters;
  
  /* K-means クラスタリング用の特徴量行列の作成 */
  CvMat* features = cvCreateMat(num_segments, dimension, CV_32F);

  /* PCA [dim x frame] -> [dim x 1] */
  CvMat* mean       = cvCreateMat(1, dimension, CV_32F);
  CvMat* eig_values = cvCreateMat(1, 1, CV_32F);
  CvMat* eig_vecs   = cvCreateMat(1, dimension, CV_32F);

  for (int t = 0; t < num_segments; ++t)
  { /* 各セグメント内のフレーム特徴量について PCA を行い，1フレームの大きさに情報圧縮を行う */
    cvCalcPCA(getSegment(t), mean, eig_values, eig_vecs, CV_PCA_DATA_AS_ROW);
    for (int d = 0; d < dimension; ++d)
      cvmSet(features, t, d, cvmGet(eig_vecs, 0, d));
  }
  
  // サンプルに対するクラスタ番号を格納する行列 (# of samples x 1)
  CvMat* clusters = cvCreateMat(num_segments, 1, CV_32S);
  
  /* K-means クラスタリング */
  cvKMeans2(features, num_mixtures, clusters, 
	    cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 1.0));
  
  DoubleVect::iterator iter_i = _index->begin();
  for (int n = 0; n < num_segments; ++n, ++iter_i)
    (*iter_i) = clusters->data.i[n];
  
  cvReleaseMat(&features);
  cvReleaseMat(&clusters);
  cvReleaseMat(&mean);
  cvReleaseMat(&eig_values);
  cvReleaseMat(&eig_vecs);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::initX(const int _initOpt,
                          const char* _latent_filename)
{
  int num_segments = (int)getNumSegments();
  vector<DoubleVect>::iterator iter_X = m_X.begin();
  for (;iter_X != m_X.end(); ++iter_X)
  /* iterate num of mixtures times */
    iter_X->resize(num_segments, 0.0);

  if (_initOpt & INIT_X_PARAM)
  {
    updateX();
  }
  else
  {
    int num_clusters = (int)getNumClusters();
    
    DoubleVect::iterator         iter_Xt;
    
    if (_initOpt & INIT_X_RANDOM)
    {
      // 乱数で発話レベル潜在変数の変分事後期待値の初期値を与える
      //double weight = 1.0 / static_cast<double>(num_clusters);
      DoubleVect sum_rnd(num_segments, 0.0);
      iter_X = m_X.begin();
      for (;iter_X != m_X.end(); ++iter_X)
      { /* iterate num of mixtures times */
        DoubleVect::iterator iter_Xt = iter_X->begin();
        DoubleVect::iterator iter_sumrnd = sum_rnd.begin();
        for (;iter_Xt != iter_X->end(); ++iter_Xt, ++iter_sumrnd)
        {
          /* iterate num of segments times */
          double weight = (*m_rand)();
          (*iter_sumrnd) += weight;
          (*iter_Xt) = weight;
        }
      }
      /* 正規化 */
      iter_X = m_X.begin();
      for (;iter_X != m_X.end(); ++iter_X)
      { /* iterate num of mixtures times */
        DoubleVect::iterator iter_Xt = iter_X->begin();
        DoubleVect::iterator iter_sumrnd = sum_rnd.begin();
        for (;iter_Xt != iter_X->end(); ++iter_Xt, ++iter_sumrnd)
        /* iterate num of segments times */
          (*iter_Xt) /= (*iter_sumrnd);
      }
    }
    else if (_initOpt & INIT_X_KMEANS)
    {
      // K-means 法でクラスタリングした結果を用いて発話レベル潜在変数の変分事後期待値の初期値を与える
      DoubleVect index(num_segments);
      kMeansClustering(&index);
      DoubleVect::iterator iter_inx = index.begin();
      for (int t = 0; iter_inx != index.end(); ++iter_inx, ++t)
        /* iterate num of segments */
        m_X[(*iter_inx)][t] = 1.0;
    }
    else if (_initOpt & INIT_X_MANUAL)
    {
      ifstream ifs(m_latent_filename.c_str());
      if (ifs == NULL)
      {
        cerr << "[vbM3trainer.cc]: cannot read file: " << m_latent_filename;
        exit(-1);
      }
      vector<DoubleVect>::iterator iter_X = m_X.begin();
      int buf;
      int cnt = 0;
      while (ifs >> buf)
      {
        if (cnt >= num_segments) {++cnt; continue;}
        if (buf < 0 || buf >= num_clusters)
          Error(1111, "[vbM3trainer.cc]: Invalid assignment of speaker");

        m_X[buf][cnt] = 1.0;
        ++cnt;
      }
      if (cnt != num_segments)
        Error(1111, "[vbM3trainer.cc]: Num of utterances is mismatched %d, %d", cnt, num_segments);
    }
    else
      Error(1111, "vbM3trainer.cc: unknown parameter for initialization of Z");
  }
  /*
  for (int u = 0; u < num_segments; ++u)
  {
    iter_X = m_X.begin();
    for (;iter_X != m_X.end(); ++iter_X)
      printf("%.3f, ", (*iter_X)[u]);
    (*m_ros) << endl;
  }
   */
}

//----------------------------------------------------------------------

void CVBMmmTrainer::initCluster(const int _initOpt)
{
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  vector<DoubleVect>::iterator iter_X       = m_X.begin();
  
  for (; iter_cl != m_cluster.end(); ++iter_cl, ++iter_X)
  { /* iterator for # of clusters */
    if (_initOpt & INIT_Z_PARAM)
    {
      /* パラメタの初期化 */
      (*iter_cl)->initParam();
      /* 初期化したパラメタを用いて，フレームレベル潜在変数の初期化 */
      (*iter_cl)->initZ(_initOpt);
    }
    else
    {
      /* フレームレベル潜在変数の初期化 */
      (*iter_cl)->initZ(_initOpt);
      /* 先に初期化した発話・フレームレベル潜在変数を用いて，パラメタを初期化する */
      (*iter_cl)->initParam(*iter_X);
    }
  }
}

//----------------------------------------------------------------------

void CVBMmmTrainer::initParam(const int _initOpt)
{
  DoubleVect::iterator iter_a = m_alpha.begin();
  if (_initOpt & INIT_Z_PARAM)
  {
    for (; iter_a != m_alpha.end(); ++iter_a)
      (*iter_a) = m_alpha0;
  }
  else
  {
    vector<DoubleVect>::iterator iter_X = m_X.begin();
    DoubleVect::iterator iter_a         = m_alpha.begin();
    for (; iter_a != m_alpha.end(); ++iter_a, ++iter_X)
      (*iter_a) = _sum(*iter_X) + m_alpha0;
  }}

//----------------------------------------------------------------------

void CVBMmmTrainer::init()
{
  init(INIT_X_RANDOM);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::init(const int _initOpt,
                         const char* _latent_filename)
{
  if (_initOpt & INIT_X_PARAM)
  {
    /* (0-1) 話者クラスタレベルのパラメタの初期化 */
    initParam(_initOpt);
    /* (0-2) 話者クラスタ潜在変数の初期化 */
    initX(_initOpt, _latent_filename);
    /* (0-3) 話者クラスタの初期化 */
    initCluster(_initOpt);
  }
  else
  {
    /* (0-1) 話者クラスタ潜在変数の初期化 */
    initX(_initOpt, _latent_filename);
    /* (0-2) 話者クラスタの初期化 */
    initCluster(_initOpt);
    /* (0-3) 話者クラスタレベルのパラメタの初期化 */
    initParam(_initOpt);
  }
  

}

//----------------------------------------------------------------------

void CVBMmmTrainer::setBasisFromFile(const char* _filename,
                                     const float& _a0,
                                     const float& _xi0,
                                     const float& _eta0)
{
  m_alpha0 = _a0;
  
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (; iter_cl != m_cluster.end(); ++iter_cl)
    (*iter_cl)->initGlobalParam(_filename, _xi0, _eta0);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::setBasisFromData(CSegment* _feature,
                                     const float& _a0,
                                     const float& _xi0,
                                     const float& _eta0)
{
  m_alpha0 = _a0;
  
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (; iter_cl != m_cluster.end(); ++iter_cl)
    (*iter_cl)->initGlobalParam(_feature, _a0, _xi0, _eta0);
}

//----------------------------------------------------------------------

void CVBMmmTrainer::updateX(void)
{
  int num_segments = getNumSegments();
  int num_clusters = getNumClusters();
  DoubleVect sumX(num_segments, -DBL_MAX); // sumX: 1 x T

  IntVect alignment(num_segments, -1);       // 各セグメントの強制アライメント
  DoubleVect max_Xt(num_segments, -DBL_MAX); // 各セグメントの最大事後確率値

  vector<DoubleVect>::iterator iter_X       = m_X.begin();
  DoubleVect::iterator iter_a               = m_alpha.begin();
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();

#if FoLS
  for (int u = 0; u < num_segments; ++u)
  { /* iterate for num of Segments: u = 1, ..., U */
//    (*m_ros) << u <<": "<<endl;
    (*m_ros) << ".";
    (*m_ros).flush();
    iter_X  = m_X.begin();
    iter_a  = m_alpha.begin();
    iter_cl = m_cluster.begin();
    double sumX = -DBL_MAX;
    double maxX = -DBL_MAX;
    double algX = -1;
    
    DoubleVect tmpX(num_clusters);
    DoubleVect::iterator iter_tmpX = tmpX.begin();

//    (*m_ros) << "B      =";
//    for (int i=0;i<10;++i)
//      (*m_ros) << cvmGet(m_cluster.at(i)->getGmm()->getB(0),0,5) << ",";
//    (*m_ros) << endl;
//    (*m_ros) << "gamma  =";
    for (int i = 0;iter_X != m_X.end(); ++iter_cl, ++iter_X, ++iter_a, ++iter_tmpX, ++i)
    { /* iterate by num of clusters: j = 1, ..., S */
//      if(u==8) (*m_ros) << (*iter_X)[u] <<endl;
      
      // (1) update {Theta^{/u}, h^{/u}}
      (*iter_a) = (*iter_a) - (*iter_X)[u];
      ((CFoLSVBGmmTrainer*)(*iter_cl))->updateZ((*iter_X)[u], u);
//      ((CFoLSVBGmmTrainer*)(*iter_cl))->updateZ((*iter_X), u);
      
      // (2) update {gamma_{u}, h_{u}}
      (*iter_tmpX)    = log(*iter_a) + (*iter_cl)->getProdSumLogZ(u);
      sumX            = LAddS(sumX, (*iter_tmpX));
      if ((*iter_tmpX) > max_Xt[u])
      { maxX = (*iter_tmpX); algX = i;}

//      printf("%.4f,",(*iter_X)[u]);
    }
    max_Xt[u] = maxX; alignment[u] = algX;

//    if(u==0) (*m_ros) << endl;
//    (*m_ros) << endl;
//    (*m_ros) << "B_{/u} =";
//    for (int i=0;i<10;++i)
//      (*m_ros) << cvmGet(m_cluster.at(i)->getGmm()->getB(0),0,5) << ",";
//    (*m_ros) << endl;
//    (*m_ros) << "gamma  =";
    
    iter_X = m_X.begin();     iter_tmpX = tmpX.begin();
    iter_a = m_alpha.begin(); iter_cl   = m_cluster.begin();
    for (int i = 0; iter_X != m_X.end(); ++iter_cl, ++iter_X, ++iter_a, ++iter_tmpX, ++i)
    { /* iterate for # of clusters */
      // (3) normalize {gamma_{u} = i, a_{i}}
      /* 正規化 */
      (*iter_X)[u] = exp(*iter_tmpX - sumX);
      if ((*iter_X)[u] < 1.0E-3) (*iter_X)[u] = 0.0;
      if ((*iter_X)[u] > 0.999)  (*iter_X)[u] = 1.0;
      (*iter_a) = (*iter_a) + (*iter_X)[u];
//      printf("%.4f,",(*iter_X)[u]);

      // (4) update {Theta, h}
      ((CFoLSVBGmmTrainer*)(*iter_cl))->update((*iter_X)[u], u);
      
    } // # of clusters
//    (*m_ros) << endl;


//    iter_X    = m_X.begin();
//    iter_cl   = m_cluster.begin();
//    for (; iter_X != m_X.end(); ++iter_X, ++iter_cl)
//    { /* iterate for # of clusters */
//      ((CFoLSVBGmmTrainer*)(*iter_cl))->update((*iter_X)[u], u);
//      ((CFoLSVBGmmTrainer*)(*iter_cl))->update((*iter_X), u);
//    }
//    (*m_ros) << "B      =";
//    for (int i=0;i<10;++i)
//      (*m_ros) << cvmGet(m_cluster.at(i)->getGmm()->getB(0),0,5) << ",";
//    (*m_ros) << endl;
//    break;
  } // # of segments


#else
  double sumAlpha  = _sum(m_alpha);
  for (int i = 0;iter_X != m_X.end(); ++iter_cl, ++iter_X, ++iter_a, ++i)
  { /* iterate by num of mixtures: j = 1, ..., S */
    DoubleVect::iterator iter_Xu   = iter_X->begin();
    DoubleVect::iterator iter_sumX = sumX.begin();
    double logalpha = digamma(*iter_a) - digamma(sumAlpha);

    IntVect::iterator iter_al = alignment.begin();
    DoubleVect::iterator iter_mXt = max_Xt.begin();
    for (int u = 0; iter_Xu != iter_X->end(); ++iter_Xu, ++iter_sumX, ++iter_al, ++iter_mXt, ++u)
    {
      /* iterate for num of Segments: u = 1, ..., U */

      (*iter_Xu)   =  logalpha + (*iter_cl)->getProdSumLogZ(u);
      (*iter_sumX) = LAddS((*iter_sumX), (*iter_Xu));
      if (*iter_Xu > *iter_mXt)
      {
        *iter_mXt = *iter_Xu;
        *iter_al  = i;
      }
    }
  }
  /* 正規化 */
  iter_X = m_X.begin();
  for (int i=0;iter_X != m_X.end(); ++iter_X,++i)
  { /* iterate num of mixtures times */
    DoubleVect::iterator iter_Xu   = iter_X->begin();
    DoubleVect::iterator iter_sumX = sumX.begin();
    IntVect::iterator iter_al      = alignment.begin();
    DoubleVect::iterator iter_mXt  = max_Xt.begin();
    for (;iter_Xu != iter_X->end(); ++iter_Xu, ++iter_sumX, ++iter_al, ++iter_mXt)
    {
      /* iterate num of Segments times */
#if ISLOG
      (*iter_Xu) = (*iter_Xu) - (*iter_sumX);
#else
      (*iter_Xu) = exp((*iter_Xu) - (*iter_sumX));
      if ((*iter_Xu) < 1.0E-3) (*iter_Xu) = 0.0;
      if ((*iter_Xu) > 0.999)  (*iter_Xu) = 1.0;
#endif
//    (*m_ros) << (*iter_Xu) << ",";
    }
//    (*m_ros) << endl;
  }
#endif
  
}

//----------------------------------------------------------------------

double CVBMmmTrainer::addAlpha(void)
{
  double sum = 0.0;

  DoubleVect::iterator it = m_alpha.begin();
  for (; it != m_alpha.end(); ++it)
    sum += *it;
  
  return sum;
}

//----------------------------------------------------------------------

void CVBMmmTrainer::updateAlpha(void)
{
  DoubleVect::iterator iter_a = m_alpha.begin();
  vector<DoubleVect>::iterator iter_X = m_X.begin();

  for (; iter_a != m_alpha.end(); ++iter_a, ++iter_X)
  {  /* iterate num of Clusters times */
    double Nj;
#if ISLOG
    Nj = _logsum(*iter_X);
#else
    Nj = _sum(*iter_X);
#endif
    (*iter_a) = Nj + m_alpha0;
  }
}

//----------------------------------------------------------------------

double CVBMmmTrainer::calcBIC(void)
{
  int covtype = m_cluster.at(0)->getCovType();
  int dimension    = getDimension();
  int num_clusters = getNumClusters();
  if (covtype == FULLC)
    return calcLowerBound() 
      - 0.5 * (dimension * dimension * dimension + 1)
            * num_clusters * m_cluster.at(0)->getNumMixtures() * 2;
  else
    return calcLowerBound() 
      - 0.5 * (dimension * 2 + 1)
            * num_clusters * m_cluster.at(0)->getNumMixtures() * 2;
}

//----------------------------------------------------------------------

double CVBMmmTrainer::calcFreeEnergy(void)
{
  return calcLowerBound() - calcKLDivergence();
}

//----------------------------------------------------------------------

double CVBMmmTrainer::calcKLDivergence(void)
{
  double KLDivergence = 0.0;
  int num_clusters    = getNumClusters();
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (; iter_cl != m_cluster.end(); ++iter_cl)
    /* iterate num of clusters */    
    KLDivergence += (*iter_cl)->getKLDivergence();

  vector<double> alpha0(num_clusters, m_alpha0);
  KLDivergence += _KLDir(m_alpha, alpha0);
  
  return KLDivergence;
}

//----------------------------------------------------------------------

double CVBMmmTrainer::calcLowerBound(void)
{
  double lowerBound = 0.0;
  int num_clusters  = getNumClusters();
  
  vector <CVBGmmTrainer*>::iterator iter_cl;
  vector<DoubleVect>::iterator iter_X;
  DoubleVect::iterator iter_Xt;
  DoubleVect::iterator iter_a;
  vector<double> alpha0(num_clusters, m_alpha0);

  iter_a  = m_alpha.begin();
  iter_X  = m_X.begin();
  iter_cl = m_cluster.begin();
  for (; iter_cl != m_cluster.end(); ++iter_cl, ++iter_X, ++iter_a)
  { /* iterate num of clusters */
    iter_Xt = iter_X->begin();
    for (int t = 0;iter_Xt != iter_X->end(); ++iter_Xt, ++t)
    { /* iterate num of segments */
      lowerBound += ((*iter_Xt)==0.0) ? 0.0:
        (*iter_Xt) * (log((*iter_a) * (*iter_Xt)) + (*iter_cl)->getLowerLimit(t));
    }
  }
  return lowerBound;
}

//----------------------------------------------------------------------

int CVBMmmTrainer::run(void)
{
  int num_clusters = getNumClusters();
  int iteration    = 0;
  double dif_N     = 0.0;
  double pre_fe    = -DBL_MAX;
  double delta_fe  = DBL_MAX;
  double t1, t2, s_time = 0;

  /* (1) メインループ */
  do
  {    
    vector <CVBGmmTrainer*>::iterator iter_cl;
    vector<DoubleVect>::iterator iter_X;
    //----------------------------------
    /* (1-1) E-step */
    (*m_ros) << "=================" << endl;
    (*m_ros) << "processing E-step";
    (*m_ros).flush();
    t1 = gettimeofday_sec();

#if !FoLS
    iter_cl = m_cluster.begin();
    iter_X  = m_X.begin();
    for (int j = 0; iter_cl != m_cluster.end(); ++iter_cl, ++iter_X, ++j)
    {
      (*m_ros) << ".";
      (*m_ros).flush();
      /// (1-1-1) フレームレベル潜在変数の更新
      (*iter_cl)->updateZ();
    }
    (*m_ros) << endl;
    (*m_ros) << "  Lower bound: "   << getLowerBound()   << endl;
    (*m_ros) << "  KL Divergence: " << getKLDivergence() << endl;
#endif
    /// (1-1-2) 話者クラスタレベル潜在変数の更新
    updateX();
    /*
    (*m_ros) <<endl;
    iter_X  = m_X.begin();
    for (int i = 0; i < getNumSegments(); ++i)
    {
      for (int j = 0; j < getNumClusters(); ++j)
        (*m_ros) << m_X[j][i] << ",";
      (*m_ros) << endl;
    }
*/
    //----------------------------------
    /* (1-2) M-step */
#if !FoLS
    (*m_ros) << "processing M-step";
    /// (1-2-1) 話者レベルクラスタのモデルパラメータの更新
    updateAlpha();
    /// (1-2-2) フレームレベルクラスタのモデルパラメータ・統計量の更新
    iter_cl = m_cluster.begin();
    iter_X  = m_X.begin();
    dif_N = 0.0;
    for (int j = 0; iter_cl != m_cluster.end(); ++iter_cl,++iter_X,  ++j)
    {
      (*iter_cl)->update(*iter_X);
      (*m_ros) << ".";
      (*m_ros).flush();
    }
#endif
    /*
    IntVect segment_cc;
    int num_clusters = getClusteringResult(&segment_cc);
    Spkr_cl_evaluation* spkr_eval =
      new Spkr_cl_evaluation(num_segments, num_clusters, g_true_num_spkr);
    spkr_eval->readMlf(g_mlf_filename);
    Result result = spkr_eval->evaluate(segment_cc);
    (*m_ros) << "acp: "<<result.acp<<", asp: "<<result.asp << ", K: "<<result.K<<endl;
    (*m_ros) << "BIC: " << getBIC() << endl;
    delete spkr_eval;*/

    (*m_ros) << endl;
    (*m_ros) << "  Lower bound: "   << getLowerBound()   << endl;
    (*m_ros) << "  KL Divergence: " << getKLDivergence() << endl;

    double fe = calcFreeEnergy();
    delta_fe  = fe - pre_fe;
    (*m_ros) << "  FreeEnergy = "<< fe
             << "(delta: "       << delta_fe
             << ")"              << endl;
    pre_fe = fe;
    dif_N /= static_cast<double>(num_clusters);

    ++iteration;
    t2 = gettimeofday_sec();
    s_time = s_time + t2 - t1;
//    showResult();
  } while (static_cast<double>(abs(delta_fe)) > m_thresh_N &&
           iteration < m_max_iteration);
//  } while(iteration < m_max_iteration);
  (*m_ros) << "====================" << endl;
  (*m_ros) << "num of iterations: " << iteration << endl;
  (*m_ros) << "time:              " << s_time << endl;
  (*m_ros) << "time / ite:        "
           << (static_cast<double>(s_time) / static_cast<double>(iteration))
           << endl;
  (*m_ros) << "====================" << endl;  
  return 0;
}

//----------------------------------------------------------------------

void CVBMmmTrainer::getMAP_X(IntVect* _segment_cc)
{
  int num_segments = getNumSegments();
//  int speaker_id = 0;
  _segment_cc->resize(num_segments);
  
  DoubleVect maxXt(num_segments, -DBL_MAX);

  vector<DoubleVect>::iterator iter_X;
  DoubleVect::iterator iter_maxXt;
  DoubleVect::iterator iter_Xt;
  IntVect::iterator iter_cc;
  
  iter_X     = m_X.begin();
  for (int j = 0; iter_X != m_X.end(); ++iter_X, ++j)
  { /* iterate num of clusters: j = 0, ..., S */
    iter_Xt    = iter_X->begin();
    iter_maxXt = maxXt.begin();
    iter_cc   = _segment_cc->begin();
    for (;iter_Xt != iter_X->end(); ++iter_Xt, ++iter_maxXt, ++iter_cc)
    {/* iterate num of segments times: t = 0, ..., T */
      if ((*iter_Xt) > (*iter_maxXt))
      {
        (*iter_maxXt) = (*iter_Xt);
        (*iter_cc)    = j;
      }
    }
  }
}

//----------------------------------------------------------------------

int CVBMmmTrainer::getClusteringResultSeg(IntVect* _segment_cc)
{
  getMAP_X(_segment_cc);
  
  /* 空のクラスタを削除 */
  vector<IntVect > clusters(getNumClusters());
  IntVect::iterator  iter_cc = _segment_cc->begin();
  for (int t = 0; iter_cc != _segment_cc->end(); ++iter_cc, ++t)
    /* iterate num of segments: t = 0, ..., T */
    clusters[(*iter_cc)].push_back(t);

  int num_clusters = 0;
  vector<IntVect>::iterator iter_cl = clusters.begin();
  for (; iter_cl != clusters.end(); ++iter_cl)
  {
    if (iter_cl->size() == 0) continue;
    IntVect::iterator iter_clx = iter_cl->begin();
    for (; iter_clx != iter_cl->end(); ++iter_clx)
      _segment_cc->at(*iter_clx) = num_clusters;
    ++num_clusters;
  }
  
  return num_clusters;
}

//----------------------------------------------------------------------

int CVBMmmTrainer::getClusteringResultCl(vector<IntVect>* _cluster)
{
  IntVect segment_cc;
  int numclusters = getClusteringResultSeg(&segment_cc);

  _cluster->resize(numclusters, IntVect());
  
  vector<IntVect>::iterator iter_cl  = _cluster->begin();
  for(int i = 0; iter_cl != _cluster->end(); ++iter_cl, ++i)
  {
    IntVect::iterator iter_cc = segment_cc.begin();
    for (int j = 0; iter_cc != segment_cc.end(); ++iter_cc, ++j)
      if ((*iter_cc) == i)
        _cluster->at(*iter_cc).push_back(j);
  }
  
  return getNumClusters();
}

  
  
//----------------------------------------------------------------------

void CVBMmmTrainer::setRunParameter(const float _thresh_N,
                                    const int   _max_iteration)
{
  m_thresh_N      = _thresh_N;
  m_max_iteration = _max_iteration;
}

//----------------------------------------------------------------------

string CVBMmmTrainer::printInfo()
{
  ostringstream os;
  
  os << "Num segment  = " << getNumSegments() << endl;
  os << "Dimension    = " << getDimension()   << endl;
  os << "Num clusters = " << getNumClusters() << endl;
  
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  for (int i = 0; iter_cl != m_cluster.end(); ++iter_cl, ++i)
  {
    os << "-------------------" <<endl;
    os << "cluster: " << i + 1 << endl;
    os << (*iter_cl)->info();
    os << endl;
  }
  return os.str();
}

//----------------------------------------------------------------------

string CVBMmmTrainer::outputModel()
{
  ostringstream os;
  int numclusters = getNumClusters();
  
  vector <CVBGmmTrainer*>::iterator iter_cl = m_cluster.begin();
  DoubleVect::iterator              iter_a  = m_alpha.begin();

  os << "<BEGIN_PRIOR> "  << endl;
  os << "<ALPHA> "        << m_alpha0 << endl;
  os << m_cluster[0]->outputPrior();
  os << "<END_PRIOR>"     << endl;
  
  os << "<BEGIN_CLUSTER> " << endl;
  os << "<NUMCLUSTERS> "   << numclusters << "\n";

  for (int i = 0; iter_cl != m_cluster.end(); ++iter_cl, ++iter_a, ++i)
  {
    os << "<CLUSTER> " << i + 1     << "\n";
    os << "<ALPHA> "   << (*iter_a) << "\n";
    os << (*iter_cl)->outputGMM();
  }

  os << "<END_CLUSTER>" << endl;
  return os.str();
}

//----------------------------------------------------------------------

void CVBMmmTrainer::getModelAllignment(vector<IntVect>* _segment_fcc, const int _cl)
{
  _segment_fcc->resize(getNumSegments());
  vector<IntVect>::iterator iter_fccu    = _segment_fcc->begin();
  for (int i = 0; iter_fccu != _segment_fcc->end(); ++iter_fccu, ++i)
  {
    int numframes = getNumFrames(i);
    iter_fccu->resize(numframes);
  }
  
  m_cluster[_cl]->getMAP_Z(_segment_fcc);
  
}

//----------------------------------------------------------------------
//                    End: vbtrainer.cc
//----------------------------------------------------------------------
