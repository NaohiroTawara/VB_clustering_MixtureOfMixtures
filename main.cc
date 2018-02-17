//
//  main.cc
//  VB. ver 1.2.0
//
//  (1) First order latent space collapsed VB 法を実装
//  (2) Sub Dimension に対応
//  (3) Free energy 算出に関するバグフィックス
//  Created by 俵 直弘 on 13/3/26.
//  Copyright (c) 2012年 俵 直弘. All rights reserved.
//

#include "vbM3trainer.h"
#include "evalK.h"


using namespace std;

//----------------------------------------------------------------------

const float DEF_INIT_ALPHA    = 1;
const float DEF_INIT_BETA     = 1;
const float DEF_INIT_XI       = 1;
const float DEF_INIT_ETA      = 1;
const float DEF_THRESH_N      = 0;
const int   DEF_MAX_ITERATION = 0;   // 反復回数の最大値
const int   DEF_SUB_DIMENSION = -1;

int g_subdimension = DEF_SUB_DIMENSION;

string g_script_filename = "";
string g_output_dirname = "gmmdefs";
string g_label_name      = "";
string g_mlf_filename    = "";
string g_model_filename  = "UNI";

float  g_alpha0      = DEF_INIT_ALPHA;
float  g_beta0       = DEF_INIT_BETA;
float  g_xi0         = DEF_INIT_XI;
float  g_eta0        = DEF_INIT_ETA;
float  g_thresh_N    = DEF_THRESH_N;
float  g_B0          = -1;

int g_output_model = 0;  //パラメタの事後ハイパーパラメタを出力するか

int    g_max_iteration   = DEF_MAX_ITERATION; // 最大イタレーション数
int    g_num_clusters    = 1;                 // 話者クラスタ数
int    g_num_mixtures    = 1;                 // 混合分布数
int    g_covtype         = DIAGC;             // covariance type (full:1, diagonal:0)
int    g_trace           = 0;                 // debug 出力の有無
int    g_true_num_spkr   = 1;

int    g_thread_num   = 1; // 並列実行するスレッド数
int    g_serialnumber = 0; // スレッドの通し番号

vector<string> g_spkr_label; // 正解ラベル系列

pthread_mutex_t g_mutex; //2つのスレッド間で変数の保護を行う

float g_rand   = 0; // 乱数シード

string g_flv_latent_filename  = "RANDOM"; // フレームレベル潜在変数の初期値の与え方
string g_ulv_latent_filename  = "RANDOM"; // 発話レベル潜在変数の初期値の与え方

//----------------------------------------------------------------------
//----------------------------------------------------------------------

static const char* USAGE_MESSAGE =
"\nUSAGE: vbtrainer [options] ...\n"
"\n"
" Option                                      Default\n"
"\n"
" -a f      Set initial alpha to f            1\n"
" -b f      Set initial beta to f             1\n"
" -c N      Set serial number to N            0\n"
" -d N      Set subdimension to N             data_dimension¥n"
" -e f      Set initial eta to f              1\n"
" -f N      Set covariance type flag to N     0\n"
" -h        show this message\n"
" -i N      Set max iterations to N           100\n"
" -l N      set the number of thread to N     1\n"
" -m i      number of mixtures                1\n"
" -n N      Set number of init clusters to N  1\n"
" -r        Initialize random seed            \n"
" -s n      *Set true number of speakers      1\n"
" -t f      EM count threshold                0.0\n"
" -x f      Set initial xi to f               1\n"
" -H b      Set model ouput flag to b         0\n"
" -B f      Set initial B  to f               none\n"
" -I f      set mlf file to f                 none\n"
" -L s      Set segment label to s            none\n"
" -M mmf    Set GMM model file to mmf         KMEANS\n"
" -O mmf    *Set output file to mmf           none\n"
" -S f      *Set script file to f             none\n"
" -T N      Set trace flags to N              0\n"
" -X s      Set initialized parameter for f-lv latent variables none\n"
" -Z s      Set initialized parameter for u-lv latent variables none\n"
"\n"
"*: required\n"
;

//----------------------------------------------------------------------

void Usage(void)
{
  cerr << USAGE_MESSAGE;
  exit(1);
}

//----------------------------------------------------------------------

void GetOption(int _argc, char* _argv[])
{
  extern char* optarg;
  
  int opt;
  while ((opt = getopt(_argc, _argv, "a:b:c:d:e:f:i:l:m:n:r:s:t:x:B:H:I:L:M:O:S:T:X:Z:h:")) != -1)
  {
    switch (opt)
    {
      case 'a':
        g_alpha0 = atof(optarg);
        break;
      case 'b':
        g_beta0 = atof(optarg);
        break;
      case 'c':
        g_serialnumber = atoi(optarg);
        break;
      case 'd':
        g_subdimension = atoi(optarg);
        break;
      case 'e':
        g_eta0 = atof(optarg);
        break;
      case 'f':
        g_covtype = atoi(optarg);
        break;
      case 'i':
        g_max_iteration = atoi(optarg);
        break;
      case 'l':
        g_thread_num = atoi(optarg);
        break;
      case 'm':
        g_num_mixtures = atoi(optarg);
        break;
      case 'n':
        g_num_clusters = atoi(optarg);
        break;
      case 'r':
      {
        float s = (float)atof(optarg);
        if (s == -1.0)
          g_rand = static_cast<unsigned long>(time(0));
        else
          g_rand = static_cast<unsigned long>(s);
        break;
      }
      case 's':
        g_true_num_spkr = atoi(optarg);
        break;
      case 't':
        g_thresh_N = atof(optarg);
        break;
      case 'x':
        g_xi0 = atof(optarg);
        break;
      case 'B':
        g_B0  = atof(optarg);
        break;
      case 'H':
        g_output_model = atoi(optarg);
        break;
      case 'I':
        g_mlf_filename = optarg;
        break;
      case 'L':
        g_label_name = optarg;
        break;
      case 'M':
        g_model_filename = optarg;
        break;
      case 'O':
        g_output_dirname = optarg;
        break;
      case 'S':
        g_script_filename = optarg;
        break;
      case 'T':
        g_trace = atoi(optarg);
        break;
      case 'X':
        g_ulv_latent_filename = optarg;
        break;
      case 'Z':
        g_flv_latent_filename = optarg;
        break;
      case 'h':
      default:
        Usage();
    }
  }
  if (g_true_num_spkr <= 0)
    Error(1111, "main.cc: please set the valid value for option '-s'");
  if (g_output_dirname == "")
    Error(1111, "main.cc: please set the valid value for option '-O'");
  if (g_script_filename == "")
    Error(1111, "main.cc: please set the valid value for option '-S'");
}

//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

struct thread_info
{
  CVBMmmTrainer* vbt;
  ofstream*       logfilename;
};

//----------------------------------------------------------------------

void *childThread(void* _info)
{
  CVBMmmTrainer* vbt = ((thread_info *)_info)->vbt;

  /* ================================ */
  /* イテレーションを開始する */
  /* ================================ */
  vbt->run();
  return 0;
}

//----------------------------------------------------------------------

int run(vector <CVBMmmTrainer*> _v_vbt,
        vector<ofstream*> _v_log)
{
  cout << "[" << getpid() << "]start" <<endl;
  
  vector <struct thread_info> info_vec(g_thread_num);

  // vbt instance
  vector<CVBMmmTrainer*>::iterator iter_vbt     = _v_vbt.begin();
  // log file
  vector<ofstream*>::iterator iter_log          = _v_log.begin();
  // Information file for each thread
  vector <struct thread_info>::iterator iter_if = info_vec.begin();
  
  for (; iter_vbt != _v_vbt.end(); ++iter_vbt, ++iter_log, ++iter_if)
  {
    iter_if->vbt         = (*iter_vbt);
    iter_if->logfilename = (*iter_log);
  }

  /* スレッドの生成 */
  vector <pthread_t> thread_id(g_thread_num);
  vector <pthread_t>::iterator iter_id = thread_id.begin();
  iter_if = info_vec.begin();
  for (; iter_id != thread_id.end(); ++iter_id, ++iter_if)
  {
    pthread_mutex_lock(&g_mutex);
    int status = pthread_create(&(*iter_id), NULL,
                                childThread, (void *)&(*iter_if));
    if(status != 0)
      cerr << "  pthread_create :" << strerror(status) << endl;
    else
      cout << "  thread_id = "
           << (*iter_id) << " start" << endl;
    pthread_mutex_unlock(&g_mutex);
  }
  /* スレッドの待ち合わせ */
  iter_id = thread_id.begin();
  for (; iter_id != thread_id.end(); ++iter_id)
  {
    void* result;
    pthread_join((*iter_id), &result);
    cout << "  thread_id = "
    << (*iter_id) << " end" << endl;
  }

  return 0;
}

//----------------------------------------------------------------------
/* ================================ */
/* 評価 */
/* ================================ */
int eval(vector <CVBMmmTrainer*> _v_vbt,
         vector<ofstream*> _v_log)
{
  int num_thread = _v_vbt.size();
  // vbt instance
  vector<CVBMmmTrainer*>::iterator iter_vbt     = _v_vbt.begin();
  // log file
  vector<ofstream*>::iterator iter_log          = _v_log.begin();

  double acc_K  = 0;
  double acc_cl = 0;
  double acc_FE = 0;

  for (; iter_vbt != _v_vbt.end(); ++iter_vbt, ++iter_log)
  {
    CVBMmmTrainer* vbt = (*iter_vbt);
    ofstream* ofs      = (*iter_log);
    IntVect segment_cc;

    //    cout <<vbt->outputModel();
    int num_clusters  = vbt->getClusteringResultSeg(&segment_cc);
    int num_segments  = vbt->getNumSegments();
    float lowerBound  = vbt->getLowerBound();
    float KL          = vbt->getKLDivergence();
    float freeEnergy  = vbt->getFreeEnergy();
  
    // 発話長ベクトルの算出
    DoubleVect length_vec(num_segments);
    DoubleVect::iterator iter_l = length_vec.begin();
    for (int i = 0; iter_l != length_vec.end(); ++iter_l, ++i)
      (*iter_l) = vbt->getNumFrames(i);
  
    CSpkrClEvaluation_K* evalK =
      new CSpkrClEvaluation_K(num_segments, num_clusters, g_true_num_spkr);
  
    evalK->setMlf(g_spkr_label, length_vec);
    Result resultK = evalK->evaluate(segment_cc);
  
    (*ofs) << "acp: "           << resultK.result["acp"]  << endl;
    (*ofs) << "asp: "           << resultK.result["asp"]  << endl;
    (*ofs) << "K: "             << resultK.result["K"]    << endl;
    (*ofs) << "clusters: "      << num_clusters           << endl;
    (*ofs) << "Lower bound: "   << lowerBound             << endl;
    (*ofs) << "KL Divergence: " << KL                     << endl;
    (*ofs) << "Free energy: "   << freeEnergy             << endl;
  
    vector<IntVect> cluster_cc;
    vbt->getClusteringResultCl(&cluster_cc);
    (*ofs) << endl;
  
    vector<IntVect>::iterator iter_c = cluster_cc.begin();
    for (int i = 0; iter_c != cluster_cc.end(); ++iter_c, ++i)
    {
      (*ofs) << "cluster" << i << ": ";
      IntVect::iterator iter_c_c = iter_c->begin();
      for (; iter_c_c != iter_c->end(); ++iter_c_c)
        (*ofs) << (*iter_c_c) << ", ";
      (*ofs) << endl;
    }
  
    (*ofs) << endl;
    IntVect::iterator iter_sc = segment_cc.begin();
    (*ofs) << "Allignment: ";
    for (; iter_sc != segment_cc.end(); ++iter_sc)
      (*ofs) << (*iter_sc) << ",";
    (*ofs) << endl;
  
    if (g_output_model)
    {
      (*ofs) << endl;
      (*ofs) << "Model: " << endl;
      (*ofs) << vbt->outputModel() << endl;
    }
  
    vector<IntVect> _segment_fcc;
    int cl = vbt->getNumClusters();
    for (int j = 0; j < cl; ++j)
    {
      (*ofs) << "cluster " << j << endl;
      vbt->getModelAllignment(&_segment_fcc, 0);
      vector<IntVect>::iterator iter_fccu    = _segment_fcc.begin();
      for (; iter_fccu != _segment_fcc.end(); ++iter_fccu)
      {
        IntVect::iterator iter_fccut = iter_fccu->begin();
        for (; iter_fccut != iter_fccu->end(); ++iter_fccut)
          (*ofs) << (*iter_fccut) << ",";
        (*ofs) << endl;
      }
    }

    cout << "----------------------------------"   << endl;
    cout << "K:               " << resultK.result["K"] << endl;
    cout << "clusters:        " << num_clusters        << endl;
    cout << "Free energy:     " << freeEnergy          << endl;

    acc_K  += resultK.result["K"];
    acc_cl += num_clusters;
    acc_FE += freeEnergy;
    
    delete evalK;
  }
  cout << "=================================="   << endl;
  cout << "ave_K:           " << acc_K  / static_cast<double>(num_thread) << endl;
  cout << "ave_clusters:    " << acc_cl / static_cast<double>(num_thread) << endl;
  cout << "ave_Free energy: " << acc_FE / static_cast<double>(num_thread) << endl;

  return 0;
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

int main(int _argc, char* _argv[])
{
  /* ================================ */
  /* オプションの読み込み */
  /* ================================ */

  GetOption(_argc, _argv);
  
  /* ================================ */
  /* 特徴量の読み込み */
  /* ================================ */
  
  cout << "Feature files:" << endl;
  cout << "  filename:  " << g_script_filename << endl;
  CSegment* feature = new CSegment(g_script_filename.c_str());
  int numsegments = feature->getNumSegments();
  for (int i = 0; i < numsegments; ++i)
    g_spkr_label.push_back(feature->getSpeakerLabel(i));
  
  if (g_subdimension != DEF_SUB_DIMENSION)
    feature->compression(g_subdimension);
  /* ================================ */
  /* オプションで指定されたスレッドの個数と同じ数の */
  /* CVBMmmTrainer インスタンスを生成する */
  /* ================================ */
  
  vector <CVBMmmTrainer*> v_vbt;
  vector <ofstream*> v_log;
  for (int i = 0; i < g_thread_num; ++i)
  {
    /* ================================ */
    /* log ファイルを作成 */
    /* ================================ */
    char tmp_str[10];
    sprintf(tmp_str,"%03d",i + g_serialnumber);
    stringstream ss;
    ss << g_output_dirname << "/" << "log_" << tmp_str;
    ofstream* ofs = new ofstream(ss.str().c_str());
    v_log.push_back(ofs);
    
    /* ================================ */
    /* インスタンスの生成 */
    /* ================================ */
    CVBMmmTrainer* vbt;
    cout << endl;
    cout << "Creating new instance for vbt[" << i << "]" << endl;
    if (i==0)
      vbt = new CVBMmmTrainer(g_num_clusters, g_num_mixtures, g_covtype, &cout);
    else
      vbt = new CVBMmmTrainer(g_num_clusters, g_num_mixtures, g_covtype, ofs);
    /* ================================ */
    /* 特徴量をセット */
    /* ================================ */
    vbt->setFeature(feature);
  
    /* ================================ */
    /* 基底ファイルの読み込み */
    /* ================================ */
    cout << "  # of clusters:                   " << g_num_clusters << endl;
    cout << "  # of mixtures:                   " << g_num_mixtures << endl;
    cout << "  Prior:                           ";
    if (g_model_filename == "UNI")
    {
      cout << "DATA mean and covariance" << endl;
      vbt->setBasisFromData(feature, g_alpha0, g_xi0, g_eta0);
    }
    else
    {
      cout << "Model file "
           << " << " << endl
           << "   "  << g_model_filename << endl;
      vbt->setBasisFromFile(g_model_filename.c_str(), g_alpha0, g_xi0, g_eta0);
    }

    /* ===============================  */
    /* イテレーションの設定 */
    /* ===============================  */
    vbt->setRunParameter(g_thresh_N, g_max_iteration);
    //vbt->setRunParameter(g_thresh_N, 1);
    
    /* ================================ */
    /* 初期化 */
    /* ================================ */
    vbt->initRandomSeed(g_rand + i);
    int opflag = 0;
    cout << "  UTTERANCE level latent variable: ";
    if (g_ulv_latent_filename == "RANDOM")
    {
      opflag |= CVBMmmTrainer::INIT_Z_RANDOM;
      cout << "RANDOM assignment" << endl;
    }
    else if (g_ulv_latent_filename == "KMEANS")
    {
      opflag |= CVBMmmTrainer::INIT_Z_KMEANS;
      cout << "KMEANS assignment" << endl;
    }
    else if (g_ulv_latent_filename == "PARAM")
    {
      opflag |= CVBMmmTrainer::INIT_Z_PARAM;
      cout << "PARAMETER FIRST" << endl;
    }
    else
    {
      opflag |= CVBMmmTrainer::INIT_Z_MANUAL;
      cout << "MANUAL assignment:" << endl;
      cout << "  << " << g_ulv_latent_filename << endl;
    }
    cout << "  FRAME level latent variable:     ";
    if (g_flv_latent_filename == "RANDOM")
    {
      opflag |= CVBMmmTrainer::INIT_X_RANDOM;
      cout << "RANDOM assignment" << endl;
    }
    else if (g_flv_latent_filename == "KMEANS")
    {
      opflag |= CVBMmmTrainer::INIT_X_KMEANS;
      cout << "KMEANS assignment" << endl;
    }
    else if (g_flv_latent_filename == "PARAM")
    {
      opflag |= CVBMmmTrainer::INIT_X_PARAM;
      cout << "PARAMETER FIRST" << endl;
    }
    
    if (g_ulv_latent_filename == "PARAM")
      vbt->init(opflag, g_ulv_latent_filename.c_str());
    else
      vbt->init(opflag);
    
    
    v_vbt.push_back(vbt);
  }
  cout << endl;
  
  /* ================================ */
  /* クラスタリングの実行 */
  /* ================================ */
  cout << "Staring clustering..." << endl << endl;
  
  run(v_vbt, v_log);
  
  cout << "Stop. "   << endl;

  /* ================================ */
  /* クラスタリングの評価 */
  /* ================================ */
  eval(v_vbt, v_log);
  
  /* ================================ */
  /* 後始末 */
  /* ================================ */
  vector<CVBMmmTrainer*>::iterator iter_vbt = v_vbt.begin();
  for (;iter_vbt != v_vbt.end(); ++iter_vbt)
    delete (*iter_vbt);
  
  cout << "exit" << endl;
  delete feature;
  return 0;
}

//----------------------------------------------------------------------
//                    End: main.cc
//----------------------------------------------------------------------
