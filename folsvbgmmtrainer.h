//
//  folsvbgmmtrainer.h
//  CVB
//
//  Single Gaussian のみに対応
//
//  Created by 俵 直弘 on 13/01/08.
//  Copyright (c) 2013年 俵 直弘. All rights reserved.
//

#ifndef __VB__folsvbgmmtrainer__
#define __VB__folsvbgmmtrainer__

#include "vbgmmtrainer.h"
#include "util_stat.h"

class CFoLSVBGmmTrainer : public CVBGmmTrainer
{
public:

  CFoLSVBGmmTrainer(const string& _name, const int _num_mixtures)
    : CVBGmmTrainer(_name, _num_mixtures)
  { if (_num_mixtures >1) Error(1111, "[CFoLSVBGmmTrainer]: mixture of GMM is not supported");}
  
  ~CFoLSVBGmmTrainer() {}
  
public:
  
  /// フレームレベル潜在変数の事後平均値を更新する関数
  void updateZ(const double& _x, const int _u);
  
  
  void updateZ(DoubleVect _X, const int _u);
  
  /// モデルパラメタを更新する関数
  void update(const double& _x, const int _u);
  
  void update(DoubleVect _X, const int _u);

};

#endif /* defined(__VB__folsvbgmmtrainer__) */
