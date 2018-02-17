//
//  folsvbM3trainer.h
//  VB
//
//  Created by 俵 直弘 on 13/01/08.
//  Copyright (c) 2013年 俵 直弘. All rights reserved.
//

#ifndef __VB__folsvbM3trainer__
#define __VB__folsvbM3trainer__

#include <iostream>


#include "vbM3trainer.h"

/**
 * @class CVBTrainer vbtrainer.h
 * @brief 変分ベイズ学習を行うクラス
 * @memo  データ集合全体を管理
 * @author Naohiro TAWARA
 * @date 2013-01-08
 */
class CFoLSVBMmmTrainer : public CVBMmmTrainer
{
public:
  
  CFoLSVBMmmTrainer(const int _num_clusters,
                    const int _num_mixtures,
                    const int _covtype,
                    ostream* _ros)
  : CVBMmmTrainer(_num_clusters, _num_mixtures, _covtype, _ros)
  {}
  
  ~CFoLSVBMmmTrainer()
  {}
  
private:
  

  void updateX(void);
  
};

#endif /* defined(__VB__folsvbM3trainer__) */
