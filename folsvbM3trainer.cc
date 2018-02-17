//
//  folsvbM3trainer.cc
//  VB
//
//  Created by 俵 直弘 on 13/01/08.
//  Copyright (c) 2013年 俵 直弘. All rights reserved.
//

#include "folsvbM3trainer.h"

//----------------------------------------------------------------------

void CFoLSVBMmmTrainer::updateX(void)
{
  int num_segments = getNumSegments();
  double sumAlpha = _sum(*getAlphas());
  DoubleVect sumX(num_segments, -DBL_MAX); // sumX: 1 x T
  
  bool first = true;
  
  IntVect alignment(num_segments, -1);       // 各セグメントの強制アライメント
  DoubleVect max_Xt(num_segments, -DBL_MAX); // 各セグメントの最大事後確率値
  
  vector<DoubleVect>::iterator iter_X       = getXs()->begin();
  DoubleVect::iterator iter_a               = getAlphas()->begin();
  vector <CVBGmmTrainer*>::iterator iter_cl = getClusters()->begin();
  
  for (int i = 0;iter_X != getXs()->end(); ++iter_cl, ++iter_X, ++iter_a, ++i)
  { /* iterate num of mixtures times : j = 1, ..., S */
    DoubleVect::iterator iter_Xt   = iter_X->begin();
    DoubleVect::iterator iter_sumX = sumX.begin();
    double logalpha = digamma(*iter_a) - digamma(sumAlpha);
    
    IntVect::iterator iter_al = alignment.begin();
    DoubleVect::iterator iter_mXt = max_Xt.begin();
    
    for (int t = 0; iter_Xt != iter_X->end(); ++iter_Xt, ++iter_sumX, ++iter_al, ++iter_mXt, ++t)
    {
      /* iterate num of Segments times : t = 1, ..., T */
      double gamma =  logalpha + (*iter_cl)->getProdSumLogZ(t);
      (*iter_Xt) = gamma;
      (*iter_sumX) = first ? gamma:
      LAddS((*iter_sumX), gamma);
      if (*iter_Xt > *iter_mXt)
      {
        *iter_mXt = *iter_Xt;
        *iter_al  = i;
      }
    }
    first = false;
  }
  
  /* 正規化 */
  iter_X = getXs()->begin();
  for (int i=0;iter_X != getXs()->end(); ++iter_X,++i)
  { /* iterate num of mixtures times */
    DoubleVect::iterator iter_Xt   = iter_X->begin();
    DoubleVect::iterator iter_sumX = sumX.begin();
    IntVect::iterator iter_al      = alignment.begin();
    DoubleVect::iterator iter_mXt  = max_Xt.begin();
    for (;iter_Xt != iter_X->end(); ++iter_Xt, ++iter_sumX, ++iter_al, ++iter_mXt)
    {
      /* iterate num of Segments times */
#if ISLOG
      (*iter_Xt) = (*iter_Xt) - (*iter_sumX);
#else
      (*iter_Xt) = exp((*iter_Xt) - (*iter_sumX));
      if ((*iter_Xt) < 1.0E-3) (*iter_Xt) = 0.0;
      if ((*iter_Xt) > 0.999)  (*iter_Xt) = 1.0;
#endif/*
if (i == *iter_al)
(*iter_Xt) = 1;
else
(*iter_Xt) = 0;*/
    }
  }
  
}



