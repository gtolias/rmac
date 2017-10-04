/* 
MEX implementation of integral max pooling
Authored by G. Tolias, 2015. 
*/

#include "mex.h"
#include <math.h>

/* compile me simply by mex aml.c */

typedef struct {
   int i1, i2, j1, j2;
} box;

typedef struct {
  double s;
  box b;
} ebox;

ebox max_score(const double *iim, const int * pDims, const int p, const double * a, double qratio, double qratio_t, int step, int *c);
ebox box_refine(const double *iim, const int * pDims, const int p, const double * a, const ebox eb, const int rf_steps, const int rf_iter, int *c);
double get_boxscore(const double *iim, const box bx, const int * pDims, const int p, const double * a);
double* create_integralimage(const double *x, const int *pDims, int p);

double l2norm(const double *x, int dim);
double fast_root_10(const double v);
int find_nearest (const double v);

/* ---------------------------------------------------------------------- */
void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if ( (nlhs != 1) || (nrhs != 8) )
    mexErrMsgTxt ("Usage: y = aml (cnn_act_3d, p, mac_query, qratio, qratio_thres, step, refine_step, refine_niter)");

  const int *pDims0 = mxGetDimensions(prhs[0]);
  double *x = (double *) mxGetPr (prhs[0]);
  int p = (int) mxGetScalar (prhs[1]);
  double * a = (double *) mxGetPr (prhs[2]);      
  double qratio = (double) mxGetScalar (prhs[3]);
  double qratiot = (double) mxGetScalar (prhs[4]);
  int step = (int) mxGetScalar (prhs[5]);
  int rf_steps = (int) mxGetScalar (prhs[6]);
  int rf_iter = (int) mxGetScalar (prhs[7]);

  if (p != 10)
    mexErrMsgTxt ("Sorry, version for hard-coded p = 10 only");

  double *iim = create_integralimage(x, pDims0, p);
  int pDims[3] = {pDims0[2], pDims0[0] + 1, pDims0[1] + 1};
  
  int c;
  ebox eb = max_score(iim, pDims, p, a, qratio, qratiot, step, &c);
  eb =  box_refine(iim, pDims, p, a, eb, rf_steps, rf_iter, &c);

  plhs[0] = mxCreateNumericMatrix (1, 6, mxSINGLE_CLASS, mxREAL);
  float * ret = (float *) mxGetPr (plhs[0]);
  ret[0] = (float)eb.s; ret[1] = (float)eb.b.i1; ret[2] = (float)eb.b.i2; ret[3] = (float)eb.b.j1; ret[4] = (float)eb.b.j2; ret[5] = (float)c;

  free(iim);  
}


/* ---------------------------------------------------------------------- */
static double pow_10[256] = {
0.0, 1.0, 1024.0, 59049.0, 1048576.0, 9765625.0, 60466176.0, 282475249.0, 1073741824.0, 3486784401.0, 10000000000.0, 25937424601.0, 61917364224.0, 137858491849.0, 289254654976.0, 576650390625.0, 1099511627776.0, 2015993900449.0, 3570467226624.0, 6131066257801.0, 10240000000000.0, 16679880978201.0, 26559922791424.0, 41426511213649.0, 63403380965376.0, 95367431640625.0, 141167095653376.0, 205891132094649.0, 296196766695424.0, 420707233300201.0, 590490000000000.0, 819628286980801.0, 1125899906842624.0, 1531578985264449.0, 2064377754059776.0, 2758547353515625.0, 3656158440062976.0, 4808584372417849.0, 6278211847988224.0, 8140406085191601.0, 10485760000000000.0, 13422659310152400.0, 17080198121677824.0, 21611482313284248.0, 27197360938418176.0, 34050628916015624.0, 42420747482776576.0, 52599132235830048.0, 64925062108545024.0, 79792266297612000.0, 97656250000000000.0, 119042423827612992.0, 144555105949057024.0, 174887470365513056.0, 210832519264920576.0, 253295162119140608.0, 303305489096114176.0, 362033331456891264.0, 430804206899405824.0, 511116753300641408.0, 604661760000000000.0, 713342911662882560.0, 839299365868340224.0, 984930291881790848.0, 1152921504606846976.0, 1346274334462890496.0, 1568336880910795776.0, 1822837804551761408.0, 2113922820157210624.0, 2446194060654759936.0, 2824752490000000000.0, 3255243551009881088.0, 3743906242624487424.0, 4297625829703557632.0, 4923990397355877376.0, 5631351470947265536.0, 6428888932339941376.0, 7326680472586201088.0, 8335775831236199424.0, 9468276082626846720.0, 10737418240000000000.0, 12157665459056928768.0, 13744803133596057600.0, 15516041187205853184.0, 17490122876598091776.0, 19687440434072264704.0, 22130157888803069952.0, 24842341419143569408.0, 27850097600940212224.0, 31181719929966182400.0, 34867844009999998976.0, 38941611811810746368.0, 43438845422363213824.0, 48398230717929316352.0, 53861511409489969152.0, 59873693923837886464.0, 66483263599150104576.0, 73742412689492828160.0, 81707280688754688000.0, 90438207500880445440.0, 100000000000000000000.0, 110462212541120446464.0, 121899441999475703808.0, 134391637934412185600.0, 148024428491834392576.0, 162889462677744123904.0, 179084769654285369344.0, 196715135728956538880.0, 215892499727278669824.0, 236736367459211739136.0, 259374246009999982592.0, 283942098606901559296.0, 310584820834420916224.0, 339456738992222306304.0, 370722131411856654336.0, 404555773570790981632.0, 441143507864991563776.0, 480682838924478840832.0, 523383555379856801792.0, 569468379011812491264.0, 619173642240000000000.0, 672749994932559937536.0, 730463141542791741440.0, 792594609605189173248.0, 859442550649180389376.0, 931322574615478534144.0, 1008568618886953828352.0, 1091533853073393582080.0, 1180591620717411303424.0, 1276136419117121667072.0, 1378584918489999867904.0, 1488377021731616063488.0, 1605976966052654874624.0, 1731874467807835717632.0, 1866585911861003681792.0, 2010655586861806780416.0, 2164656967840983678976.0, 2329194047563391827968.0, 2504902718110474174464.0, 2692452204196940218368.0, 2892546549760000000000.0, 3105926159393528741888.0, 3333369396234118234112.0, 3575694237941010268160.0, 3833759992447475122176.0, 4108469075197275668480.0, 4400768849616443015168.0, 4711653532607690833920.0, 5042166166892418433024.0, 5393400662063408218112.0, 5766503906249999908864.0, 6162677950336718602240.0, 6583182266716099969024.0, 7029336084596720140288.0, 7502520803928269914112.0, 8004182490046885003264.0, 8535834451185868210176.0, 9099059901039401500672.0, 9695514708609891041280.0, 10326930237613179666432.0, 10995116277760000000000.0, 11701964070276793630720.0, 12449449430074295058432.0, 13239635967018159570944.0, 14074678408802362982400.0, 14956826027973136089088.0, 15888426175698793660416.0, 16871927924929096318976.0, 17909885825636445978624.0, 19004963774880800571392.0, 20159939004489999056896.0, 21377706189197972340736.0, 22661281678134343630848.0, 24013807852610947907584.0, 25438557613203015073792.0, 26938938999176024817664.0, 28518499943362777317376.0, 30180933165649573707776.0, 31930081208285370777600.0, 33769941616283277590528.0, 35704672266239998951424.0, 37738596846955705401344.0, 39876210495294204280832.0, 42122185590781553672192.0, 44481377712499930955776.0, 46958831761893059723264.0, 49559788255159619944448.0, 52289689788971841748992.0, 55154187683317728411648.0, 58159148805327866036224.0, 61310662578009995739136.0, 64615048177878504046592.0, 68078861925529707085824.0, 71708904873278933303296.0, 75512230594040656035840.0, 79496153175699230818304.0, 83668255425284800512000.0, 88036397287336249917440.0, 92608724480901576130560.0, 97393677359695037202432.0, 102400000000000000000000.0, 107636749520976964747264.0, 113113305642107337179136.0, 118839380482575359279104.0, 124825028607463120699392.0, 131080657325707040391168.0, 137617037244838078054400.0, 144445313087602912395264.0, 151577014775638417997824.0, 159024068785448662597632.0, 166798809782009982877696.0, 174913992535407979397120.0, 183382804125988218208256.0, 192218876443582470815744.0, 201436298986451495813120.0, 211049631965666500149248.0, 221073919720733357899776.0, 231524704452345201688576.0, 242418040278232820875264.0, 253770507618165381398528.0, 265599227914239982174208.0, 277921878692682180591616.0, 290756708973467196719104.0, 304122555034158445363200.0, 318038856534447018213376.0, 332525673007965060202496.0, 347603700728035641655296.0, 363294289954110647042048.0, 379619462565741214040064.0, 396601930091015148929024.0, 414265112136489965191168.0, 432633155225742533984256.0, 451730952053751361306624.0, 471584161164422532825088.0, 492219227058666333011968.0, 513663400740527846457344.0, 535944760708973365035008.0, 559092234403032671977472.0, 583135620108095991054336.0, 608105609331265052344320.0, 634033809653760000000000.0, 660952768068482248998912.0, 688895994810941376036864.0, 717897987691852578422784.0, 747994256939818743234560.0, 779221350562617201000448.0, 811616880235713713405952.0, 845219547726738047893504.0, 880069171864760718721024.0, 916206716063318519316480.0, 953674316406250018963456.0, 992515310305509705252864.0, 1032774265740240720232448.0, 1074497011086501968084992.0, 1117730665547155028049920.0, 1162523670191533209944064.0
};

/* ---------------------------------------------------------------------- */
static double root_10[64] = {
0.0, 1.0, 1024.0, 59049.0, 1048576.0, 9765625.0, 60466176.0, 282475249.0, 1073741824.0, 3486784401.0, 10000000000.0, 25937424601.0, 61917364224.0, 137858491849.0, 289254654976.0, 576650390625.0, 1099511627776.0, 2015993900449.0, 3570467226624.0, 6131066257801.0, 10240000000000.0, 16679880978201.0, 26559922791424.0, 41426511213649.0, 63403380965376.0, 95367431640625.0, 141167095653376.0, 205891132094649.0, 296196766695424.0, 420707233300201.0, 590490000000000.0, 819628286980801.0, 1125899906842624.0, 1531578985264449.0, 2064377754059776.0, 2758547353515625.0, 3656158440062976.0, 4808584372417849.0, 6278211847988224.0, 8140406085191601.0, 10485760000000000.0, 13422659310152400.0, 17080198121677824.0, 21611482313284248.0, 27197360938418176.0, 34050628916015624.0, 42420747482776576.0, 52599132235830048.0, 64925062108545024.0, 79792266297612000.0, 97656250000000000.0, 119042423827612992.0, 144555105949057024.0, 174887470365513056.0, 210832519264920576.0, 253295162119140608.0, 303305489096114176.0, 362033331456891264.0, 430804206899405824.0, 511116753300641408.0, 604661760000000000.0, 713342911662882560.0, 839299365868340224.0, 984930291881790848.0
};

/* ---------------------------------------------------------------------- */
double* create_integralimage(const double *x, const int *pDims, int p) 
{
  int row, column, fm;  
  double v = 0;
  int s1 = pDims[0], s2 = pDims[1], s3 = pDims[2];
  
  double *iim = (double *) malloc (sizeof(*iim) * (s1+1) * (s2+1) * s3);

/* avoid to loop over all of them ******** */  
  for (fm = 0; fm < s3; fm++){
    row = 0;
    for (column = 0; column < (s2+1); column ++) 
      iim[(column) * s3 * (s1+1) + (row) * s3 + fm] = 0;
    column = 0;
    for (row = 0; row < (s1+1); row ++) 
      iim[(column) * s3 * (s1+1) + (row) * s3 + fm] = 0;
  }
  
  for (fm = 0; fm < s3; fm++)
    for (row = 0; row < s1; row++) 
      for (column = 0; column < s2; column++) {
        /* hard coded, in the power of 10 */ 
        v = pow_10[(int)x[fm * s1 * s2 + column * s1 + row]]; 
       /* for the leftmost pixel, just copy value from original */
       if (row == 0 & column == 0)
        iim[(column+1) * s3 * (s1+1) + (row+1) * s3 + fm] = v;
       /* For the first row, just add the value to the left of this pixel */
       else if (row == 0)
        iim[(column+1) * s3 * (s1+1) + (row+1) * s3 + fm] = v + iim[(column) * s3 * (s1+1) + (row+1) * s3 + fm];
       /* For the first column, just add the value to the top of this pixel */
       else if (column == 0)
        iim[(column+1) * s3 * (s1+1) + (row+1) * s3 + fm] = v + iim[(column+1) * s3 * (s1+1) + (row) * s3 + fm];
       /* For other pixels, add the left and above values and subtract the value to the left and above diagonally */
       else 
        iim[(column+1) * s3 * (s1+1) + (row+1) * s3 + fm] = v +         iim[(column) * s3 * (s1+1) + (row+1) * s3 + fm] +         iim[(column+1) * s3 * (s1+1) + (row) * s3 + fm] -         iim[(column) * s3 * (s1+1) + (row) * s3 + fm];        
      }

return iim;
}

/* ---------------------------------------------------------------------- */
ebox box_refine(const double *iim, const int * pDims, const int p, const double * a, const ebox eb, const int rf_steps, const int rf_iter, int *c_ext)
{
  ebox eb_current;
  ebox eb_best = eb;
  int c, step, step_init = rf_steps, iter = rf_iter;
  int change = 1;

  double qnorm = l2norm(a, pDims[0]);
  
  for (step = step_init; step >= 1; step--) {
    c = 0;
    change = 1;

    while (change & c < iter) {
      change = 0;

      if (eb_best.b.i1 > step) {
        eb_current = eb_best;
        eb_current.b.i1 -= step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.i2 >= eb_best.b.i1 + step + 1) {
        eb_current = eb_best;
        eb_current.b.i1 += step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.i2 >= eb_best.b.i1 + step + 1) {
        eb_current = eb_best;
        eb_current.b.i2 -= step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.i2 <= pDims[1] - 1 - step) {
        eb_current = eb_best;
        eb_current.b.i2 += step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.j1 > step) {
        eb_current = eb_best;
        eb_current.b.j1 -= step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.j2 >= eb_best.b.j1 + step + 1) {
        eb_current = eb_best;
        eb_current.b.j1 += step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.j2 >= eb_best.b.j1 + step + 1) {
        eb_current = eb_best;
        eb_current.b.j2 -= step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      }

      if (eb_best.b.j2 <= pDims[2] - 1 - step) {
        eb_current = eb_best;
        eb_current.b.j2 += step;
        eb_current.s = get_boxscore(iim, eb_current.b, pDims, p, a) / qnorm;
        if (eb_current.s > eb_best.s) {
          eb_best = eb_current;
          change = 1;
        }
      } 
      c++;
      (*c_ext) += 8;
    }
  }

  return eb_best;
}

/* ---------------------------------------------------------------------- */
ebox max_score(const double *iim, const int * pDims, const int p, const double * a, double qratio, double qratio_t, int step, int *c)
{

  int i1 =0, i2 = 0, j1 = 0, j2 = 0;
  double s = -10e+10;

  ebox eb = (ebox){s, 1, pDims[1]-1,1, pDims[2]-1};

  (*c) = 0;

  double qnorm = l2norm(a, pDims[0]);

  for (i1 = 1; i1<=pDims[1]-1-1; i1+=step){
    for (i2 = i1+1; i2<=pDims[1]-1; i2+=step){
      for (j1 = 1; j1<=pDims[2]-1-1; j1+=step){
        for (j2 = j1+1; j2<=pDims[2]-1; j2+=step){

          double ratio = fabs(log(qratio / ((double)(i2-i1+1)/(double)(j2-j1+1))));
          if ((qratio>0) & (ratio > log(qratio_t))) continue;           

          box bx = (box){i1,i2,j1,j2};
          s = get_boxscore(iim, bx, pDims, p, a) / qnorm;
          (*c) ++;
          
          if (s > eb.s) {
            eb.s = s;
            eb.b = bx;
          }

        }
      }
    }
  }

return eb;
}


/* ---------------------------------------------------------------------- */
double get_boxscore(const double *iim, const box bx, const int * pDims, const int p, const double * a)
{ 
  int i;
  int s1 = pDims[0];
  int s2 = pDims[1];
  double *vec = (double *) malloc (sizeof(*vec) * s1);
  double s = 0;

  for (i = 0; i < s1; i++)
    vec[i] = iim[bx.i2*s1 + bx.j2*s1*s2 + i];
    
  for (i = 0; i < s1; i++)
    vec[i] -= iim[(bx.i1-1)*s1 + bx.j2*s1*s2 + i];

  for (i = 0; i < s1; i++)
    vec[i] -= iim[bx.i2*s1 + (bx.j1-1)*s1*s2 + i];

  for (i = 0; i < s1; i++)
    vec[i] += iim[(bx.i1-1)*s1 + (bx.j1-1)*s1*s2 + i];

  double n = 0;
  for (i = 0; i < s1; i++)
    {
      /* hard coded 10-th root */
      vec[i] = (vec[i] > 0) ? fast_root_10(vec[i]) : 0; 
      s += a[i]*vec[i];
      n += pow(vec[i], 2.0);
    }

  free(vec);
  s /= pow(n, 0.5); 

  return s;
}

/* ---------------------------------------------------------------------- */
double l2norm(const double *x, int dim){
  double n = 0;
  int i;
  for (i = 0; i < dim; i++) n+= pow(x[i], 2.0);

  return pow(n, 0.5);
}

/* ---------------------------------------------------------------------- */
double fast_root_10(const double v){
  return (double)find_nearest (v);
}

/* ---------------------------------------------------------------------- */
int find_nearest (const double v)
{
  int first, last, middle, size = 256;

  first = 0;
  last = size - 1;
  middle = (first+last)/2;

  while (first < last) {
    if (pow_10[middle] < v){
      first = middle + 1;    
    }
    else {
      last = middle - 1;
    }
    middle = (first + last)/2;
  }

  return first;
}
