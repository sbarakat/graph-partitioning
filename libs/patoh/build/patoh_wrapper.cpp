#include "patoh.h"

#ifdef __cplusplus
extern "C" {
#endif

int Patoh_Initialize_Parameters(PPaToH_Parameters pargs, int cuttype,
                                int SuggestByProblemType) {
  return PaToH_Initialize_Parameters(pargs, cuttype, SuggestByProblemType);
}

int Patoh_Process_Arguments(PPaToH_Parameters pargs, int startargc, int argc, char *argv[], SingleArgumentCallBack func) {
  return PaToH_Process_Arguments(pargs, startargc, argc, argv, func);
}

int Patoh_Alloc(PPaToH_Parameters pargs, int _c, int _n, int _nconst, int *cwghts, int *nwghts, int *xpins, int *pins) {
  return PaToH_Alloc(pargs, _c, _n, _nconst, cwghts, nwghts, xpins, pins);
}

int Patoh_Free() {
  return PaToH_Free();
}

/* --- partition --- */

    /* Input: pargs, _c, _n, cwghts, nwghts, xpins, pins
       Ouput: partvec, partweights and cut */
    /* Deprecated: use PaToH_Part instead */
int Patoh_Partition(PPaToH_Parameters pargs, int _c, int _n, int *cwghts,
                    int *nwghts, int *xpins, int *pins, int *partvec,
                    int *partweights, int *cut)
{
  return PaToH_Partition(pargs, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, cut);
}
    /* Input: pargs, _c, _n, cwghts, nwghts, xpins, pins, partvec
       Ouput: partvec, partweights and cut */
    /* Deprecated: use PaToH_Part instead */
int Patoh_Partition_with_FixCells(PPaToH_Parameters pargs, int _c, int _n,
                                  int *cwghts, int *nwghts, int *xpins,
                                  int *pins, int *partvec, int *partweights,
                                  int *cut)
{
  return PaToH_Partition_with_FixCells(pargs, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, cut);
}

    /* Input: pargs, _c, _n, _nconst, cwghts, xpins, pins
       Ouput: partvec, partweights and cut */
    /* Deprecated: use PaToH_Part instead */
int Patoh_MultiConst_Partition(PPaToH_Parameters pargs, int _c, int _n,
                               int _nconst, int *cwghts,
                               int *xpins, int *pins, int *partvec,
                               int *partweights, int *cut)
{
  return PaToH_MultiConst_Partition(pargs, _c, _n, _nconst, cwghts, xpins, pins, partvec, partweights, cut);
}
    /* Unified interface for PaToH with target part weights

    _nconst > 1: calls PaToH_MultiConst_Partition (net weights discarded)
    otherwise    if useFixCells=1  calls PaToH_Partition_with_FixCells
                 otherwise calls PaTOH_Partition

       Input: pargs, _c, _n, _nconst, cwghts, nwghts, xpins, pins, targetweights,
       Ouput: partvec, partweights and cut

       if targetweights is NULL each targetweights[i] will be assigned to
           1/pargs->_k

     */
 int Patoh_Part(PPaToH_Parameters pargs, int _c, int _n, int _nconst, int useFixCells,
                int *cwghts, int *nwghts, int *xpins, int *pins, float *targetweights,
                int *partvec, int *partweights, int *cut)
{
  return PaToH_Part(pargs, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, cut);
}
 /* --- refine --- */
     /* Input: pargs, _c, _n, cwghts, nwghts, xpins, pins, partvec
        Ouput: partvec, partweights and cut */
 int Patoh_Refine_Bisec(PPaToH_Parameters pargs, int _c, int _n,
                  int *cwghts, int *nwghts, int *xpins,
                  int *pins, int *partvec, int *partweights,
                  int *cut)
{
  return PaToH_Refine_Bisec(pargs, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, cut);
}
/* --- utility --- */
char *Patoh_VersionStr() {
  return PaToH_VersionStr();
}

int Patoh_Print_Parameter_Abrv(void)
{
  return PaToH_Print_Parameter_Abrv();
}


void Patoh_Print_Parameters(PPaToH_Parameters p)
{
  return PaToH_Print_Parameters(p);
}


int Patoh_Check_User_Parameters(PPaToH_Parameters pargs, int verbose)
{
  return PaToH_Check_User_Parameters(pargs, verbose);
}


int Patoh_Read_Hypergraph(char *filename, int *_c, int *_n, int *_nconst,
                          int **cwghts, int **nwghts, int **xpins, int **pins)
{
  return PaToH_Read_Hypergraph(filename, _c, _n, _nconst, cwghts, nwghts, xpins, pins);
}

int Patoh_Write_Hypergraph(char *filename, int numbering, int _c, int _n, int _nconst,
                           int *cwghts, int *nwghts, int *xpins, int *pins)
{
  return PaToH_Write_Hypergraph(filename, numbering, _c, _n, _nconst, cwghts, nwghts, xpins, pins);
}


int Patoh_Check_Hypergraph(int _c, int _n, int _nconst,
                           int *cwghts, int *nwghts, int *xpins, int *pins)
{
  return PaToH_Check_Hypergraph(_c, _n, _nconst, cwghts, nwghts, xpins, pins);
}


int Patoh_Compute_Cut(int _k, int cuttype, int _c, int _n, int *nwghts,
                      int *xpins, int *pins, int *partvec)
{
  return PaToH_Compute_Cut(_k, cuttype, _c, _n, nwghts, xpins, pins, partvec);
}


int Patoh_Compute_Part_Weights(int _k, int _c, int _nconst,
                               int *cwghts, int *partvec, int *partweights)
{
  return PaToH_Compute_Part_Weights(_k, _c, _nconst, cwghts, partvec, partweights);
}

int Patoh_Compute_Part_NetWeights(int _k, int _n, int *nwghts,
                                  int *xpins, int *pins,
                                  int *partvec, int *partinweights)
{
  return PaToH_Compute_Part_NetWeights(_k, _n, nwghts, xpins, pins, partvec, partinweights);
}


#ifdef __cplusplus
}
#endif
