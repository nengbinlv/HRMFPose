/************************************************************************
*
*
*   changed by sheng wang
*   2017.10.11
*
*
*  lap.h
   version 1.0 - 21 june 1996
   author  Roy Jonker, MagicLogic Optimization Inc.
   
   header file for LAP
*
**************************************************************************/

/*************** CONSTANTS  *******************/

#define BIG 100000
typedef int row;
typedef int col;
typedef double cost;
#if !defined TRUE
#define	 TRUE		1
#endif
#if !defined FALSE
#define  FALSE		0
#endif
typedef int boolean;
#include <vector>
  namespace srt3d {
	  
	  //using namespace std;
	  //extern double lap(int dim, double **assigncost,
	  //               int *rowsol, int *colsol, double *u, double *v);
	  //extern double lap(int dim, std::vector< std::vector<double> >& assigncost,
	  //                  int *rowsol, int *colsol, double *u, double *v);

	  extern double lap(int dim, std::vector< std::vector<double> >& assigncost,
		  std::vector<int>& rowsol, std::vector<int>& colsol, std::vector<double>& u, std::vector<double>& v);
  }
//extern void checklap(int dim, double **assigncost,
//                     int *rowsol, int *colsol, double *u, double *v);

