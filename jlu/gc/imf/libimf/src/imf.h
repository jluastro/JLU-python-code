/*
libimf -  Programming functions for the initial mass function
Copyright (C) 2005  Jan Pflamm-Altenburg

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
USA.

Additionally:
	  
  When publishing results based on this software
  or parts of it (executable and/ or source code
  cite:
       Pflamm-Altenburg, J., Kroupa P., 2006

  See manpage for details.
*/

#ifndef _imf_h
#define _imf_h _imf_h
#include <float.h>
#define PARA_MAX 10
#define IMF_MASSINF DBL_MAX

/*** General ***/

struct _IMF{
  int n;
  double m_max_physical;
  double m_cl;
  double m_max;
  double k;
  double *m;
  double *psi;
  double (**f)(double,double *);
  double (**a_f);
  double (**mf)(double,double *);
  double (**a_mf);
  double (**F)(double,double *);
  double (**a_F);
  double (**mF)(double,double *);
  double (**a_mF);
  double *lamda;
  double (**invF)(double,double *);
  double (**a_invF);
};

typedef struct _IMF IMF;

/*** Creating general imfs ***/

int imf_init_salpeter_1955(IMF *imf);

int imf_init_miller_scalo_1979(IMF *imf);

int imf_init_kennicutt_1983(IMF *imf);

int imf_init_kroupa_2001(IMF *imf);

int imf_init_ktg_1993(IMF *imf);

int imf_init_weidner_kroupa_2004(IMF *imf);

int imf_init_chabrier_2003(IMF *imf);

/***  Creating user defined general imfs ***/

int imf_init_multi_power(IMF *imf,int n,double *m,double *alpha);

int imf_shift_left(IMF *imf,double left);

int imf_shift_right(IMF *imf,double right);

int imf_free(IMF *imf);

int imf_init_user(IMF *imf,char *imf_string);

/*** Getting values and integrals of an 
     initialized general imf ***/

double imf_xi(IMF *imf,double m);

double imf_mxi(IMF *imf,double m);

double imf_int_xi(IMF *imf,double left,double right);

double imf_int_mxi(IMF *imf,double left,double right);



/*** Normalize an imf for an individual cluster ***/

int imf_norm_cl(IMF *imf,double m_cl,double m_max_physical);

int imf_norm_cl_wk04(IMF *imf,double m_cl,double m_max_physical);

/*** Getting values and integrals of an initialized 
     imf  for  an  individual cluster after normalizing ***/

double imf_xi_cl(IMF *imf,double m);

double imf_mxi_cl(IMF *imf,double m);

double imf_int_xi_cl(IMF *imf,double left,double right);

double imf_int_mxi_cl(IMF *imf,double left,double right);

double imf_dice_star_cl(IMF *imf,double x);

/*** Segment functions ***/

double imf_power(double m,double *a);

double imf_prim_power(double m,double *a);

double imf_inv_prim_power(double x,double *a);

double imf_log_normal(double m,double *a);

double imf_prim_log_normal(double m,double *a);

double imf_inv_prim_log_normal(double x,double *a);

double imf_mlog_normal(double m,double *a);

double imf_prim_mlog_normal(double m,double *a);

#endif
