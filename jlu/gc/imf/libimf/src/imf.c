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

#include "imf.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/***************************************/
/***                                 ***/
/*** Deklaration of static functions ***/
/***                                 ***/
/***************************************/

static double prim_xi(IMF *imf,double a);
static double prim_mxi(IMF *imf,double a);
static double gamma_closed(double m,double left,double right);
static double theta_closed(double x);
static double theta_open(double x);
static double delta(double x);
static int init_psi(IMF *imf);
static int init_lamda(IMF *imf);
static double error(double x);
static double inv_error(double x);
static char *string_parse(const char *line,char *to,const char *delim);
static double assign_double(char *parameter);


/***************************************/
/***                                 ***/
/***         General IMFs            ***/
/***                                 ***/
/***************************************/

int imf_init_salpeter_1955(IMF *imf)
{
  double m[2] = {0.40,10.0};
  double a[4] = {-2.3};
  imf_init_multi_power(imf,1,m,a); 
  return 0;
}

int imf_init_miller_scalo_1979(IMF *imf)
{
  double m[4] = {0.1,1,10,IMF_MASSINF};
  double a[3] = {-1.4,-2.5,-3.3};
  imf_init_multi_power(imf,3,m,a); 
  return 0;
}

int imf_init_kennicutt_1983(IMF *imf)
{
  double m[3] = {0.1,1,IMF_MASSINF};
  double a[2] = {-1.4,-2.5};
  imf_init_multi_power(imf,2,m,a); 
  return 0;
}

int imf_init_kroupa_2001(IMF *imf)
{
  double m[5] = {0.01,0.08,0.5,1.0,IMF_MASSINF};
  double a[4] = {-0.3,-1.3,-2.3,-2.3};
  imf_init_multi_power(imf,4,m,a); 
  return 0;
}

int imf_init_ktg_1993(IMF *imf)
{
  double m[4] = {0.08,0.50,1.0,IMF_MASSINF};
  double a[3] = {-1.3,-2.2,-2.7};
  imf_init_multi_power(imf,3,m,a); 
  return 0;
}

int imf_init_weidner_kroupa_2004(IMF *imf)
{
  double m[5] = {0.01,0.08,0.5,1.0,IMF_MASSINF};
  double a[4] = {-0.3,-1.3,-2.3,-2.35};
  imf_init_multi_power(imf,4,m,a); 
  return 0;
}



int imf_init_multi_power(IMF *imf,int n,double *m,double *alpha)
{
  int i;
  imf->n = n;
  imf->k = 1;
  imf->m_max_physical = IMF_MASSINF;
  imf->m_cl = 0;
 
  imf->m = (double *) malloc((imf->n+1) * sizeof(double));
  imf->f = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->mf = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->F = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->mF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->invF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_f = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->a_mf = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->a_F = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->a_mF = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->a_invF = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->m[0] = m[0];
  for(i=1;i<=imf->n;i++)
    {
      imf->f[i] = &imf_power;
      imf->mf[i] = &imf_power;
      imf->F[i] = &imf_prim_power;
      imf->mF[i] = &imf_prim_power;
      imf->invF[i] = &imf_inv_prim_power;
      imf->a_f[i] = (double *) malloc(PARA_MAX * sizeof(double));
      imf->a_mf[i] = (double *) malloc(PARA_MAX * sizeof(double));
      imf->a_F[i] = (double *) malloc(PARA_MAX * sizeof(double));
      imf->a_mF[i] = (double *) malloc(PARA_MAX * sizeof(double));
      imf->a_invF[i] = (double *) malloc(PARA_MAX * sizeof(double));
      imf->m[i] = m[i];
      imf->a_f[i][0] = alpha[i-1];
      imf->a_mf[i][0] = alpha[i-1]+1;
      imf->a_F[i][0] = alpha[i-1];
      imf->a_mF[i][0] = alpha[i-1]+1;
      imf->a_invF[i][0] = alpha[i-1];
   }
  init_psi(imf);
  return 0;
}

int imf_init_chabrier_2003(IMF *imf){

  int i;
  
  imf->n = 2;
  imf->k = 1.;
  imf->m_max_physical = IMF_MASSINF;
  imf->m_cl = 0.;
  
    
  imf->m = (double *) malloc((imf->n+1) * sizeof(double));
  imf->f = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_f = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->mf = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_mf = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->F = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_F = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->mF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_mF = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->invF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_invF = (double **) malloc((imf->n+1) * sizeof(double*));
  for(i=1;i<=imf->n;i++){
    imf->a_f[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_mf[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_F[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_mF[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_invF[i] = (double *) malloc(PARA_MAX * sizeof(double));
  }
  
  
  imf->m[0] = 0.01;
  imf->m[1] = 1.0;
  imf->m[2] = IMF_MASSINF;
  
  
  imf->f[1] = &imf_log_normal;
  imf->a_f[1][0] = log10(0.079);
  imf->a_f[1][1] = 0.69;
  
  imf->mf[1] = &imf_mlog_normal;
  imf->a_mf[1][0] = log10(0.079);
  imf->a_mf[1][1] = 0.69;
  
  imf->F[1] = &imf_prim_log_normal;
  imf->a_F[1][0] = log10(0.079);
  imf->a_F[1][1] = 0.69;
  
  imf->mF[1] = &imf_prim_mlog_normal;
  imf->a_mF[1][0] = log10(0.079);
  imf->a_mF[1][1] = 0.69;

  imf->invF[1] = &imf_inv_prim_log_normal;
  imf->a_invF[1][0] = log10(0.079);
  imf->a_invF[1][1] = 0.69;

  

  imf->f[2] = &imf_power;
  imf->a_f[2][0] = -1.3-1.;

  imf->mf[2] = &imf_power;
  imf->a_mf[2][0] = imf->a_f[2][0]+1.;

  imf->F[2] = &imf_prim_power;
  imf->a_F[2][0] = imf->a_f[2][0];

  imf->mF[2] = &imf_prim_power;
  imf->a_mF[2][0] = imf->a_f[2][0]+1;

  imf->invF[2] = &imf_inv_prim_power;
  imf->a_invF[2][0] = imf->a_f[2][0];
  
  init_psi(imf);
  return 0;
}

int imf_shift_left(IMF *imf,double left){
  if((imf->m[0]) > left)
    imf->m[0] = left;
  return 0;
}

int imf_shift_right(IMF *imf,double right){
  if((imf->m[0]) < right)
    imf->m[0] = right;
  return 0;
}

int imf_free(IMF *imf){
  int i;
  for(i=1;i<=imf->n;i++){
    free(imf->a_f[i]); 
    free(imf->a_mf[i]);
    free(imf->a_F[i]); 
    free(imf->a_mF[i]);
    free(imf->a_invF[i]);
  }
  free(imf->m);
  free(imf->f);
  free(imf->mf);
  free(imf->F);
  free(imf->mF);
  free(imf->invF);
  free(imf->a_f);
  free(imf->a_mf);
  free(imf->a_F);
  free(imf->a_mF);
  free(imf->a_invF);
  free(imf->psi);
  return 0;
}

int imf_init_user(IMF *imf,char *imf_string_original){

  int n,i;
  double value;
  double value2;
  char *pos;
  char *seg_pos;
  char * imf_string;
  char *parameter;
  char *segment;
  char *function;

  segment = (char *) malloc((strlen(imf_string_original)+1)*sizeof(char));
  parameter = (char *) malloc((strlen(imf_string_original)+1)*sizeof(char));
  imf_string = (char *) malloc((strlen(imf_string_original)+1)*sizeof(char));
  function = (char *) malloc((strlen(imf_string_original)+1)*sizeof(char));
  strcpy(imf_string,imf_string_original);

  n = 0;
  pos = imf_string; 


  /** count bracketts **/
  
  while((*pos) != 0){
    if((*pos) == '(')
      n++;
    pos++;
  }
  
  imf->n = n;
  imf->k = 1.;
  imf->m_max_physical = IMF_MASSINF;
  imf->m_cl = 0.;
  
  
    
  imf->m = (double *) malloc((imf->n+1) * sizeof(double));
  imf->f = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_f = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->mf = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_mf = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->F = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_F = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->mF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_mF = (double **) malloc((imf->n+1) * sizeof(double*));
  imf->invF = (double(**)(double,double *)) 
    malloc((imf->n+1) * sizeof(double(*)(double,double *)));
  imf->a_invF = (double **) malloc((imf->n+1) * sizeof(double*));

  for(i=1;i<=imf->n;i++){
    imf->a_f[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_mf[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_F[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_mF[i] = (double *) malloc(PARA_MAX * sizeof(double));
    imf->a_invF[i] = (double *) malloc(PARA_MAX * sizeof(double));
  }

  pos = imf_string;
  
  pos = string_parse(pos,parameter,"(");
  imf->m[0] = assign_double(parameter);
  for(i=1;i<=imf->n;i++){
    pos = string_parse(pos,segment,")");
    pos = string_parse(pos,parameter,"(");
    imf->m[i] = assign_double(parameter);
    seg_pos = segment;
    seg_pos = string_parse(seg_pos,function,":");
    if(strcmp(function,"pow") == 0){
      seg_pos = string_parse(seg_pos,parameter,")");
      value = assign_double(parameter);
      imf->f[i] = &imf_power;
      imf->a_f[i][0] = value;
      
      imf->mf[i] = &imf_power;
      imf->a_mf[i][0] = value+1.;
      
      imf->F[i] = &imf_prim_power;
      imf->a_F[i][0] = value;

      imf->mF[i] = &imf_prim_power;
      imf->a_mF[i][0] = value+1;
      
      imf->invF[i] = &imf_inv_prim_power;
      imf->a_invF[i][0] = value;
    }
    if(strcmp(function,"log-norm") == 0){
      seg_pos = string_parse(seg_pos,parameter,":");
      value = assign_double(parameter);
      seg_pos = string_parse(seg_pos,parameter,")");
      value2 = assign_double(parameter);
      
      imf->f[i] = &imf_log_normal;
      imf->a_f[i][0] = log10(value);
      imf->a_f[i][1] = value2;
      
      imf->mf[i] = &imf_mlog_normal;
      imf->a_mf[i][0] = log10(value);
      imf->a_mf[i][1] = value2;
      
      imf->F[i] = &imf_prim_log_normal;
      imf->a_F[i][0] = log10(value);
      imf->a_F[i][1] = value2;
      
      imf->mF[i] = &imf_prim_mlog_normal;
      imf->a_mF[i][0] = log10(value);
      imf->a_mF[i][1] = value2;
      
      imf->invF[i] = &imf_inv_prim_log_normal;
      imf->a_invF[i][0] = log10(value);
      imf->a_invF[i][1] = value2;
      
    }
  }


  init_psi(imf);

  free(segment);
  free(imf_string);
  free(parameter);
  free(function);
  
  return 0;
}


/*********************************************/
/***                                       ***/
/***  Getting values and integrals of an   ***/ 
/***     initialized general IMF           ***/
/***                                       ***/
/*********************************************/

double imf_xi(IMF *imf,double m)
{
  int i;
  double y = 0;
  double z = 1;
  for(i=1;i<=imf->n;i++)
    {
      y += gamma_closed(m,imf->m[i-1],imf->m[i]) 
	* imf->psi[i]
	* (*(imf->f[i]))(m,imf->a_f[i]);
    }
  for(i=1;i<=imf->n;i++)
    {
      z *= delta(m-imf->m[i]);
    }
  return imf->k * z * y;
}

double imf_mxi(IMF *imf,double m)
{
  int i;
  double y = 0;
  double z = 1;
  for(i=1;i<=imf->n;i++)
    {
      y += gamma_closed(m,imf->m[i-1],imf->m[i]) 
	* imf->psi[i]
	* (*(imf->mf[i]))(m,imf->a_mf[i]);
    }
  for(i=1;i<=imf->n;i++)
    {
      z *= delta(m-imf->m[i]);
    }
  return imf->k * z * y;
}

double imf_int_xi(IMF *imf,double left,double right)
{
  return prim_xi(imf,right) -
    prim_xi(imf,left);
}

double imf_int_mxi(IMF *imf,double left,double right)
{
  return prim_mxi(imf,right) -
    prim_mxi(imf,left);
}
/**************************************************/
/***                                            ***/
/***          Normalize an IMF for              ***/ 
/***          an individual cluster             ***/
/***                                            ***/
/**************************************************/

int imf_norm_cl(IMF *imf,double m_cl,double m_max_physical)
{
  imf->k = 1;
  imf->m_cl = m_cl;
  imf->m_max_physical = (m_max_physical < m_cl) ?
    m_max_physical : m_cl;
  imf->m_max_physical = (imf->m[imf->n] < imf->m_max_physical) ? 
    imf->m[imf->n] : imf->m_max_physical;
  imf->m_max = imf->m_max_physical;
  imf->k = m_cl/imf_int_mxi(imf,imf->m[0],imf->m_max);
  init_lamda(imf);
  return 0;
}

int imf_norm_cl_wk04(IMF *imf,double m_cl,double m_max_physical)
{
  double a,b,c;
  double mb;
  imf->k = 1;
  imf->m_cl = m_cl;
  imf->m_max_physical = (m_max_physical < m_cl) ?
    m_max_physical : m_cl;
  imf->m_max_physical = (imf->m[imf->n] < imf->m_max_physical) ? 
    imf->m[imf->n] : imf->m_max_physical;
  a = imf->m[0];
  c = imf->m_max_physical;
  b = (c+a)/2;
  while(((c/b)-(a/b)) > 0.00001)
    {
      mb = imf_int_mxi(imf,imf->m[0],b)/imf_int_xi(imf,b,imf->m_max_physical);
      if(mb < imf->m_cl)
	a = b;
      else
	c = b;
      b = (c+a)/2;
    }
  imf->m_max = b;
  imf->k = imf->m_cl  / imf_int_mxi(imf,imf->m[0],imf->m_max);
  init_lamda(imf);
  return 0;
}

/**********************************************************/
/***                                                    ***/
/***   Getting values and integrals of an initialized   ***/
/***   IMF for an individual cluster after normalizing  ***/
/***                                                    ***/
/**********************************************************/

double imf_xi_cl(IMF *imf,double m)
{

  return theta_closed(imf->m_max-m)*imf_xi(imf,m);
}

double imf_mxi_cl(IMF *imf,double m)
{

  return theta_closed(imf->m_max-m)*imf_mxi(imf,m);
}

double imf_int_xi_cl(IMF *imf,double left,double right)
{
  return (prim_xi(imf,right) -
    theta_closed(right-imf->m_max) * imf_int_xi(imf,imf->m_max,right))
    - 
    (prim_xi(imf,left) -
    theta_closed(left-imf->m_max) * imf_int_xi(imf,imf->m_max,left))
    ;
}

double imf_int_mxi_cl(IMF *imf,double left,double right)
{
  return (prim_mxi(imf,right) -
    theta_closed(right-imf->m_max) * imf_int_mxi(imf,imf->m_max,right))
    -
    (prim_mxi(imf,left) -
     theta_closed(left-imf->m_max) * imf_int_mxi(imf,imf->m_max,left))
    ;
}

double imf_dice_star_cl(IMF *imf,double r)
{
  int i;
  double y,z,x;
  double y_i,aux,aux1,aux2;
  x = r * (imf->lamda)[imf->n];
  y = 0;
  for(i=1;i<=imf->n;i++)
    {
      aux = (x-imf->lamda[i-1]);
      if(aux < 0)
	break;
      z = aux/(imf->psi[i]*imf->k);
      z += (*(imf->F[i]))(imf->m[i-1],imf->a_F[i]);
      aux1 = gamma_closed(x,imf->lamda[i-1],imf->lamda[i]);
      aux2 = (*(imf->invF[i]))(z,imf->a_invF[i]);
      y_i = gamma_closed(x,imf->lamda[i-1],imf->lamda[i])
	* (*(imf->invF[i]))(z,imf->a_invF[i]);
      y += y_i;
    }
  z = 1.;
  for(i=1;i<imf->n;i++)
    {
      z *= delta(x-imf->lamda[i]); 
    }
  return y*z;
}

/*********************************************/
/***                                       ***/
/***          Segment functions            ***/
/***                                       ***/
/*********************************************/

double imf_power(double m,double *a)
{
  if(a[0] == 0)
    return 1;
  else
    return pow(m,a[0]);
}

double imf_prim_power(double m,double *a)
{
  double z;
  if(a[0] == -1.)
    return log(m);
  else
    {
      z = 1.+a[0];
      return pow(m,z)/z;
    }
}

double imf_inv_prim_power(double x,double *a)
{
  double z;
  if(a[0] == -1.)
    return exp(x);
  else
    {
      z = 1.+a[0];
      return pow(z*x,1./z);
    }
}

double imf_log_normal(double m,double *a)
{
  double z = log10(m)-a[0];
  return exp(-z*z/(2.*a[1]*a[1]))/m;
}

double imf_prim_log_normal(double m,double *a)
{
  double mu,aux;
  mu = (log10(m)-a[0])/(1.4142135623731*a[1]);
  aux = 2.88586244942136*a[1]*error(mu);
  return aux;
}

double imf_inv_prim_log_normal(double x,double *a)
{
  double mu;
  mu = inv_error(0.346516861952484*x/a[1]);
  return pow(10.,1.4142135623731*a[1]*mu+a[0]);
}

double imf_mlog_normal(double m,double *a)
{
  double z = log10(m)-a[0];
  return exp(-z*z/(2.*a[1]*a[1]));
}

double imf_prim_mlog_normal(double m,double *a)
{
  double y,eta;
  eta = (log10(m)-a[0]-a[1]*a[1]*2.30258509299405)/(1.4142135623731*a[1]);
  y = error(eta);
  y *= 2.88586244942136*a[1]*
    exp(2.30258509299405*(1.15129254649702*a[1]*a[1]+a[0]));
  return y;
}

/***************************************************/
/***                                             ***/
/***              Static functions               ***/
/***                                             ***/
/***************************************************/

static double gamma_closed(double m,double left,double right)
{
  return theta_closed(m-left) * theta_closed(right-m);
}

static double theta_closed(double x)
{
  if(x<0)
    return 0;
  else
    return 1;
}

static double theta_open(double x)
{
  if(x>0)
    return 1;
  else
    return 0;
}

static double delta(double x)
{
  if(x==0)
    return 0.5;
  else
    return 1;
}

static double prim_mxi(IMF *imf,double a)
{
  double y1,y2;
  int i;
  y1 = 0;
  y2 = 0;
  for(i=1;i<=imf->n;i++)
    {
      y1 += theta_open(a-imf->m[i])*imf->psi[i]*
	((*(imf->mF[i]))(imf->m[i],imf->a_mF[i])
	 -(*(imf->mF[i]))(imf->m[i-1],imf->a_mF[i]));
      y2 += gamma_closed(a,imf->m[i-1],imf->m[i])*imf->psi[i]*
	((*(imf->mF[i]))(a,imf->a_mF[i])
	 -(*(imf->mF[i]))(imf->m[i-1],imf->a_mF[i]));
    }
  return imf->k*(y1+y2);
}

static double prim_xi(IMF *imf,double a)
{
  double y1,y2;
  int i;
  y1 = 0;
  y2 = 0;
  for(i=1;i<=imf->n;i++)
    {
      y1 += theta_open(a-imf->m[i])*imf->psi[i]*
	((*(imf->F[i]))(imf->m[i],imf->a_F[i])
	 -(*(imf->F[i]))(imf->m[i-1],imf->a_F[i]));
      y2 += gamma_closed(a,imf->m[i-1],imf->m[i])*imf->psi[i]*
	((*(imf->F[i]))(a,imf->a_F[i])
	 -(*(imf->F[i]))(imf->m[i-1],imf->a_F[i]));
    }
  return imf->k*(y1+y2);
}

static int init_psi(IMF *imf)
{
  int i;
  double y,z;
  imf->psi = (double *) malloc((imf->n+1) * sizeof(double));
  imf->psi[1] = 1;
  for(i=2;i<=imf->n;i++)
    {
      y = (*(imf->f[i-1]))(imf->m[i-1],imf->a_f[i-1]);
      z = (*(imf->f[i]))(imf->m[i-1],imf->a_f[i]);
      imf->psi[i] = imf->psi[i-1] 
	* y
	/ z;
    }
  return 0;
}

static int init_lamda(IMF *imf)
{
  int i;
  imf->lamda = (double *) malloc((imf->n+1) * sizeof(double));
  for(i=0;i<=imf->n;i++)
    {
      imf->lamda[i] = imf_int_xi_cl(imf,imf->m[0],imf->m[i]);
    }
  return 0;
}

static double error(double x){
  double ax2,x2;
  x2 = x*x;
  ax2 = 0.140012288686666*x2; 
  if(x>=0)
    return sqrt(1.-exp(-x2*(1.27323954473516+ax2)/(1+ax2)));
  else
    return -sqrt(1.-exp(-x2*(1.27323954473516+ax2)/(1+ax2)));
}

static double inv_error(double x){
  double x2,lnx2,y,aux;
  x2 = x*x;
  lnx2 = log(1.-x2);
  aux = 4.54688497944829 + lnx2/2.;
  y = -4.54688497944829-lnx2/2.+sqrt(aux*aux-lnx2/0.140012288686666);
  if(x>=0)
    return sqrt(y);
  else
    return -sqrt(y);
}

static double assign_double(char *parameter){
  if(strcmp(parameter,"inf") == 0)
    return IMF_MASSINF;
  else
    return atof(parameter);
}

/** extracted from JP-A's libchar_string **/


static char *string_parse(const char *line,char *to,const char *delim)
{
    const  char * delim_pt = delim;
    bool found_delim = false;
    while(*line != '\0')
    {
        found_delim = false;
        delim = delim_pt;
        while(*delim != '\0')
        {
            if(*line == *delim)
                found_delim = true;
            delim++;
        }
      
        if(found_delim == false)
            break;
        line++;
    }
    while(*line != '\0')
    {
        delim = delim_pt;
        found_delim = false;
        while(*delim != '\0')
        {
            if(*line == *delim)
                found_delim = true;
            delim++;
        }
        if(found_delim == true)
            break;
        *to = *line;
        line++;
        to++;
    }
    *to = '\0';
    while(*line != '\0')
    {
        found_delim = false;
        delim = delim_pt;
        while(*delim != '\0')
        {
            if(*line == *delim)
                found_delim = true;
            delim++;
        }
      
        if(found_delim == false)
            break;
        line++;
    }
    return (char *) line;
}
