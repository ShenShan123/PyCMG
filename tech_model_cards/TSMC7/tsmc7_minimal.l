* TSMC7 7nm NMOS SVT Minimal Modelcard
* Extracted from: cln7_1d8_sp_v1d2_2p2_clean_fixed.l
* Contains: nch_svt_mac.global + nch_svt_mac.5
* Source lines: 13913-13950 (global), 14850-14900 (.5)
* Model type changed from 'nmos' to 'bsimcmg' for OSDI compatibility
*
.model nch_svt_mac.global bsimcmg (
+level = 72 version = 106.1
+a1 = 0 a11 = 0 a2 = 0
+a21 = 0 aigbacc = 0.025 aigbacc1 = 0
+aigbinv = 0.0258 aigbinv1 = 0 aigc1 = 0
+aigd1 = 0 aigs1 = 0 alpha0 = 0
+alpha01 = 0 alpha1 = 0 alpha11 = 0
+alphaii0 = 5.5e-22 alphaii01 = 0 alphaii1 = 1.67e-14
+alphaii11 = 0 beta0 = 0
+betaii0 = 5.0000000e-03
+betaii1 = 0.002 betaii2 = 5.5000000e-02
+bgidl = 5.983795e+09
+bgisl = 5.983795e+09 bigbacc = 0.005
+bigbinv = 1e-05
+bigc = 0.005 bigd = 0.0012 bigs = 0.0012
+bulkmod = 1 bvd = 3.100000e+00
+bvs = 3.100000e+00
+capmod = 1 cdsc = 0 cdscd = 0
+cfd = 5.857141e-18 cfs = 5.857141e-18 cgbn = 3.8998567e-11
+cgbo = 0 cgdl = 1.357872e-10 cgdo = 1.180742e-12
+cgeomod = 0 cgidl = 0.5 cgisl = 0.5
+cgsl = 1.357872e-10 cgso = 1.180742e-12 cigbacc = 0.03
+cigbinv = 0.005 cigc = 0.03 cigd = 0.01
+cigs = 0.01 cjd = 1.384083e-03 cjs = 1.384083e-03
+cjswd = 1e-18 cjswgd = 1e-18 cjswgs = 1e-18
+cjsws = 1e-18 ckappad = 1.2 ckappas = 1.2
+coremod = 0 deltavsat = 0.01 deltavsatcv = 1
+deltaw = 0 deltawcv = 0 delvfbacc = 0.055
+devtype = 1 dlbin = 0 dlc = 0
+dlcacc = 0 dlcigd = 2.5e-09 dlcigs = 2.5e-09
+drout = 1 dsub = 0.186873 dvt0 = 2.9273
+dvt1 = 0.5071 egidl = 0.098
+egisl = 0.098 eigbinv = 0.2
+emobt = 0 eot = 1.16e-09 eotacc = 1.16e-09
+eotbox = 1.4e-07 epsrox = 3.9 esatii = 10000000
+etamob = 1.41 etaqm = 0.54 fpitch = 3e-08
+geomod = 1 gidlmod = 1 hfin = 4.2e-08
+igbmod = 1 igcmod = 1 igt = 0.6
+iimod = 2 iit = 0 ijthdfwd = 0.1
+ijthdrev = 0.1 ijthsfwd = 0.1 ijthsrev = 0.1
+jsd = 8.000000e-05 jss = 8.000000e-05
+jswd = 0.000000e+00
+jswgd = 0.000000e+00 jswgs = 0.000000e+00
+jsws = 0.000000e+00
+jtss = 2.400000e-01 jtsswgs = 0.000000e+00
+jtssws = 0
+k0 = 0.095 k01 = 0 k0si1 = 0
+k1 = 0.02 k11 = 0 k1rsce = 0.14156406
+k1sat = 0 k1sat1 = 0 k1si1 = 0
+lii = 8.5e-08 lint = 0
+ll = 0 llc = 0 lln = 1
+lpe0 = 8.0625e-09 lrsd = 4.9e-08
+mjd = 2.290000e-01
+mjs = 2.290000e-01 mjswd = 2.290000e-01
+mjswgd = 2.290000e-01
+mjswgs = 2.290000e-01 mjsws = 2.290000e-01
+nbody = 1e+23
+ngate = 0 nigbacc = 2.4 nigbinv = 2
+njd = 1.150000e+00 njs = 1.150000e+00
+njts = 8.000000e+00
+njtssw = 20 njtsswg = 2.000000e+01
+nqsmod = 0
+nsd = 2e+26 ntox = 1.0
+pbd = 5.870000e-01 pbs = 5.870000e-01
+pbswd = 5.870000e-01
+pbswgd = 5.870000e-01 pbswgs = 5.870000e-01
+pbsws = 5.870000e-01
+pclmcv = 0 pclmg = 0 pdibl1 = 0
+pgidl = 1 pgisl = 1 phibe = 0.7
+phig = 4.461 phin = 0.05 pigcd = 0.1
+poxedge = 1 pqm = 0.5 pqmacc = 0.9
+psatcv = 2 pvag = 50
+qm0 = 1 qm0acc = 1 qmfactor = 0
+qmtcencv = 0.1 qmtcencva = 0.08 qmtceniv = 0
+rdsmod = 0 rdsw = 15 rdswmin = 0
+rdw = 20 rdwmin = 0 rgatemod = 0
+rgeomod = 0 rgext = 0 rgfin = 0.001
+rshd = 142 rshs = 142 rsw = 20
+rswmin = 0 shmod = 0
+sii0 = 6.7000000e-01
+sii1 = 0.1 sii2 = 0 siid = 0
+tcj = 5.000000e-04 tcjsw = 5.000000e-04
+tcjswg = 5.000000e-04
+tfin = 6e-09 tgidl = -0.000975 tii = 0
+tmexp = 0 tnjts = 1.300000e+00
+tnjtsd = 1.300000e+00
+tnjtssw = 0 tnjtsswd = 0
+tnjtsswg = 0.000000e+00
+tnjtsswgd = 0.000000e+00 tnom = 25
+toxg = 1.16e-09
+toxp = 1.002253e-09 toxref = 3e-09
+tpb = 5.560000e-04
+tpbsw = 5.560000e-04 tpbswg = 5.560000e-04
+tss = 0.0002565
+uc = 0.01 uc1 = 5.6e-11 ucs = 0.8
+ucste = 0 ud = 0.75 ud1 = 0
+vsatcv = 1000000 vtsd = 7.750000e+00
+vtss = 7.750000e+00
+vtsswd = 10 vtsswgd = 1.000000e+01
+vtsswgs = 1.000000e+01
+vtssws = 10 xjbvd = 1 xjbvs = 1
+xl = 1.0e-8 xrcrg1 = 12
+xtid = 3.000000e+00
+xtis = 3.000000e+00 xtsd = 4.500000e-01
+xtss = 4.500000e-01
+xtsswd = 0 xtsswgd = 0.000000e+00
+xtsswgs = 0.000000e+00
+xtssws = 0 tfindrw = 8e-09
+jtsd = 2.400000e-01
+jtsswd = 0 jtsswgd = 0.000000e+00
+njtsd = 8.000000e+00
+njtsswd = 20 njtsswgd = 2.000000e+01 
************************
*   Aging parameters   *
************************
+tmi_ver_age=113062.050050
+device_id=1.1
+tmimod='tmiageflag'
+etmimod='tmiageflag'
+phys_id='phys_id_mos_svt'
+dagetime='dagetime_mos_svt'
+age_unit='age_unit_mos_svt'
+tr_start='tr_start_mos_svt'
+tr_end='tr_end_mos_svt'
+r_hci='r_hcin_mos_svt'
+r_bti='r_btin_mos_svt'
+r_bti_max_delta='r_bti_max_delta_mos_svt'
+multiwave='multiwave_mos_svt'
+vgsdmax='vgsdmax_mos_svt'
+tmishe='tmishe_mos_svt'
+shevmod='shevmod_mos_svt'
+sheimod='sheimod_mos_svt'
+alter_flag='alter_flag_mos_svt'
+sorting_num='sorting_num'
+sorting_phys='sorting_phys'
+sorting_item='sorting_item'
+lifetime_item='lifetime_item'
+lifetime_value='lifetime_value_mos_svt'
+age_specmod='age_specmod'
+age_spotflag='age_spotflag'
+etmiagechkflag='etmiagechkflag'
+age_globalflag='age_globalflag'
+age_stopmod='age_stopmod'
+age_med_flag='age_med_flag'
**********************************
*   Lifetime Corner parameters   *
**********************************
+lcflag='lcflag'
+inputflag='inputflag'
+lc_dvtlin_hci='lc_dvtlin_hcin_mos_svt'
+lc_dvtlin_bti='lc_dvtlin_btin_mos_svt'
****************************************
*   Thermal Time Constant parameters   *
****************************************
+tthflag='tthflag'
+tth_r='tthn_r_mos_svt'
+tth_f='tthn_f_mos_svt'
***************************
*   Recovery parameters   *
***************************
+recovbtiflag='recovbtiflag'
+rec_delay_flag='rec_delay_flag'
***********************
*   TDDB parameters   *
***********************
+category_id=1.1
+tmitddbflag='tmitddbflag'
+tddbmod='tddbmod'
+area_factor='area_factorn_core'
+area_predict='area_predictn_core'
+tddb_stress_time='tddb_stress_time'
+tddb_fail_perc='tddb_fail_perc_mos_svt'
**********************
*   COA parameters   *
**********************
+tmicoaflag='tmicoaflag'
+coa_sorting_num='coa_sorting_num'
+vgs_coa_max='coa_nonstress_max_mos_svt'
+vgsr_coa_max='coa_nonstress_max_mos_svt'
+vgd_coa_max='coa_nonstress_max_mos_svt'
+vgdr_coa_max='coa_nonstress_max_mos_svt'
+vgb_coa_max='coa_bulk_max_mos_svt'
+vgbr_coa_max='coa_bulk_max_mos_svt'
+vds_coa_max='coa_nonstress_max_mos_svt'
+vbs_coa_max='coa_bulk_max_mos_svt'
+vbsr_coa_max='coa_bulk_max_mos_svt'
+vbd_coa_max='coa_bulk_max_mos_svt'
+vbdr_coa_max='coa_bulk_max_mos_svt'
+vgs_coa_sts_max='coa_stress_max_mos_svt'
+vgsr_coa_sts_max='coa_stress_max_mos_svt'
+vgd_coa_sts_max='coa_stress_max_mos_svt'
+vgdr_coa_sts_max='coa_stress_max_mos_svt'
+vgb_coa_sts_max='coa_bulk_max_mos_svt'
+vgbr_coa_sts_max='coa_bulk_max_mos_svt'
+vds_coa_sts_max='coa_stress_max_mos_svt'
+vbs_coa_sts_max='coa_bulk_max_mos_svt'
+vbsr_coa_sts_max='coa_bulk_max_mos_svt'
+vbd_coa_sts_max='coa_bulk_max_mos_svt'
+vbdr_coa_sts_max='coa_bulk_max_mos_svt'

******************************
*   FinFET PARAMETER START   *
******************************
*****  cc Vth   ****
+ithx  = 1.0e-8
+im    = 0
+dlthx = 0

***********************
*   general setting   *
***********************
+tmi_ver = 108061.120005
+tmi_ver_inst = 2.6                       tmi_ver_rdr = 0.2                        tmi_ver_nfin = 0.5
+tmi_ver_rshg = 0.1                       tmi_ver_adaspdps = 0.2                   tmi_ver_nrsd = 4.6
+tmi_ver_nctx = 0.0                       tmi_ver_dxti = 0.1                       tmi_ver_iboff = 0.5
+tmi_ver_mclib = 0.1                      tmi_ver_mcratio = 3.4                    tmi_ver_mcglobal = 0.6
+tmi_ver_mclocal = 1.6                    tmi_ver_mcdelta = 0.4                    tmi_ver_mccorsum = 0.5
+tmi_ver_cj = 0.2                         tmi_ver_cf = 0.4                         tmi_ver_cg = 0.0
+tmi_ver_xl = 0.3                         tmi_ver_lef = 0.0                        tmi_ver_eflod = 0.0
+tmi_ver_oje = 0.0                        tmi_ver_lod = 1.6                        tmi_ver_ose = 2.0
+tmi_ver_wpe = 0.0                        tmi_ver_ple = 0.3                        tmi_ver_ppe = 0.5
+tmi_ver_mbe = 0.6                        tmi_ver_cpo = 1.0                        tmi_ver_podg = 0.0
+tmi_ver_tnoi=0.4
+tmi_ver_fnoimc=1.3
+tmiclampflag=1
+tmiwarnflag=0
*
+ccodflag     = 'ccodflag_mos_svt'
+ccosflag     = 'ccosflag_mos_svt'
+rcodflag     = 'rcodflag_mos_svt'
+rcosflag     = 'rcosflag_mos_svt'
+rgflag       = 'rgflag_mos_svt'
+ngconflag    = 'ngconflag_mos_svt'
+igflag       = 'igflag_mos_svt'
+iboffwcflag  = 'iboffwcflag_mos_svt'
+ccoskewflag  = 'ccoskewflag_mos_svt'
+rgskewflag   = 'rgskewflag_mos_svt'
+sigma_factor = 'sigma_factorn_svt'
+mismatchflag = 'mismatchflag_mos_svt'
+matchingflag='matchingflag_mos_svt'
+edgeflag='edgeflag_mos_svt'
+local_factor = 'local_factorn_svt'
+globalflag   = 'globalflag_mos_svt'
+totalflag    = 'totalflag_mos_svt'
+global_factor= 'global_factorn_svt'
+nidsmin='nidsminn_mos_svt'
+noiseflag='noiseflagn_mos_svt'
+noiseflag_mc='noiseflagn_mc_mos_svt'
+fnc_sigma_factor='fnc_sigma_factorn_svt'
+fnc_global_factor='fnc_global_factorn_svt'
+noiseflagwmc='noiseflagwmcn_mos_svt'
**********************
*   FinFET setting   *
**********************
+_nfin_cj   = 20
+_nfin_isub = 2
+_nfin_min_mcratio = 2
***********************
*   cco corner skew   *
***********************
+ccoskew_cb=-0.1716
+ccoskew_cw=0.1588
***********************
*   rco corner skew   *
***********************
+rcoskew_rb=-0.1191
+rcoskew_rw=0.1798
***********************
*   rg corner skew    *
***********************
+rgskew_rb=-0.50
+rgskew_rw=0.50
****************************
*   rdr rule restriction   *
****************************
+lrdr1=0.008    lrdr2=0.011    lrdr3=0.02     lrdr4=0.036
+lrdr5=0        lrdr6=0        lrdr7=0        lrdr8=0
*
+lrdr_low =0.072
+lrdr_high=1
*
+nfinfbd_low=0            nfinfbd_high=1.99
+lfbd_low=0               lfbd_high=10
*****************************
*   nrd & nrs calculation   *
*****************************
+md_rule_1=11e-9          md_rule_2=24e-9          md_rule_3=30e-9
+md_rule_4=0e-9           md_rule_5=160e-9
+m0_rule_1=2              m0_rule_2=20e-9          m0_rule_3=20e-9
+vd_rule_1=140e-9         rs_rule_1=6              rm_rule_1=6
*
+md_param_1=0.02400       md_param_2=0.00700       md_param_3=16e-9
+md_param_4=-8e-9         md_param_5=-10e-9
+m0_param_1=4e-9          vd_param_1=70            vd_param_2=39
+rcsd_param_1=0.12000     rcsd_param_2=1e-9        rcsd_param_3=15e-9
+rcsd_param_4=15e-9       rcsd_param_5=54.7
+odtap_param_1=310.0      odtap_param_2=108.2
*
+r_rs_nc=10.62             l_rs_nc=0                ll_rs_nc=1
+n_rs_nc=0                nn_rs_nc=1               p_rs_nc=0
+r_rs_wc=1.728             l_rs_wc=0                ll_rs_wc=1
+n_rs_wc=85              nn_rs_wc=1.26            p_rs_wc=0
*
+r_rm_nc=66.71            l_rm_nc=0                ll_rm_nc=1
+n_rm_nc=0                nn_rm_nc=1               p_rm_nc=0
+r_rm_wc=2.31             l_rm_wc=0                ll_rm_wc=1
+n_rm_wc=270              nn_rm_wc=0.8             p_rm_wc=0
*
+nfin_param_high=20
+nfin_param_1=12          nfin_param_2=6          nfin_param_3=3
+nfin_param_low =2
*
+r0_param_1=0.35          r0_param_2=0             r0_param_3=0
+rd0_param_1=0            rd0_param_2=0            rd0_param_3=0
+rs0_param_1=0            rs0_param_2=0            rs0_param_3=0
**************************
*   iboff size scaling   *
**************************
+r_rjtsswgsc=1            l_rjtsswgsc=0            ll_rjtsswgsc=1
+n_rjtsswgsc=0            nn_rjtsswgsc=1           p_rjtsswgsc=0
+r_rjtsswglc=1            l_rjtsswglc=0            ll_rjtsswglc=1
+n_rjtsswglc=0            nn_rjtsswglc=1           p_rjtsswglc=0
+nfinmin_iboff=1
+lbin_iboff=0.00e-6
************************
*   iboff worst case   *
************************
+r_rjtsswgwc=2            n_rjtsswgwc=0            nn_rjtsswgwc=1
+r_ragidlwc=2             n_ragidlwc=0             nn_ragidlwc=1

************************
*   rshg calculation   *
************************
+rsheet=25.00
+drsh=7.7000              rshh=-0.4000             rshn=0.9300
+dlr=1.6400e-08           llr=-7.9531e-05          llnr=0.9900
***********************
*   dxti definition   *
***********************
+dxti=-6
*********************
*   xl definition   *
*********************
+xl_lc=9e-009
+xl2_lc=15e-009
+lbin_xl=8e-009
+lbin_xl2=11e-009
*********************
*   cf definition   *
*********************
+nfinbin_cf=2
+cfs_nw=8.198300e-18 cfd_nw=8.198300e-18
********************
*   iddq setting   *
********************
+dldvtshift_iddq=-0.0001947
+dlkt1_iddq=-8.5e-05
******************************
*   FinFET PARAMETER END     *
******************************

*******************************
*#    Noise PARAMETER START    *
*******************************
*******************************
*#*   thermal noise definition  *
*******************************
+rnoia=0
+tnoib=0
+tnoia=0
+rnoib=0
+ntnoi=1e-10
+tnoimod=91
+tmemod=1.2
+tnoia_min=0
+tnoia_par1=1040000
+tnoia_par2=5002000
+rnoia_min=0
+rnoia_par1=-0.0008
+rnoia_par2=0.545
+tnoib_min=0
+tnoib_par1=0
+tnoib_par2=0
+rnoib_min=0
+rnoib_par1=0
+rnoib_par2=0
*******************************
*#*   flicker noise definition  *
*******************************
+fn_noiss=1
+etminoiseupdate=1
+etmimodel=1
+noia=1.080000e+42
+noib=2.200000e+25
+noic=1.810000e+9
+ef=0.91
+em=1000000
+lintnoi=-0.0000000191
+noiss_amp_l=0
+noiss_amp_nfin=0
+noiss_amp_r=1
+noiss_amp_total=0
+noiss_clamp=0.4
+noiss_temp=-0.4047
+noiss_vg_l=0
+noiss_vg_nfin=0
+noiss_vg_r=9
+noiss_vg_total=0
+noiss_vth_l=0
+noiss_vth_nfin=0
+noiss_vth_r=0.5
+noiss_vth_total=0
+fn_fac_a=0
+fn_fac_b=1
+arearatio_a=4
+arearatio_b=10
+arearatio_c=-0.24
*******************************
*#    Noise PARAMETER END      *
*******************************
******************************
*   GLOBAL PARAMETER START   *
******************************
***************************************
*   user-friendly parameter setting   *
***************************************
+tmidtemp     = 'dtempn_mos_svt'
+tmidelvtrand = 'delvtrandn_mos_svt'
+tmiu0mult    = 'u0multn_mos_svt'
+tmiids0mult  = 'ids0multn_mos_svt'

*****************************
*   function flag setting   *
*****************************
+rcovarflag='rcovarflag_mos_svt'
+devrdrflag='devrdrflag_mos_svt'
+iddqflag  ='iddqflag_mos_svt'
+crflag    ='crflag_mos_svt'
+gnpskew   ='gnpskewn_mos_svt'

**********************************
*       layout effect parameter  *
**********************************
+lodflag='lodflag_mos_svt' osexflag='osexflag_mos_svt' oseyflag='oseyflag_mos_svt' pleflag='pleflag_mos_svt' mbeflag='mbeflag_mos_svt' cpoflag='cpoflag_mos_svt' ppeflag='ppeflag_mos_svt'
**********************************
*       lod   parameter          *
**********************************
+lodflag0 = 1
+sa_index = 2                sabl     = 0.054e-6
+sarefmod = 1                sarefc   = 0.332e-6           sbrefc    = 0.332e-6
+samax    = 0.6e-06            sbmax    = 0.6e-06
+samin    = 0.057e-6         sbmin    = 0.057e-6
+rllodflag= 0                lreflod  = 1e-6               llodref   = 2
+lod_clamp= -1e50            lmax_lod = 0.240e-6           lmin_lod  = 0.011e-6          minwlod  = 2
********************************************************************
+wlod     = 5
+ku0      = 0                lku0     = 0                  wku0     = 0
+pku0     = 0                tku0     = 0              	   llodku0  = 1
+wlodku0  = 1                kvsat    = 0.78
+kvth0    = 0                lkvth0   = 0                  wkvth0   = 0
+pkvth0   = 0                llodvth  = 2            	   wlodvth  = 2
+stk2     = 0                lodk2    = 1              	   steta0   = 0
+lodeta0  = 1
+wlod0    = 5
+ku00     = 5.044e-09        lku00    = '4.431e-17*1.3'    wku00    = '-1e-09*0'             pku00   = -5.54e-16
+llodku00 = 1                wlodku00 = 1
+kvth00   = -4.34e-11        lkvth00  = -4.78e-33          wkvth00  = 1.152e-08          pkvth00 = 1.08e-32
+llodvth0 = 3                wlodvth0 = 2                  steta00  = 0
+stk20    = 0                lodk20   = 1                  lodeta00 = 1
**********************************************************************
+wlod1    = 5                llod1    = 0
+ku01     = '2.53e-08*6.55' lku01    = '-6.11e-10*53.6'        wku01    = '-7.6e-08*0.8'     pku01   = '3.389e-08*0.96'
+llodku01 = 0.1              wlodku01 = 1
+kvsat1   = 0.87
+kvth01   = '-2.71e-10*70.0'  lkvth01  = '1.82e-17*20.85'    wkvth01  = -1.11e-07          pkvth01 = '1.778e-15*1.75'
+llodvth1 = 1                wlodvth1 = 2
+steta01  = 0                lodeta01 = 1
+stk21    = 0                lodk21   = 1
*********************************************************************
+wlod3    = 3
+ku03     = 0                lku03    = 0                  wku03    = 0
+pku03    = 0                tku03    = 0                  llodku03 = 1
+wlodku03 = 0                kvsat3   = 1
+kvth03   = 0                lkvth03  = 0                  wkvth03  = 0
+pkvth03  = 0                llodvth3 = 1                  wlodvth3 = 1
+stk23    = 0                lodk23   = 1                  steta03  = 0
+lodeta03 = 2
+wlod03   = 6
+ku003    = '2.40e-04*-20.1'         lku003   = '1.5e-10*-1.5'            wku003   = 1.12e-02           pku003  = -1.80e-10
+llodku003= 1                wlodku003= 1
+kvth003  = '2.32e-03*1.5'         lkvth003 = '-1.96e-11*0.9*1.2'    wkvth003 = -3.13e-03           pkvth003= '1.39e-10*1.2'
+llodvth03= 1                wlodvth03= 1                  steta003 = 0
+stk203   = 0                lodk203  = 1
+lodeta003= 1
*
+steta01p1  = -4.2               lodeta01p1 = 1
+kvsat1p1   = 1.3
+ku003p1    = '2.40e-04*-20.1'         lku003p1   = '1.5e-10*-1.5'            wku003p1   = 1.12e-02           pku003p1  = -1.80e-10

+kvth003p1  = '2.32e-03*1.5'         lkvth003p1 = '1.96e-11*0.9*6.2'    wkvth003p1 = '-3.13e-03*10.5'           pkvth003p1= '1.39e-10*12.5'

+wkvth01p1  = '-1.11e-07*0.995'
+wkvth00p1  = '1.152e-08*-0.85'
+pku01p1   = '3.389e-08*0.86' pkvth01p1 = '1.778e-15*1.75'
*
+kvth003p2  = '2.32e-03*1.5*0'         lkvth003p2 = '1.96e-11*0.9*34.5'    wkvth003p2 = '-3.13e-03*0'           pkvth003p2= '1.39e-10*12.5*0'
+ku003p2    = '2.40e-04*-20.1*0'         lku003p2   = '1.5e-10*0.6'            wku003p2   = '1.12e-02*0'           pku003p2  = '-1.80e-10*0'
*************************
*     constant pitch      *
*************************
+spa_l=0.0115e-6                 pitch_spa=0.057e-6
********************************************************************
+sparef1   = 0.080e-06          sparef2   = 0.094e-06           spabl     = 0.036e-06
********************************************************************
**********************************
*       ose sfx  parameter          *
**********************************
+oseflag0 = 1
+osexflag0 = 1
+oseyflag0 = 1
+sfxmax      = 2.09e-6             minwodx     = 2                 sfxmin      = 0.046e-6
+lrefodx     = 0.036e-6         lodxref     = 2                 wodx        = 0
+kvth0odxa   = 0                lkvth0odxa  = 0                 lodxakvth0  = 1
+wkvth0odxa  = 0                wodxakvth0  = 1.3               pkvth0odxa  = 0
+ku0odxa     = 0                lku0odxa    = 0                 lodxaku0    = 1
+wku0odxa    = 0                wodxaku0    = 1.2               pku0odxa    = 0
+keta0odx    = 0                wketa0odx   = 0                 wodxketa0   = 1
+kvsatodx    = 0
+kvth0odxb   = '-0.0294-0.006'          lkvth0odxb  = '0.15e-17*6.2'         lodxbkvth0  = 2
+wkvth0odxb  = -1.44e-03        wodxbkvth0  = 1                 pkvth0odxb  = 0
+ku0odxb     = 0.04             lku0odxb    = -5.0e-06    	lodxbku0    = 0.5
+wku0odxb    = 0.042            wodxbku0    = 1                 pku0odxb    = -1.1e-05
********************************************************************************************
+lrefodx1    = 0.05e-6         lodx1ref    = 1                 r_sodx1     = 0.0
+rsodxa      = 0.5
+kvth0odx1a  = -9e-4            lkvth0odx1a = 0                 lodx1akvth0 = 2
+wkvth0odx1a = 0                wodx1akvth0 = 3                 pkvth0odx1a = 0
+ku0odx1a    = 0                lku0odx1a   = 0                 lodx1aku0   = 1
+wku0odx1a   = 0                wodx1aku0   = 1                 pku0odx1a   = 0
+keta0odx1   = 0                wketa0odx1  = 0                 wodx1keta0  = 1
+kvsatodx1   = 0.0
+kvth0odx1b  = '4.4e-4+0.6e-04'	        lkvth0odx1b = '1e-11*0.75'             lodx1bkvth0 = 1
+wkvth0odx1b = -5e-05	        wodx1bkvth0 = 2                 pkvth0odx1b = 0
+ku0odx1b    = -6.72e-4      	lku0odx1b   = 8.125e-8	        lodx1bku0   = 0.5
+wku0odx1b   = -1.2e-4          wodx1bku0   = 0.5               pku0odx1b   = 5e-08
+csaodx3     = 0                csfxodx3    = 0                 sfxpeak     = 1e-013
+kvth0odx3b  = 0                lkvth0odx3b = 0                 lodx3bkvth0 = 1
+wkvth0odx3b = 0                wodx3bkvth0 = 1                 pkvth0odx3b = 0
+ku0odx3b    = 0                lku0odx3b   = 0                 lodx3bku0   = 1
+wku0odx3b   = 0                wodx3bku0   = 1                 pku0odx3b   = 0
+keta0odx3   = 0                kvsatodx3   = 0
********************************************************************************************
+sfxmax4      = 2.09e-6
+kvth0odx4b  = 0                lkvth0odx4b = '-2.31821e-41*5.2'      lodx4bkvth0 = 5
+wkvth0odx4b = 0           	wodx4bkvth0 = 1                 pkvth0odx4b = 0
+ku0odx4b    = 0                lku0odx4b   = 1.25e-3 	        lodx4bku0   = 0.25
+wku0odx4b   = 0                wodx4bku0   = 1                 pku0odx4b   = -0.0006
+keta0odx4   = 0                kvsatodx4   = 0
+wketa0odx4  = 0                wodx4keta0  = 1
+lrefodx4    = 5e-6             lodx4ref    = 2
+kvth0odx4a  = -2.42            lkvth0odx4a = '6.264e-14*0.85'          lodx4akvth0 = 2
+wkvth0odx4a = -2.8             wodx4akvth0 = 2                  pkvth0odx4a = 0
+ku0odx4a    = 4.41             lku0odx4a   = 1.56e-14           lodx4aku0   = 2
+wku0odx4a   = -3.5             wodx4aku0   = 2                  pku0odx4a   = 0
********************************************************************************************
+lrefodx5     = 5e-8             lodx5ref     = 0
+kvth0odx5a   = 0                lkvth0odx5a  = 0                 lodx5akvth0  = 1
+wkvth0odx5a  = 0                wodx5akvth0  = -0.23             pkvth0odx5a  = 0
+ku0odx5a     = 0                lku0odx5a    = 0                 lodx5aku0    = 1
+wku0odx5a    = 0                wodx5aku0    = 1                 pku0odx5a    = 0
+keta0odx5    = 0                wketa0odx5   = 0                 wodx5keta0   = 1
+kvsatodx5    = 0
+kvth0odx5b   = 0                lkvth0odx5b  = 0                 lodx5bkvth0  = 3
+wkvth0odx5b  = 0           	 wodx5bkvth0  = 1.5               pkvth0odx5b  = 0
+ku0odx5b     = 0                lku0odx5b    = 0                 lodx5bku0    = 3
+wku0odx5b    = 0                wodx5bku0    = 1.5               pku0odx5b    = 0
********************************************************************************************
+lrefodx6    = 5e-8             lodx6ref    = 0
+kvth0odx6a  = 0                lkvth0odx6a = 0                 lodx6akvth0 = 1
+wkvth0odx6a = 0                wodx6akvth0 = 1                 pkvth0odx6a = 0
+ku0odx6a    = 0                lku0odx6a   = 0                 lodx6aku0   = 1
+wku0odx6a   = 0                wodx6aku0   = 1                 pku0odx6a   = 0
+keta0odx6   = 0                wketa0odx6  = 0                 wodx6keta0  = 1
+kvsatodx6   = 0
+kvth0odx6b  = 0                lkvth0odx6b = 0                 lodx6bkvth0 = 3
+wkvth0odx6b = 0                wodx6bkvth0 = 3                 pkvth0odx6b = 0
+ku0odx6b    = 0         	lku0odx6b   = 0                 lodx6bku0   = 3
+wku0odx6b   = 0                wodx6bku0   = 3                 pku0odx6b   = 0
**********************************************************************************************
+kvth0odx8b  = 0                lkvth0odx8b = 0                  lodx8bkvth0 = 1
+wkvth0odx8b = 0                wodx8bkvth0 = 1.3                pkvth0odx8b = 0
+ku0odx8b    = 0                lku0odx8b   = 0                  lodx8bku0   = 3
+wku0odx8b   = 0                wodx8bku0   = 1.3                pku0odx8b   = 0
+keta0odx8   = 0                kvsatodx8   = 0
+wketa0odx8  = 0                wodx8keta0  = 1
+lrefodx8    = 5e-8             lodx8ref    = 0
+kvth0odx8a  = 0                lkvth0odx8a = 0                 lodx8akvth0 = 1
+wkvth0odx8a = 0                wodx8akvth0 = 1                 pkvth0odx8a = 0
+ku0odx8a    = 0                lku0odx8a   = 0                 lodx8aku0   = 1
+wku0odx8a   = 0                wodx8aku0   = 1                 pku0odx8a   = 0
********************************************************************************************
**********************************
*       ose sfy  parameter          *
**********************************
+sfyref1     = 0.202e-6           sfyref2     = 0.292e-6
+sfytref1    = 0.202e-6           sfybref1    = 0.202e-6
+sfymin      = 0.172e-6
+sfymax      = 0.382e-6           minwody     = 2                  wody        = 0e-7
+lmax_osey   = 0.072e-6		  lmin_osey   = 0.010e-6
+lmax_osey1  = 0.240e-6		  lmin_osey1  = 0.010e-6
+rsodyt      = 0.5
+kvth0odya   = 0                  lkvth0odya  = 0                  lodyakvth0  = 1
+wkvth0odya  = 0                  wodyakvth0  = 1                  pkvth0odya  = 0
+ku0odya     = 0                  lku0odya    = 0                  lodyaku0    = 1
+wku0odya    = 0                  wodyaku0    = 1                  pku0odya    = 0
+keta0ody    = 0                  wketa0ody   = 0                  wodyketa0   = 1
+kvsatody    = 0
+lrefody     = 0                  lodyref     = 0
+kvth0odyb   = 0                  lkvth0odyb  = 0.6e-8             lodybkvth0  = 1
+wkvth0odyb  = 0		  wodybkvth0  = 1                  pkvth0odyb  = 3.5e-8
+ku0odyb     = 0                  lku0odyb    = 0                  lodybku0    = 1
+wku0odyb    = -1e+1              wodybku0    = 0.5                pku0odyb    = '15e-8*1.35'
********************************************************************************************
+expody1     = 0.45
+kvth0ody1a  = 0                  lkvth0ody1a = 0                  lody1akvth0 = 1
+wkvth0ody1a = 0                  wody1akvth0 = 1                  pkvth0ody1a = 0
+ku0ody1a    = 0                  lku0ody1a   = 0                  lody1aku0   = 1
+wku0ody1a   = 0                  wody1aku0   = 1                  pku0ody1a   = 0
+keta0ody1   = 0                  wketa0ody1  = 0                  wody1keta0  = 1
+kvsatody1   = -0.8
+lrefody1    = 5e-8               lody1ref    = 0.0
+kvth0ody1b  = -2.0e-03           lkvth0ody1b = -4.5e-40           lody1bkvth0 = 5
+wkvth0ody1b = -1.5e-01           wody1bkvth0 = 1.5                pkvth0ody1b = '-0.92e-39*1.25'
+ku0ody1b    =0			lku0ody1b   =0			 lody1bku0   = 1
+wku0ody1b   =0			wody1bku0   = 1                  pku0ody1b   = -1.05e-8
+sfypk       = 0.326e-6           sfypkw      = 0                  sfypkl      = 0
+sfypkwn     = 1                  sfypkln     = 1                  sfypkp      = 0
********************************************************************************************
+expody11     = 0.5
+sfymin11     = 0.083e-6
+sfymax11     = 0.174e-6
+kvth0ody11a  = 0                 lkvth0ody11a = 0                 lody11akvth0 = 1
+wkvth0ody11a = 0                 wody11akvth0 = 1                 pkvth0ody11a = 0
+ku0ody11a    = 0                 lku0ody11a   = 0                 lody11aku0   = 1
+wku0ody11a   = 0                 wody11aku0   = 1                 pku0ody11a   = 0
+keta0ody11   = 0                 wketa0ody11  = 0                 wody11keta0  = 1
+kvsatody11   = 0
+lrefody11    = 5e-8              lody11ref    = 0
+kvth0ody11b  = 12.5e-03          lkvth0ody11b = 15.5e-35           lody11bkvth0 = 4
+wkvth0ody11b = '0.62e-01*3.2'          wody11bkvth0 = 2.8               pkvth0ody11b = '4.56e-33*-0.6'
+ku0ody11b    = 0.0		 lku0ody11b   = -2.55e-33          lody11bku0   = 4
+wku0ody11b   = 0.12            wody11bku0   = 0.1               pku0ody11b   = 16.05e-33
********************************************************************************************
+expody12     = 0.3
+sfymin12     = 0.064e-6
+sfymax12     = 0.081e-6
+kvth0ody12a  = 0                 lkvth0ody12a = 0                 lody12akvth0 = 1
+wkvth0ody12a = 0                 wody12akvth0 = 1                 pkvth0ody12a = 0
+ku0ody12a    = 0                 lku0ody12a   = 0                 lody12aku0   = 1
+wku0ody12a   = 0                 wody12aku0   = 1                 pku0ody12a   = 0
+keta0ody12   = 0                 wketa0ody12  = 0                 wody12keta0  = 1
+kvsatody12   = 0
+lrefody12    = 5e-8              lody12ref    = 0
+kvth0ody12b  = '-0.87e-01*0.250' lkvth0ody12b = 0          lody12bkvth0 = 4
+wkvth0ody12b = 0                 wody12bkvth0 = 1                 pkvth0ody12b = 0
+ku0ody12b    = '-1.5e-02*-2*-40.25*1.25'     lku0ody12b   = 0                 lody12bku0   = 4
+wku0ody12b   = 0                 wody12bku0   = 1                 pku0ody12b   = 0
********************************************************************************************
+wmaxsody2   = 20
+expody2     = 0.3
+kvth0ody2a  = 0                  lkvth0ody2a = 0                  lody2akvth0 = 1
+wkvth0ody2a = 0                  wody2akvth0 = 1                  pkvth0ody2a = 0
+ku0ody2a    = 0                  lku0ody2a   = 0                  lody2aku0   = 1
+wku0ody2a   = 0                  wody2aku0   = 1                  pku0ody2a   = 0
+keta0ody2   = 0                  wketa0ody2  = 0                  wody2keta0  = 1
+kvsatody2   = 1
+lrefody2    = 5e-8               lody2ref    = 0
+kvth0ody2b  = 0                  lkvth0ody2b = 0                  lody2bkvth0 = 1
+wkvth0ody2b = 0                  wody2bkvth0 = 1                  pkvth0ody2b = 1.45e-25
+ku0ody2b    = 0                  lku0ody2b   = 0                  lody2bku0   = 5
+wku0ody2b   = 0                  wody2bku0   = 1                  pku0ody2b   = 1.35e-56
********************************************************************************************
+wmaxsody21   = 20
+expody21     = 0.3
+kvth0ody21a  = 0                 lkvth0ody21a = 0                 lody21akvth0 = 1
+wkvth0ody21a = 0                 wody21akvth0 = 1                 pkvth0ody21a = 0
+ku0ody21a    = 0                 lku0ody21a   = 0                 lody21aku0   = 1
+wku0ody21a   = 0                 wody21aku0   = 1                 pku0ody21a   = 0
+keta0ody21   = 0                 wketa0ody21  = 0                 wody21keta0  = 1
+kvsatody21   = -0.2
+lrefody21    = 5e-8              lody21ref    = 0
+kvth0ody21b  = 0                 lkvth0ody21b = 0                 lody21bkvth0 = 1
+wkvth0ody21b = 0                 wody21bkvth0 = 1                 pkvth0ody21b = -0.42e-25
+ku0ody21b    = 0                 lku0ody21b   = 0                 lody21bku0   = 1
+wku0ody21b   = 0                 wody21bku0   = 1                 pku0ody21b   = '-2.0e-25*0.2'
********************************************************************************************
+expody3     = 0.5
+kvth0ody3a  = 0                  lkvth0ody3a = 0                  lody3akvth0 = 1
+wkvth0ody3a = 0                  wody3akvth0 = 1                  pkvth0ody3a = 0
+ku0ody3a    = 0                  lku0ody3a   = 0                  lody3aku0   = 1
+wku0ody3a   = 0                  wody3aku0   = 1                  pku0ody3a   = 0
+keta0ody3   = 0                  wketa0ody3  = 0                  wody3keta0  = 1
+kvsatody3   = 1
+lrefody3    = 5e-8               lody3ref    = 0
+kvth0ody3b  = 0	          lkvth0ody3b = 0	           lody3bkvth0 = 1
+wkvth0ody3b = 0                  wody3bkvth0 = 1                  pkvth0ody3b = 0e-6
+ku0ody3b    = 0                  lku0ody3b   = 0                  lody3bku0   = 1
+wku0ody3b   = 0                  wody3bku0   = 1                  pku0ody3b   = 0e-6
********************************************************************************************
**********************************
*       ple parameter            *
**********************************
+l_ple=7.2e-08
**********************************
*       ple od parameter         *
**********************************
********************************************************************************************
+pleflag0=1
+pl_odflag=1
+wplod        = 0
+plodaref     = 0.072e-6          plodbref     = 0.072e-6
+plodamin     = 0.072e-6          plodbmin     = 0.072e-6
+plodamax     = 2.4e-07           plodbmax     = 2.4e-07
**+plodamax     = 1.4e-07           plodbmax     = 1.4e-07
+rplod1       = 1                 rplod2       = 0                rplod3       = 0
+rplod12      = 1                 rplod13      = 1                rplod23      = 1

+kvth0ploda   = 0                 lkvth0ploda =  1.0                lplodakvth0  = 1
+wkvth0ploda  = 0                 wplodakvth0  = 1.0              pkvth0ploda  = 0.0

+ku0ploda     = 0                 lku0ploda    = 0                lplodaku0    = 1.0
+wku0ploda    = 0.0               wplodaku0    = 1.0              pku0ploda    = '60*3.83e-12'

+keta0plod    = 0                 wketa0plod   = 0.0              wplodketa0   = 1.0
+lrefploda    = 5e-8              lplodaref    = 0
+lrefplodb    = 5e-8              lplodbref    = 0
*+kvsatplod    = 2

+kvth0plodb   = -0.000142        lkvth0plodb  = '1.4e-11'         lplodbkvth0  = 1.0
+wkvth0plodb  = 0.000355         wplodbkvth0  = 1                pkvth0plodb  = -0.55e-10

+ku0plodb     = 0		lku0plodb    = 0         lplodbku0    = 1
+wku0plodb    = 0                wplodbku0    = 1            pku0plodb    = '-220*3.83e-12'

+rplod        = 0.5
********************************************************************************************
**********************************
*       mbe parameter            *
**********************************
+mbeflag0    = 1
+smbref      = 10e-6
+smbmin      = 0.048e-6
+smbmax      = 10e-6            minwsmb     = 2.6           wsmb        = 0
+lmax_mbe    = 0.240e-6		lmin_mbe    = 0.011e-6
+rsmbt       = 0.5
+smbpwr      = 1
+kvth0smba   = 0	        lkvth0smba  = 0			  lsmbakvth0  = 1
+wkvth0smba  = 0		wsmbakvth0  = 1			  pkvth0smba  = 0
+ku0smba     = 0	        lku0smba    = 0		          lsmbaku0    = 1
+wku0smba    = 0.0              wsmbaku0    = 1	                  pku0smba    = 0
+keta0smb    = 0                wketa0smb   = 0                   wsmbketa0   = 1
+kvsatsmb    = '-0.3*0+0.5'
+lrefsmb     = 5e-8             lsmbref     = 0
+kvth0smbb   = 1                lkvth0smbb  = -7.28e-24           lsmbbkvth0  = 3
+wkvth0smbb  = 2                wsmbbkvth0  = 1                   pkvth0smbb  = '4.77e-23*0.96'
+ku0smbb     = 7                lku0smbb    = -3e-23              lsmbbku0    = 3
+wku0smbb    = -2               wsmbbku0    = 1			  pku0smbb    = '-3e-23*-1.2'
********************************************************************************************
**********************************
*       cpo   parameter          *
**********************************
+cpoflag0     = 1
+sporef       = 0.326e-6
+l_icpo       = 0.24e-6             w_icpo         = 0.578e-6
+sel_nlicpo   = 2e-10               nlrticpo       = 1.2              nlrticpo3      = 1.36	   tbrticpo       = 0.78
+rticpo       = 1                   rticpo3        = '0.43-0.32'	      icpopwr        = 0.5
+sicpomin     =	0.024e-6
+sicpomax     =	0.18e-6             minwsicpo      = 2.0              wsicpo	     = 0
+kvth0sicpoa  =	0                   lkvth0sicpoa   = 0                lsicpoakvth0   = 1
+wkvth0sicpoa =	0                   wsicpoakvth0   = 1                pkvth0sicpoa   = 0
+ku0sicpoa    =	0                   lku0sicpoa     = 0                lsicpoaku0     = 1
+wku0sicpoa   =	0                   wsicpoaku0     = 1                pku0sicpoa     = 0
+kvsatsicpo   =	0.55
+lrefsicpo    =	5e-8                lsicporef      = 0

+kvth0sicpob  =	0                   lkvth0sicpob   = -2.2e-01           lsicpobkvth0   = 0.5
+wkvth0sicpob =	-1.5e+04            wsicpobkvth0   = 2.0              pkvth0sicpob   = '10.65e-01'

+ku0sicpob    =	1e+03               lku0sicpob     = -1e-04           lsicpobku0     = 1
+wku0sicpob   =	-8e+04              wsicpobku0     = 1                pku0sicpob     = '-4e-04*0.2'

+rticpol      = 0.44                icpol          = 0.171e-6         icpop          = 2.1
+discpode     = 0.106e-6            rticpodes      = 0                rticpode       = 1

+l_gcpo       = 0.24e-6             w_gcpo         = 0.578e-6
+sel_nlgcpo   = 2e-10               nlrtgcpo       = 0.6              nlrtgcpo3      = 0.55
+rtgcpo       = 2.7                 rtgcpo3        = 1.9	      gcpopwr        = 1.45
+sgcpomin     =	0.028e-6
+sgcpomax     =	0.18e-6             minwsgcpo      = 2                wsgcpo	     = 0
+kvth0sgcpoa  =	0                   lkvth0sgcpoa   = 0                lsgcpoakvth0   = 2
+wkvth0sgcpoa =	0	            wsgcpoakvth0   = 2                pkvth0sgcpoa   = 0
+ku0sgcpoa    =	0                   lku0sgcpoa     = 0                lsgcpoaku0     = 1
+wku0sgcpoa   =	0.0                 wsgcpoaku0     = 2.0              pku0sgcpoa     = 0
+kvsatsgcpo   =	1
+lrefsgcpo    =	5e-8                lsgcporef      = 0

+kvth0sgcpob  =	-0.75e-5            lkvth0sgcpob   = -2e-22            lsgcpobkvth0   = 2
+wkvth0sgcpob =	-2.6208e-4  	    wsgcpobkvth0   = 2                 pkvth0sgcpob   = '-2.2e-20*4.5'

+ku0sgcpob    =	-1.2e-4             lku0sgcpob     = -0.3e-19	       lsgcpobku0     = 2
+wku0sgcpob   =	-2.9325e-3          wsgcpobku0     = 1.5               pku0sgcpob     = '-0.69e-18*-0.5'
********************************************************************************************
**********************************
*       ppe   parameter          *
**********************************
+ppeflag0  = 1                     l_ppe     = 0.020e-6                    pitch_spa2=0.064e-6
+minwppe   = 3                     wppe      = 0
+kvth0ppe  = '-0.0005835*27'         lkvth0ppe = '-4.5e-26*2.75*-0.013e9'             lppekvth0 = 2
+wkvth0ppe = 0                     wppekvth0 = 1                           pkvth0ppe = 0
+ku0ppe    = '3e-2*9.23*1.625'                 lku0ppe   = '1.55e-25*3.1*-1.55'                    lppeku0   = 3
+wku0ppe   = 450                  wppeku0   = 8                           pku0ppe   = 0
+kvsatppe  = 0.42
+keta0ppe  = 0                     wketa0ppe = 0                           wppeketa0 = 0
+rsa1ppe   = '-2.0*0.25'
+kcitppe   = '1.30e-3*1.45'
+kppe_utl  = 0.42
********************************************************************************************
+ppe2flag0  = 1                      l_ppe2     = 0.020e-6                    pitch_spa3=0.076e-6
+minwppe2   = 3                      wppe2      = 0
*+kvth0ppe2  = '0.014937*0.27'         lkvth0ppe2 = '-6.08e-11*3'                   lppe2kvth0 = 1
+kvth0ppe2  = '-0.014937*0.27*8.2+0.001'         lkvth0ppe2 = '6.08e-27*25.5'                   lppe2kvth0 = 3
+wkvth0ppe2 = 0                      wppe2kvth0 = 1                           pkvth0ppe2 = 0
*+ku0ppe2    = '0.079025*0.83'               lku0ppe2   = '-3e-11*3'                    lppe2ku0   = 1
+ku0ppe2    = '0.079025*2.45'               lku0ppe2   = '-5e-26*10.5'                    lppe2ku0   = 3
+wku0ppe2   = 1e-2                   wppe2ku0   = 1                           pku0ppe2   = '0-8e-26'
*+kvsatppe2  = '0.2*13'
+kvsatppe2  = '0.2*19.0'
*+keta0ppe2  = 0.036                      wketa0ppe2 = 0                           wppe2keta0 = 0
+keta0ppe2  = 0.020                     wketa0ppe2 = 0                           wppe2keta0 = 0
+rsa1ppe2   = '-5.25*0.45'
*+kcitppe2   = '1.80e-3'
+kcitppe2   = '2.40e-3'
+kppe2_utl  = 0.3
********************************************************************************************
****************************
*   GLOBAL PARAMETER END   *
****************************

*************************************
*   statistical/corner parameters   *
*************************************
+tox_c      = 'toxn_svt'         dxl_c      = 'dxln_svt'        dtfin_c    = 'dtfinn_svt'
+dhfin_c    = 'dhfinn_svt'       cjdrw_c    = 'cjdrwn_svt'      cjsw_c     = 'cjswn_svt'
+cjswg_c    = 'cjswgn_svt'       cgo_c      = 'cgon_svt'        cgl_c      = 'cgln_svt'
+ddlc_c     = 'ddlcn_svt'        cfd_c      = 'cfdn_svt'        cfs_c      = 'cfsn_svt'
+ccod_c     = 'ccodn_svt'        ccos_c     = 'ccosn_svt'       ddvtshift_c = 'ddvtshiftn_svt'
+dldvtshift_c = 'dldvtshiftn_svt'                               dndvtshift_c = 'dndvtshiftn_svt'
+dpdvtshift_c = 'dpdvtshiftn_svt'                               dcit_c     = 'dcitn_svt'
+dlcit_c    = 'dlcitn_svt'       dncit_c    = 'dncitn_svt'      dpcit_c    = 'dpcitn_svt'
+dtss_c     = 'dtssn_svt'        dltss_c    = 'dltssn_svt'      dntss_c    = 'dntssn_svt'
+dptss_c    = 'dptssn_svt'       dk0si_c    = 'dk0sin_svt'      dlk0si_c   = 'dlk0sin_svt'
+dnk0si_c   = 'dnk0sin_svt'      dpk0si_c   = 'dpk0sin_svt'     deta0_c    = 'deta0n_svt'
+dleta0_c   = 'dleta0n_svt'      dneta0_c   = 'dneta0n_svt'     dpeta0_c   = 'dpeta0n_svt'
+du0_c      = 'du0n_svt'         dlu0_c     = 'dlu0n_svt'       dnu0_c     = 'dnu0n_svt'
+dpu0_c     = 'dpu0n_svt'        dua_c      = 'duan_svt'        dlua_c     = 'dluan_svt'
+dnua_c     = 'dnuan_svt'        dpua_c     = 'dpuan_svt'       dvsat_c    = 'dvsatn_svt'
+dlvsat_c   = 'dlvsatn_svt'      dnvsat_c   = 'dnvsatn_svt'     dpvsat_c   = 'dpvsatn_svt'
+dksativ_c  = 'dksativn_svt'     dlksativ_c = 'dlksativn_svt'   dnksativ_c = 'dnksativn_svt'
+dpksativ_c = 'dpksativn_svt'    dpclm_c    = 'dpclmn_svt'      dlpclm_c   = 'dlpclmn_svt'
+dnpclm_c   = 'dnpclmn_svt'      dppclm_c   = 'dppclmn_svt'     dpdibl2_c  = 'dpdibl2n_svt'
+dlpdibl2_c = 'dlpdibl2n_svt'    dnpdibl2_c = 'dnpdibl2n_svt'   dppdibl2_c = 'dppdibl2n_svt'
+dk1_c      = 'dk1n_svt'         dlk1_c     = 'dlk1n_svt'       dnk1_c     = 'dnk1n_svt'
+dpk1_c     = 'dpk1n_svt'        dk1sat_c   = 'dk1satn_svt'     dlk1sat_c  = 'dlk1satn_svt'
+dnk1sat_c  = 'dnk1satn_svt'     dpk1sat_c  = 'dpk1satn_svt'    duc_c      = 'ducn_svt'
+dluc_c     = 'dlucn_svt'        dnuc_c     = 'dnucn_svt'       dpuc_c     = 'dpucn_svt'
+dkt1_c     = 'dkt1n_svt'        dlkt1_c    = 'dlkt1n_svt'      dnkt1_c    = 'dnkt1n_svt'
+dpkt1_c    = 'dpkt1n_svt'       dutl_c     = 'dutln_svt'       dlutl_c    = 'dlutln_svt'
+dnutl_c    = 'dnutln_svt'       dputl_c    = 'dputln_svt'      duc1_c     = 'duc1n_svt'
+dluc1_c    = 'dluc1n_svt'       dnuc1_c    = 'dnuc1n_svt'      dpuc1_c    = 'dpuc1n_svt'
+dat_c      = 'datn_svt'         dlat_c     = 'dlatn_svt'       dnat_c     = 'dnatn_svt'
+dpat_c     = 'dpatn_svt'        ntox_c     = 'ntoxn_svt'       rcoskew_c  = 'rcoskewn_svt'
+ss_flag_c  = 'ss_flagn_svt'     ff_flag_c  = 'ff_flagn_svt'    sf_flag_c  = 'sf_flagn_svt'
+fs_flag_c  = 'fs_flagn_svt'
+monte_flag_c = 'monte_flagn_svt'
+ssa_flag_c = 'ssa_flagn_svt'   ffa_flag_c = 'ffa_flagn_svt'

*************************
*   variables setting   *
*************************
+par1='par1' par2='par2' par3='par3' par4='par4' par5='par5' par6='par6' par7='par7' par8='par8' par9='par9' par10='par10'
+par11='par11' par12='par12' par13='par13' par14='par14' par15='par15' par16='par16' par17='par17' par18='par18' par19='par19' par20='par20'
+par21='par21' par22='par22' par23='par23' par24='par24' par25='par25' par26='par26' par27='par27' par28='par28' par29='par29' par30='par30'
+par31='par31' par32='par32'
+plo_tox='plo_tox' plo_dxl='plo_dxl' plo_dhfin='plo_dhfin' plo_dtfin='plo_dtfin' parl1='parl1' parl2='parl2'
+w1='w1_c' w2='w2_c' w3='w3_c' w4='w4_c' w5='w5_c' w6='w6_c' w7='w7_c' w8='w8_c' w9='w9_c' w10='w10_c'





+par1_age='par1_age_c'
+parl1_age='parl1_age_c'

+par1fn_mc='par1fn_mc_c'
+par2fn_mc='par2fn_mc_c'
**** ratio parameter
+a_1        = 1
+b_1        = 0                          c_1        = 0                          d_1        = 0
+e_1_1      = 0                          e_1_2      = 0                          f_1_1      = 0
+f_1_2      = 0                          g_1_1      = 0                          g_1_2      = 0


+a_3        = 1                          b_3        = 0                          c_3        = 0
+d_3        = 0                          e_3_1      = 0                          e_3_2      = 0
+f_3_1      = 0                          f_3_2      = 0                          g_3_1      = 0
+g_3_2      = 0

+a_4        = 0.945                      b_4 = -0.000182                   c_4        = 13351.414303
+d_4        = 428951.076286		 e_4_1      = -1.265891                  e_4_2      = 6.2e-05
+f_4_1      = 0.433098                   f_4_2      = 0.06181                    g_4_1      = -424048.205818
+g_4_2      = -0.998304

**** local parameter
+r_mis_dvtshift   = 0.00246                ll_mis_dvtshift   = 0.32               nn_mis_dvtshift   = -0.022
+r_mis_lgrp       = 0.00378              ll_mis_lgrp       = -0.024             nn_mis_lgrp       = 0.275
+r_mis_tgrp       = '0.003685*0.8'             ll_mis_tgrp       = 0.0                 nn_mis_tgrp       = '0.12*0'
+r_mis_corr       = 0.54                 ll_mis_corr       = -0.05               nn_mis_corr       = 0.0075
+r_mis_ogrp       = 3.7010780e-04        ll_mis_ogrp       = 0.0                 nn_mis_ogrp       = 0.0
+r_mis_u0         = 0.004              ll_mis_u0         = -0.04               nn_mis_u0         = 0.12
+u0_limit1        = 0.1                  u0_limit2         = 2
+r_mis_eta0       = 0.0                  ll_mis_eta0       = 0.0                 nn_mis_eta0       = 0.0
+avtgm_min        = 0.0                  geo_local_min     = 2.3                 geo_local_base    = 2.7182818
+lanr_low=0.072
+lanr_high=1
+geo_edge_factor=0.55
+geo_local_factor=1.7




*

+global_mc = 'global_mc_flag_svt' tox_g = 'toxn_svt_ms_global' dxl_g = 'dxln_svt_ms_global'
+dtfin_g = 'dtfinn_svt_ms_global' dhfin_g = 'dhfinn_svt_ms_global' cjdrw_g = 'cjdrwn_svt_ms_global'
+cjsw_g = 'cjswn_svt_ms_global' cjswg_g = 'cjswgn_svt_ms_global' cgo_g = 'cgon_svt_ms_global'
+cgl_g = 'cgln_svt_ms_global' ddvtshift_g = 'ddvtshiftn_svt_ms_global' dldvtshift_g = 'dldvtshiftn_svt_ms_global'
+dndvtshift_g = 'dndvtshiftn_svt_ms_global' dpdvtshift_g = 'dpdvtshiftn_svt_ms_global' deta0_g = 'deta0n_svt_ms_global'
+dleta0_g = 'dleta0n_svt_ms_global' dneta0_g = 'dneta0n_svt_ms_global' dpeta0_g = 'dpeta0n_svt_ms_global'
+du0_g = 'du0n_svt_ms_global' dlu0_g = 'dlu0n_svt_ms_global' dnu0_g = 'dnu0n_svt_ms_global'
+dpu0_g = 'dpu0n_svt_ms_global' dvsat_g = 'dvsatn_svt_ms_global' dlvsat_g = 'dlvsatn_svt_ms_global'
+dnvsat_g = 'dnvsatn_svt_ms_global' dpvsat_g = 'dpvsatn_svt_ms_global' dpclm_g = 'dpclmn_svt_ms_global'
+dlpclm_g = 'dlpclmn_svt_ms_global' dnpclm_g = 'dnpclmn_svt_ms_global' dppclm_g = 'dppclmn_svt_ms_global'
+dpdibl2_g = 'dpdibl2n_svt_ms_global' dlpdibl2_g = 'dlpdibl2n_svt_ms_global' dnpdibl2_g = 'dnpdibl2n_svt_ms_global'
+dppdibl2_g = 'dppdibl2n_svt_ms_global' dkt1_g = 'dkt1n_svt_ms_global' dlkt1_g = 'dlkt1n_svt_ms_global'
+dnkt1_g = 'dnkt1n_svt_ms_global' dpkt1_g = 'dpkt1n_svt_ms_global' dutl_g = 'dutln_svt_ms_global'
+dlutl_g = 'dlutln_svt_ms_global' dnutl_g = 'dnutln_svt_ms_global' dputl_g = 'dputln_svt_ms_global'
+dat_g = 'datn_svt_ms_global' dlat_g = 'dlatn_svt_ms_global' dnat_g = 'dnatn_svt_ms_global'
+dpat_g = 'dpatn_svt_ms_global' ntox_g = 'ntoxn_svt_ms_global' rcoskew_g = 'rcoskewn_svt_ms_global'
+ss_flag_g = 'ss_flagn_svt_ms_global' ff_flag_g = 'ff_flagn_svt_ms_global' sf_flag_g = 'sf_flagn_svt_ms_global'
+fs_flag_g = 'fs_flagn_svt_ms_global' dcit_g = 'dcitn_svt_ms_global' dlcit_g = 'dlcitn_svt_ms_global'
+dncit_g = 'dncitn_svt_ms_global' dpcit_g = 'dpcitn_svt_ms_global' dtss_g = 'dtssn_svt_ms_global'
+dltss_g = 'dltssn_svt_ms_global' dntss_g = 'dntssn_svt_ms_global' dptss_g = 'dptssn_svt_ms_global'
+dksativ_g = 'dksativn_svt_ms_global' dlksativ_g = 'dlksativn_svt_ms_global' dnksativ_g = 'dnksativn_svt_ms_global'
+dpksativ_g = 'dpksativn_svt_ms_global' monte_flag_g = 'monte_flagn_svt_ms_global'
+ssa_flag_g = 'ssa_flagn_svt_ms_global'   ffa_flag_g = 'ffa_flagn_svt_ms_global'

**********************************
*   statistical pca parameters   *
**********************************
*
+stretch1=1.608
*
+weight1=4.84063687697
+weight2=3.63808063183
+weight3=2.36738629631
+weight4=1.66153960695
+weight5=1.21758011213

+tox_1 = -1.74188176915e-12    tox_2 = 2.20365680018e-12    tox_3 = -1.77640573833e-13    tox_4 = 2.5354399566e-13    tox_5 = 1.47905648126e-13
+dxl_1 = -3.90108778278e-11    dxl_2 = -5.38919548434e-11    dxl_3 = 6.48902298192e-12    dxl_4 = -6.37462880418e-12    dxl_5 = -2.40202018325e-10
+dtfin_1 = 3.4025905019e-11    dtfin_2 = 4.89065691494e-11    dtfin_3 = -6.90766341751e-12    dtfin_4 = 1.2295206929e-11    dtfin_5 = -1.86631162755e-10    dtfin_max = -4e-09
+dhfin_1 = 8.16622458187e-11    dhfin_2 = 1.17376910208e-10    dhfin_3 = -1.65785278883e-11    dhfin_4 = 2.9508743352e-11    dhfin_5 = -4.4791932084e-10
+cjdrw_1 = -8.99641084726e-06    cjdrw_2 = -3.57503308194e-07    cjdrw_3 = 3.64518859862e-07    cjdrw_4 = -5.85734331038e-07    cjdrw_5 = -6.1807704056e-07
+cjsw_1 = -7.22214619222e-21    cjsw_2 = -2.86996803482e-22    cjsw_3 = 2.92628753893e-22    cjsw_4 = -4.70216294073e-22    cjsw_5 = -4.96180401359e-22
+cjswg_1 = -7.22214619222e-21    cjswg_2 = -2.86996803482e-22    cjswg_3 = 2.92628753893e-22    cjswg_4 = -4.70216294073e-22    cjswg_5 = -4.96180401359e-22
+cgo_1 = 8.52749133929e-15    cgo_2 = 3.38869179737e-16    cgo_3 = -3.45519060129e-16    cgo_4 = 5.55204127497e-16    cgo_5 = 5.85861039461e-16
+cgl_1 = 9.0845354751e-13    cgl_2 = 3.6100524319e-14    cgl_3 = -3.68089515917e-14    cgl_4 = 5.91471910258e-14    cgl_5 = 6.24131433818e-14
+ddvtshift_1 = -0.00317713342916    ddvtshift_2 = 0.00141936227277    ddvtshift_3 = 0.00012020970877    ddvtshift_4 = -0.000277368807807    ddvtshift_5 = -7.84971790011e-05
+dldvtshift_1 = -2.21687479524e-06    dldvtshift_2 = 3.3074324535e-05    dldvtshift_3 = -6.28324373891e-06    dldvtshift_4 = 1.0181148317e-05    dldvtshift_5 = 3.33485358319e-06
+dndvtshift_1 = -0.00188060621526    dndvtshift_2 = 0.0011701285084    dndvtshift_3 = -0.000126129269408    dndvtshift_4 = 0.000194579051293    dndvtshift_5 = -1.21040336315e-06
+dpdvtshift_1 = -5.98465893184e-05    dpdvtshift_2 = 7.1399196675e-05    dpdvtshift_3 = 6.27694141191e-06    dpdvtshift_4 = -1.54070977821e-05    dpdvtshift_5 = 2.2247692904e-06
+du0_1 = -0.000511954461606    du0_2 = 0.000101794739982    du0_3 = -3.45026960027e-05    du0_4 = 6.54627030032e-05    du0_5 = -1.98240293445e-05
+dlu0_1 = 1.74568509371e-06    dlu0_2 = -1.74491360951e-06    dlu0_3 = -1.59685715175e-06    dlu0_4 = 3.13417783836e-06    dlu0_5 = 7.69706699783e-08
+dnu0_1 = 0.000422569154351    dnu0_2 = 0.0012408986297    dnu0_3 = -0.000196900303763    dnu0_4 = 0.000302574152019    dnu0_5 = 0.000153377939928
+dpu0_1 = '5.12307061677e-06*1.1'    dpu0_2 = '1.24706633529e-05*1.1'    dpu0_3 = '3.14538391049e-06*1.1'    dpu0_4 = '-6.76777612397e-06*1.1'    dpu0_5 = '1.19166566675e-06*1.1'
+dvsat_1 = -533.087120415    dvsat_2 = 1801.04548957    dvsat_3 = -44.967369831    dvsat_4 = -9.75041621898    dvsat_5 = 132.627929124
+dlvsat_1 = 7.46164078469    dlvsat_2 = -9.91257827344    dlvsat_3 = 3.01762815419    dlvsat_4 = -5.29000169517    dlvsat_5 = -0.667962853534
+dnvsat_1 = 217.446990345    dnvsat_2 = 754.941128914    dnvsat_3 = 239.749095854    dnvsat_4 = -503.114330611    dnvsat_5 = 62.5100078388
+dpvsat_1 = '-18.2166397582*1.1'    dpvsat_2 = '-6.72156552396*1.1'    dpvsat_3 = '-11.2122471245*1.1'    dpvsat_4 = '22.0051311566*1.1'    dpvsat_5 = '-0.849113228092*1.1'
+dpclm_1 = -0.000154356805081    dpclm_2 = 0.00625600233001    dpclm_3 = -0.00065444261331    dpclm_4 = 0.000901770718579    dpclm_5 = 0.000605091166411
+dlpclm_1 = 1.73258766614e-06    dlpclm_2 = -7.02208915938e-05    dlpclm_3 = 7.34583227106e-06    dlpclm_4 = -1.01219821431e-05    dlpclm_5 = -6.79188385163e-06
+dpdibl2_1 = 1.68547434499e-06    dpdibl2_2 = -6.83114127939e-05    dpdibl2_3 = 7.14608101938e-06    dpdibl2_4 = -9.84674054655e-06    dpdibl2_5 = -6.6071958203e-06
+dlpdibl2_1 = 1.46702281143e-08    dlpdibl2_2 = 8.40879455987e-08    dlpdibl2_3 = 2.92635949301e-06    dlpdibl2_4 = 1.52942160027e-06    dlpdibl2_5 = 1.71726193846e-08     dlpdibl2_max = -3.4949e-5
+dkt1_1 = 0.000212861324919    dkt1_2 = 1.47162055099e-05    dkt1_3 = 5.21999005371e-05    dkt1_4 = -0.000102815960938    dkt1_5 = 1.03916271467e-05
+dlkt1_1 = 4.78366621774e-06    dlkt1_2 = -3.5031891144e-07    dlkt1_3 = -5.44686546472e-06    dlkt1_4 = 1.0387916427e-05    dlkt1_5 = 6.9418491701e-07
+dnkt1_1 = 0.0    dnkt1_2 = 0.0    dnkt1_3 = 0.0    dnkt1_4 = 0.0    dnkt1_5 = 0.0
+dutl_1 = -7.8403697084e-07    dutl_2 = -3.52342567638e-08    dutl_3 = -7.87052224952e-09    dutl_4 = 2.49881413321e-08    dutl_5 = -5.11071516608e-08
+dlutl_1 = 1.71893658495e-08    dlutl_2 = 1.80535584099e-09    dlutl_3 = 1.0212493315e-08    dlutl_4 = -1.98066131432e-08    dlutl_5 = 4.2184984792e-10
+dnutl_1 = 2.86363293149e-07    dnutl_2 = 1.79424214044e-08    dnutl_3 = 5.2189973003e-08    dnutl_4 = -1.03724159638e-07    dnutl_5 = 1.5234859079e-08
+dputl_1 = -4.32584670256e-08    dputl_2 = -2.66688217311e-09    dputl_3 = -7.46079753616e-09    dputl_4 = 1.48571337229e-08    dputl_5 = -2.33084197876e-09
+dat_1 = 1.31207086481e-05    dat_2 = 3.44845939892e-07    dat_3 = -2.2477686157e-06    dat_4 = 4.14618491098e-06    dat_5 = 1.02084531995e-06
+dlat_1 = -6.77000159106e-07    dlat_2 = -1.87748692737e-08    dlat_3 = 1.06438792729e-07    dlat_4 = -1.95632081737e-07    dlat_5 = -5.20094806907e-08
+dnat_1 = -7.55003411717e-06    dnat_2 = -2.40557304644e-08    dnat_3 = 2.9884590814e-06    dnat_4 = -5.63726421515e-06    dnat_5 = -7.05372929463e-07
+dpat_1 = 4.57676014203e-07    dpat_2 = -2.76331387117e-08    dpat_3 = -4.63937190398e-07    dpat_4 = 8.84157969228e-07    dpat_5 = 6.24363588465e-08
*+ntox_1 = 0.0049413998703    ntox_2 = 0.000245974682155    ntox_3 = 0.00028202320489    ntox_4 = -0.000603318099522    ntox_5 = 0.000305930306334
+ntox_1 = 0.044834    ntox_2 = -0.0023341    ntox_3 = -0.0044107    ntox_4 = -0.0026517    ntox_5 = 0
+rcoskew_1 = -0.0998025009214    rcoskew_2 = -0.140725512434    rcoskew_3 = 0.0184728586576    rcoskew_4 = -0.0264112506002    rcoskew_5 = -0.0203029473817
+ss_flag_1 = -0.0528191357924    ss_flag_2 = -0.00778752182431    ss_flag_3 = -0.0531550151745    ss_flag_4 = 0.102629113075    ss_flag_5 = 0.000218918204763
+ff_flag_1 = 0.0503543812393    ff_flag_2 = -0.00368756748885    ff_flag_3 = -0.0573354259444    ff_flag_4 = 0.109346488705    ff_flag_5 = 0.00730720965274
+monte_flag_1 = -0.00337101611029    monte_flag_2 = 0.136625558099    monte_flag_3 = -0.0142924478877    monte_flag_4 = 0.0196938749706    monte_flag_5 = 0.0132146559337
+dcit_1 = 0    dcit_2 = 0    dcit_3 = 0    dcit_4 = 0    dcit_5 = 0
+dlcit_1 = 0    dlcit_2 = 0    dlcit_3 = 0    dlcit_4 = 0    dlcit_5 = 0
+dncit_1 = 0    dncit_2 = 0    dncit_3 = 0    dncit_4 = 0    dncit_5 = 0
+dpcit_1 = 0    dpcit_2 = 0    dpcit_3 = 0    dpcit_4 = 0    dpcit_5 = 0
+dnpclm_1 = 6.33282457496e-05    dnpclm_2 = -0.002566661397    dnpclm_3 = 0.000268499355263    dnpclm_4 = -0.000369971104585    dnpclm_5 = -0.000248251847836
+dppclm_1 = -2.52510344063e-06    dppclm_2 = 0.000102341150426    dppclm_3 = -1.07059438921e-05    dppclm_4 = 1.47519530671e-05    dppclm_5 = 9.89860981768e-06
+dnpdibl2_1 = 4.6820404336e-06    dnpdibl2_2 = 2.68368806677e-05    dnpdibl2_3 = 0.000933955038924    dnpdibl2_4 = 0.000488118774753    dnpdibl2_5 = 5.48068494119e-06 dnpdibl2_max = -3.4949e-3
+dppdibl2_1 = 9.24308172847e-10    dppdibl2_2 = 5.29802091347e-09    dppdibl2_3 = 1.84377364483e-07    dppdibl2_4 = 9.63622974262e-08    dppdibl2_5 = 1.08197311744e-09
+deta0_1 = 8.24395203247e-06    deta0_2 = 4.72533204409e-05    deta0_3 = 0.00164447117674    deta0_4 = 0.000859460276407    deta0_5 = 9.65017376526e-06 deta0_max = -9.0949e-3
+dleta0_1 = 0    dleta0_2 = 0    dleta0_3 = 0    dleta0_4 = 0    dleta0_5 = 0
+dneta0_1 = 6.56890003947e-07    dneta0_2 = 3.76521281646e-06    dneta0_3 = 0.000131033838325    dneta0_4 = 6.84830360654e-05    dneta0_5 = 7.68939782496e-07
+dpeta0_1 = 0    dpeta0_2 = 0    dpeta0_3 = 0    dpeta0_4 = 0    dpeta0_5 = 0
+dtss_1 = 0    dtss_2 = 0    dtss_3 = 0    dtss_4 = 0    dtss_5 = 0
+dltss_1 = 0    dltss_2 = 0    dltss_3 = 0    dltss_4 = 0    dltss_5 = 0
+dntss_1 = 0    dntss_2 = 0    dntss_3 = 0    dntss_4 = 0    dntss_5 = 0
+dptss_1 = 0    dptss_2 = 0    dptss_3 = 0    dptss_4 = 0    dptss_5 = 0
+dksativ_1 = 0    dksativ_2 = 0    dksativ_3 = 0    dksativ_4 = 0    dksativ_5 = 0
+dlksativ_1 = 0    dlksativ_2 = 0    dlksativ_3 = 0    dlksativ_4 = 0    dlksativ_5 = 0
+dnksativ_1 = 0    dnksativ_2 = 0    dnksativ_3 = 0    dnksativ_4 = 0    dnksativ_5 = 0
+dpksativ_1 = 0    dpksativ_2 = 0    dpksativ_3 = 0    dpksativ_4 = 0    dpksativ_5 = 0
+dpkt1_1 = 2.37686111066e-05    dpkt1_2 = 3.50438482094e-06    dpkt1_3 = 2.39197568285e-05    dpkt1_4 = -4.61831008839e-05    dpkt1_5 = -9.85131921433e-08 )

.model nch_svt_mac.5 bsimcmg ( level = 72
+lmin_flag = 0 lmax_flag = 0 nfinmin = 12
+nfinmax = 20.888 lmin = 1.1e-08 lmax = 2e-08
+version = 106.1 agidl = 7.558000e-06
+lagidl = 0.000000e+00
+nagidl = 0.000000e+00 pagidl = 0.000000e+00
+agisl = 7.558000e-06
+lagisl = 0.000000e+00 nagisl = 0.000000e+00
+pagisl = 0.000000e+00
+aigc = 2.2679722e-02 laigc = -2.7016367e-05
+naigc = 3.7200667e-03
+paigc = 5.3422450e-06 aigd = 2.5765107e-02
+laigd = -3.5525375e-05
+naigd = -2.7198698e-03 paigd = -7.4772767e-05
+aigs = 2.5765107e-02
+laigs = -3.5525375e-05 naigs = -2.7198698e-03
+paigs = -7.4772767e-05
+at = 1.5883342e-04 lat = 2.5123609e-05
+nat = 2.7666181e-02
+pat = -6.3510945e-04 cit = -0.00013004753
+lcit = 5.9851311e-05
+ncit = -0.0047995507 pcit = 6.2984264e-05
+dvtshift = 5.6730479e-03
+ldvtshift = 8.9209524e-04 ndvtshift = -3.0292612e-01
+pdvtshift = 2.2546155e-03
+eta0 = 2.5996914e-02 leta0 = -1.5743829e-04
+neta0 = -9.1962854e-02
+peta0 = 2.5892571e-03 eu = 0.955
+leu = 0.002625
+neu = 4.5 peu = -0.1575
+ksativ = 2.5361422e+00 pksativ = 2.2376843e-01
+nksativ = -1.2033706e+01 lksativ = -9.3140318e-03
+kt1 = 0.059227553 lkt1 = -0.0012338213 nkt1 = -0.42302427
+pkt1 = 0.014805849 mexp = 2.1
+pdibl2 = 1.9248425e-03 ppdibl2 = -6.5803780e-04
+npdibl2 = 3.2901890e-02 lpdibl2 = 3.1503150e-05
+psat = 6.4652400e+00 ppsat = -2.5927084e-01
+npsat = 1.3217132e+01 lpsat = 3.3272549e-02
+ptwg = 8.8798890e+01
+lptwg = -1.1958448e+00 nptwg = -8.8958712e+02
+pptwg = 3.0030154e+01
+tmik0si = 1.5137601 bltmik0si = -0.017981605 bntmik0si = -14.865121
+bptmik0si = 0.31027922 tmik1si = 1.5137601 bltmik1si = -0.017981605
+bntmik1si = -14.865121 bptmik1si = 0.31027922 tmipclm = 0.07
+bltmipclm = 4.666675e-13 bntmipclm = 2.66e-08 bptmipclm = -5.32e-10
+u0 = 8.6654318e-03 lu0 = 8.8911784e-04
+nu0 = 6.9862596e-01
+pu0 = -1.6807405e-02 ua = -1.6770528e+00
+lua = 8.0519393e-02
+nua = 3.8204633e+01 pua = -1.1370326e+00
+ua1 = 1.6133963e-02
+lua1 = -1.7267927e-04 nua1 = -1.3160753e-01
+pua1 = 2.6321506e-03
+ute = 5.8536800e-01 lute = -1.7512411e-02
+nute = -1.1864413e+01
+pute = 3.1654887e-01 utl = 2.9093929e-05
+lutl = -2.1462084e-06
+nutl = -1.3306657e-03 putl = 3.3741594e-05
+vsat = 1.4957767e+04
+lvsat = 7.6088071e+02 nvsat = 6.2819468e+05
+pvsat = -7.6873351e+03 vsat_ffa = 3.2648483e+04
+pvsat_ffa = 1.2301969e+04 nvsat_ffa = -3.9604795e+05
+lvsat_ffa = -1.2821937e+03 vsat_mcl = -1.9464146e+03
+pvsat_mcl = 1.0528605e+03 nvsat_mcl = -5.2643023e+04
+lvsat_mcl = 2.8928296e+01 dvtshift_ss = -1.6518737e-03
+pdvtshift_ss = -1.4544188e-03 ndvtshift_ss = 5.4325371e-02
+ldvtshift_ss = -2.7555033e-05 pdibl2_ffa = -1.3832056e-02
+ppdibl2_ffa = -5.6664531e-03 npdibl2_ffa = 2.0801381e-01
+lpdibl2_ffa = 3.3679525e-04 u0_ffa = 1.3070780e-03
+pu0_ffa = 3.9187335e-04 nu0_ffa = -2.5010885e-03
+lu0_ffa = -1.0535909e-04 cit_ffa = 4.1449462e-04
+pcit_ffa = 1.9381509e-04 ncit_ffa = -5.4915039e-03
+lcit_ffa = -1.2310623e-05 cit_ssa = -4.1854809e-05
+pcit_ssa = -6.3830000e-10 ncit_ssa = -1.8727386e-03
+lcit_ssa = 9.1150119e-07 utl_ss = -3.7334234e-05
+putl_ss = -6.5803790e-06 nutl_ss = 1.8801082e-04
+lutl_ss = 7.8169816e-07 u0_ss = -2.6461035e-03
+pu0_ss = -3.7073270e-04 nu0_ss = 1.6231132e-02
+lu0_ss = 4.8014384e-05 eta0_ffa = 8.4861422e-02
+peta0_ffa = 2.3903177e-02 neta0_ffa = -1.0148570e+00
+leta0_ffa = -1.8946539e-03 u0_ff = 4.1936637e-03
+pu0_ff = 1.1592952e-03 nu0_ff = -6.3415663e-02
+lu0_ff = -7.0294324e-05 utl_ff = 3.7334234e-05
+putl_ff = 6.5803790e-06 nutl_ff = -1.8801082e-04
+lutl_ff = -7.8169816e-07 pdibl2_ssa = 3.1291934e-04
+ppdibl2_ssa = 1.3983270e-04 npdibl2_ssa = -8.8644011e-03
+lpdibl2_ssa = -1.1652737e-05 tmipclm_ssa = -3.6361832e-04
+bltmipclm_ssa = -3.4528735e-04 bntmipclm_ssa = -3.1255010e-01
+bptmipclm_ssa = 8.4497578e-03 eta0_ssa = 2.3085918e-03
+peta0_ssa = -3.8716957e-03 neta0_ssa = 1.6749936e-01
+leta0_ssa = 3.9734867e-05 vsat_ff = 5.2028362e+03
+pvsat_ff = 3.7889271e+02 nvsat_ff = -3.9837219e+04
+lvsat_ff = -6.4472392e+01 dvtshift_mcl = 2.3333335e-03
+pdvtshift_mcl = -2.4936077e-11 ndvtshift_mcl = -2.9611208e-10
+ldvtshift_mcl = -4.6666669e-05 vsat_ss = -1.6652638e+04
+pvsat_ss = -5.8446327e+03 nvsat_ss = 1.9468395e+05
+lvsat_ss = 4.9083402e+02 tmipclm_ffa = -1.3678593e-01
+bltmipclm_ffa = 3.9608837e-03 bntmipclm_ffa = 2.2223133e+00
+bptmipclm_ffa = -6.2685735e-02 dvtshift_ssa = 4.5993026e-03
+pdvtshift_ssa = 5.3644564e-04 ndvtshift_ssa = -8.8056769e-03
+ldvtshift_ssa = -3.0519351e-06 dvtshift_ffa = 8.5289489e-03
+pdvtshift_ffa = 1.1234806e-03 ndvtshift_ffa = 1.0319723e-02
+ldvtshift_ffa = -5.9501199e-04 dvtshift_ff = -4.1829604e-03
+pdvtshift_ff = -9.9155557e-04 ndvtshift_ff = -1.2246930e-02
+ldvtshift_ff = 2.5008235e-04 vsat_ssa = 1.0314538e+04
+pvsat_ssa = -6.6527585e+01 nvsat_ssa = -8.3024447e+04
+lvsat_ssa = 1.0325231e+02 ptwgt = 6.6601260e-03
+lptwgt = -5.3202520e-05 nptwgt = -2.6321512e-02
+pptwgt = 5.2643024e-04 vsat_mc = -1.6450945e+04
+pvsat_mc = -3.9482268e+03 nvsat_mc = 1.9741134e+05
+lvsat_mc = 3.2901890e+02 u0_ssa = -8.3118585e-04
+pu0_ssa = 1.9832602e-04 nu0_ssa = -5.6664580e-03
+lu0_ssa = 2.9091506e-05 )

