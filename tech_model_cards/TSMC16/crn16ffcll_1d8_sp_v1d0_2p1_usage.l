**************************************************************
*  TSMC SPICE MODEL --- Security B - TSMC Restricted Secret  *
**************************************************************
.LIB pre_simu
.param ccoflag_cap=1
.param ccoflag_cap_18=1
.param ccosflag_mos_svt=1
.param ccosflag_mos_hvt=1
.param ccosflag_mos_lvt=1
.param ccosflag_mos_ulvt=1
.param ccosflag_mos_lnvt=1
.param ccosflag_mos_18=1
.param ccosflag_mos_18ud15=1
.param ccosflag_mos_18ud12=1
.param ccosflag_mos_hia18=1
.param ccosflag_mos_mbscr=1
.param ccodflag_mos_svt=1
.param ccodflag_mos_hvt=1
.param ccodflag_mos_lvt=1
.param ccodflag_mos_ulvt=1
.param ccodflag_mos_lnvt=1
.param ccodflag_mos_18=1
.param ccodflag_mos_18ud15=1
.param ccodflag_mos_18ud12=1
.param ccodflag_mos_hia18=1
.param ccodflag_mos_mbscr=1
.param rgflag_mos_svt=1
.param rgflag_mos_hvt=1
.param rgflag_mos_lvt=1
.param rgflag_mos_ulvt=1
.param rgflag_mos_lnvt=1
.param rgflag_mos_18=1
.param rgflag_mos_18ud15=1
.param rgflag_mos_18ud12=1
.param rgflag_mos_hia18=1
.param rgflag_mos_mbscr=1
.param rgflag_cap=1
.param rgflag_cap_18=1
.param rcoflag_res=1
.param rcoflag_disres=1
.param rcosflag_mos_svt=1
.param rcosflag_mos_hvt=1
.param rcosflag_mos_lvt=1
.param rcosflag_mos_ulvt=1
.param rcosflag_mos_lnvt=1
.param rcosflag_mos_18=1
.param rcosflag_mos_18ud15=1
.param rcosflag_mos_18ud12=1
.param rcosflag_mos_hia18=1
.param rcosflag_mos_mbscr=1
.param rcodflag_mos_svt=1
.param rcodflag_mos_hvt=1
.param rcodflag_mos_lvt=1
.param rcodflag_mos_ulvt=1
.param rcodflag_mos_lnvt=1
.param rcodflag_mos_18=1
.param rcodflag_mos_18ud15=1
.param rcodflag_mos_18ud12=1
.param rcodflag_mos_hia18=1
.param rcodflag_mos_mbscr=1
.param capflag_mim=1
.param ccosflag_mos_mpodesvt=1
.param ccosflag_mos_mpodehvt=1
.param ccosflag_mos_mpodelvt=1
.param ccosflag_mos_mpodeulvt=1
.param ccosflag_mos_mpodelnvt=1
.param ccodflag_mos_mpodesvt=1
.param ccodflag_mos_mpodehvt=1
.param ccodflag_mos_mpodelvt=1
.param ccodflag_mos_mpodeulvt=1
.param ccodflag_mos_mpodelnvt=1
.param rgflag_mos_mpodesvt=1
.param rgflag_mos_mpodehvt=1
.param rgflag_mos_mpodelvt=1
.param rgflag_mos_mpodeulvt=1
.param rgflag_mos_mpodelnvt=1
.param rcosflag_mos_mpodesvt=1
.param rcosflag_mos_mpodehvt=1
.param rcosflag_mos_mpodelvt=1
.param rcosflag_mos_mpodeulvt=1
.param rcosflag_mos_mpodelnvt=1
.param rcodflag_mos_mpodesvt=1
.param rcodflag_mos_mpodehvt=1
.param rcodflag_mos_mpodelvt=1
.param rcodflag_mos_mpodeulvt=1
.param rcodflag_mos_mpodelnvt=1
.param rcmflag_mos_svt_rf=1
.param rcmflag_mos_lvt_rf=1
.param rcmflag_mos_ulvt_rf=1
.param rcmflag_mos_18_rf=1
.param rcmflag_mos_18ud15_rf=1
.param rcmflag_mos_18ud12_rf=1
.ENDL pre_simu

.LIB tmiAge_ON
.option tmiAge=1
.param tmiageflag=1
.ENDL tmiAge_ON

.lib hci_only
.param phys_id=1
.endl hci_only

.lib bti_only
.param phys_id=2
.endl bti_only

.lib age_unit_day
.param age_unit=1
.endl age_unit_day

.lib age_unit_minute
.param age_unit=2
.endl age_unit_minute

.lib she
.param tmishe=1
.param tmiageflag=1
.param sheimod=0
.option tmiAge=1
.endl she

.lib she_2
.param tmishe=1
.param tmiageflag=1
.param sheimod=2
.option tmiAge=1
.endl she_2

.lib age_she
.option tmiAge=1
.param tmishe=2
.param tmiageflag=1
.param sheimod=0
.param phys_id=0
.param age_unit=0
.endl age_she

.lib age_she_2
.option tmiAge=1
.param tmishe=2
.param tmiageflag=1
.param sheimod=2
.param phys_id=0
.param age_unit=0
.endl age_she_2

.LIB fnoise_mos_typ
.param noiseflagn_mos_svt=0
.param noiseflagp_mos_svt=0
.param noiseflagn_mos_lvt=0
.param noiseflagp_mos_lvt=0
.param noiseflagn_mos_hvt=0
.param noiseflagp_mos_hvt=0
.param noiseflagn_mos_ulvt=0
.param noiseflagp_mos_ulvt=0
.param noiseflagp_mos_lnvt=0
.param noiseflagn_mos_lnvt=0
.param noiseflagn_mos_18=0
.param noiseflagp_mos_18=0
.param noiseflagn_mos_18ud15=0
.param noiseflagp_mos_18ud15=0
.param noiseflagn_mos_18ud12=0
.param noiseflagp_mos_18ud12=0
.param noiseflagn_mos_mpodesvt=0
.param noiseflagp_mos_mpodesvt=0
.param noiseflagn_mos_mpodelvt=0
.param noiseflagp_mos_mpodelvt=0
.param noiseflagn_mos_mpodeulvt=0
.param noiseflagp_mos_mpodeulvt=0
.param noiseflagn_mos_mpodehvt=0
.param noiseflagp_mos_mpodehvt=0
.param noiseflagn_mos_mpodelnvt=0
.param noiseflagp_mos_mpodelnvt=0
.param noiseflagn_mos_svt_rf=0
.param noiseflagp_mos_svt_rf=0
.param noiseflagn_mos_lvt_rf=0
.param noiseflagp_mos_lvt_rf=0
.param noiseflagn_mos_ulvt_rf=0
.param noiseflagp_mos_ulvt_rf=0
.param noiseflagn_mos_18_rf=0
.param noiseflagp_mos_18_rf=0
.param noiseflagn_mos_18ud15_rf=0
.param noiseflagp_mos_18ud15_rf=0
.param noiseflagn_mos_18ud12_rf=0
.param noiseflagp_mos_18ud12_rf=0
.ENDL fnoise_mos_typ

.LIB fnoise_mos_wc
.param noiseflagn_mos_svt=1
.param noiseflagp_mos_svt=1
.param noiseflagn_mos_lvt=1
.param noiseflagp_mos_lvt=1
.param noiseflagn_mos_hvt=1
.param noiseflagp_mos_hvt=1
.param noiseflagn_mos_ulvt=1
.param noiseflagp_mos_ulvt=1
.param noiseflagp_mos_lnvt=1
.param noiseflagn_mos_lnvt=1
.param noiseflagn_mos_18=1
.param noiseflagp_mos_18=1
.param noiseflagn_mos_18ud15=1
.param noiseflagp_mos_18ud15=1
.param noiseflagn_mos_18ud12=1
.param noiseflagp_mos_18ud12=1
.param noiseflagn_mos_mpodesvt=1
.param noiseflagp_mos_mpodesvt=1
.param noiseflagn_mos_mpodelvt=1
.param noiseflagp_mos_mpodelvt=1
.param noiseflagn_mos_mpodeulvt=1
.param noiseflagp_mos_mpodeulvt=1
.param noiseflagn_mos_mpodehvt=1
.param noiseflagp_mos_mpodehvt=1
.param noiseflagn_mos_mpodelnvt=1
.param noiseflagp_mos_mpodelnvt=1
.param noiseflagn_mos_svt_rf=1
.param noiseflagp_mos_svt_rf=1
.param noiseflagn_mos_lvt_rf=1
.param noiseflagp_mos_lvt_rf=1
.param noiseflagn_mos_ulvt_rf=1
.param noiseflagp_mos_ulvt_rf=1
.param noiseflagn_mos_18_rf=1
.param noiseflagp_mos_18_rf=1
.param noiseflagn_mos_18ud15_rf=1
.param noiseflagp_mos_18ud15_rf=1
.param noiseflagn_mos_18ud12_rf=1
.param noiseflagp_mos_18ud12_rf=1
.ENDL fnoise_mos_wc

.LIB fnoise_mos_bc
.param noiseflagn_mos_svt=-1
.param noiseflagp_mos_svt=-1
.param noiseflagn_mos_lvt=-1
.param noiseflagp_mos_lvt=-1
.param noiseflagn_mos_hvt=-1
.param noiseflagp_mos_hvt=-1
.param noiseflagn_mos_ulvt=-1
.param noiseflagp_mos_ulvt=-1
.param noiseflagp_mos_lnvt=-1
.param noiseflagn_mos_lnvt=-1
.param noiseflagn_mos_18=-1
.param noiseflagp_mos_18=-1
.param noiseflagn_mos_18ud15=-1
.param noiseflagp_mos_18ud15=-1
.param noiseflagn_mos_18ud12=-1
.param noiseflagp_mos_18ud12=-1
.param noiseflagn_mos_mpodesvt=-1
.param noiseflagp_mos_mpodesvt=-1
.param noiseflagn_mos_mpodelvt=-1
.param noiseflagp_mos_mpodelvt=-1
.param noiseflagn_mos_mpodeulvt=-1
.param noiseflagp_mos_mpodeulvt=-1
.param noiseflagn_mos_mpodehvt=-1
.param noiseflagp_mos_mpodehvt=-1
.param noiseflagn_mos_mpodelnvt=-1
.param noiseflagp_mos_mpodelnvt=-1
.param noiseflagn_mos_svt_rf=-1
.param noiseflagp_mos_svt_rf=-1
.param noiseflagn_mos_lvt_rf=-1
.param noiseflagp_mos_lvt_rf=-1
.param noiseflagn_mos_ulvt_rf=-1
.param noiseflagp_mos_ulvt_rf=-1
.param noiseflagn_mos_18_rf=-1
.param noiseflagp_mos_18_rf=-1
.param noiseflagn_mos_18ud15_rf=-1
.param noiseflagp_mos_18ud15_rf=-1
.param noiseflagn_mos_18ud12_rf=-1
.param noiseflagp_mos_18ud12_rf=-1
.ENDL fnoise_mos_bc

.LIB fnoise_mos_stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_usage.l' fnoise_mos_typ
.param noiseflagn_mc_mos_svt=1
.param noiseflagp_mc_mos_svt=1
.param noiseflagn_mc_mos_lvt=1
.param noiseflagp_mc_mos_lvt=1
.param noiseflagn_mc_mos_hvt=1
.param noiseflagp_mc_mos_hvt=1
.param noiseflagn_mc_mos_ulvt=1
.param noiseflagp_mc_mos_ulvt=1
.param noiseflagn_mc_mos_lnvt=1
.param noiseflagp_mc_mos_lnvt=1
.param noiseflagn_mc_mos_18=1
.param noiseflagp_mc_mos_18=1
.param noiseflagn_mc_mos_18ud15=1
.param noiseflagp_mc_mos_18ud15=1
.param noiseflagn_mc_mos_18ud12=1
.param noiseflagp_mc_mos_18ud12=1
.param noiseflagn_mc_mos_mpodesvt=1
.param noiseflagp_mc_mos_mpodesvt=1
.param noiseflagn_mc_mos_mpodelvt=1
.param noiseflagp_mc_mos_mpodelvt=1
.param noiseflagn_mc_mos_mpodehvt=1
.param noiseflagp_mc_mos_mpodehvt=1
.param noiseflagn_mc_mos_mpodeulvt=1
.param noiseflagp_mc_mos_mpodeulvt=1
.param noiseflagn_mc_mos_mpodelnvt=1
.param noiseflagp_mc_mos_mpodelnvt=1
.param noiseflagn_mc_mos_svt_rf=1
.param noiseflagp_mc_mos_svt_rf=1
.param noiseflagn_mc_mos_lvt_rf=1
.param noiseflagp_mc_mos_lvt_rf=1
.param noiseflagn_mc_mos_ulvt_rf=1
.param noiseflagp_mc_mos_ulvt_rf=1
.param noiseflagn_mc_mos_18_rf=1
.param noiseflagp_mc_mos_18_rf=1
.param noiseflagn_mc_mos_18ud15_rf=1
.param noiseflagp_mc_mos_18ud15_rf=1
.param noiseflagn_mc_mos_18ud12_rf=1
.param noiseflagp_mc_mos_18ud12_rf=1
.ENDL fnoise_mos_stat

.lib fnoise_res_typ
.param rnoiseflag_res=0
.param rnoiseflag_disres=0
.param rnoiseflag_disres_rf=0
.endl fnoise_res_typ

.lib fnoise_res_wc
.param rnoiseflag_res=1
.param rnoiseflag_disres=1
.param rnoiseflag_disres_rf=1
.endl fnoise_res_wc

.lib fnoise_res_bc
.param rnoiseflag_res=-1
.param rnoiseflag_disres=-1
.param rnoiseflag_disres_rf=-1
.endl fnoise_res_bc

.LIB TTMacro_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param iboff_flag_mos_svt=0
.param iboff_flag_mos_hvt=0
.param iboff_flag_mos_lvt=0
.param iboff_flag_mos_ulvt=0
.param iboff_flag_mos_lnvt=0
.param iboff_flag_mos_18=0
.param iboff_flag_mos_18ud15=0
.param iboff_flag_mos_18ud12=0
.param iboff_flag_mos_hia18=0
.param iboff_flag_mos_mpodesvt=0
.param iboff_flag_mos_mpodehvt=0
.param iboff_flag_mos_mpodelvt=0
.param iboff_flag_mos_mpodeulvt=0
.param iboff_flag_mos_mpodelnvt=0
.param iboff_flag_mos_svt_rf=0
.param iboff_flag_mos_lvt_rf=0
.param iboff_flag_mos_ulvt_rf=0
.param iboff_flag_mos_18_rf=0
.param iboff_flag_mos_18ud15_rf=0
.param iboff_flag_mos_18ud12_rf=0
.ENDL TTMacro_MOS_MOSCAP

.LIB SSMacro_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=1
.param rcskewflag_mos_lvt_rf=1
.param rcskewflag_mos_ulvt_rf=1
.param rcskewflag_mos_18_rf=1
.param rcskewflag_mos_18ud15_rf=1
.param rcskewflag_mos_18ud12_rf=1
.ENDL SSMacro_MOS_MOSCAP

.LIB FFMacro_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=-1
.param rcskewflag_mos_lvt_rf=-1
.param rcskewflag_mos_ulvt_rf=-1
.param rcskewflag_mos_18_rf=-1
.param rcskewflag_mos_18ud15_rf=-1
.param rcskewflag_mos_18ud12_rf=-1
.ENDL FFMacro_MOS_MOSCAP

.LIB SFMacro_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL SFMacro_MOS_MOSCAP

.LIB FSMacro_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL FSMacro_MOS_MOSCAP

.LIB SSGNPGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_usage.l' SSGlobalCorner_LocalMC_MOS_MOSCAP
.param global_factorn_svt=0.833
.param global_factorp_svt=0.833
.param global_factorn_hvt=0.833
.param global_factorp_hvt=0.833
.param global_factorn_lvt=0.833
.param global_factorp_lvt=0.833
.param global_factorn_ulvt=0.833
.param global_factorp_ulvt=0.833
.param global_factorn_lnvt=0.833
.param global_factorp_lnvt=0.833
.param global_factorn_mpodesvt=0.833
.param global_factorp_mpodesvt=0.833
.param global_factorn_mpodehvt=0.833
.param global_factorp_mpodehvt=0.833
.param global_factorn_mpodelvt=0.833
.param global_factorp_mpodelvt=0.833
.param global_factorn_mpodeulvt=0.833
.param global_factorp_mpodeulvt=0.833
.param global_factorn_mpodelnvt=0.833
.param global_factorp_mpodelnvt=0.833
.ENDL SSGNPGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFGNPGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_usage.l' FFGlobalCorner_LocalMC_MOS_MOSCAP
.param global_factorn_svt=0.866
.param global_factorp_svt=0.866
.param global_factorn_hvt=0.857
.param global_factorp_hvt=0.857
.param global_factorn_lvt=0.845
.param global_factorp_lvt=0.845
.param global_factorn_ulvt=0.847
.param global_factorp_ulvt=0.847
.param global_factorn_lnvt=0.88
.param global_factorp_lnvt=0.88
.param global_factorn_mpodesvt=0.866
.param global_factorp_mpodesvt=0.866
.param global_factorn_mpodehvt=0.845
.param global_factorp_mpodehvt=0.845
.param global_factorn_mpodelvt=0.845
.param global_factorp_mpodelvt=0.845
.param global_factorn_mpodeulvt=0.847
.param global_factorp_mpodeulvt=0.847
.param global_factorn_mpodelnvt=0.88
.param global_factorp_mpodelnvt=0.88
.ENDL FFGNPGlobalCorner_LocalMC_MOS_MOSCAP

.LIB TTGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ttg_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ttg_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ttg_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ttg_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ttg_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ttg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=0
.param iboff_flag_mos_hvt=0
.param iboff_flag_mos_lvt=0
.param iboff_flag_mos_ulvt=0
.param iboff_flag_mos_lnvt=0
.param iboff_flag_mos_18=0
.param iboff_flag_mos_18ud15=0
.param iboff_flag_mos_18ud12=0
.param iboff_flag_mos_hia18=0
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=0
.param iboff_flag_mos_mpodehvt=0
.param iboff_flag_mos_mpodelvt=0
.param iboff_flag_mos_mpodeulvt=0
.param iboff_flag_mos_mpodelnvt=0
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=0
.param iboff_flag_mos_lvt_rf=0
.param iboff_flag_mos_ulvt_rf=0
.param iboff_flag_mos_18_rf=0
.param iboff_flag_mos_18ud15_rf=0
.param iboff_flag_mos_18ud12_rf=0
.ENDL TTGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssg_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssg_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssg_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssg_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssg_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=1
.param rcskewflag_mos_lvt_rf=1
.param rcskewflag_mos_ulvt_rf=1
.param rcskewflag_mos_18_rf=1
.param rcskewflag_mos_18ud15_rf=1
.param rcskewflag_mos_18ud12_rf=1
.ENDL SSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffg_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffg_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffg_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffg_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffg_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=-1
.param rcskewflag_mos_lvt_rf=-1
.param rcskewflag_mos_ulvt_rf=-1
.param rcskewflag_mos_18_rf=-1
.param rcskewflag_mos_18ud15_rf=-1
.param rcskewflag_mos_18ud12_rf=-1
.ENDL FFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' sfg_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' sfg_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' sfg_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' sfg_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' sfg_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SF_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' sfg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL SFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' fsg_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' fsg_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' fsg_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' fsg_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' fsg_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FS_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' fsg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL FSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssag_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssag_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssag_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssag_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ssag_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' SS_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssag_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ssg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ss_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=2
.param rcskewflag_mos_lvt_rf=2
.param rcskewflag_mos_ulvt_rf=2
.param rcskewflag_mos_18_rf=2
.param rcskewflag_mos_18ud15_rf=2
.param rcskewflag_mos_18ud12_rf=2
.ENDL SSAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffag_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffag_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffag_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffag_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' ffag_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' FF_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffag_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ffg_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' ff_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.param rcskewflag_mos_svt_rf=-2
.param rcskewflag_mos_lvt_rf=-2
.param rcskewflag_mos_ulvt_rf=-2
.param rcskewflag_mos_18_rf=-2
.param rcskewflag_mos_18ud15_rf=-2
.param rcskewflag_mos_18ud12_rf=-2
.ENDL FFAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB LocalMCOnly_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MOS_CAP_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMVAR_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param globalflag_mos_svt=0
.param globalflag_mos_hvt=0
.param globalflag_mos_lvt=0
.param globalflag_mos_ulvt=0
.param globalflag_mos_lnvt=0
.param globalflag_mos_18=0
.param globalflag_mos_18ud15=0
.param globalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param globalflag_mos_mpodesvt=0
.param globalflag_mos_mpodehvt=0
.param globalflag_mos_mpodelvt=0
.param globalflag_mos_mpodeulvt=0
.param globalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param globalflag_mos_svt_rf=0
.param globalflag_mos_lvt_rf=0
.param globalflag_mos_ulvt_rf=0
.param globalflag_mos_18_rf=0
.param globalflag_mos_18ud15_rf=0
.param globalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL LocalMCOnly_MOS_MOSCAP

.LIB GlobalMC_LocalMC_MOS_MOSCAP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_mos_cap
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_mos_cap_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_hvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_lnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' TT_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' global_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodesvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodehvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodelnvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' Total_mpodeulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1_aux.l' stat_mpode
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFMOS_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_rfmvar
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_rfmvar_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' global_rfmos_18ud12
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_svt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_lvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_ulvt
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud15
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' Total_RF_MOS_18ud12
.param totalflag_mos_svt=0
.param totalflag_mos_hvt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lnvt=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboff_flag_mos_svt=1
.param iboff_flag_mos_hvt=1
.param iboff_flag_mos_lvt=1
.param iboff_flag_mos_ulvt=1
.param iboff_flag_mos_lnvt=1
.param iboff_flag_mos_18=1
.param iboff_flag_mos_18ud15=1
.param iboff_flag_mos_18ud12=1
.param iboff_flag_mos_hia18=1
.param totalflag_mos_mpodesvt=0
.param totalflag_mos_mpodehvt=0
.param totalflag_mos_mpodelvt=0
.param totalflag_mos_mpodeulvt=0
.param totalflag_mos_mpodelnvt=0
.param iboff_flag_mos_mpodesvt=1
.param iboff_flag_mos_mpodehvt=1
.param iboff_flag_mos_mpodelvt=1
.param iboff_flag_mos_mpodeulvt=1
.param iboff_flag_mos_mpodelnvt=1
.param totalflag_mos_svt_rf=0
.param totalflag_mos_lvt_rf=0
.param totalflag_mos_ulvt_rf=0
.param totalflag_mos_18_rf=0
.param totalflag_mos_18ud15_rf=0
.param totalflag_mos_18ud12_rf=0
.param iboff_flag_mos_svt_rf=1
.param iboff_flag_mos_lvt_rf=1
.param iboff_flag_mos_ulvt_rf=1
.param iboff_flag_mos_18_rf=1
.param iboff_flag_mos_18ud15_rf=1
.param iboff_flag_mos_18ud12_rf=1
.ENDL GlobalMC_LocalMC_MOS_MOSCAP

.LIB TT_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES_rf
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.param mismatchflag_disres_rf=0
.ENDL TT_RES_BIP_DIO_DISRES

.LIB SS_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DISRES_rf
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.param mismatchflag_disres_rf=0
.ENDL SS_RES_BIP_DIO_DISRES

.LIB FF_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DISRES_rf
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.param mismatchflag_disres_rf=0
.ENDL FF_RES_BIP_DIO_DISRES

.LIB TTGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES_rf
.ENDL TTGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB SSGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_DISRES_rf
.ENDL SSGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB FFGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_DISRES_rf
.ENDL FFGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB LocalMCOnly_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_BIP_NPN
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DISRES_rf
.ENDL LocalMCOnly_RES_BIP_DIO_DISRES

.LIB GlobalMC_LocalMC_RES_BIP_DIO_DISRES
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_res
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_disres
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_bip
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_bip_npn
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_dio
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_dio_18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_dio_dnw
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat_res
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat_bip_dio
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FOD
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_hia18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_DIO_gated18_rf
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_disres_rf
.ENDL GlobalMC_LocalMC_RES_BIP_DIO_DISRES

.LIB TT_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_fmom_p80_rf=0
.ENDL TT_MOM

.LIB SS_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_fmom_p80_rf=0
.ENDL SS_MOM

.LIB FF_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_fmom_p80_rf=0
.ENDL FF_MOM

.LIB TTGlobalCorner_LocalMC_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFFMOM_p80
.ENDL TTGlobalCorner_LocalMC_MOM

.LIB SSGlobalCorner_LocalMC_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFFMOM_p80
.ENDL SSGlobalCorner_LocalMC_MOM

.LIB FFGlobalCorner_LocalMC_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFFMOM_p80
.ENDL FFGlobalCorner_LocalMC_MOM

.LIB LocalMCOnly_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_FMOM_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFFMOM_p80
.ENDL LocalMCOnly_MOM

.LIB GlobalMC_LocalMC_MOM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_fmom_p80
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat_mom
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_MIM
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' total_rffmom_p80
.ENDL GlobalMC_LocalMC_MOM

.LIB TT_SCR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_mbscr
.param iboff_flag_mos_mbscr=0
.ENDL TT_SCR

.LIB SS_SCR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL SS_SCR

.LIB FF_SCR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL FF_SCR

.LIB SF_SCR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SF_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL SF_SCR

.LIB FS_SCR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FS_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL FS_SCR

.lib TT_R_METAL
.lib 'res_metal.l' TT_RES_METAL
.endl TT_R_METAL

.lib SS_R_METAL
.lib 'res_metal.l' SS_RES_METAL
.endl SS_R_METAL

.lib FF_R_METAL
.lib 'res_metal.l' FF_RES_METAL
.endl FF_R_METAL

.lib TT_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFIND
.endl TT_IND_JVAR

.lib SS_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFIND
.endl SS_IND_JVAR

.lib FF_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFIND
.endl FF_IND_JVAR

.lib TTGlobalCorner_LocalMC_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFIND
.endl TTGlobalCorner_LocalMC_IND_JVAR

.lib SSGlobalCorner_LocalMC_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' SS_RFIND
.endl SSGlobalCorner_LocalMC_IND_JVAR

.lib FFGlobalCorner_LocalMC_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' FF_RFIND
.endl FFGlobalCorner_LocalMC_IND_JVAR

.lib LocalMCOnly_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' TT_RFIND
.endl LocalMCOnly_IND_JVAR

.lib GlobalMC_LocalMC_IND_JVAR
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' stat_ind
.lib 'crn16ffcll_1d8_sp_v1d0_2p1.l' GLOBAL_RFIND
.endl GlobalMC_LocalMC_IND_JVAR
