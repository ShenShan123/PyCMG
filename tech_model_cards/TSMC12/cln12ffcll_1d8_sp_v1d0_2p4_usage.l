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
.param ccosflag_mos_hv18=1
.param ccodflag_mos_hv18=1
.param rgflag_mos_hv18=1
.param rcosflag_mos_hv18=1
.param rcodflag_mos_hv18=1
.ENDL pre_simu

.LIB fnoise_mos_typ
.param noiseflagn_mos_svt=0
.param noiseflagp_mos_svt=0
.param noiseflagp_mos_hvt=0
.param noiseflagn_mos_hvt=0
.param noiseflagn_mos_lvt=0
.param noiseflagp_mos_lvt=0
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
.param noiseflagp_mos_mpodehvt=0
.param noiseflagn_mos_mpodehvt=0
.param noiseflagn_mos_mpodelvt=0
.param noiseflagp_mos_mpodelvt=0
.param noiseflagn_mos_mpodeulvt=0
.param noiseflagp_mos_mpodeulvt=0
.param noiseflagp_mos_mpodelnvt=0
.param noiseflagn_mos_mpodelnvt=0
.param noiseflagn_mos_hv18=0
.param noiseflagp_mos_hv18=0
.ENDL fnoise_mos_typ

.LIB fnoise_mos_wmc
.param noiseflagwmcn_mos_svt=1
.param noiseflagwmcp_mos_svt=1
.param noiseflagwmcp_mos_hvt=1
.param noiseflagwmcn_mos_hvt=1
.param noiseflagwmcn_mos_lvt=1
.param noiseflagwmcp_mos_lvt=1
.param noiseflagwmcn_mos_ulvt=1
.param noiseflagwmcp_mos_ulvt=1
.param noiseflagwmcp_mos_lnvt=1
.param noiseflagwmcn_mos_lnvt=1
.param noiseflagwmcn_mos_18=1
.param noiseflagwmcp_mos_18=1
.param noiseflagwmcn_mos_18ud15=1
.param noiseflagwmcp_mos_18ud15=1
.param noiseflagwmcn_mos_18ud12=1
.param noiseflagwmcp_mos_18ud12=1
.param noiseflagwmcn_mos_mpodesvt=1
.param noiseflagwmcp_mos_mpodesvt=1
.param noiseflagwmcp_mos_mpodehvt=1
.param noiseflagwmcn_mos_mpodehvt=1
.param noiseflagwmcn_mos_mpodelvt=1
.param noiseflagwmcp_mos_mpodelvt=1
.param noiseflagwmcn_mos_mpodeulvt=1
.param noiseflagwmcp_mos_mpodeulvt=1
.param noiseflagwmcp_mos_mpodelnvt=1
.param noiseflagwmcn_mos_mpodelnvt=1
.param noiseflagn_mos_svt=1
.param noiseflagp_mos_svt=1
.param noiseflagp_mos_hvt=1
.param noiseflagn_mos_hvt=1
.param noiseflagn_mos_lvt=1
.param noiseflagp_mos_lvt=1
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
.param noiseflagp_mos_mpodehvt=1
.param noiseflagn_mos_mpodehvt=1
.param noiseflagn_mos_mpodelvt=1
.param noiseflagp_mos_mpodelvt=1
.param noiseflagn_mos_mpodeulvt=1
.param noiseflagp_mos_mpodeulvt=1
.param noiseflagp_mos_mpodelnvt=1
.param noiseflagn_mos_mpodelnvt=1
.param noiseflagwmcn_mos_hv18=1
.param noiseflagwmcp_mos_hv18=1
.param noiseflagn_mos_hv18=1
.param noiseflagp_mos_hv18=1
.ENDL fnoise_mos_wmc

.LIB fnoise_mos_bc
.param noiseflagn_mos_svt=-1
.param noiseflagp_mos_svt=-1
.param noiseflagp_mos_hvt=-1
.param noiseflagn_mos_hvt=-1
.param noiseflagn_mos_lvt=-1
.param noiseflagp_mos_lvt=-1
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
.param noiseflagp_mos_mpodehvt=-1
.param noiseflagn_mos_mpodehvt=-1
.param noiseflagn_mos_mpodelvt=-1
.param noiseflagp_mos_mpodelvt=-1
.param noiseflagn_mos_mpodeulvt=-1
.param noiseflagp_mos_mpodeulvt=-1
.param noiseflagp_mos_mpodelnvt=-1
.param noiseflagn_mos_mpodelnvt=-1
.param noiseflagn_mos_hv18=-1
.param noiseflagp_mos_hv18=-1
.ENDL fnoise_mos_bc

.LIB fnoise_mos_stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_usage.l' fnoise_mos_typ
.param noiseflagn_mc_mos_svt=1
.param noiseflagp_mc_mos_svt=1
.param noiseflagp_mc_mos_hvt=1
.param noiseflagn_mc_mos_hvt=1
.param noiseflagn_mc_mos_lvt=1
.param noiseflagp_mc_mos_lvt=1
.param noiseflagn_mc_mos_ulvt=1
.param noiseflagp_mc_mos_ulvt=1
.param noiseflagp_mc_mos_lnvt=1
.param noiseflagn_mc_mos_lnvt=1
.param noiseflagn_mc_mos_18=1
.param noiseflagp_mc_mos_18=1
.param noiseflagn_mc_mos_18ud15=1
.param noiseflagp_mc_mos_18ud15=1
.param noiseflagn_mc_mos_18ud12=1
.param noiseflagp_mc_mos_18ud12=1
.param noiseflagn_mc_mos_mpodesvt=1
.param noiseflagp_mc_mos_mpodesvt=1
.param noiseflagp_mc_mos_mpodehvt=1
.param noiseflagn_mc_mos_mpodehvt=1
.param noiseflagn_mc_mos_mpodelvt=1
.param noiseflagp_mc_mos_mpodelvt=1
.param noiseflagn_mc_mos_mpodeulvt=1
.param noiseflagp_mc_mos_mpodeulvt=1
.param noiseflagp_mc_mos_mpodelnvt=1
.param noiseflagn_mc_mos_mpodelnvt=1
.param noiseflagn_mc_mos_hv18=1
.param noiseflagp_mc_mos_hv18=1
.ENDL fnoise_mos_stat

.LIB fnoise_res_typ
.param rnoiseflag_res=0
.param rnoiseflag_disres=0
.ENDL fnoise_res_typ

.LIB fnoise_res_wc
.param rnoiseflag_res=1
.param rnoiseflag_disres=1
.ENDL fnoise_res_wc

.LIB fnoise_res_bc
.param rnoiseflag_res=-1
.param rnoiseflag_disres=-1
.ENDL fnoise_res_bc

.LIB age0
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=0
.ENDL age0

.LIB age1
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=1
.ENDL age1

.LIB age2
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=2
.ENDL age2

.LIB hci_only
.option tmiAge=1
.param tmiageflag=1
.param phys_id=1
.ENDL hci_only

.LIB bti_only
.option tmiAge=1
.param tmiageflag=1
.param phys_id=2
.ENDL bti_only

.LIB age_unit_day
.option tmiAge=1
.param tmiageflag=1
.param age_unit=1
.ENDL age_unit_day

.LIB age_unit_minute
.option tmiAge=1
.param tmiageflag=1
.param age_unit=2
.ENDL age_unit_minute

.LIB she
.option tmiAge=1
.param tmiageflag=1
.param tmishe=1
.param sheimod=2
.ENDL she

.LIB age0_she
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=0
.param tmishe=2
.param sheimod=2
.param phys_id=0
.param age_unit=0
.ENDL age0_she

.LIB age1_she
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=1
.param tmishe=2
.param sheimod=2
.param phys_id=0
.param age_unit=0
.ENDL age1_she

.LIB age2_she
.option tmiAge=1
.param tmiageflag=1
.param age_specmod=2
.param tmishe=2
.param sheimod=2
.param phys_id=0
.param age_unit=0
.ENDL age2_she

.LIB TTMacro_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' tt_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' tt_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param iboff_flag_mos_hv18=0
.ENDL TTMacro_MOS_MOSCAP

.LIB SSMacro_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ss_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ss_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param iboff_flag_mos_hv18=1
.ENDL SSMacro_MOS_MOSCAP

.LIB FFMacro_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ff_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ff_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param iboff_flag_mos_hv18=1
.ENDL FFMacro_MOS_MOSCAP

.LIB SFMacro_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sf_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sf_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param iboff_flag_mos_hv18=1
.ENDL SFMacro_MOS_MOSCAP

.LIB FSMacro_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fs_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fs_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param iboff_flag_mos_hv18=1
.ENDL FSMacro_MOS_MOSCAP

.LIB SSGNPGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_usage.l' SSGlobalCorner_LocalMC_MOS_MOSCAP
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
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_usage.l' FFGlobalCorner_LocalMC_MOS_MOSCAP
.param global_factorn_svt=0.857
.param global_factorp_svt=0.857
.param global_factorn_hvt=0.857
.param global_factorp_hvt=0.857
.param global_factorn_lvt=0.874
.param global_factorp_lvt=0.874
.param global_factorn_ulvt=0.871
.param global_factorp_ulvt=0.871
.param global_factorn_lnvt=0.88
.param global_factorp_lnvt=0.88
.param global_factorn_mpodesvt=0.857
.param global_factorp_mpodesvt=0.857
.param global_factorn_mpodehvt=0.857
.param global_factorp_mpodehvt=0.857
.param global_factorn_mpodelvt=0.874
.param global_factorp_mpodelvt=0.874
.param global_factorn_mpodeulvt=0.871
.param global_factorp_mpodeulvt=0.871
.param global_factorn_mpodelnvt=0.88
.param global_factorp_mpodelnvt=0.88
.ENDL FFGNPGlobalCorner_LocalMC_MOS_MOSCAP

.LIB TTGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' tt_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ttg_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ttg_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ttg_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ttg_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ttg_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' tt_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ttg_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=0
.ENDL TTGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ss_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssg_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssg_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssg_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssg_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssg_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ss_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssg_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL SSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ff_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffg_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffg_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffg_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffg_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffg_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ff_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffg_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL FFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sf_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sfg_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sfg_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sfg_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sfg_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sfg_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' sf_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SF_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' sfg_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL SFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fs_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fsg_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fsg_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fsg_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fsg_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fsg_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' fs_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FS_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' fsg_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL FSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ss_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssag_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssag_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssag_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssag_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ssag_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ss_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' SS_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ssag_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL SSAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ff_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffag_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffag_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffag_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffag_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ffag_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' ff_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' FF_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' ffag_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL FFAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB LocalMCOnly_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' tt_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MOS_CAP_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' tt_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFMVAR_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param globalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL LocalMCOnly_MOS_MOSCAP

.LIB GlobalMC_LocalMC_MOS_MOSCAP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' tt_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_mos_cap
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_mos_cap_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_svt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_ulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_lnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud15
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_18ud12
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' tt_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' TT_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' global_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodesvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodehvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodeulvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' Total_mpodelnvt
.lib 'cln12ffcll_1d8_sp_v1d0_2p4_aux.l' stat_mpode
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' total_rfmvar
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' total_rfmvar_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' global_hv18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' Total_hv18
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
.param totalflag_mos_hv18=0
.param iboff_flag_mos_hv18=1
.ENDL GlobalMC_LocalMC_MOS_MOSCAP

.LIB TT_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FOD
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.ENDL TT_RES_BIP_DIO_DISRES

.LIB SS_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FOD
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.ENDL SS_RES_BIP_DIO_DISRES

.LIB FF_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FOD
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.param mismatchflag_res=0
.param mismatchflag_disres=0
.ENDL FF_RES_BIP_DIO_DISRES

.LIB TTGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FOD
.ENDL TTGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB SSGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FOD
.ENDL SSGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB FFGlobalCorner_LocalMC_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FOD
.ENDL FFGlobalCorner_LocalMC_RES_BIP_DIO_DISRES

.LIB LocalMCOnly_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_BIP_NPN
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FOD
.ENDL LocalMCOnly_RES_BIP_DIO_DISRES

.LIB GlobalMC_LocalMC_RES_BIP_DIO_DISRES
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_res
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_disres
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_bip
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_bip_npn
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_dio
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_dio_18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_dio_dnw
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat_res
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat_bip_dio
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_hia18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_DIO_gated18
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FOD
.ENDL GlobalMC_LocalMC_RES_BIP_DIO_DISRES

.LIB TT_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_rtmom=0
.param mismatchflag_fmom_p80_rf=0
.ENDL TT_MOM

.LIB SS_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_rtmom=0
.param mismatchflag_fmom_p80_rf=0
.ENDL SS_MOM

.LIB FF_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_rtmom=0
.param mismatchflag_fmom_p80_rf=0
.ENDL FF_MOM

.LIB TTGlobalCorner_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFFMOM_p80
.ENDL TTGlobalCorner_LocalMC_MOM

.LIB SSGlobalCorner_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' SS_RFFMOM_p80
.ENDL SSGlobalCorner_LocalMC_MOM

.LIB FFGlobalCorner_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' FF_RFFMOM_p80
.ENDL FFGlobalCorner_LocalMC_MOM

.LIB LocalMCOnly_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' TT_RFFMOM_p80
.ENDL LocalMCOnly_MOM

.LIB GlobalMC_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_fmom_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_rtmom
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat_mom
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' total_rffmom_p80
.ENDL GlobalMC_LocalMC_MOM

.LIB QBEST_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' QBEST_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_rtmom=0
.param mismatchflag_fmom_p80_rf=0
.ENDL QBEST_MOM

.LIB QWORST_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' QWORST_RFFMOM_p80
.param mismatchflag_fmom_p80=0
.param mismatchflag_rtmom=0
.param mismatchflag_fmom_p80_rf=0
.ENDL QWORST_MOM

.LIB QBESTGlobalCorner_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' QBEST_RFFMOM_p80
.ENDL QBESTGlobalCorner_LocalMC_MOM

.LIB QWORSTGlobalCorner_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_FMOM_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_RTMOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' QWORST_RFFMOM_p80
.ENDL QWORSTGlobalCorner_LocalMC_MOM

.LIB QGlobalMC_LocalMC_MOM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_fmom_p80
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' total_rtmom
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' stat_mom
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_MIM
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_vmom
.lib 'crn12ffcll_lct_1d8_sp_v1d0_2p4.l' total_rffmom_p80
.param qtotalflag_fmom_rf=1
.ENDL QGlobalMC_LocalMC_MOM

.LIB TT_SCR
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' TT_mbscr
.param iboff_flag_mos_mbscr=0
.ENDL TT_SCR

.LIB SS_SCR
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SS_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL SS_SCR

.LIB FF_SCR
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FF_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL FF_SCR

.LIB SF_SCR
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' SF_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL SF_SCR

.LIB FS_SCR
.lib 'cln12ffcll_1d8_sp_v1d0_2p4.l' FS_mbscr
.param iboff_flag_mos_mbscr=1
.ENDL FS_SCR

.LIB TT_R_METAL
.lib 'res_metal.l' TT_RES_METAL
.ENDL TT_R_METAL

.LIB SS_R_METAL
.lib 'res_metal.l' SS_RES_METAL
.ENDL SS_R_METAL

.LIB FF_R_METAL
.lib 'res_metal.l' FF_RES_METAL
.ENDL FF_R_METAL

