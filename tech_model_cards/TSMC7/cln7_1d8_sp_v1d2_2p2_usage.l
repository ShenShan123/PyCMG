**************************************************************
*  TSMC SPICE MODEL --- Security B - TSMC Restricted Secret  *
**************************************************************
.LIB pre_simu
.param ccoflag_cap=1
.param ccoflag_cap_18=1
.param ccosflag_mos_svt=1
.param ccosflag_mos_lvt=1
.param ccosflag_mos_ulvt=1
.param ccosflag_mos_lvt15=1
.param ccosflag_mos_lvt15ud12=1
.param ccosflag_mos_18=1
.param ccosflag_mos_18ud15=1
.param ccosflag_mos_18ud12=1
.param ccosflag_mos_hia18=1
.param ccodflag_mos_svt=1
.param ccodflag_mos_lvt=1
.param ccodflag_mos_ulvt=1
.param ccodflag_mos_lvt15=1
.param ccodflag_mos_lvt15ud12=1
.param ccodflag_mos_18=1
.param ccodflag_mos_18ud15=1
.param ccodflag_mos_18ud12=1
.param ccodflag_mos_hia18=1
.param rgflag_mos_svt=1
.param rgflag_mos_lvt=1
.param rgflag_mos_ulvt=1
.param rgflag_mos_lvt15=1
.param rgflag_mos_lvt15ud12=1
.param rgflag_mos_18=1
.param rgflag_mos_18ud15=1
.param rgflag_mos_18ud12=1
.param rgflag_mos_hia18=1
.param rgflag_cap=1
.param rgflag_cap_18=1
.param rcoflag_res=1
.param rcoflag_disres=1
.param rcosflag_mos_svt=1
.param rcosflag_mos_lvt=1
.param rcosflag_mos_ulvt=1
.param rcosflag_mos_lvt15=1
.param rcosflag_mos_lvt15ud12=1
.param rcosflag_mos_18=1
.param rcosflag_mos_18ud15=1
.param rcosflag_mos_18ud12=1
.param rcosflag_mos_hia18=1
.param rcodflag_mos_svt=1
.param rcodflag_mos_lvt=1
.param rcodflag_mos_ulvt=1
.param rcodflag_mos_lvt15=1
.param rcodflag_mos_lvt15ud12=1
.param rcodflag_mos_18=1
.param rcodflag_mos_18ud15=1
.param rcodflag_mos_18ud12=1
.param rcodflag_mos_hia18=1
.param capflag_disres=1
.param capflag_mim=1
.ENDL pre_simu

.LIB fnoise_mos_typ
.param noiseflagn_mos_svt=0
.param noiseflagp_mos_svt=0
.param noiseflagn_mos_lvt=0
.param noiseflagp_mos_lvt=0
.param noiseflagn_mos_ulvt=0
.param noiseflagp_mos_ulvt=0
.param noiseflagn_mos_lvt15=0
.param noiseflagp_mos_lvt15=0
.param noiseflagn_mos_lvt15ud12=0
.param noiseflagp_mos_lvt15ud12=0
.param noiseflagn_mos_18=0
.param noiseflagp_mos_18=0
.param noiseflagn_mos_18ud15=0
.param noiseflagp_mos_18ud15=0
.param noiseflagn_mos_18ud12=0
.param noiseflagp_mos_18ud12=0
.ENDL fnoise_mos_typ

.LIB fnoise_mos_wmc
.param noiseflagwmcn_mos_svt=1
.param noiseflagwmcp_mos_svt=1
.param noiseflagwmcn_mos_lvt=1
.param noiseflagwmcp_mos_lvt=1
.param noiseflagwmcn_mos_ulvt=1
.param noiseflagwmcp_mos_ulvt=1
.param noiseflagwmcn_mos_lvt15=1
.param noiseflagwmcp_mos_lvt15=1
.param noiseflagwmcn_mos_lvt15ud12=1
.param noiseflagwmcp_mos_lvt15ud12=1
.param noiseflagwmcn_mos_18=1
.param noiseflagwmcp_mos_18=1
.param noiseflagwmcn_mos_18ud15=1
.param noiseflagwmcp_mos_18ud15=1
.param noiseflagwmcn_mos_18ud12=1
.param noiseflagwmcp_mos_18ud12=1
.param noiseflagn_mos_svt=1
.param noiseflagp_mos_svt=1
.param noiseflagn_mos_lvt=1
.param noiseflagp_mos_lvt=1
.param noiseflagn_mos_ulvt=1
.param noiseflagp_mos_ulvt=1
.param noiseflagn_mos_lvt15=1
.param noiseflagp_mos_lvt15=1
.param noiseflagn_mos_lvt15ud12=1
.param noiseflagp_mos_lvt15ud12=1
.param noiseflagn_mos_18=1
.param noiseflagp_mos_18=1
.param noiseflagn_mos_18ud15=1
.param noiseflagp_mos_18ud15=1
.param noiseflagn_mos_18ud12=1
.param noiseflagp_mos_18ud12=1
.ENDL fnoise_mos_wmc

.LIB fnoise_mos_bc
.param noiseflagn_mos_svt=-1
.param noiseflagp_mos_svt=-1
.param noiseflagn_mos_lvt=-1
.param noiseflagp_mos_lvt=-1
.param noiseflagn_mos_ulvt=-1
.param noiseflagp_mos_ulvt=-1
.param noiseflagn_mos_lvt15=-1
.param noiseflagp_mos_lvt15=-1
.param noiseflagn_mos_lvt15ud12=-1
.param noiseflagp_mos_lvt15ud12=-1
.param noiseflagn_mos_18=-1
.param noiseflagp_mos_18=-1
.param noiseflagn_mos_18ud15=-1
.param noiseflagp_mos_18ud15=-1
.param noiseflagn_mos_18ud12=-1
.param noiseflagp_mos_18ud12=-1
.ENDL fnoise_mos_bc

.LIB fnoise_mos_stat
.lib 'cln7_1d8_sp_v1d2_2p2_usage.l' fnoise_mos_typ
.param noiseflagn_mc_mos_svt=1
.param noiseflagp_mc_mos_svt=1
.param noiseflagn_mc_mos_lvt=1
.param noiseflagp_mc_mos_lvt=1
.param noiseflagn_mc_mos_ulvt=1
.param noiseflagp_mc_mos_ulvt=1
.param noiseflagn_mc_mos_lvt15=1
.param noiseflagp_mc_mos_lvt15=1
.param noiseflagn_mc_mos_lvt15ud12=1
.param noiseflagp_mc_mos_lvt15ud12=1
.param noiseflagn_mc_mos_18=1
.param noiseflagp_mc_mos_18=1
.param noiseflagn_mc_mos_18ud15=1
.param noiseflagp_mc_mos_18ud15=1
.param noiseflagn_mc_mos_18ud12=1
.param noiseflagp_mc_mos_18ud12=1
.ENDL fnoise_mos_stat

.LIB fnoise_res_typ
.param rnoiseflag_disres=0
.ENDL fnoise_res_typ

.LIB fnoise_res_wc
.param rnoiseflag_disres=1
.ENDL fnoise_res_wc

.LIB fnoise_res_bc
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
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param iboffwcflag_mos_svt=0
.param iboffwcflag_mos_lvt=0
.param iboffwcflag_mos_ulvt=0
.param iboffwcflag_mos_lvt15=0
.param iboffwcflag_mos_lvt15ud12=0
.param iboffwcflag_mos_18=0
.param iboffwcflag_mos_18ud15=0
.param iboffwcflag_mos_18ud12=0
.param iboffwcflag_mos_hia18=0
.ENDL TTMacro_MOS_MOSCAP

.LIB SSMacro_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR_18
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL SSMacro_MOS_MOSCAP

.LIB FFMacro_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR_18
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL FFMacro_MOS_MOSCAP

.LIB SFMacro_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL SFMacro_MOS_MOSCAP

.LIB FSMacro_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL FSMacro_MOS_MOSCAP

.LIB IDDQ_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR_18
.param gnpskewn_mos_svt=0.827
.param gnpskewp_mos_svt=0.827
.param gnpskewn_mos_lvt=0.82
.param gnpskewp_mos_lvt=0.82
.param gnpskewn_mos_ulvt=0.78
.param gnpskewp_mos_ulvt=0.78
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param mismatchflag_mos_svt=0
.param mismatchflag_mos_lvt=0
.param mismatchflag_mos_ulvt=0
.param iddqflag_mos_svt=1
.param iddqflag_mos_lvt=1
.param iddqflag_mos_ulvt=1
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL IDDQ_MOS_MOSCAP

.LIB SSGNPGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2_usage.l' SSGlobalCorner_LocalMC_MOS_MOSCAP
.param gnpskewn_mos_svt=0.782
.param gnpskewp_mos_svt=0.782
.param gnpskewn_mos_lvt=0.782
.param gnpskewp_mos_lvt=0.782
.param gnpskewn_mos_ulvt=0.782
.param gnpskewp_mos_ulvt=0.782
.ENDL SSGNPGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFGNPGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2_usage.l' FFGlobalCorner_LocalMC_MOS_MOSCAP
.param gnpskewn_mos_svt=0.827
.param gnpskewp_mos_svt=0.827
.param gnpskewn_mos_lvt=0.82
.param gnpskewp_mos_lvt=0.82
.param gnpskewn_mos_ulvt=0.78
.param gnpskewp_mos_ulvt=0.78
.ENDL FFGNPGlobalCorner_LocalMC_MOS_MOSCAP

.LIB TTGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ttg_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=0
.param iboffwcflag_mos_lvt=0
.param iboffwcflag_mos_ulvt=0
.param iboffwcflag_mos_lvt15=0
.param iboffwcflag_mos_lvt15ud12=0
.param iboffwcflag_mos_18=0
.param iboffwcflag_mos_18ud15=0
.param iboffwcflag_mos_18ud12=0
.param iboffwcflag_mos_hia18=0
.ENDL TTGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssg_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL SSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffg_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL FFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SFGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' sfg_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL SFGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FSGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' fsg_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FS_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL FSGlobalCorner_LocalMC_MOS_MOSCAP

.LIB SSAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ssag_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL SSAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB FFAGlobalCorner_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' ffag_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL FFAGlobalCorner_LocalMC_MOS_MOSCAP

.LIB LocalMCOnly_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MOS_CAP_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFMVAR_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param globalflag_mos_svt=0
.param globalflag_mos_lvt=0
.param globalflag_mos_ulvt=0
.param globalflag_mos_lvt15=0
.param globalflag_mos_lvt15ud12=0
.param globalflag_mos_18=0
.param globalflag_mos_18ud15=0
.param globalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL LocalMCOnly_MOS_MOSCAP

.LIB GlobalMC_LocalMC_MOS_MOSCAP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_mos_cap
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_mos_cap_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' global_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_svt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_ulvt
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_lvt15ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud15
.lib 'cln7_1d8_sp_v1d2_2p2.l' Total_18ud12
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' total_rfmvar
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' total_rfmvar_18
.param totalflag_mos_svt=0
.param totalflag_mos_lvt=0
.param totalflag_mos_ulvt=0
.param totalflag_mos_lvt15=0
.param totalflag_mos_lvt15ud12=0
.param totalflag_mos_18=0
.param totalflag_mos_18ud15=0
.param totalflag_mos_18ud12=0
.param iboffwcflag_mos_svt=1
.param iboffwcflag_mos_lvt=1
.param iboffwcflag_mos_ulvt=1
.param iboffwcflag_mos_lvt15=1
.param iboffwcflag_mos_lvt15ud12=1
.param iboffwcflag_mos_18=1
.param iboffwcflag_mos_18ud15=1
.param iboffwcflag_mos_18ud12=1
.param iboffwcflag_mos_hia18=1
.ENDL GlobalMC_LocalMC_MOS_MOSCAP

.LIB TT_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DISRES
.param mismatchflag_disres=0
.ENDL TT_RES_DISRES

.LIB SS_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DISRES
.param mismatchflag_disres=0
.ENDL SS_RES_DISRES

.LIB FF_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DISRES
.param mismatchflag_disres=0
.ENDL FF_RES_DISRES

.LIB TTGlobalCorner_LocalMC_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DISRES
.ENDL TTGlobalCorner_LocalMC_RES_DISRES

.LIB SSGlobalCorner_LocalMC_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DISRES
.ENDL SSGlobalCorner_LocalMC_RES_DISRES

.LIB FFGlobalCorner_LocalMC_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DISRES
.ENDL FFGlobalCorner_LocalMC_RES_DISRES

.LIB LocalMCOnly_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RES
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DISRES
.ENDL LocalMCOnly_RES_DISRES

.LIB GlobalMC_LocalMC_RES_DISRES
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_res
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_disres
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat_res
.ENDL GlobalMC_LocalMC_RES_DISRES

.LIB TT_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP_NPN
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.ENDL TT_BIP_DIO

.LIB SS_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_BIP_NPN
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.ENDL SS_BIP_DIO

.LIB FF_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_hia18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_BIP_NPN
.param mismatchflag_bip=0
.param mismatchflag_bip_npn=0
.ENDL FF_BIP_DIO

.LIB TTGlobalCorner_LocalMC_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP_NPN
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_hia18
.ENDL TTGlobalCorner_LocalMC_BIP_DIO

.LIB SSGlobalCorner_LocalMC_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_BIP_NPN
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_DIO_hia18
.ENDL SSGlobalCorner_LocalMC_BIP_DIO

.LIB FFGlobalCorner_LocalMC_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_BIP_NPN
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_DIO_hia18
.ENDL FFGlobalCorner_LocalMC_BIP_DIO

.LIB LocalMCOnly_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_BIP_NPN
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_hia18
.ENDL LocalMCOnly_BIP_DIO

.LIB GlobalMC_LocalMC_BIP_DIO
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_bip
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_bip_npn
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_dio
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_dio_18
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_dio_dnw
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat_bip_dio
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_DIO_hia18
.ENDL GlobalMC_LocalMC_BIP_DIO

.LIB TT_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFRTMOM
.param mismatchflag_rtmom=0
.param mismatchflag_rtmom_rf=0
.ENDL TT_MOM

.LIB SS_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFRTMOM
.param mismatchflag_rtmom=0
.param mismatchflag_rtmom_rf=0
.ENDL SS_MOM

.LIB FF_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFRTMOM
.param mismatchflag_rtmom=0
.param mismatchflag_rtmom_rf=0
.ENDL FF_MOM

.LIB QWORST_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' QWORST_RFRTMOM
.param mismatchflag_rtmom=0
.param mismatchflag_rtmom_rf=0
.ENDL QWORST_MOM

.LIB QBEST_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' QBEST_RFRTMOM
.param mismatchflag_rtmom=0
.param mismatchflag_rtmom_rf=0
.ENDL QBEST_MOM

.LIB TTGlobalCorner_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFRTMOM
.ENDL TTGlobalCorner_LocalMC_MOM

.LIB SSGlobalCorner_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' SS_RFRTMOM
.ENDL SSGlobalCorner_LocalMC_MOM

.LIB FFGlobalCorner_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' FF_RFRTMOM
.ENDL FFGlobalCorner_LocalMC_MOM

.LIB QWORSTGlobalCorner_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' FF_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' QWORST_RFRTMOM
.ENDL QWORSTGlobalCorner_LocalMC_MOM

.LIB QBESTGlobalCorner_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' SS_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' QBEST_RFRTMOM
.ENDL QBESTGlobalCorner_LocalMC_MOM

.LIB LocalMCOnly_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_RTMOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' TT_RFRTMOM
.ENDL LocalMCOnly_MOM

.LIB GlobalMC_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_rtmom
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat_mom
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' total_rfrtmom
.param qtotalflag_rtmom_rf=0
.ENDL GlobalMC_LocalMC_MOM

.LIB QGlobalMC_LocalMC_MOM
.lib 'cln7_1d8_sp_v1d2_2p2.l' total_rtmom
.lib 'cln7_1d8_sp_v1d2_2p2.l' stat_mom
.lib 'cln7_1d8_sp_v1d2_2p2.l' TT_MIM
.lib 'crn7_lct_1d8_sp_v1d2_2p2.l' total_rfrtmom
.param qtotalflag_rtmom_rf=1
.ENDL QGlobalMC_LocalMC_MOM

.LIB TT_R_METAL
.lib 'res_metal.l' TT_res_metal
.ENDL TT_R_METAL

.LIB SS_R_METAL
.lib 'res_metal.l' SS_res_metal
.ENDL SS_R_METAL

.LIB FF_R_METAL
.lib 'res_metal.l' FF_res_metal
.ENDL FF_R_METAL

