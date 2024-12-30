"""
###########################################################################
###########################################################################
Created on Mon Nov 30 2020

@author: Lisa Eick

Prepares Data for Modell input
###########################################################################
###########################################################################
"""
#install.packages("/home/ivm/R/reticulate_1.22.tar.gz", '/home/ivm/R/x86_64-pc-linux-gnu-library/4.1', repos = NULL, type="source")
#install.packages("/home/ivm/R/here_1.0.1.tar.gz", '/home/ivm/R/x86_64-pc-linux-gnu-library/4.1', repos = NULL, type="source")
#library()

library(data.table)
library(tidyverse)
library(lubridate)
library(dplyr)
#library(reticulate)
library(readxl)
library("readxl")
library(TwoSampleMR)


################################################################################################################################
#################################################### ENDPOINT OF INTEREST ######################################################
################################################################################################################################
#Endpoints:
#I9_CHD T2D C3_LUNG_NONSMALL_EXALLC
endpoint<-"C_STROKE"

endInfoDir="/home/ivm/Documents/TuomoKiskinen/Data/DF11-Infodata/finngen_R11_endpoint_core_noncore_1.0.xlsx"
endpointPathlong="/finngen/library-red/finngen_R11/phenotype_1.0/data/finngen_R11_endpoint_longitudinal_1.0.txt.gz"

######## Make Mendilian Randomization with finemapping results ###############
#get all summary statistics of finngen
files <- list.files(path="/finngen/library-green/finngen_R12/finngen_R12_analysis_data/summary_stats/release", pattern="*.gz", full.names=TRUE, recursive=TRUE)
#grep the right sumstats
myfilesfinished <- files[!grepl(".tbi", files)]

#get the files of finemap results of phenotypes 
condFiles <- list.files(path="/finngen/library-green/finngen_R12/finngen_R12_analysis_data/finemap/summary", pattern="*.SUSIE_99.cred.summary.tsv", full.names=TRUE, recursive=TRUE)
#head(condFiles,100)

endpointlist<-c(endpoint)
#Takes the sumstats of the phenotype of interest (stroke) and puts it in the right format
for (endpoint in endpointlist){
  endpointsumstatspath <- myfilesfinished[grepl(endpoint, myfilesfinished)]
  endpoiSumStat <- fread(cmd=paste("zcat", endpointsumstatspath))
  #add colmns needed for mr
  endpoiSumStat$pheno<-endpoint
  names(endpoiSumStat)[1]<-"chr"
  endpoiSumStat$samplesize<-453733
  endpoiSumStat$SNPID<- paste0("chr",endpoiSumStat$chr,"_",endpoiSumStat$pos,"_",endpoiSumStat$ref,"_",endpoiSumStat$alt)
  endpoiSumStat<-as.data.frame(endpoiSumStat)
  #make MR table using "format_data"
  endpoiSumStat1<-format_data(
    endpoiSumStat,
    type = "outcome",
    phenotype_col = "pheno",
    snps = NULL,
    header = TRUE,
    snp_col = "SNPID",
    beta_col = "beta",
    se_col = "sebeta",
    samplesize_col = "samplesize",
    eaf_col = "af_alt",
    effect_allele_col = "alt",
    other_allele_col = "ref",
    pval_col = "pval",
    gene_col = "nearest_genes",
    chr_col = "chr",
    pos_col = "pos",
    log_pval = FALSE
  )

  #MR of clumped table
  head(endpoiSumStat)
  #create final table to store results
  endTable<-data.frame(matrix(ncol=7))
  names(endTable)<-c("Endpoint", "Exposure", "MR_Egger", "IVW", "WeightedMode", "correct_causal_direction", "steiger_pval")
  #Go through each finmapping result in finngen
  for (condpath in condFiles){
    #Get name of exposure endpoint
    endpoinname <- tail(strsplit(condpath, split="/|.SUSIE_99.cred.summary.tsv")[[1]],1)
    #Check the endpoints which are corr with the endpoint of interest
    #read in exposure file
    expoSumStat <- fread(condpath)
    head(expoSumStat)
    #Add necessary coloumns
    expoSumStat$CHR<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,1]
    expoSumStat$BP<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,2]
    expoSumStat$A1<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,3]
    expoSumStat$A2<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,4]
    #check if the clump file exists already
    expoSumStat$samplesize<-453733
    expoSumStat<-as.data.frame(expoSumStat)
    #Make format table using TwoSampleMR
    expoSumStat1<-format_data(
      expoSumStat,
        phenotype_col = "trait" ,
        type = "exposure",
        snps = NULL,
        header = TRUE,
        snp_col = "rsid",
        beta_col = "beta",
        se_col = "sd",
        eaf_col = "prob",
        effect_allele_col = "A2",
        other_allele_col = "A1",
        pval_col = "p",
        chr_col = "CHR",
        pos_col = "BP",
        log_pval = FALSE
      )
      dat <- harmonise_data(expoSumStat1, endpoiSumStat1)
      res <- mr(dat)
      #ddh<-get_r_from_lor(dat)
      out <- directionality_test(dat)
      knitr::kable(out)
      endpoiSumStat1
      #Store all info in table
      if(nrow(res)==0){
        endTable<-rbind(endTable,c(endpoint,endpoinname,NaN,NaN,NaN,NaN,NaN))
      }
      else{
        if (is.null(out)){
          endTable<-rbind(endTable,c(endpoint,endpoinname,res$pval[1],res$pval[2],res$pval[3],NaN,NaN))
        }
        else{
          endTable<-rbind(endTable,c(endpoint,endpoinname,res$pval[1],res$pval[2],res$pval[3],out$correct_causal_direction[1],out$steiger_pval[1]))
        }
     }
  }
  fwrite(endTable, file = paste0("/home/ivm/Documents/CausalPrediction/Data/R12_MendilianTable_", endpoint,"_finemap.txt"))
}


######## Make mendalion randomisation for top hits of c_stroke Sophie ###############
#Same ad above but making a distance table for all related endpoints
endpoiSumStat <- fread("/home/ivm/Documents/CausalPrediction/Data/R12_MendilianTable_C_STROKE_finemap.txt")
endpoiSumStat<-endpoiSumStat[order(IVW)]
endpoiSumStat<-endpoiSumStat[IVW<1e-10]

files <- list.files(path="/finngen/library-green/finngen_R12/finngen_R12_analysis_data/summary_stats/release", pattern="*.gz", full.names=TRUE, recursive=TRUE)
myfilesfinished <- files[!grepl(".tbi", files)]

condFiles <- list.files(path="/finngen/library-green/finngen_R12/finngen_R12_analysis_data/finemap/summary", pattern="*.SUSIE_99.cred.summary.tsv", full.names=TRUE, recursive=TRUE)
head(condFiles,100)

endpointlist<-endpoiSumStat$Exposure
for (endpoint in endpointlist){
  endpointsumstatspath <- myfilesfinished[grepl(endpoint, myfilesfinished)]
  endpoiSumStat <- fread(cmd=paste("zcat", endpointsumstatspath))
  endpoiSumStat$pheno<-endpoint
  names(endpoiSumStat)[1]<-"chr"
  endpoiSumStat$samplesize<-453733
  endpoiSumStat$SNPID<- paste0("chr",endpoiSumStat$chr,"_",endpoiSumStat$pos,"_",endpoiSumStat$ref,"_",endpoiSumStat$alt)
  endpoiSumStat<-as.data.frame(endpoiSumStat)
  endpoiSumStat1<-format_data(
    endpoiSumStat,
    type = "outcome",
    phenotype_col = "pheno",
    snps = NULL,
    header = TRUE,
    snp_col = "SNPID",
    beta_col = "beta",
    se_col = "sebeta",
    samplesize_col = "samplesize",
    eaf_col = "af_alt",
    effect_allele_col = "alt",
    other_allele_col = "ref",
    pval_col = "pval",
    gene_col = "nearest_genes",
    chr_col = "chr",
    pos_col = "pos",
    log_pval = FALSE
  )
  
  #MR of clumped table
  head(endpoiSumStat)
  endTable<-data.frame(matrix(ncol=7))
  names(endTable)<-c("Endpoint", "Exposure", "MR_Egger", "IVW", "WeightedMode", "correct_causal_direction", "steiger_pval")
  for (subendpoint in endpointlist){
    condpath <- condFiles[grepl(paste0(subendpoint,".SUSIE"), condFiles)]
    #Get name of exposure endpoint
    #endpoinname <- tail(strsplit(condpath, split="/|.SUSIE_99.cred.summary.tsv")[[1]],1)
    #Check the endpoints which are corr with the endpoint of interest
    #read in exposure file
    expoSumStat <- fread(condpath)
    head(expoSumStat)
    expoSumStat$CHR<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,1]
    expoSumStat$BP<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,2]
    expoSumStat$A1<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,3]
    expoSumStat$A2<-matrix(unlist(strsplit(expoSumStat$v, split=":")),ncol=4,byrow=T)[,4]
    #check if the clump file exists already
    expoSumStat$samplesize<-453733
    expoSumStat<-as.data.frame(expoSumStat)
    expoSumStat1<-format_data(
      expoSumStat,
      phenotype_col = "trait" ,
      type = "exposure",
      snps = NULL,
      header = TRUE,
      snp_col = "rsid",
      beta_col = "beta",
      se_col = "sd",
      eaf_col = "prob",
      effect_allele_col = "A2",
      other_allele_col = "A1",
      pval_col = "p",
      chr_col = "CHR",
      pos_col = "BP",
      log_pval = FALSE
    )
    dat <- harmonise_data(expoSumStat1, endpoiSumStat1)
    res <- mr(dat)
    #ddh<-get_r_from_lor(dat)
    out <- directionality_test(dat)
    knitr::kable(out)
    endpoiSumStat1
    #Store all info in table
    if(nrow(res)==0){
      endTable<-rbind(endTable,c(endpoint,subendpoint,NaN,NaN,NaN,NaN,NaN))
    }
    else{
      if (is.null(out)){
        endTable<-rbind(endTable,c(endpoint,subendpoint,res$pval[1],res$pval[2],res$pval[3],NaN,NaN))
      }
      else{
        endTable<-rbind(endTable,c(endpoint,subendpoint,res$pval[1],res$pval[2],res$pval[3],out$correct_causal_direction[1],out$steiger_pval[1]))
      }
    }
  }
  fwrite(endTable, file = paste0("/home/ivm/Documents/CausalPrediction/Data/R12_MendilianTable_", endpoint,"_Sophie_finemap.txt"))
}