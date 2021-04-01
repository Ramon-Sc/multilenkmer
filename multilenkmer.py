
#imports:
import sys
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
from Bio import pairwise2
from matplotlib import colors
import matplotlib.image as mpimg
import argparse
################################################################################
parser=argparse.ArgumentParser()

parser.add_argument("-m","--matchbonus",help="Alignment match bonus, default=1", type=int, default=1,nargs='+')

parser.add_argument("-mm","--mismatchpenalty",help="Alignment mismatch penalty, default=1", type=int, default=1,nargs='+')

parser.add_argument("-go","--gapopeningcost",help="gap opening cost, default=100",type=int, default=100,nargs='+')

parser.add_argument("-ge","--gapextendcost",help="gap extension cost, default=100",type=int, default=100,nargs='+')

parser.add_argument("-SC","--scorecutoff",help="cutoff for considering/ommitting kmers, default=1.8",type=float, default=1.8,nargs='+')

parser.add_argument("-MI","--maxiter",help="maximum number of iterations per set of parameters",type=int, default=3)

#file I/O
parser.add_argument("-i","--input",help="input kmer lists",type=str,required=True)

parser.add_argument("-o","--outputdir",help="plot and report output directory",type=str,required=True)

args = parser.parse_args()

################################################################################
################################################################################

#MEGA IMPORTANT VARIABLES

if type(args.scorecutoff) == float:
    SCORECUTOFF_list=[args.scorecutoff]
else:
    SCORECUTOFF_list=args.scorecutoff
#set maximum number of iterations
MAX_ITER=args.maxiter

#Alignment scoring parameters
if type(args.matchbonus) == int:
    match_list=[args.matchbonus]
else:
    match_list=args.matchbonus

#these values are SUBTRACTED from score
if type(args.mismatchpenalty) == int:
    mismatch_list=[args.mismatchpenalty]
else:
    mismatch_list=args.mismatchpenalty

if type(args.gapopeningcost) == int:
    gapo_list=[args.gapopeningcost]
else:
    gapo_list=args.gapopeningcost

if type(args.gapextendcost) == int:
    gape_list=[args.gapextendcost]
else:
    gape_list=args.gapextendcost


dataset=args.input

logo_outdir=args.outputdir

################################################################################
dsname=dataset.split("/")[-1]

dsname_for_latex="".join([letter for letter in dsname if letter.isalnum()])


file =open(logo_outdir+"/"+"report"+str(dsname)+".tex","w+")

################################################################################
#LATEX part
file.write("""
\\documentclass{article}\n
\\usepackage{graphicx}\n
\\usepackage{longtable}\n
\\usepackage[margin=0.4in]{geometry}
\\usepackage{color, colortbl}
\\definecolor{GrayOne}{gray}{0.9}
\\definecolor{GrayTwo}{gray}{0.75}
\\newcommand{\\centered}[1]{\\begin{tabular}{c} #1 \\end{tabular}}\n
\\begin{document}\n
\\begin{center}\n

\\begin{longtable}{|c|c|c|c|c|c|c|}\n

\\caption{"""+dsname_for_latex+"""}\\\\\n

\hline
\\centered{\\textbf{Match} \\\\ \\textbf{bonus} \\\\ \\quad \\\\ \\textbf{Mism.} \\\\ \\textbf{penalty}} &
\\centered{\\textbf{Gap} \\\\ \\textbf{open.} \\\\ \\textbf{cost} \\\\ \\quad \\\\ \\textbf{Gap} \\\\ \\textbf{ext.} \\\\ \\textbf{cost}} &
\\centered{\\textbf{Score}\\\\ \\textbf{cutoff}} &
\\centered{\\textbf{Iter.}} &
\\centered{\\textbf{Number} \\\\ \\textbf{of} \\\\ \\textbf{kmers} \\\\ \\textbf{contained}} &
\\centered{\\textbf{proposed} \\\\ \\textbf{consensus} \\\\ \\textbf{sequence}} &
\\centered{\\textbf{Logo}}
\\\\ \n
\\hline\n\\endfirsthead

\\centered{\\textbf{Match} \\\\ \\textbf{bonus} \\\\ \\quad \\\\ \\textbf{Mism.} \\\\ \\textbf{penalty}} &
\\centered{\\textbf{Gap} \\\\ \\textbf{open.} \\\\ \\textbf{cost} \\\\ \\quad \\\\ \\textbf{Gap} \\\\ \\textbf{ext.} \\\\ \\textbf{cost}} &
\\centered{\\textbf{Score}\\\\ \\textbf{cutoff}} &
\\centered{\\textbf{Iter.}} &
\\centered{\\textbf{Number} \\\\ \\textbf{of} \\\\ \\textbf{kmers} \\\\ \\textbf{contained}} &
\\centered{\\textbf{proposed} \\\\ \\textbf{consensus} \\\\ \\textbf{sequence}} &
\\centered{\\textbf{Logo}}
\\\\ \n

\\hline\n
\\endhead
\\endfoot
\\hline\\hline\\endlastfoot\n
    """)


################################################################################

def main(SCORECUTOFF_list,MAX_ITER,match_list,mismatch_list,gapo_list,gape_list,dataset,logo_outdir):

    #for rowcolors in lateX table
    colorcount=1

    consensussequence=""

    for SCORECUTOFF in SCORECUTOFF_list:
        for match in match_list:
            for mismatch in mismatch_list:
                for gapo in gapo_list:
                    for gape in gape_list:

                        #rowcolors for Latex table - every other run (one run: one combination of params and all its iterations) has the same color in table (colorcoded seperation of different runs - for visual clarity)
                        if colorcount % 2 == 0:
                            color="GrayOne"
                        else:
                            color="GrayTwo"

                        ITER=1
                        raw_kmer_list=[]

                        with open (dataset,"r") as f:
                            for line in f:
                                splitline=line.split("\t")
                                raw_kmer_list.append((splitline[0],splitline[1][:-1]))


                        while ITER <= MAX_ITER and len(raw_kmer_list)!=0:

                            #matrix setup and initialization:
                            masterseq=raw_kmer_list[0][0]
                            thelinelength=len(masterseq)*3
                            THELIST=[" " for i in range(thelinelength)]
                            len_masterseq=len(masterseq)

                            mtx=[]

                            ###put masterseq in THELIST
                            for i in range(len(masterseq)):
                                THELIST[i+len(masterseq)]=masterseq[i]

                            #normalization factor for kmer noPeakscores:
                            highestscorevalue=raw_kmer_list[0][1]
                            normalizationfactor=1/float(highestscorevalue)
                            addedkmercount=1

                            #Position count matrix and Position weight matrix setup:
                            #[A[],C[],G[],T[]]
                            position_count_matrix=[]
                            #PWM
                            position_weight_matrix=[]

                            for i in range(thelinelength):
                                position_count_matrix.append([0,0,0,0])
                                position_weight_matrix.append([0.0,0.0,0.0,0.0,])


                            position_count_matrix,mtx=update_pcm_mtx("-"*len_masterseq+masterseq+"-"*len_masterseq,position_count_matrix,[i+len_masterseq for i in range(len_masterseq)],1,mtx,thelinelength)


                            SecondList=[]


                            for rawkmindex,kmertup in enumerate(raw_kmer_list):

                                #omit first kmer in list as this is masterseq
                                if rawkmindex == 0:
                                    pass

                                else:
                                    currentkmer=kmertup[0]
                                    currentkmerNopeakscore=float(kmertup[1])

                                    emptystr=""
                                    thelist_as_str=emptystr.join(THELIST)

                                    seqB,lst_pos_to_update,alignment_score=align_cons_kmer(thelist_as_str,currentkmer,match,mismatch,gapo,gape)

                                    consensuslencounter=0
                                    for p in thelist_as_str:
                                        if p != " ":
                                            consensuslencounter+=1


                                    if alignment_score > consensuslencounter/float(SCORECUTOFF):

                                        addedkmercount+=1

                                        basecountincrement=currentkmerNopeakscore*normalizationfactor

                                        position_count_matrix,mtx=update_pcm_mtx(seqB,position_count_matrix,lst_pos_to_update,basecountincrement,mtx,thelinelength)

                                        #update position_weight_matrix:
                                        for pos,acgt_count_list in enumerate(position_count_matrix):

                                            for i in range(4):
                                                position_weight_matrix[pos][i] = acgt_count_list[i]/addedkmercount

                                        consensussequence,emptyfields=get_consesus_from_pwm(position_weight_matrix)

                                        for i,letter in enumerate(consensussequence):
                                            THELIST[i+emptyfields]=letter

                                    else:
                                        SecondList.append((currentkmer,currentkmerNopeakscore))

                            raw_kmer_list=[]
                            raw_kmer_list=SecondList

                            print("Number of iteration: ",ITER)
                            print("Number of kmers added: ",addedkmercount)
                            print("consensussequence: ",consensussequence)


                            consensus_for_latex=""

                            if len(consensussequence)>8:
                            ###
                                if len(consensussequence) %2 == 0:

                                    consensus_for_latex=consensussequence[int(len(consensussequence)/2)-5:int(len(consensussequence)/2)+5]

                                else:
                                    consensus_for_latex=consensussequence[int(len(consensussequence)/2)-3:int(len(consensussequence)/2)+6]
                            else:
                                consensus_for_latex=consensussequence





                    #PLOTS###
                            #Position count and position weight matrices for logomaker
                            bases=["a","c","g","t"]
                            pcm_d={"a":[],"c":[],"g":[],"t":[]}

                            for i in range(4):
                                for position in position_count_matrix:
                                    pcm_d[bases[i]].append(position[i])

                            position_count_matrix_df=pd.DataFrame(data=pcm_d)


                            pwm_d={"a":[],"c":[],"g":[],"t":[]}
                            for i in range(4):
                                for position in position_weight_matrix:
                                    pwm_d[bases[i]].append(position[i])

                            position_weight_matrix_df=pd.DataFrame(data=pwm_d)


                            # "heatmap type thingy - assign numerical value to each base acgt - convert a,c,g,t to 1,2,3,4 respectively
                            conversion_dict={"-":0,"a":1,"c":2,"g":3,"t":4}
                            converted_mtx=[]
                            for row in mtx:
                                tmprow=[]
                                for letter in row:
                                    tmprow.append(conversion_dict[letter])
                                converted_mtx.append(tmprow)

                            # heatmap colors for N,A,C,G,T
                            colormap = colors.ListedColormap(['White','Green','Blue', 'Orange', 'Red'])

                            data=np.array(converted_mtx)

                            #actual plot generated here
                            f, (ax1, ax2) = plt.subplots(2,gridspec_kw={"height_ratios":[1,5]},figsize=(12,5),sharex=True)

                            #pcolormesh generates heatmap-type plot as vectorgraphic

                            # horizontal confinements of heatmap quadriliterals (essentially widths of boxes)
                            cornersX=[i-0.5 for i in range(data.shape[1]+1)]

                            # vertical confinements of heatmap quadriliterals (essentially heights of boxes)
                            cornersY=[i for i in range(data.shape[0]+1)]

                            # draw heatmap
                            ax2.pcolormesh(cornersX,cornersY,data, cmap = colormap)

                            ax1.set_axis_off()

                            plt.subplots_adjust(wspace=0.0, hspace=0.0)

                            #add seqlogo to subpots on axis 1
                            sequenceLogo=lm.Logo(position_count_matrix_df,color_scheme='classic',ax=ax1)
                            #ax1.set_xlim(xmin=0.0)

                            #variables for filename
                            sco1,sco2=(str(SCORECUTOFF)).split(".")
                            datasetname=dataset.split("/")[-1]

                            #save plot as vectorgraphic and as jpg for .tex document
                            plt.savefig(logo_outdir+"/"+datasetname+"_"+"m"+str(match)+"_"+"mm"+str(mismatch)+"_"+"gapo"+str(gapo)+"_"+"gape"+str(gape)+"_"+sco1+"comma"+sco2+"_"+"iter"+str(ITER)+"LOGOANDMAP"+".svg", format="svg")
                            plt.savefig(logo_outdir+"/"+datasetname+"_"+"m"+str(match)+"_"+"mm"+str(mismatch)+"_"+"gapo"+str(gapo)+"_"+"gape"+str(gape)+"_"+sco1+"comma"+sco2+"_"+"iter"+str(ITER)+"LOGOANDMAP"+".jpg", format="jpg")

                            plt.close(f)
                    ################################################################################
                            #LateX table rows
                            file.write("\\rowcolor{"+color+"}")
                            file.write("\t\t\t \\centered{"+str(match)+" \\\\ \\quad \\\\ "+str(mismatch)+"} & \\centered{"+str(gapo)+" \\\\ \\quad \\\\ "+str(gape)+"} & "+str(sco1)+"."+str(sco2)+" & "+str(ITER)+" & "+ str(addedkmercount)+" & "+consensus_for_latex+" & "+"\\includegraphics[width=0.3\\textwidth]"+"{"+logo_outdir+"/"+datasetname+"_"+"m"+str(match)+"_"+"mm"+str(mismatch)+"_"+"gapo"+str(gapo)+"_"+"gape"+str(gape)+"_"+str(sco1)+"comma"+str(sco2)+"_"+"iter"+str(ITER)+"LOGOANDMAP"+".jpg}\\\\\n\\hline")
                    ################################################################################
                    #increment ITER
                            ITER+=1
################################################################################
                    colorcount+=1
    file.write("\\end{longtable}\n")
    file.write("\\end{center}\n")
    file.write("\\end{document}\n")
    file.close()
################################################################################

def align_cons_kmer(consensusseq,currentkmer,match,mismatch,gapo,gape):

    #from bio.pairwise2 manual "localmd" :
    #   m     A match score is the score of identical chars, otherwise mismatchscore.
    #   d     The sequences have different open and extend gap penalties. (tweak: very high opening and extend costs for consensussequence to prevent gaps in consensus)
    alignments = pairwise2.align.localmd(consensusseq, currentkmer,match,-mismatch,-100,-100,-gapo,-gape)

    # best: alignment with highest score - sort list based on score - reverse - first element is the one with highest score
    best = sorted(alignments, key=lambda x : x.score,reverse=True)[0]
    seqB,lst_pos_to_update=best.seqB,[tup[0] for tup in enumerate(best.seqB) if tup[1] != "-"]

    return seqB,lst_pos_to_update,best.score

def update_pcm_mtx(s,pcm,lst_pos_to_update,bc_increment,mtx,thelinelength):

    #update mtx part 1: add empty row
    mtx.append(["-" for i in range(thelinelength)])

    for index in lst_pos_to_update:

        pcm_update_dict={"a":0,"c":1,"g":2,"t":3}
        #catch matrix overflow (only update pcm if index not out of range)
        if index < len(pcm):
            #update mtx part 2: fill in letters
            mtx[-1][index]=s[index]
            #update position count matrix
            pcm[index][pcm_update_dict[s[index]]]+=bc_increment

    return pcm,mtx

def get_consesus_from_pwm(position_weight_matrix):

    consesussequence=""
    letterlist=["a","c","g","t"]

    efcounterON = True
    emptyfields = 0

    for entry in position_weight_matrix:
        if max(entry)!=0.0:
            efcounterON=False
            consesussequence+=letterlist[entry.index(max(entry))]
        else:
            if efcounterON:
                emptyfields+=1

    return consesussequence,emptyfields


if __name__ == '__main__':

    main(SCORECUTOFF_list,MAX_ITER,match_list,mismatch_list,gapo_list,gape_list,dataset,logo_outdir)
