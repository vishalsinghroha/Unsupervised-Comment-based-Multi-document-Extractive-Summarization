# Unsupervised-Comment-based-Multi-document-Extractive-Summarization

# Microblog_Summ_two_objectives
 
This work proposes a novel multi-objective optimization-based framework for  Unsupervised-Comment-based-Multi-document-Extractive-Summarization. A subset of relevant news sentences will be automatically selected from an available set of sentences by utilizing the user-comments. Different statistical quality functions measuring various aspects of summary, namely, diversity, user attention score, density based score and, user-attention with syntactic score, are optimized simultaneously using the search capability of a multi-objective binary differential evolution technique. 

Input Files: 
--------------------------------------------------------------------
1)	WMD matrix which is the distance matrix having tweet to tweet distance in semantic space [Line-19]
2)	Tf-idf score of the tweets [Line-28]
3)	The file having Length of the tweets [Line-38]
4)	Original set of Tweet [Line 58]
5)	Gold summaries [Line-70, Line-82, Line-97]


 User input:
 -----------------------------------------------------------------------
1)	Population size [Line 112]
2)	Mating pool size [Line 113]
3)	Minimum number of tweets to be in the summary [Line 115]
4)	Maximum number of tweets to be in the summary [Line 116]
5)	Maximum number of generations [Line 117]

Output Files: 
-----------------------------------------------------------------------
1)	Folder ‘generation_wise_details’: It includes summaries obtained for each solution in the population + Rouge scores for each summary
2)	Folder ‘Pareto_front’: It include Pareto fronts obtained at the end of each generation. 
3)	Files:
(a)	 ‘Annotator1_solutionwise_summary_score_overview’,
(b)	‘Annotator2_solutionwise_summary_score_overview’, 
(c)	‘Annotator2_solutionwise_summary_score_overview’
These files contains gold summaries scores corresponding to each solution in the final population (at the end of the execution)
(d)	Plots: 
i.	‘Generation_wise_Objective_values’: It shows the maximum values of objective functions at each generation.
ii.	‘New Sols_vs_Generations’: It shows the number of new good solutions obtained at the end of each generation. 
iii.	‘Generation Wise Rouge score’:  It shows the maximum ROUGE score values (obtained using the gold summary) at each generation.
How to Run:
a)	Create a folder with any name in the ‘Output’ folder and input the same name when asked for “Enter dataset name” while execution. All the outputs will be stored in this created folder. 
b)	To run the program, go to ‘examples’ folder and run the file ‘main_hblast1.py’ and give the required number parameters from an user end.  Note that this file is for ‘Bomb blast in Hyderabad’ dataset. Other files exist of different datasets. 
