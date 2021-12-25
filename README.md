# Unsupervised-Comment-based-Multi-document-Extractive-Summarization
 
This work proposes a novel multi-objective optimization-based framework for  Unsupervised-Comment-based-Multi-document-Extractive-Summarization. A subset of relevant news sentences will be automatically selected from an available set of sentences by utilizing the user-comments. Different statistical quality functions measuring various aspects of summary, namely, diversity, user attention score, density based score and, user-attention with syntactic score, are optimized simultaneously using the search capability of a multi-objective binary differential evolution technique. 

Input Files: 
--------------------------------------------------------------------
1)	WMD matrix which is the distance matrix having tweet to tweet distance in semantic space [Line-28]
2)	Reader Attention score of news sentence [Line-38]
3)	Density based score of news sentence [Line-49]
4)	Reader Attention with syntatcic score of news sentence [Line-60]
5)	Length of news sentences [Line 84]
6)	Original set of news sentences [Line 96]
7)	Reference/Actual/Gold summaries [Line-123, Line-152, Line-177, Line-202]

Note: All the above input files are present in the preprocessing directory.


 User input:
 -----------------------------------------------------------------------
 Since, the code is automated for multiple topics, you have to update the below values before running the main program.
1)	Population size [Line 229]
2)	Mating pool size [Line 232]
3)	Minimum number of tweets to be in the summary [Line 237]
4)	Maximum number of tweets to be in the summary [Line 240]
5)	Maximum number of generations [Line 243]

Output Files: 
-----------------------------------------------------------------------
1)	Folder ‘generation_wise_details’: It includes summaries obtained for each solution in the population + Rouge scores for each summary
2)	Folder ‘Pareto_front’: It include Pareto fronts obtained at the end of each generation. 
3)	Files:
(a)	 ‘Annotator1_solutionwise_summary_score_overview’,
(b)	‘Annotator2_solutionwise_summary_score_overview’, 
(c)	‘Annotator3_solutionwise_summary_score_overview’
(d) ‘Annotator4_solutionwise_summary_score_overview’
These files contains gold summaries scores corresponding to each solution in the final population (at the end of the execution)
(e)	Plots: 
i)	‘Generation_wise_Objective_values’: It shows the maximum values of objective functions at each generation.
ii)	‘New Sols_vs_Generations’: It shows the number of new good solutions obtained at the end of each generation. 
iii)	‘Generation Wise Rouge score’:  It shows the maximum ROUGE score values (obtained using the gold summary) at each generation.

How to Run:
-----------------------------------------------------------------------
1)	Create a text file and provide all the topics names in that file separated by '\n' and provide the text file path in [Line 618]. All the outputs will be stored in the output folder in the folder with the same topic name you have provided in the input text file. 
2)	To run the program, go to ‘examples’ folder and run the file ‘comment_based_summarization_main.py’ and give the required number parameters before running the program.  Note that there we have utilized 2 datasets one belonging to English language and another belonging to French language. For the testing purpose result of only 3 topics out of 45 topics of the english dataset are present. For running the code on french dataset execute 'french_dataset_comment_based_summarization_main.py' and provide the path of all the required input files to the program. 
