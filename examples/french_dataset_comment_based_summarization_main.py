from SMEA.evolution import Evolution
from SMEA.problems.zdt import ZDT
from SMEA.problems.zdt.zdt3_definitions import ZDT3Definitions
from plotter import Plotter
import numpy as np
import pandas as pd
import time
import os
# from rouge import Rouge
from rouge_metric import PyRouge


def print_generation(population, generation_num):
    print("Generation: {}".format(generation_num))


def run_comment_based_summarization(dataset):
    # dataset = input("Enter dataset name : ")
    print('*****************************************************************')
    print('Topic name: ' + str(dataset))
    from numpy import genfromtxt, asarray, unique

    """---------------Fetch Tweet to Tweet Similarity (WMD Distance)  Matrix --------------"""

    # EMD_matrix = genfromtxt('../preprocessing/T2T_WMD_matrices/hblast2_T2T_WMD_matrix.txt',
    #                        skip_header=0)  # load EMD matrix for sentences

    my_wmd = pd.read_csv(
        '../preprocessing/french/' + str(dataset) + '/' + str(
            dataset) + '_document_wmd.csv', header=None)  # load EMD matrix for sentences
    temp_EMD_matrix = np.array(my_wmd)
    EMD_matrix = temp_EMD_matrix.transpose()
    # print(EMD_matrix[0])
    print("MATRIX SHAPE : ", EMD_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """---------------objective 2: Fetch Reader Attention value of each tweet --------------"""
    my_ra = pd.read_csv(
        '../preprocessing/french/' + str(dataset) + '/' + str(
            dataset) + '_reader_attention.csv')  # load EMD matrix for sentences
    MAX_reader_attention_matrix = np.array(my_ra['average_reader_attention'])

    # print(MAX_reader_attention_matrix)
    print("first tweet reader attention score :", MAX_reader_attention_matrix[0])
    print("MAX reader attention MATRIX SHAPE : ", MAX_reader_attention_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """--------------- objective 3: Fetch Density based score of each tweet --------------"""
    density_based_score = pd.read_csv(
        '../preprocessing/french/' + str(dataset) + '/' + str(
            dataset) + '_sentence_score.csv')  # load EMD matrix for sentences
    MAX_density_based_score_matrix = np.array(density_based_score['dbs_score'])

    # print(MAX_reader_attention_matrix)
    print("first tweet density based score :", MAX_density_based_score_matrix[0])
    print("MAX density based MATRIX SHAPE : ", MAX_density_based_score_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """--------------- objective 4: reader attention with tf-idf --------------"""
    objective4_score = pd.read_csv(
        '../preprocessing/french/' + str(dataset) + '/' + str(
            dataset) + '_rawt_score.csv')  # load EMD matrix for sentences
    MAX_objective4_score_matrix = np.array(objective4_score['rawt_score'])

    # print(MAX_reader_attention_matrix)
    print("first sentence objective4  score :", MAX_objective4_score_matrix[0])
    print("MAX objective4 MATRIX SHAPE : ", MAX_objective4_score_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """--------------- objective 5: named entity recognition --------------"""
    # objective5_score = pd.read_csv(
    #    'C:/Users/VISHAL/Documents/french_dataset/objective_5_version_1/' + str(
    #        dataset) + '.csv')  # load EMD matrix for sentences
    # MAX_objective5_score_matrix = np.array(objective5_score['entity_weight'])
    #
    # print(MAX_reader_attention_matrix)
    # print("first sentence objective5  score :", MAX_objective5_score_matrix[0])
    # print("MAX objective5 MATRIX SHAPE : ", MAX_objective5_score_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """---------------Fetch tweet length of each tweet ------------------------------------------"""
    # MAX_TWEET_length_matrix = genfromtxt('../preprocessing/MAX_TWEET_LENGTH/hblast2_max_tweet_length.txt',
    #                                     skip_header=0)  # load EMD matrix for sentences
    MAX_TWEET_length_matrix = my_ra['sentence_length']

    # print((MAX_TWEET_length_matrix))
    # MAX_TWEET_length_matrix = MAX_TWEET_length_matrix[:, 1:]
    print("first tweet length :", MAX_TWEET_length_matrix[0])
    print("MAX Tweet Length MATRIX SHAPE : ", MAX_TWEET_length_matrix.shape)
    """----------------------------------END-----------------------------------------------------"""

    """----------------------------Fetch cleaned Tweets ----------------------------------------- """
    clean_text_data = []

    for i in range(len(my_ra['clean_document_sentences'])):
        clean_text_data.append(my_ra['clean_document_sentences'][i])  # load the actual tweets from file
    print("Clean fist tweet :" + str(clean_text_data[0]))
    """----------------------------------END-----------------------------------------------------"""

    """----------------------------Fetch actual Tweets ----------------------------------------- """
    actual_text_data = []
    my_document_sentences = pd.read_csv(
        'C:/Users/VISHAL/Documents/french_dataset/processed_documents/' + str(dataset)  + '.csv')

    for i in range(len(my_document_sentences['document_sentence'])):
        actual_text_data.append(my_document_sentences['document_sentence'][i])  ##load the actual tweets from file

    print("first actual tweet : ", actual_text_data[0])
    print("total number of tweet : ", len(actual_text_data))
    """----------------------------------END-----------------------------------------------------"""

    """----------------------------Fetch actual summary1 ----------------------------------------- """
    count_summary_line = 0
    filepath2 = '../french_dataset/actual_summary/' + str(
        dataset) + '.csv'
    actual_summary1_df = pd.read_csv(filepath2)
    temp_actual_summary1 = actual_summary1_df['summary']
    lower_actual_summary1 = []

    for sent in temp_actual_summary1:
        lower_actual_summary1.append(str(sent))
        count_summary_line += 1

    # print(count_summary_line)
    # print(lower_actual_summary1)

    actual_summary1 = str(lower_actual_summary1)

    # actual_summary1 = actual_summary1.replace('.','\n ')

    print("actual summary1 :", actual_summary1)
    '''with open(filepath2, encoding="utf8") as fp:
        for cnt2, line2 in enumerate(fp):
            actual_summary1 += '' + line2.lower()
            count_summary_line += 1
    print("actual summary1 :", actual_summary1)'''
    """----------------------------------END-----------------------------------------------------"""

    max_len_solution = len(clean_text_data)
    print("maximum length of solution : ", max_len_solution)
    print("no. of sentence in the article : ", len(clean_text_data))
    print("********no. of line in summary :******", count_summary_line)

    # pop_size = input("Enter size of population : ")
    pop_size = 25

    # H = input("Enter mating pool size : ")
    H = 5

    print("no. of line in summary :", count_summary_line)

    # smin = int(input("Enter the minimum number of tweets in the summary: "))
    smin = 2

    # smax = int(input("Enter the maximum number of tweets in the summary: "))
    smax = 5

    # T = input("Enter maximum no. of generation : ")
    T = 5

    start = time.time()
    print("starting time :", start)

    SMEA_clustering = ZDT3Definitions(max_len_solution, clean_text_data, EMD_matrix, MAX_TWEET_length_matrix,
                                      #MAX_reader_attention_matrix,
                                      #MAX_density_based_score_matrix,
                                      MAX_objective4_score_matrix,
                                      #MAX_objective5_score_matrix
                                      )

    problem = ZDT(SMEA_clustering, max_len_solution)
    evolution = Evolution(problem, T, pop_size, H)
    evolution.register_on_new_generation(print_generation)

    final_population, K = evolution.evolve(EMD_matrix,
                                           #MAX_reader_attention_matrix,
                                           #MAX_density_based_score_matrix,
                                           MAX_objective4_score_matrix,
                                           #MAX_objective5_score_matrix,
                                           MAX_TWEET_length_matrix, smin, smax,
                                           max_len_solution, dataset, actual_text_data, actual_summary1,
                                           )

    print("length final population:", len(final_population))

    end = time.time()
    print("\n ending time : " + str(end))
    print("\n Total execution time :" + str(end - start))
    total_time = end - start

    fname1 = '../output/French/final_output1234/' + str(dataset) + '/running_time'
    text_file1 = open(fname1, "w")
    text_file1.write("Starting time : " + str(start) + '\n')
    text_file1.write("Starting time : " + str(end) + '\n')
    text_file1.write("Total execution time : " + str(total_time) + '\n')
    text_file1.close()

    fname2 = '../output/French/final_output1234/' + str(dataset) + '/Min_max_sentence'
    text_file2 = open(fname2, "w")
    text_file2.write("Minimum number of sentence taken : " + str(smin) + '\n')
    text_file2.write("Maximum number of sentence taken : " + str(smax) + '\n')
    text_file2.write("Number of sentence in the summary : " + str(count_summary_line) + '\n')
    text_file2.close()

    All_summary = []
    # record solution no. (same solution number for different annotators)

    solution_no = 0

    annotator1_sol_no = []
    ann1_datasetname = []
    ann1_rouge_1_p, ann1_rouge_1_r, ann1_rouge_1_f = [], [], []
    ann1_rouge_2_p, ann1_rouge_2_r, ann1_rouge_2_f = [], [], []
    ann1_rouge_su4_p, ann1_rouge_su4_r, ann1_rouge_su4_f = [], [], []

    avg_sol_no = []  # record solution number
    avg_dataset_name = []
    avg_rouge_su4_p, avg_rouge_su4_r, avg_rouge_su4_f = [], [], []
    avg_rouge_2_p, avg_rouge_2_r, avg_rouge_2_f = [], [], []
    avg_rouge_1_p, avg_rouge_1_r, avg_rouge_1_f = [], [], []

    for individual in final_population:
        Summary = ''
        features = individual.features

        # print "Summary {0}  : \n".format(solution_no)
        for j in range(len(features)):
            if features[j] == 1:
                if (len(str(Summary).split()) + len(str(actual_text_data[j]).split())) <= 30:
                    # position=Clean_tweet_positions[j]
                    # print(type(position))
                    # ones_position.append(j)
                    # print " Sentence number {0} :".format(j), actual_text_data[j]
                    if Summary == '':
                        Summary = actual_text_data[j]
                    else:
                        Summary += ' \n' + actual_text_data[j]  # .decode('utf-8', 'ignore')
        # print("Summary {0} :  ".format(solution_no), Summary)
        All_summary.append(Summary)
        if not os.path.isdir(
                '../output/French/final_output1234/' + str(
                    dataset) + '/' + 'Predicted_summary'):
            os.makedirs(
                '../output/French/final_output1234/' + str(dataset) + '/Predicted_summary')

        fname = '../output/French/final_output1234/' + str(
            dataset) + '/Predicted_summary/' + "Summary-{0}".format(solution_no)

        text_file = open(fname, "w")
        f_summary = Summary
        text_file.write(f_summary)

        rouge = PyRouge(rouge_n=(1, 2), rouge_l=False, rouge_su=True, skip_gap=4)

        # rouge = Rouge()

        Summary_for_test = Summary.lower().replace('.', '')
        # Summary_for_test = [Summary_for_test[:len(Summary_for_test) - 1]]

        actual_summary1_for_test = actual_summary1.lower().replace('.', '\n ')

        actual1_scores = rouge.evaluate([Summary_for_test], [[actual_summary1_for_test]])

        # actual1_scores = rouge.get_scores(Summary, actual_summary1)
        print("summary1 score :", actual1_scores)
        text_file.write('\n\nRouge score with annotator1 : \n' + str(actual1_scores) + '\n')

        """Record annotator1 score of all solutions to store in .csv"""
        annotator1_sol_no.append(solution_no)
        # dataset_name.append(dataset)
        # Annotator_no.append(1)
        ann1_datasetname.append(dataset)
        ann1_rouge_su4_p.append(actual1_scores['rouge-su4']['p'])
        ann1_rouge_su4_r.append(actual1_scores['rouge-su4']['r'])
        ann1_rouge_su4_f.append(actual1_scores['rouge-su4']['f'])

        # actual1_scores['rouge-su']['f']

        ann1_rouge_2_p.append(actual1_scores['rouge-2']['p'])
        ann1_rouge_2_r.append(actual1_scores['rouge-2']['r'])
        ann1_rouge_2_f.append(actual1_scores['rouge-2']['f'])

        ann1_rouge_1_p.append(actual1_scores['rouge-1']['p'])
        ann1_rouge_1_r.append(actual1_scores['rouge-1']['r'])
        ann1_rouge_1_f.append(actual1_scores['rouge-1']['f'])

        """Now calculating average score of annotators for current solution number to store into .csv file"""
        # print(type(actual1_scores))
        # print(type(actual1_scores[0]))

        rouge_avg_score_score = {'rouge-su4': {}, 'rouge-2': {}, 'rouge-1': {}}
        for k in range(len(actual1_scores.keys())):
            new_actual1_scores = list(actual1_scores.keys())
            key = new_actual1_scores[k]
            # print "key :", key
            m_key_val = actual1_scores[key]

            score = {
                k: (m_key_val.get(k, 0))
                for k in
                set(m_key_val) }

            # print score
            rouge_avg_score_score[key] = score

        avg_sol_no.append(solution_no)
        avg_dataset_name.append(dataset)
        avg_rouge_su4_p.append(rouge_avg_score_score['rouge-su4']['p'])
        avg_rouge_su4_r.append(rouge_avg_score_score['rouge-su4']['r'])
        avg_rouge_su4_f.append(rouge_avg_score_score['rouge-su4']['f'])

        avg_rouge_2_p.append(rouge_avg_score_score['rouge-2']['p'])
        avg_rouge_2_r.append(rouge_avg_score_score['rouge-2']['r'])
        avg_rouge_2_f.append(rouge_avg_score_score['rouge-2']['f'])

        avg_rouge_1_p.append(rouge_avg_score_score['rouge-1']['p'])
        avg_rouge_1_r.append(rouge_avg_score_score['rouge-1']['r'])
        avg_rouge_1_f.append(rouge_avg_score_score['rouge-1']['f'])
        """End of storing average rouge score for current solution number"""

        print("total rouge score of solution-{0} : ".format(solution_no), rouge_avg_score_score)
        print("==============================================================")
        solution_no += 1

    # ann1_max_R2_recall_index=ann1_rouge_2_r.index(max(ann1_rouge_2_r))
    # ann1_max_RL_recall_index=ann1_rouge_su4_r.index(max(ann1_rouge_su4_r))
    # ann1_datasetname.append('Max_R2_recal({0})'.format(annotator1_sol_no[ann1_max_R2_recall_index]))
    f1name_summ = '../output/French/final_output1234/' + str(
        dataset) + '/' + 'Annotator1_solutionwise_summary_score_overview.csv'
    df1 = pd.DataFrame({'dataset': ann1_datasetname, 'Solution no': annotator1_sol_no, 'rouge_1_p': ann1_rouge_1_p,
                        'rouge_1_r': ann1_rouge_1_r, 'rouge_1_f': ann1_rouge_1_f, 'rouge_2_p': ann1_rouge_2_p,
                        'rouge_2_r': ann1_rouge_2_r, 'rouge_2_f': ann1_rouge_2_f, 'rouge_su4_p': ann1_rouge_su4_p,
                        'rouge_su4_r': ann1_rouge_su4_r, 'rouge_su4_f': ann1_rouge_su4_f})
    df1.to_csv(f1name_summ)

    f5name_summ = '../output/French/final_output1234/' + str(
        dataset) + '/' + 'Average_summary_score_overview.csv'
    df5 = pd.DataFrame(
        {'dataset': avg_dataset_name, 'Solution no': avg_sol_no, 'rouge_1_p': avg_rouge_1_p, 'rouge_1_r': avg_rouge_1_r,
         'rouge_1_f': avg_rouge_1_f, 'rouge_2_p': avg_rouge_2_p, 'rouge_2_r': avg_rouge_2_r, 'rouge_2_f': avg_rouge_2_f,
         'rouge_su4_p': avg_rouge_su4_p, 'rouge_su4_r': avg_rouge_su4_r, 'rouge_su4_f': avg_rouge_su4_f})
    df5.to_csv(f5name_summ)

    results = ''

    # print('article No : {0} Fig No : {1}  Pop-size : {2}'.format(articleno, fig_number, pop_size))
    results += "Best Solution as per Avg. Max Rouge_1_precision: Solution no={0}, Rouge-1 precision score={1}, No. of tweet={2} \n".format(
        avg_rouge_1_p.index(max(avg_rouge_1_p)), avg_rouge_1_p[avg_rouge_1_p.index(max(avg_rouge_1_p))],
        K[avg_rouge_1_p.index(max(avg_rouge_1_p))])
    results += "Best Solution as per Avg. Max Rouge_1_recall: Solution no={0}, Rouge-1 recall score={1}, No. of tweet={2}\n".format(
        avg_rouge_1_r.index(max(avg_rouge_1_r)), avg_rouge_1_r[avg_rouge_1_r.index(max(avg_rouge_1_r))],
        K[avg_rouge_1_r.index(max(avg_rouge_1_r))])
    results += "Best Solution as per Avg. Max Rouge_1_F1: Solution no={0}, Rouge-1 F1 score={1}, No. of tweet={2}\n".format(
        avg_rouge_1_f.index(max(avg_rouge_1_f)), avg_rouge_1_f[avg_rouge_1_f.index(max(avg_rouge_1_f))],
        K[avg_rouge_1_f.index(max(avg_rouge_1_f))])

    results += "Best Solution as per Avg. Max Rouge_2_precision: Solution no={0}, Rouge-2 precision score={1}, No. of tweet={2}\n".format(
        avg_rouge_2_p.index(max(avg_rouge_2_p)), avg_rouge_2_p[avg_rouge_2_p.index(max(avg_rouge_2_p))],
        K[avg_rouge_2_p.index(max(avg_rouge_2_p))])
    results += "Best Solution as per Avg. Max Rouge_2_recall: Solution no={0}, Rouge-2 recall score={1}, No. of tweets={2}\n".format(
        avg_rouge_2_r.index(max(avg_rouge_2_r)), avg_rouge_2_r[avg_rouge_2_r.index(max(avg_rouge_2_r))],
        K[avg_rouge_2_r.index(max(avg_rouge_2_r))])
    results += "Best Solution as per Avg. Max Rouge_2_F1: Solution no={0}, Rouge-2 F1 score={1}, No. of tweet={2}\n".format(
        avg_rouge_2_f.index(max(avg_rouge_2_f)), avg_rouge_2_f[avg_rouge_2_f.index(max(avg_rouge_2_f))],
        K[avg_rouge_2_f.index(max(avg_rouge_2_f))])

    results += "Best Solution as per Avg. Max rouge_su4_precision: Solution no={0}, rouge-su4 precision score={1}, No. of tweet={2}\n".format(
        avg_rouge_su4_p.index(max(avg_rouge_su4_p)), avg_rouge_su4_p[avg_rouge_su4_p.index(max(avg_rouge_su4_p))],
        K[avg_rouge_su4_p.index(max(avg_rouge_su4_p))])
    results += "Best Solution as per Avg. Max rouge_su4_recall: Solution no={0}, rouge-su4 recall score={1}, No. of tweets={2}\n".format(
        avg_rouge_su4_r.index(max(avg_rouge_su4_r)), avg_rouge_su4_r[avg_rouge_su4_r.index(max(avg_rouge_su4_r))],
        K[avg_rouge_su4_r.index(max(avg_rouge_su4_r))])
    results += "Best Solution as per Avg. Max rouge_su4_F1: Solution no={0}, rouge-su4 F1 score={1}, No. of tweet={2}\n".format(
        avg_rouge_su4_f.index(max(avg_rouge_su4_f)), avg_rouge_su4_f[avg_rouge_su4_f.index(max(avg_rouge_su4_f))],
        K[avg_rouge_su4_f.index(max(avg_rouge_su4_f))])

    results += str(K)

    print(results)
    fname = '../output/French/final_output1234/' + str(
        dataset) + '/Best_resulting_Solutions.txt'

    text_file = open(fname, "w")
    ff_results = str(results.encode('utf-8'))
    text_file.write(ff_results)
    text_file.close()

    best_result_df = pd.DataFrame(
        {'Topic_name': dataset,
         'Rouge-1_Solution_no': avg_rouge_1_f.index(max(avg_rouge_1_f)),
         'Rouge-1_F1_score': avg_rouge_1_f[avg_rouge_1_f.index(max(avg_rouge_1_f))],
         'Rouge-1_no_of_tweet': K[avg_rouge_1_f.index(max(avg_rouge_1_f))],

         'Rouge-2_Solution_no': avg_rouge_2_f.index(max(avg_rouge_2_f)),
         'Rouge-2_F1_score': avg_rouge_2_f[avg_rouge_2_f.index(max(avg_rouge_2_f))],
         'Rouge-2_no_of_tweet': K[avg_rouge_2_f.index(max(avg_rouge_2_f))],

         'Rouge-su4_Solution_no': avg_rouge_su4_f.index(max(avg_rouge_su4_f)),
         'Rouge-su4_F1_score': avg_rouge_su4_f[avg_rouge_su4_f.index(max(avg_rouge_su4_f))],
         'Rouge-su4_no_of_tweet': K[avg_rouge_su4_f.index(max(avg_rouge_su4_f))]
         }, index=[0])

    best_result_df.to_csv(
        '../output/French/final_output1234/' + str(dataset) + '/Best_resulting_Solutions.csv',
        index=False)


if __name__ == "__main__":
    # Get the list of all files and directories
    #root_path = "C:\\Users\\VISHAL\\Documents\\french_dataset\\"

    topics_df = pd.read_csv('../french_topics_test.csv') #  'french_topics_text' for development, and 'french_topics_all' for all topics

    files_list = []
    files_name = []
    for ii in range(len(topics_df.topics)):
        files_list.append(topics_df.topics[ii])
        files_name.append(topics_df.topic_no[ii])

    for j in range(len(files_list)):
        print('Starting topic ' + str( j + 1) + ' : ' + str(files_list[j]))

    for i in range(len(files_list)):
    #for i in range(int(len(files_list)/2)):
    #for i in range(int(len(files_list) / 2), len(files_list)):
        print('\n *********************************************************\n')
        print('Starting topic ' + str( files_name[i]) + ' : ' + str(files_list[i]))
        run_comment_based_summarization(files_name[i])
        print('Finish!')
