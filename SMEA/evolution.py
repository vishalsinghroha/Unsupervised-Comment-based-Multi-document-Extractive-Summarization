"""Module with main parts of NSGA-II algorithm.
Contains main loop"""
from numpy import dtype

from SMEA.utils import NSGA2Utils
from SMEA.population import Population
# from SOM_Training.som_train import SOM_Training, SOM_Testing
# from som.SOM_Training import SOM_Training, SOM_Testing
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_3d
from functools import cmp_to_key
from mpl_toolkits.mplot3d import Axes3D
from copy import copy, deepcopy
import os
import plotly
import plotly.graph_objs as go
import numpy as np


class Evolution(object):
    def __init__(self, problem, num_of_generations, num_of_individuals, mating_pool_size):
        self.utils = NSGA2Utils(problem, num_of_individuals)
        self.population = None
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mating_pool_size = mating_pool_size

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)

    def my_new_cmp(self, a, b):
        result = (a > b) - (a < b)
        return result

    def evolve(self, EMD_matrix,
               MAX_reader_attention_matrix,
               MAX_density_based_score_matrix,
               MAX_objective4_score_matrix,
               MAX_objective5_score_matrix,
               MAX_TWEET_length_matrix,
               min_sen, max_sen,
               max_len_solution, dataset,
               actual_text_data, actual_summary1
               , actual_summary2
               , actual_summary3
               , actual_summary4
               ):

        self.population = self.utils.create_initial_population(min_sen, max_sen, max_len_solution)
        print("Population Initialization finished..........")
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)

        MAX_OBJ1_VALUE = []
        MAX_OBJ2_VALUE = []
        MAX_OBJ3_VALUE = []
        MAX_OBJ4_VALUE = []
        MAX_OBJ5_VALUE = []
        MAX_ROUGE1_VALUE = []
        MAX_ROUGE1_VALUE_INDEX = []  # store solution no. having avg. max. Rouge-1 score per generation
        MAX_ROUGE2_VALUE = []
        MAX_ROUGE2_VALUE_INDEX = []  # store solution no. having avg. max. Rouge-2 score per generation
        MAX_ROUGESU4_VALUE = []
        MAX_ROUGESU4_VALUE_INDEX = []  # store solution no. having avg. max. Rouge-SU4 score per generation
        GENERATION_NUMBER = []
        K1 = []
        results11 = ''
        new_sol_generated = []

        for i in range(0, int(self.num_of_generations)):
            old_population = deepcopy(self.population)
            old_pop = np.array([ii.features for ii in old_population])
            # print(old_pop)
            # print('article No : {0} Fig No : {1}  Pop-size : {2}\n\n\n\n\n\n'.format(articleno, fig_number, len(old_population)))

            print("Generation Number : ", i)

            children = self.utils.create_children(min_sen, max_sen, self.population, self.mating_pool_size, EMD_matrix,
                                                  MAX_reader_attention_matrix,
                                                  MAX_density_based_score_matrix,
                                                  MAX_objective4_score_matrix,
                                                  MAX_objective5_score_matrix,
                                                  MAX_TWEET_length_matrix, max_len_solution)
            # print('Generating children finished.................')
            self.population.extend(children)
            # print("length of merge population = ", (len(self.population)))
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            # some changes

            # fig = plt.figure(figsize=(10, 8))
            # ax1 = fig.add_subplot(221)
            # ax2 = fig.add_subplot(222)
            # ax3 = fig.add_subplot(223)
            # ax4 = fig1.add_subplot(224, projection='3d')

            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(111)

            fig_3d = plt_3d.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(222, projection='3d')

            while len(new_population) + len(self.population.fronts[front_num]) <= int(self.num_of_individuals):
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])

                xline = [jj.objectives[0] for jj in self.population.fronts[front_num]]
                yline = [ii.objectives[1] for ii in self.population.fronts[front_num]]
                zline = [ii.objectives[2] for ii in self.population.fronts[front_num]]
                wline = [ii.objectives[3] for ii in self.population.fronts[front_num]]
                pline = [ii.objectives[4] for ii in self.population.fronts[front_num]]

                magnified_pline = []
                for k in range(len(pline)):
                    magnified_pline.append(pline[k]*200)

                #ax1.scatter(xline, yline, label="fr-{}".format(front_num))

                # Creating plot
                ax_3d.scatter3D(xline, yline, zline,
                                c=wline,
                                cmap=plt.hot(),
                                s= magnified_pline,
                                label="fr-{}".format(front_num))


                front_num += 1

            if len(new_population) < int(self.num_of_individuals):
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                self.population.fronts[front_num] = sorted(self.population.fronts[front_num],
                                                           key=cmp_to_key(self.utils.crowding_operator), reverse=True)
                new_population.extend(
                    self.population.fronts[front_num][0:int(self.num_of_individuals) - len(new_population)])

                xline = [jj.objectives[0] for jj in self.population.fronts[front_num]]
                yline = [ii.objectives[1] for ii in self.population.fronts[front_num]]
                zline = [ii.objectives[2] for ii in self.population.fronts[front_num]]
                wline = [ii.objectives[3] for ii in self.population.fronts[front_num]]
                pline = [ii.objectives[4] for ii in self.population.fronts[front_num]]

                magnified_pline = []
                for k in range(len(pline)):
                    magnified_pline.append(pline[k] * 200)
                # Creating plot
                ax_3d.scatter3D(xline, yline, zline,
                                c=wline,
                                cmap=plt.hot(),
                                s= magnified_pline,
                                label="fr-{}".format(front_num))
                plt_3d.title(
                    "generation - " + str(i) + "\n" + "Anti-redundancy vs"
                                                       " Reader attention \n vs "
                                                      " Density based score \n "
                                                      "Reader attention with tf-idf (RAWT) score \n"
                                                      " vs Named entity score"
                )
                #fig_3d.colorbar(img_3d)
                ax_3d.set_xlabel("Anti-redundancy")
                ax_3d.set_ylabel("Reader attention")
                ax_3d.set_zlabel("Density based")
                ax_3d.legend()

                #ax1.scatter(xline, yline, label="fr-{}".format(front_num))
                #ax1.title.set_text("Anti-redundancy vs. Named entity score")
                #ax1.set_xlabel('Anti-redundancy')
                #ax1.set_ylabel('Named entity score')

                #fig.suptitle('generation - {}'.format(i))
                #plt.legend()

                name = '/Pareto_front/Pareto_fronts_generations_no' + str(i)
                if not os.path.isdir(
                        'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/Pareto_front'):
                    os.makedirs(
                        'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/Pareto_front')

                plt.savefig(
                    'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/' + name,
                    dpi=300, bbox_inches='tight')
                #plt.subplots_adjust(top=0.9, bottom=0.09, left=0.10, right=0.95, hspace=0.8, wspace=0.9)
                del ax_3d
                #del ax1
                # fig.tight_layout()
                # plt_3d.close()
                # plt_3d.clf()
                # plt_3d.cla()
                # del fig_3d
            # returned_population = self.population)

            prev_rep = 0
            new_pop = np.array([ii.features for ii in new_population])
            # print('new_pop',new_pop)
            for x in new_pop.tolist():
                if x in old_pop.tolist():
                    # print(x)
                    prev_rep += 1
            # print(len(new_population), len(old_population), prev_rep)
            results11 += 'new and repeated solutions generated in this generation-{0}  ---- {1} --- {2}\n'.format(i,
                                                                                                                  len(old_population) - prev_rep,
                                                                                                                  prev_rep)
            new_sol_generated.append(len(old_population) - prev_rep)

            self.population = new_population

            # print("no. of new training data samples at the end of generation {0}  : ".format(i), len(training_data))
            name = '/generation_wise_details'  # /gen_no' + str(i)
            if not os.path.isdir('C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + name):
                os.makedirs('C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + name)
            gen_folder = '/gen2' + str(i)
            os.makedirs(
                'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + name + gen_folder)

            # following code is to generate pareto front flots and also to record ROUGE score of each solution
            counter11 = 0  # solution number
            # print('I am here')
            annotator1_sol_no = []
            ann1_datasetname = []
            ann1_rouge_1_p, ann1_rouge_1_r, ann1_rouge_1_f = [], [], []
            ann1_rouge_2_p, ann1_rouge_2_r, ann1_rouge_2_f = [], [], []
            ann1_rouge_su4_p, ann1_rouge_su4_r, ann1_rouge_su4_f = [], [], []

            annotator2_sol_no = []
            ann2_datasetname = []
            ann2_rouge_1_p, ann2_rouge_1_r, ann2_rouge_1_f = [], [], []
            ann2_rouge_2_p, ann2_rouge_2_r, ann2_rouge_2_f = [], [], []
            ann2_rouge_su4_p, ann2_rouge_su4_r, ann2_rouge_su4_f = [], [], []

            annotator3_sol_no = []
            ann3_datasetname = []
            ann3_rouge_1_p, ann3_rouge_1_r, ann3_rouge_1_f = [], [], []
            ann3_rouge_2_p, ann3_rouge_2_r, ann3_rouge_2_f = [], [], []
            ann3_rouge_su4_p, ann3_rouge_su4_r, ann3_rouge_su4_f = [], [], []

            annotator4_sol_no = []
            ann4_datasetname = []
            ann4_rouge_1_p, ann4_rouge_1_r, ann4_rouge_1_f = [], [], []
            ann4_rouge_2_p, ann4_rouge_2_r, ann4_rouge_2_f = [], [], []
            ann4_rouge_su4_p, ann4_rouge_su4_r, ann4_rouge_su4_f = [], [], []

            avg_sol_no = []  # record solution number
            avg_dataset_name = []
            avg_rouge_su4_p, avg_rouge_su4_r, avg_rouge_su4_f = [], [], []
            avg_rouge_2_p, avg_rouge_2_r, avg_rouge_2_f = [], [], []
            avg_rouge_1_p, avg_rouge_1_r, avg_rouge_1_f = [], [], []

            ob1 = []
            ob2 = []
            ob3 = []
            ob4 = []
            ob5 = []

            k11 = []
            for k1 in self.population:
                f11 = open(
                    'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                        dataset) + name + gen_folder + '/predicted_summary' + str(counter11), 'w')
                Summary = ''
                # print("objectives :", k1.objectives)
                features = k1.features
                # print "Summary {0}  : \n".format(solution_no)
                for j in range(len(features)):
                    if features[j] == 1:
                        Summary += '\n' + actual_text_data[j]  # .decode('utf-8', 'ignore')
                encoded_summary = Summary
                encoded_summary = encoded_summary + ' '
                f11.write(str(encoded_summary.encode('utf-8',  errors='replace')))

                # from rouge import Rouge
                from rouge_metric import PyRouge
                rouge = PyRouge(rouge_n=(1, 2), rouge_l=False, rouge_su=True, skip_gap=4)

                # rouge = Rouge()

                Summary_for_test = Summary.lower().replace('.', '')

                actual_summary1_for_test = actual_summary1.lower().replace('.', '')

                #actual_summary1_for_test = str(actual_summary1).lower().replace('.', '\n ')
                actual1_scores = rouge.evaluate([Summary_for_test], [[actual_summary1_for_test]])

                # actual1_scores = rouge.get_scores(Summary, actual_summary1)

                print("summary1 score :", actual1_scores)
                f11.write('\n\nRouge score with annotator1 : \n' + str(actual1_scores) + '\n')
                annotator1_sol_no.append(counter11)
                ann1_datasetname.append(dataset)

                ann1_rouge_su4_p.append(actual1_scores['rouge-su4']['p'])
                ann1_rouge_su4_r.append(actual1_scores['rouge-su4']['r'])
                ann1_rouge_su4_f.append(actual1_scores['rouge-su4']['f'])

                ann1_rouge_2_p.append(actual1_scores['rouge-2']['p'])
                ann1_rouge_2_r.append(actual1_scores['rouge-2']['r'])
                ann1_rouge_2_f.append(actual1_scores['rouge-2']['f'])

                ann1_rouge_1_p.append(actual1_scores['rouge-1']['p'])
                ann1_rouge_1_r.append(actual1_scores['rouge-1']['r'])
                ann1_rouge_1_f.append(actual1_scores['rouge-1']['f'])

                actual_summary2_for_test = actual_summary2.lower().replace('.', '')

                actual2_scores = rouge.evaluate([Summary_for_test], [[actual_summary2_for_test]])
                # print("summary1 score :", actual1_scores)
                f11.write('\n\nRouge score with annotator2 : \n' + str(actual2_scores) + '\n')
                annotator2_sol_no.append(counter11)
                ann2_datasetname.append(dataset)
                ann2_rouge_su4_p.append(actual2_scores['rouge-su4']['p'])
                ann2_rouge_su4_r.append(actual2_scores['rouge-su4']['r'])
                ann2_rouge_su4_f.append(actual2_scores['rouge-su4']['f'])

                ann2_rouge_2_p.append(actual2_scores['rouge-2']['p'])
                ann2_rouge_2_r.append(actual2_scores['rouge-2']['r'])
                ann2_rouge_2_f.append(actual2_scores['rouge-2']['f'])

                ann2_rouge_1_p.append(actual2_scores['rouge-1']['p'])
                ann2_rouge_1_r.append(actual2_scores['rouge-1']['r'])
                ann2_rouge_1_f.append(actual2_scores['rouge-1']['f'])

                actual_summary3_for_test = actual_summary3.lower().replace('.', '')

                actual3_scores = rouge.evaluate([Summary_for_test], [[actual_summary3_for_test]])
                # print("summary1 score :", actual1_scores)
                f11.write('\n\nRouge score with annotator3 : \n' + str(actual3_scores) + '\n')

                annotator3_sol_no.append(counter11)
                ann3_datasetname.append(dataset)
                ann3_rouge_su4_p.append(actual3_scores['rouge-su4']['p'])
                ann3_rouge_su4_r.append(actual3_scores['rouge-su4']['r'])
                ann3_rouge_su4_f.append(actual3_scores['rouge-su4']['f'])

                ann3_rouge_2_p.append(actual3_scores['rouge-2']['p'])
                ann3_rouge_2_r.append(actual3_scores['rouge-2']['r'])
                ann3_rouge_2_f.append(actual3_scores['rouge-2']['f'])

                ann3_rouge_1_p.append(actual3_scores['rouge-1']['p'])
                ann3_rouge_1_r.append(actual3_scores['rouge-1']['r'])
                ann3_rouge_1_f.append(actual3_scores['rouge-1']['f'])

                actual_summary4_for_test = actual_summary4.lower().replace('.', '')

                actual4_scores = rouge.evaluate([Summary_for_test], [[actual_summary4_for_test]])
                # print("summary1 score :", actual1_scores)
                f11.write('\n\nRouge score with annotator4 : \n' + str(actual4_scores) + '\n')

                annotator4_sol_no.append(counter11)
                ann4_datasetname.append(dataset)
                ann4_rouge_su4_p.append(actual4_scores['rouge-su4']['p'])
                ann4_rouge_su4_r.append(actual4_scores['rouge-su4']['r'])
                ann4_rouge_su4_f.append(actual4_scores['rouge-su4']['f'])

                ann4_rouge_2_p.append(actual4_scores['rouge-2']['p'])
                ann4_rouge_2_r.append(actual4_scores['rouge-2']['r'])
                ann4_rouge_2_f.append(actual4_scores['rouge-2']['f'])

                ann4_rouge_1_p.append(actual4_scores['rouge-1']['p'])
                ann4_rouge_1_r.append(actual4_scores['rouge-1']['r'])
                ann4_rouge_1_f.append(actual4_scores['rouge-1']['f'])

                """Now calculating average score of annotators for current solution to store into .csv file"""
                rouge_avg_score_score = {'rouge-su4': {}, 'rouge-2': {}, 'rouge-1': {}}
                for k in range(len(actual1_scores.keys())):
                    new_temp1 = list(actual1_scores.keys())
                    key = new_temp1[k]
                    # print "key :", key
                    m_key_val = actual1_scores[key]
                    n_key_val = actual2_scores[key]
                    p_key_val = actual3_scores[key]
                    q_key_val = actual4_scores[key]
                    # score = {k: (m_key_val.get(k, 0) + n_key_val.get(k, 0) + p_key_val.get(k, 0) + q_key_val.get(k,
                    #                                                                                             0)) / float(
                    #    4) for k in
                    #         set(m_key_val) & set(n_key_val) & set(p_key_val) & set(q_key_val)}

                    score = {k:  max(m_key_val.get(k, 0)
                                 , n_key_val.get(k, 0)
                                 , p_key_val.get(k, 0)
                                 , q_key_val.get(k, 0)
                                 )
                             for k in
                             set(m_key_val)
                             & set(n_key_val)
                             & set(p_key_val)
                             & set(q_key_val)
                             }
                    # print score
                    rouge_avg_score_score[key] = score

                f11.write('\n\naverage Rouge score : \n' + str(rouge_avg_score_score) + '\n')

                avg_sol_no.append(counter11)
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

                f11.write('\n\nAnti-redundancy objective value :\t' + str(k1.objectives[0]))
                f11.write('\nAverage Reader attention score objective value :\t' + str(
                    k1.objectives[1]))
                f11.write('\nAverage Density based score objective value :\t' + str(
                    k1.objectives[2]))
                f11.write('\nAverage Reader attention with tf-idf (RAWT) score objective value :\t' + str(k1.objectives[3]))
                f11.write('\nAverage Named entity score value :\t' + str(k1.objectives[4]))

                ob1.append(k1.objectives[0])
                ob2.append(k1.objectives[1])
                ob3.append((k1.objectives[2]))
                ob4.append((k1.objectives[3]))
                ob5.append((k1.objectives[4]))
                # print "ob2 :", ob2[0]

                k11.append(k1.K)
                counter11 += 1

            import pandas as pd
            f1name_summ = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Gen{0}'.format(
                i) + 'Annotator1_solutionwise_summary_score_overview.csv'
            df1 = pd.DataFrame(
                {'dataset': ann1_datasetname, 'Solution no': annotator1_sol_no, 'rouge_1_p': ann1_rouge_1_p,
                 'rouge_1_r': ann1_rouge_1_r, 'rouge_1_f': ann1_rouge_1_f, 'rouge_2_p': ann1_rouge_2_p,
                 'rouge_2_r': ann1_rouge_2_r, 'rouge_2_f': ann1_rouge_2_f, 'rouge_su4_p': ann1_rouge_su4_p,
                 'rouge_su4_r': ann1_rouge_su4_r, 'rouge_su4_f': ann1_rouge_su4_f})
            df1.to_csv(f1name_summ)

            f2name_summ = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Gen{0}'.format(
                i) + 'Annotator2_solutionwise_summary_score_overview.csv'
            df2 = pd.DataFrame(
                {'dataset': ann2_datasetname, 'Solution no': annotator2_sol_no, 'rouge_1_p': ann2_rouge_1_p,
                 'rouge_1_r': ann2_rouge_1_r, 'rouge_1_f': ann2_rouge_1_f, 'rouge_2_p': ann2_rouge_2_p,
                 'rouge_2_r': ann2_rouge_2_r, 'rouge_2_f': ann2_rouge_2_f, 'rouge_su4_p': ann2_rouge_su4_p,
                 'rouge_su4_r': ann2_rouge_su4_r, 'rouge_su4_f': ann2_rouge_su4_f})
            df2.to_csv(f2name_summ)

            f3name_summ = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Gen{0}'.format(
                i) + 'Annotator3_solutionwise_summary_score_overview.csv'
            df3 = pd.DataFrame(
                {'dataset': ann3_datasetname, 'Solution no': annotator3_sol_no, 'rouge_1_p': ann3_rouge_1_p,
                 'rouge_1_r': ann3_rouge_1_r, 'rouge_1_f': ann3_rouge_1_f, 'rouge_2_p': ann3_rouge_2_p,
                 'rouge_2_r': ann3_rouge_2_r, 'rouge_2_f': ann3_rouge_2_f, 'rouge_su4_p': ann3_rouge_su4_p,
                 'rouge_su4_r': ann3_rouge_su4_r, 'rouge_su4_f': ann3_rouge_su4_f})
            df3.to_csv(f3name_summ)

            f4name_summ = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Gen{0}'.format(
                i) + 'Annotator4_solutionwise_summary_score_overview.csv'
            df4 = pd.DataFrame(
                {'dataset': ann4_datasetname, 'Solution no': annotator4_sol_no, 'rouge_1_p': ann4_rouge_1_p,
                 'rouge_1_r': ann4_rouge_1_r, 'rouge_1_f': ann4_rouge_1_f, 'rouge_2_p': ann4_rouge_2_p,
                 'rouge_2_r': ann4_rouge_2_r, 'rouge_2_f': ann4_rouge_2_f, 'rouge_su4_p': ann4_rouge_su4_p,
                 'rouge_su4_r': ann4_rouge_su4_r, 'rouge_su4_f': ann4_rouge_su4_f})
            df4.to_csv(f4name_summ)

            f5name_summ = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Gen{0}'.format(
                i) + 'Average_summary_score_overview.csv'
            df5 = pd.DataFrame({'dataset': avg_dataset_name, 'Solution no': avg_sol_no, 'rouge_1_p': avg_rouge_1_p,
                                'rouge_1_r': avg_rouge_1_r, 'rouge_1_f': avg_rouge_1_f, 'rouge_2_p': avg_rouge_2_p,
                                'rouge_2_r': avg_rouge_2_r, 'rouge_2_f': avg_rouge_2_f, 'rouge_su4_p': avg_rouge_su4_p,
                                'rouge_su4_r': avg_rouge_su4_r, 'rouge_su4_f': avg_rouge_su4_f})
            df5.to_csv(f5name_summ)

            results = ''

            # print('article No : {0} Fig No : {1}  Pop-size : {2}'.format(articleno, fig_number, pop_size))
            results += "Best Solution as per Avg. Max Rouge_1_precision: Solution no={0}, Rouge-1 precision score={1}, No. of tweet={2} \n".format(
                avg_rouge_1_p.index(max(avg_rouge_1_p)), avg_rouge_1_p[avg_rouge_1_p.index(max(avg_rouge_1_p))],
                k11[avg_rouge_1_p.index(max(avg_rouge_1_p))])
            results += "Best Solution as per Avg. Max Rouge_1_recall: Solution no={0}, Rouge-1 recall score={1}, No. of tweet={2}\n".format(
                avg_rouge_1_r.index(max(avg_rouge_1_r)), avg_rouge_1_r[avg_rouge_1_r.index(max(avg_rouge_1_r))],
                k11[avg_rouge_1_r.index(max(avg_rouge_1_r))])
            results += "Best Solution as per Avg. Max Rouge_1_F1: Solution no={0}, Rouge-1 F1 score={1}, No. of tweet={2}\n".format(
                avg_rouge_1_f.index(max(avg_rouge_1_f)), avg_rouge_1_f[avg_rouge_1_f.index(max(avg_rouge_1_f))],
                k11[avg_rouge_1_f.index(max(avg_rouge_1_f))])

            results += "Best Solution as per Avg. Max Rouge_2_precision: Solution no={0}, Rouge-2 precision score={1}, No. of tweet={2}\n".format(
                avg_rouge_2_p.index(max(avg_rouge_2_p)), avg_rouge_2_p[avg_rouge_2_p.index(max(avg_rouge_2_p))],
                k11[avg_rouge_2_p.index(max(avg_rouge_2_p))])
            results += "Best Solution as per Avg. Max Rouge_2_recall: Solution no={0}, Rouge-2 recall score={1}, No. of tweets={2}\n".format(
                avg_rouge_2_r.index(max(avg_rouge_2_r)), avg_rouge_2_r[avg_rouge_2_r.index(max(avg_rouge_2_r))],
                k11[avg_rouge_2_r.index(max(avg_rouge_2_r))])
            results += "Best Solution as per Avg. Max Rouge_2_F1: Solution no={0}, Rouge-2 F1 score={1}, No. of tweet={2}\n".format(
                avg_rouge_2_f.index(max(avg_rouge_2_f)), avg_rouge_2_f[avg_rouge_2_f.index(max(avg_rouge_2_f))],
                k11[avg_rouge_2_f.index(max(avg_rouge_2_f))])

            results += "Best Solution as per Avg. Max rouge_su4_precision: Solution no={0}, rouge-su4 precision score={1}, No. of tweet={2}\n".format(
                avg_rouge_su4_p.index(max(avg_rouge_su4_p)),
                avg_rouge_su4_p[avg_rouge_su4_p.index(max(avg_rouge_su4_p))],
                k11[avg_rouge_su4_p.index(max(avg_rouge_su4_p))])
            results += "Best Solution as per Avg. Max rouge_su4_recall: Solution no={0}, rouge-su4 recall score={1}, No. of tweets={2}\n".format(
                avg_rouge_su4_r.index(max(avg_rouge_su4_r)),
                avg_rouge_su4_r[avg_rouge_su4_r.index(max(avg_rouge_su4_r))],
                k11[avg_rouge_su4_r.index(max(avg_rouge_su4_r))])
            results += "Best Solution as per Avg. Max rouge_su4_F1: Solution no={0}, rouge-su4 F1 score={1}, No. of tweet={2}\n".format(
                avg_rouge_su4_f.index(max(avg_rouge_su4_f)),
                avg_rouge_su4_f[avg_rouge_su4_f.index(max(avg_rouge_su4_f))],
                k11[avg_rouge_su4_f.index(max(avg_rouge_su4_f))])

            results += str(k11)

            # print(results)
            fname = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
                dataset) + name + gen_folder + '/Best_resulting_Solutions.txt'
            text_file = open(fname, "w")
            new_results = str(results.encode('utf-8'))
            text_file.write(new_results)
            text_file.close()

            MAX_OBJ1_VALUE.append(max(ob1))
            MAX_OBJ2_VALUE.append(max(ob2))
            MAX_OBJ3_VALUE.append(max(ob3))
            MAX_OBJ4_VALUE.append(max(ob4))
            MAX_OBJ5_VALUE.append(max(ob5))

            index11 = avg_rouge_1_f.index(max(avg_rouge_1_f))
            index12 = avg_rouge_2_f.index(max(avg_rouge_2_f))
            index13 = avg_rouge_su4_f.index(max(avg_rouge_su4_f))
            MAX_ROUGE1_VALUE.append(avg_rouge_1_f[index11])
            MAX_ROUGE1_VALUE_INDEX.append(index11)
            MAX_ROUGE2_VALUE.append(avg_rouge_2_f[index12])
            MAX_ROUGE2_VALUE_INDEX.append(index12)
            MAX_ROUGESU4_VALUE.append(avg_rouge_su4_f[index13])
            MAX_ROUGESU4_VALUE_INDEX.append(index13)

            GENERATION_NUMBER.append(i)
            # print('anti-redundancy : ', ob1)
            # print('Tweet_length : ', ob2)
            # print('Sent_ref_figure : ', ob3)
            # print("K : ", k11)
            K1 = k11

            print("Generation {0} finished ".format(i))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=+++++++++++++++=")
        """PLOT MAX VALUE OF OBJECTIVE FUNCTIONS"""

        fname = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
            dataset) + '/newSolnsfromMergedPop-Gen{0}.txt'.format(self.num_of_generations)
        text_file = open(fname, "w")
        final_results11 = str(results11.encode('utf-8'))
        text_file.write(final_results11)
        text_file.close()

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(GENERATION_NUMBER, MAX_ROUGE1_VALUE, label='Rouge-1')
        plt.plot(GENERATION_NUMBER, MAX_ROUGE1_VALUE)
        plt.scatter(GENERATION_NUMBER, MAX_ROUGE2_VALUE, label='Rouge-2')
        plt.plot(GENERATION_NUMBER, MAX_ROUGE2_VALUE)
        plt.scatter(GENERATION_NUMBER, MAX_ROUGESU4_VALUE, label='rouge-su4')
        plt.plot(GENERATION_NUMBER, MAX_ROUGESU4_VALUE)
        plt.legend()
        plt.xlabel('Generation Number')
        plt.ylabel('Rouge-score')
        name = str(dataset) + 'Generation Wise Rouge score'
        plt.savefig('C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/' + name, dpi=300)
        plt.close('all')

        import matplotlib.pyplot as plt1
        fig = plt1.figure(figsize=(10, 8))
        plt1.scatter(GENERATION_NUMBER, new_sol_generated)
        plt1.plot(GENERATION_NUMBER, new_sol_generated)
        plt1.xlabel("Generation Number")
        plt1.ylabel('New Sols Generated')
        fig.suptitle('Number of new good solutions proceeding to next generation', fontsize=8)
        name = str(dataset) + 'New Sols_vs_Generations'
        plt1.savefig('C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/' + name, dpi=300)
        plt1.close('all')

        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(10, 8))
        plt2.figure(figsize=(10, 8))
        plt2.plot(GENERATION_NUMBER, MAX_OBJ1_VALUE)
        plt2.plot(GENERATION_NUMBER, MAX_OBJ2_VALUE)  #
        plt2.plot(GENERATION_NUMBER, MAX_OBJ3_VALUE)
        plt2.plot(GENERATION_NUMBER, MAX_OBJ4_VALUE)
        plt2.plot(GENERATION_NUMBER, MAX_OBJ5_VALUE)
        # plt2.tight_layout()
        plt2.grid(alpha=0.6)
        # plt.set_title('MAX_OBJECTIVE FUNCTION VALUES', color='red')
        plt2.xlabel("Generation Number")
        plt2.ylabel('Objective functions')
        # 'Objective-3: Sent. ref fig '
        # plt.legend(['Objective-1: anti-redundancy', 'Objective-2: Sent. to Fig. Caption'], loc='upper left')
        # plt.legend(['Objective-1: anti-redundancy', 'Objective-2: Sent_Ref-Fig'], loc='upper left')
        plt2.legend(['Objective-1: anti-redundancy',
                     'Objective-2: Reader attention value',
                     'Objective-3: Density based score',
                     'Objective-4: Reader attention with tf-idf (RAWT) score',
                     'Objective-5: Named entity score'
                     ])
        name = '/Generation_wise_Objective_values_' + str(dataset)
        plt2.savefig('C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(dataset) + '/' + name, dpi=300)

        plt2.close('all')

        print("max obj1 value :", MAX_OBJ1_VALUE)
        print("max ob2 value :", MAX_OBJ2_VALUE)
        print("max ob3 value :", MAX_OBJ3_VALUE)
        print("max ob4 value :", MAX_OBJ4_VALUE)
        print("max ob5 value :", MAX_OBJ5_VALUE)
        print("final population length : ", len(self.population))

        import pandas as pd

        index_R1 = MAX_ROUGE1_VALUE.index(
            max(MAX_ROUGE1_VALUE))  # solution no. having Max. ROUGE-1 value overall generation
        index_R2 = MAX_ROUGE2_VALUE.index(
            max(MAX_ROUGE2_VALUE))  # solution no. having Max. ROUGE-2 value overall generation
        index_SU4 = MAX_ROUGESU4_VALUE.index(
            max(MAX_ROUGESU4_VALUE))  # solution no. having Max. rouge-su4 value overall generation

        GENERATION_NUMBER.append('Max. Score')
        MAX_ROUGE1_VALUE_INDEX.append(MAX_ROUGE1_VALUE_INDEX[index_R1])
        MAX_ROUGE1_VALUE.append(MAX_ROUGE1_VALUE[index_R1])

        MAX_ROUGE2_VALUE_INDEX.append(MAX_ROUGE2_VALUE_INDEX[index_R2])
        MAX_ROUGE2_VALUE.append(MAX_ROUGE2_VALUE[index_R2])

        MAX_ROUGESU4_VALUE_INDEX.append(MAX_ROUGESU4_VALUE_INDEX[index_SU4])
        MAX_ROUGESU4_VALUE.append(MAX_ROUGESU4_VALUE[index_SU4])

        f5name = 'C:/Users/VISHAL/Documents/final_moo_outputs/compressive/version_1/final_output12345/' + str(
            dataset) + '/Max_Avg_ROUGE_score_generation_wise.csv'
        df5 = pd.DataFrame({'generation_no': GENERATION_NUMBER, 'solution No._R1': MAX_ROUGE1_VALUE_INDEX,
                            'rouge_1_f': MAX_ROUGE1_VALUE, 'solution No._R2': MAX_ROUGE2_VALUE_INDEX,
                            'rouge_2_f': MAX_ROUGE2_VALUE, 'Solution No._RSU4': MAX_ROUGESU4_VALUE_INDEX,
                            'rouge_su4_f': MAX_ROUGESU4_VALUE})
        df5.to_csv(f5name, index=False)

        return self.population, K1
