import math
from SMEA import seq
from SMEA.problems.problem_definitions import ProblemDefinitions

from cluster_validity_indices.anti_redundancy import Sent_to_sent


# from cluster_validity_indices.sent_to_caption import Sent_to_caption
# from cluster_validity_indices.sent_ref_fig import Sent_ref_fig
class ZDT3Definitions(ProblemDefinitions):

    def __init__(self, solution_length, text_data, SS_EMD_matrix, Tweet_length_Matrix,
                 reader_attention_matrix,
                 Density_based_score_matrix,
                 objective4_score_matrix,
                 objective5_score_matrix
                 ):
        self.n = solution_length
        self.Tweet_cleaned_data = text_data
        self.SS_WMD_matrix = SS_EMD_matrix
        self.Tweet_length_Matrix = Tweet_length_Matrix
        self.Tweet_reader_attention_matrix = reader_attention_matrix
        self.Density_based_score_matrix = Density_based_score_matrix
        self.Tweet_objective4_score_matrix = objective4_score_matrix
        self.Tweet_objective5_score_matrix = objective5_score_matrix

    def f1(self, individual):
        obj1 = Sent_to_sent(self.SS_WMD_matrix, individual.features)
        return obj1
        # obj2=Sent_to_caption(self.SC_WMD_matrix, individual.features, self.Fig_number)
        # return obj2
        # return individual.features[0]
        # obj1 = Sent_ref_fig(self.text_data, individual.features, self.Fig_number)
        # # print('obj1',obj1)
        # return obj1

    """
    def f2(self, individual):       #return sum of tweet length in the solution
        average_length = 0
        counter=0
        chromosome= individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                average_length+=self.Tweet_length_Matrix[i]
                counter+=1
        average_length=average_length/float(counter)
        return average_length


	"""

    def f2(self, individual):  # return sum of reader attention value of each tweet in the solution
        tweet_reader_attention_value = 0
        counter = 0
        chromosome = individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                tweet_reader_attention_value += self.Tweet_reader_attention_matrix[i]
                counter += 1
        tweet_reader_attention_value = float(tweet_reader_attention_value) / float(counter)
        return tweet_reader_attention_value

    def f3(self, individual):  # return sum of density based score value of each tweet in the solution
        tweet_density_based_score_value = 0
        counter = 0
        chromosome = individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                tweet_density_based_score_value += self.Density_based_score_matrix[i]
                counter += 1
        tweet_density_based_score_value = float(tweet_density_based_score_value) / float(counter)
        return tweet_density_based_score_value


    def f4(self, individual):  # return sum of Reader attention with tf-idf(RAWT) score value of each tweet in the solution
        tweet_objective4_score_value = 0
        counter = 0
        chromosome = individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                tweet_objective4_score_value += self.Tweet_objective4_score_matrix[i]
                counter += 1
        tweet_objective4_score_value = float(tweet_objective4_score_value) / float(counter)
        return tweet_objective4_score_value

    def f5(self, individual):  # return sum of named entity score value of each tweet in the solution
        tweet_objective5_score_value = 0
        counter = 0
        chromosome = individual.features
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                tweet_objective5_score_value += self.Tweet_objective5_score_matrix[i]
                counter += 1
        tweet_objective5_score_value = float(tweet_objective5_score_value) / float(counter)
        return tweet_objective5_score_value