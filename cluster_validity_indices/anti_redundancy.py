
import itertools

# Anti redundancy objective function



def Sent_to_sent(matrix_T2T_wmd, chromosome):
    '''
    matrix_t2t_wmd : Preprepared matrix with WMD distance between each pair of Tweets
    chromosome : A particular chosen solution
    Returns the WMD between each pair of sentence
    '''
    # Objective is to maximise the WMD between the sentences chosen
    sentChosen = []
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            sentChosen.append(i)

    val = 0
    counter =0
    a = list(itertools.combinations(sentChosen, 2))
    for k in a:
        val +=matrix_T2T_wmd[k[0]][k[1]]
        counter +=1
    # for sent in sentChosen:
    #     for sent2 in sentChosen:
    #         if sent != sent2:
    #             val += matrix_T2T_wmd[sent][sent2]
    #             counter+=1
    val /= float(counter)
    return val
