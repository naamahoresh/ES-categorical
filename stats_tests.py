import glob, os
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
import time


def normality_for_one_file():
    main_path_to_read = "/home/naamah/Documents/result_CATES/GECCO/result GECCO/result_Surrogate_vs_DOE/"
    fileLoc_general="/home/naamah/Documents/CatES/result_All/GECCO/Histogram"


    problem_list = ["LABS", "Ising1D", "Ising2DSquare", "MIS","NQueens"]
    sigma= 0.05

    columns = ['Problem','population','Conditions', 'dim', 'K^2 P-value','K^2 Is Normality' ,'shapiro P-value', 'shapiro Is Normality','kstest P-value', 'kstest Is Normality']

    for problem in problem_list:
        df_normal = pd.DataFrame(columns=columns)
        index_df = 0
        path = main_path_to_read+problem+"/IOHProfiler"
        problem_folder_name_to_save = fileLoc_general+"/"+problem

        if not os.path.exists(problem_folder_name_to_save):
            os.makedirs(problem_folder_name_to_save)

        for file_name in glob.glob(path+"*/IOHprofiler_*.info"):
            file = open(file_name, "r")
            text= file.readlines()
            file.close()

            all_conditions_start = file_name.index("/IOHProfiler_")
            all_conditions_end =  file_name.index("/IOHprofiler_")
            all_condition = file_name[all_conditions_start+13:all_conditions_end]
            all_conditions_file_name = fileLoc_general+"/"+problem+"/"+all_condition
            if not os.path.exists(all_conditions_file_name):
                os.makedirs(all_conditions_file_name)

            exp_condition_index = text[0].index("algId = ")+9
            exp_condition_substr=text[0][exp_condition_index:]
            exp_condition_end_index = exp_condition_substr.index("'")
            exp_condition = exp_condition_substr[:exp_condition_end_index]

            for i in range(int(len(text)/3)):
                index_line = i*3+2
                index_list = [j for j, x in enumerate(text[index_line]) if x == "|"]
                space_list = [j for j, x in enumerate(text[index_line]) if x == " "]
                space_list=[j for j in space_list if j > index_list[0]]
                space_list.append(-1)

                fitness_list = []
                for ind in range(len(index_list)):
                    fitness_list.append(float(text[index_line][index_list[ind]+1:space_list[ind]-1]))

                dim_index = text[index_line-2].index("DIM = ")+6
                dim_substr = text[index_line-2][dim_index:]
                dim_end_index = dim_substr.index(" ")
                dim = dim_substr[:dim_end_index-1]

                is_norm_statistic_k, is_norm_p_k = stats.normaltest(fitness_list)
                is_norm_statistic_shapiro, is_norm_p_shapiro = stats.shapiro(fitness_list)
                is_norm_statistic_kstest, is_norm_p_kstest = stats.kstest(fitness_list, 'norm')
                print(is_norm_p_k)
                print(is_norm_p_shapiro)
                print(is_norm_p_kstest)
                print("\n")

                df_normal.loc[index_df]=[problem, str(exp_condition), all_condition, dim,round(is_norm_p_k,4),sigma<is_norm_p_k,round(is_norm_p_shapiro,4), sigma<is_norm_p_shapiro,round(is_norm_p_kstest,4), sigma<is_norm_p_kstest]
                index_df=index_df+1

                fig = plt.figure()
                plt.hist(fitness_list)
                plt.title("Normality Histogram \n{} - {} (dim {})\nk^2 - P value: {} ({})\nshapiro - P value: {} ({})\nkstest - P value: {} ({})".format(problem, str(exp_condition),str(dim),round(is_norm_p_k,4),sigma<is_norm_p_k,round(is_norm_p_shapiro,4), sigma<is_norm_p_shapiro,round(is_norm_p_kstest,4), sigma<is_norm_p_kstest),fontsize=12)
                fig.tight_layout()

                stri =all_conditions_file_name+ "/hist_" + problem + "_d" + str(dim) + "_" + str(exp_condition)
                plt.savefig(stri)
                plt.close()

        with open(fileLoc_general+"/Normality_{}.csv".format(problem), 'a') as f:
            df_normal.to_csv(f, header=True)



def normality_for_two_file():
    main_path_to_read = "/home/naamah/Documents/CatES/result_All/normality/FROM 2 FILES/data1/"
    fileLoc_general = "/home/naamah/Documents/CatES/result_All/normality/FROM 2 FILES/Histogram"

    problem_list = ["LABS", "Ising1D", "Ising2DSquare", "MIS", "NQueens"]
    sigma = 0.05

    columns = ['Problem', 'population', 'Conditions', 'dim', 'K^2 P-value', 'K^2 Is Normality', 'shapiro P-value',
               'shapiro Is Normality', 'kstest P-value', 'kstest Is Normality']

    for problem in problem_list:
        df_normal = pd.DataFrame(columns=columns)
        index_df = 0
        path = main_path_to_read + problem + "/IOHProfiler"
        problem_folder_name_to_save = fileLoc_general + "/" + problem

        if not os.path.exists(problem_folder_name_to_save):
            os.makedirs(problem_folder_name_to_save)

        for file_name in glob.glob(path + "*/IOHprofiler_*.info"):
            file = open(file_name, "r")
            text = file.readlines()
            file.close()
            file_name_second_file=file_name.replace("data1","data2")
            file_2 = open(file_name_second_file, "r")
            text_2 = file_2.readlines()
            file_2.close()

            all_conditions_start = file_name.index("/IOHProfiler_")
            all_conditions_end = file_name.index("/IOHprofiler_")
            all_condition = file_name[all_conditions_start + 13:all_conditions_end]
            all_conditions_file_name = fileLoc_general + "/" + problem + "/" + all_condition
            if not os.path.exists(all_conditions_file_name):
                os.makedirs(all_conditions_file_name)

            exp_condition_index = text[0].index("algId = ") + 9
            exp_condition_substr = text[0][exp_condition_index:]
            exp_condition_end_index = exp_condition_substr.index("'")
            exp_condition = exp_condition_substr[:exp_condition_end_index]

            for i in range(int(len(text) / 3)):
                index_line = i * 3 + 2
                index_list_1 = [j for j, x in enumerate(text[index_line]) if x == "|"]
                space_list_1 = [j for j, x in enumerate(text[index_line]) if x == " "]
                space_list_1 = [j for j in space_list_1 if j > index_list_1[0]]
                space_list_1.append(-1)

                index_list_2 = [j for j, x in enumerate(text_2[index_line]) if x == "|"]
                space_list_2 = [j for j, x in enumerate(text_2[index_line]) if x == " "]
                space_list_2 = [j for j in space_list_2 if j > index_list_2[0]]
                space_list_2.append(-1)

                fitness_list = []
                for ind in range(len(index_list_1)):
                    fitness_list.append(float(text[index_line][index_list_1[ind] + 1:space_list_1[ind] - 1]))
                for ind in range(len(index_list_2)):
                    fitness_list.append(float(text_2[index_line][index_list_2[ind] + 1:space_list_2[ind] - 1]))

                dim_index = text[index_line - 2].index("DIM = ") + 6
                dim_substr = text[index_line - 2][dim_index:]
                dim_end_index = dim_substr.index(" ")
                dim = dim_substr[:dim_end_index - 1]

                is_norm_statistic_k, is_norm_p_k = stats.normaltest(fitness_list)
                is_norm_statistic_shapiro, is_norm_p_shapiro = stats.shapiro(fitness_list)
                is_norm_statistic_kstest, is_norm_p_kstest = stats.kstest(fitness_list, 'norm')
                print(is_norm_p_k)
                print(is_norm_p_shapiro)
                print(is_norm_p_kstest)
                print("\n")

                df_normal.loc[index_df] = [problem, str(exp_condition), all_condition, dim, round(is_norm_p_k, 4),
                                           sigma < is_norm_p_k, round(is_norm_p_shapiro, 4), sigma < is_norm_p_shapiro,
                                           round(is_norm_p_kstest, 4), sigma < is_norm_p_kstest]
                index_df = index_df + 1

                fig = plt.figure()
                plt.hist(fitness_list)
                plt.title(
                    "Normality Histogram \n{} - {} (dim {})\nk^2 - P value: {} ({})\nshapiro - P value: {} ({})\nkstest - P value: {} ({})".format(
                        problem, str(exp_condition), str(dim), round(is_norm_p_k, 4), sigma < is_norm_p_k,
                        round(is_norm_p_shapiro, 4), sigma < is_norm_p_shapiro, round(is_norm_p_kstest, 4),
                        sigma < is_norm_p_kstest), fontsize=12)
                fig.tight_layout()

                stri = all_conditions_file_name + "/hist_" + problem + "_d" + str(dim) + "_" + str(exp_condition)
                plt.savefig(stri)
                plt.close()

        with open(fileLoc_general + "/Normality_{}.csv".format(problem), 'a') as f:
            df_normal.to_csv(f, header=True)


def epsilon_test():
    DIMENSION = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]

    for dim in DIMENSION:
        times = []
        for ind in range(100):
            start = time.time()

            X_train = np.random.randint(2, size=(630, dim))
            y_train = [sum(i) for i in X_train]

            X_test = np.random.randint(2, size=(30, dim))

            clf = svm.SVR(kernel='rbf', C=307, epsilon=2, gamma=0.1223, shrinking=False, degree=3)
            clf.fit(X_train, y_train)
            Y_test = clf.predict(X_test)
            stop = time.time()

            times.append((stop - start))

        print("Dim: {}  -  median Time: {} (overall time: {})".format(dim, np.median(times), np.sum(times)))




def NonParams_test():
    main_path_to_read = "/home/naamah/Documents/CatES/result_All/GECCO/result GECCO/result_Surrogate_vs_DOE/"
    file_location_to_save="/home/naamah/Documents/CatES/result_All/GECCO/extra_tests/non_parametric_tests/with_small_epsilon/"
    # epsilon = 1
    # main_path_to_read = "/home/naamah/Documents/CatES/result_All/GECCO/extra_tests/epsilon/epsilon_{}/".format(epsilon)
    # file_location_to_save="/home/naamah/Documents/CatES/result_All/GECCO/tmp/non_parametric_tests/test_{}".format(epsilon)

    problem_list = ["LABS", "Ising1D", "Ising2DSquare", "MIS", "NQueens"]
    sigma= 0.05

    columns = ['Conditions', 'dim', 'Is sig by friedman?','friedman P-value', 'friedman Stats']
    columns_wilc = ['Conditions', 'dim', 'Condition 1','Condition 2','Is sig by Wilcoxon?','Wilcoxo P-value', 'Wilcoxo Stats']
    #
    # for folder_name in glob.glob("/home/naamah/Documents/CatES/result_All/GECCO/result GECCO/result_Surrogate_vs_DOE/MIS/*/IOHprofiler_*.info"):
    #     print(folder_name)

    for problem in problem_list:
        df_friedman = pd.DataFrame(columns=columns)
        index_df = 0

        df_wilcoxon = pd.DataFrame(columns=columns_wilc)
        index_df_wilcoxon = 0

        read_path_by_problem = main_path_to_read+problem+"/IOHProfiler"
        # read_path_by_problem = main_path_to_read+problem

        if not os.path.exists(file_location_to_save):
            os.makedirs(file_location_to_save)


        for folder_name in glob.glob(read_path_by_problem + "_{}*".format(problem)):
        # for folder_name in glob.glob(read_path_by_problem + "/*/IOHProfiler"):

            fitness_list_25 = []
            fitness_list_64 = []
            fitness_list_100 = []
            fitness_list_25_condition=[]
            fitness_list_64_condition=[]
            fitness_list_100_condition=[]


            for file_name in glob.glob(folder_name+"*/IOHprofiler_*.info"):
                file = open(file_name, "r")
                text= file.readlines()
                file.close()

                all_conditions_start = file_name.index("/IOHProfiler_")
                all_conditions_end = file_name.index("/IOHprofiler_")
                all_conditions = file_name[all_conditions_start+13:all_conditions_end]


                exp_condition_index = text[0].index("algId = ")+9
                exp_condition_substr=text[0][exp_condition_index:]
                exp_condition_end_index = exp_condition_substr.index("'")
                exp_condition = exp_condition_substr[:exp_condition_end_index]



                for i in range(int(len(text)/3)):
                    index_line = i*3+2
                    index_list = [j for j, x in enumerate(text[index_line]) if x == "|"]
                    space_list = [j for j, x in enumerate(text[index_line]) if x == " "]
                    space_list=[j for j in space_list if j > index_list[0]]
                    space_list.append(-1)

                    dim_index = text[index_line - 2].index("DIM = ") + 6
                    dim_substr = text[index_line - 2][dim_index:]
                    dim_end_index = dim_substr.index(" ")
                    dim = dim_substr[:dim_end_index - 1]

                    fitness_list = []
                    for ind in range(len(index_list)):
                        fitness_list.append(float(text[index_line][index_list[ind]+1:space_list[ind]-1]))


                    if int(dim)==25:
                        fitness_list_25.append(fitness_list)
                        fitness_list_25_condition.append(exp_condition)
                    elif int(dim)==64:
                        fitness_list_64.append(fitness_list)
                        fitness_list_64_condition.append(exp_condition)
                    else:
                        fitness_list_100.append(fitness_list)
                        fitness_list_100_condition.append(exp_condition)


            #The stats tests
            if (len(fitness_list_25)>0):
                    Stats_25, p_25= stats.friedmanchisquare(*fitness_list_25)
                    df_friedman.loc[index_df] = [all_conditions,25,sigma > p_25,p_25, Stats_25]
                    index_df=index_df+1

                    for x in range(len(fitness_list_25)):
                        for y in range(x+1,len(fitness_list_25)):
                            wilc_stats, silc_p=stats.wilcoxon(fitness_list_25[x],fitness_list_25[y])
                            df_wilcoxon.loc[index_df_wilcoxon]=[all_conditions,25,fitness_list_25_condition[x], fitness_list_25_condition[y],sigma > silc_p,silc_p, wilc_stats]
                            index_df_wilcoxon=index_df_wilcoxon+1

            if (len(fitness_list_64)>0):
                    Stats_64, p_64= stats.friedmanchisquare(*fitness_list_64)
                    df_friedman.loc[index_df] = [all_conditions,64,sigma > p_64,p_64,Stats_64]
                    index_df=index_df+1

                    for x in range(len(fitness_list_64)):
                        for y in range(x+1,len(fitness_list_64)):
                            wilc_stats, silc_p=stats.wilcoxon(fitness_list_64[x],fitness_list_64[y])
                            df_wilcoxon.loc[index_df_wilcoxon]=[all_conditions,64,fitness_list_64_condition[x],fitness_list_64_condition[y],sigma > silc_p,silc_p, wilc_stats]
                            index_df_wilcoxon=index_df_wilcoxon+1

            if (len(fitness_list_100)>0):
                    Stats_100,p_100= stats.friedmanchisquare(*fitness_list_100)
                    df_friedman.loc[index_df] = [all_conditions,100,sigma > p_100,p_100,Stats_100]
                    index_df=index_df+1

                    for x in range(len(fitness_list_100)):
                        for y in range(x+1,len(fitness_list_100)):
                            wilc_stats, silc_p=stats.wilcoxon(fitness_list_100[x],fitness_list_100[y])
                            df_wilcoxon.loc[index_df_wilcoxon]=[all_conditions,100,fitness_list_100_condition[x],fitness_list_100_condition[y],sigma > silc_p,silc_p, wilc_stats]
                            index_df_wilcoxon=index_df_wilcoxon+1

        with open(file_location_to_save + "/friedman_{}.csv".format(problem), 'a') as f:
                df_friedman.to_csv(f, header=True)

        with open(file_location_to_save + "/Wilc_{}.csv".format(problem), 'a') as f:
                df_wilcoxon.to_csv(f, header=True)



NonParams_test()
# normality_for_one_file()
# normality_for_two_file()
# epsilon_test()