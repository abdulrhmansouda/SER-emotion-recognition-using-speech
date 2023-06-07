# # # from FeaturesManagement import main
# # # import os
# # import pandas as pd
# # import os
# # import sys
# # sys.path.insert(0, os.getcwd())
# # import parameters as para


# # a = pd.get_dummies(para.emotions)
# # p = pd.DataFrame([])
# # p = p.append(a.angry)
# # p = p.append(a.fear)
# # print(p)
# # exit()
# # a = a.append(pd.get_dummies(para.emotions).fear)
# # print(a)
# # # print(pd.merge([a,pd.get_dummies(para.emotions).fear]))
# # # print(os.getcwd())
# # # exit()



# # # print(main.hello())


# import pandas as pd
# import numpy as np

# # create a pandas DataFrame with some NaN and Inf values
# df = pd.DataFrame({'A': [1, 2, np.nan, 4, np.inf],
#                    'B': [5, 6, np.inf, 8, 9]})

# # check which values are infinite
# mask = df.applymap(np.isinf)

# # check which rows contain infinite values
# any_inf = mask.any(axis=1)

# # get the number of rows containing infinite values
# # num_rows = any_inf.sum()
# num_rows = any_inf.sum()

# print('Number of rows containing infinite values:', num_rows)