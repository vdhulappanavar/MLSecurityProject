import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
con = pd.read_csv('data.csv')
con.head()
# print(con)
sns.scatterplot(x="fly_ash", y="concrete_compressive_strength", data=con)
plt.savefig('save_as_a_png.png')