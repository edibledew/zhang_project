import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_exercise03.csv')

descriptor = np.array(df.drop(columns=['class']))  
target = np.array(df['class'])  

#X_train, X_test, y_train, y_test = train_test_split(descriptor, target, test_size=0.2, random_state=0)

model = RandomForestClassifier(random_state=0, n_estimators=1000)

model.fit(descriptor, target)

combinations_df = pd.read_csv('exercise03_comb_encoded.csv')

combinations_data = np.array(combinations_df)

predictions = model.predict(combinations_data)

combinations_df['Predicted_class'] = predictions

combinations_df.to_csv('predicted_exercise03.csv', index=False)
print('saved as predicted_exercise03.csv')
