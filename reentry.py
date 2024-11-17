# Load in 
satcat = pd.read_csv('satcat.csv')
reentry = pd.read_csv('reentry.csv')

# Remove unnecessary columns
df = satcat.drop(columns=["NORAD_CAT_ID", "OWNER", "LAUNCH_DATE", "LAUNCH_SITE", "DECAY_DATE", "RCS", "DATA_STATUS_CODE"])

# Remove non-debris data
deb = df[df['OBJECT_TYPE'].isin(["DEB", "R/B"])].copy()

# For all rows with "ids" that can also be found in the re-entry df, set reentry as 1 and set everything else to 0. This will be a new column
# merge dfs
result = pd.merge(deb, reentry, left_on="OBJECT_ID", right_on="International Designator")
# result = result[['']]
# print(result["OBJECT_ID"].describe())
# create new df
deb["REENTRY"] = deb["OBJECT_ID"].isin(result["OBJECT_ID"]).astype(int)
# print(deb["REENTRY"].describe())
# add column for re-entry

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Use type, period, inclination, apogee, perigee to classify debris as 0 or 1
deb = deb[["PERIOD", "INCLINATION", "APOGEE", "PERIGEE", "REENTRY"]].dropna()
X = deb[["PERIOD", "INCLINATION", "APOGEE", "PERIGEE"]]
y = deb[["REENTRY"]]
y = y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)