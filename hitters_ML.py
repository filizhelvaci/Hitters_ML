import numpy as np
import pandas as pd
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from helpers.data_prep import *
import seaborn as sns
from matplotlib import pyplot as plt
from helpers.eda import *
import seaborn as sns

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)
"""
AtBat : 1986 1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
Hits : 1986 1987 sezonundaki isabet sayısı
HmRun : 1986 1987 sezonundaki en değerli vuruş sayısı
Runs : 1986 1987 sezonunda takımına kazandırdığı sayı
RBI : Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
Walks : Karşı oyuncuya yaptırılan hata sayısı
Years : Oyuncunun major liginde oynama süresi (sene)
CAtBat : Oyuncunun kariyeri boyunca topa vurma sayısı
CHits : Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
CHmRun : Oyucunun kariyeri boyunca yaptığı en değerli sayısı
CRuns : Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
CRBI : Oyuncunun kariyeri boyunca koşu yaptırdır dığı oyuncu sayısı
CWalks : Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
League : Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
Division : 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
PutOuts : Oyun icinde takım arkadaşınla yardımlaşma(Topun ahavada tutulup oyuncunun oyun dışı bırakılması)
Assits : 1986 1987 sezonunda oyuncunun yaptığı asist sayısı(Topun oyuncuya değdirilerek oyuncunun oyundışı kalması)
Errors : 1986 1987 sezonundaki oyuncunun hata sayısı
Salary : Oyuncunun 1986 1987 sezonunda aldığı maaş(bin uzerinden)
NewLeague : 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

df=data = pd.read_csv("dataset/hitters.csv")
df.head()
check_df(df)
df.describe().T

# Eksik değerleri uçurduk
df=df.dropna()

# Değişkeler arasındaki ilişkileri gözlemliyoruz
sns.scatterplot(x="Salary",y="AtBat",hue="Hits",data=df)
plt.show()
sns.scatterplot(x="Salary",y="Years",hue="Hits",data=df)
plt.show()
sns.scatterplot(x="Salary",y="Years",hue="HmRun",data=df)
plt.show()
sns.scatterplot(x="Salary",y="Years",hue="Hits",data=df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="BuPu")
plt.show()

# Feature Engineer
df.loc[(df["AtBat"] > 18) & (df["AtBat"] < 250), "NEW_ATBAT"] = "AT_BAT_4"
df.loc[(df["AtBat"] >= 250) & (df["AtBat"] < 400), "NEW_ATBAT"] = "AT_BAT_3"
df.loc[(df["AtBat"] >= 400) & (df["AtBat"] < 525), "NEW_ATBAT"] = "AT_BAT_2"
df.loc[(df["AtBat"] >= 525), "NEW_ATBAT"] = "AT_BAT_1"

df.loc[(df["Hits"] > 0) & (df["Hits"] < 70), "NEW_HITS"] = "HITS_4"
df.loc[(df["Hits"] >= 70) & (df["Hits"] < 100), "NEW_HITS"] = "HITS_3"
df.loc[(df["Hits"] >= 100) & (df["Hits"] < 150), "NEW_HITS"] = "HITS_2"
df.loc[(df["Hits"] >= 150), "NEW_HITS"] = "HITS_1"

df.loc[(df["HmRun"] >= 0) & (df["HmRun"] < 6), "NEW_HM_RUN"] = "HM_RUN_4"
df.loc[(df["HmRun"] >= 6) & (df["HmRun"] < 10), "NEW_HM_RUN"] = "HM_RUN_3"
df.loc[(df["HmRun"] >= 10) & (df["HmRun"] < 20), "NEW_HM_RUN"] = "HM_RUN_2"
df.loc[(df["HmRun"] >= 20), "NEW_HM_RUN"] = "HM_RUN_1"

df.loc[(df["Runs"] >= 0) & (df["Runs"] < 35), "NEW_RUNS"] = "RUNS_4"
df.loc[(df["Runs"] >= 35) & (df["Runs"] < 50), "NEW_RUNS"] = "RUNS_3"
df.loc[(df["Runs"] >= 50) & (df["Runs"] < 75), "NEW_RUNS"] = "RUNS_2"
df.loc[(df["Runs"] >= 75), "NEW_RUNS"] = "RUNS_1"

df.loc[(df["RBI"] >= 0) & (df["RBI"] < 35), "NEW_RBI"] = "RBI_4"
df.loc[(df["RBI"] >= 35) & (df["RBI"] < 50), "NEW_RBI"] = "RBI_3"
df.loc[(df["RBI"] >= 50) & (df["RBI"] < 75), "NEW_RBI"] = "RBI_2"
df.loc[(df["RBI"] >= 75), "NEW_RBI"] = "RBI_1"

df.loc[(df["Walks"] >= 0) & (df["Walks"] < 35), "NEW_WALKS"] = "WALKS_4"
df.loc[(df["Walks"] >= 35) & (df["Walks"] < 50), "NEW_WALKS"] = "WALKS_3"
df.loc[(df["Walks"] >= 50) & (df["Walks"] < 75), "NEW_WALKS"] = "WALKS_2"
df.loc[(df["Walks"] >= 75), "NEW_WALKS"] = "WALKS_1"

df.loc[(df["Years"] > 0) & (df["Years"] < 6), "NEW_YEARS_CAT"] = "YOUNG_PLAYER"
df.loc[(df["Years"] >= 6) & (df["Years"] < 15), "NEW_YEARS_CAT"] = "METURE_PLAYER"
df.loc[(df["Years"] >= 15), "NEW_YEARS_CAT"] = "SENIOR_PLAYER"

df.loc[(df["Walks"] > 0) & (df["Walks"] < 35), "NEW_WALKS"] = "WALKS_4"
df.loc[(df["Walks"] >= 35) & (df["Walks"] < 50), "NEW_WALKS"] = "WALKS_3"
df.loc[(df["Walks"] >= 50) & (df["Walks"] < 75), "NEW_WALKS"] = "WALKS_2"
df.loc[(df["Walks"] >= 75), "NEW_WALKS"] = "WALKS_1"

df.loc[((df["AtBat"]>350) | (df["Hits"]>95) | (df["HmRun"]>10)), "O_Player"]=1
df.loc[~((df["AtBat"]>350) | (df["Hits"]>95) | (df["HmRun"]>10)), "O_Player"]=0

df.loc[((df["PutOuts"]>210) | (df["Assists"]>40) ), "D_Player"]=1
df.loc[~((df["PutOuts"]>210) | (df["Assists"]>40) ),"D_Player"]=0

# Feature Engineer
#df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"  # Savunma oyuncusu mu?
#df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"  # Hucum oyuncusu mu?
# İyi bir koşucu mu?
# İyi bir atıcı mı?
# TRANSFER OLDUMU???
df["MEAN_ATBAT"] = df["CAtBat"] / df["Years"]
df["MEAN_HITS"] = df["CHits"] / df["Years"]
df["MEAN_HMRUN"] = df["HmRun"] / df["Years"]
df["MEAN_RUNS"] = df["Runs"] / df["Years"]


df.loc[(df["PutOuts"] >= 0) & (df["PutOuts"] < 150), "NEW_PUT_OUTS"] = "PUT_OUTS_4"
df.loc[(df["PutOuts"] >= 130) & (df["PutOuts"] < 230), "NEW_PUT_OUTS"] = "PUT_OUTS_3"
df.loc[(df["PutOuts"] >= 230) & (df["PutOuts"] < 330), "NEW_PUT_OUTS"] = "PUT_OUTS_2"
df.loc[(df["PutOuts"] >= 330), "NEW_PUT_OUTS"] = "PUT_OUTS_1"

df.loc[(df["Assists"] >= 0) & (df["Assists"] < 10), "NEW_ASSİSTS"] = "ASSİSTS_4"
df.loc[(df["Assists"] >= 10) & (df["Assists"] < 50), "NEW_ASSİSTS"] = "ASSİSTS_3"
df.loc[(df["Assists"] >= 50) & (df["Assists"] < 200), "NEW_ASSİSTS"] = "ASSİSTS_2"
df.loc[(df["Assists"] >= 200), "NEW_ASSİSTS"] = "ASSİSTS_1"

df.loc[(df["NEW_RBI"] == "RBI_1") & (df["NEW_RUNS"] == "RUNS_1") & (df["NEW_WALKS"] == "WALKS_1"), "NEW_STRATEGIC_P"] = "S_PLAYER_1"
df.loc[(df["NEW_RBI"] == "RBI_1") & (df["NEW_RUNS"] == "RUNS_1") & (df["NEW_WALKS"] == "WALKS_2"), "NEW_STRATEGIC_P"] = "S_PLAYER_2"
df.loc[(df["NEW_RBI"] == "RBI_1") & (df["NEW_RUNS"] == "RUNS_2") & (df["NEW_WALKS"] == "WALKS_1"), "NEW_STRATEGIC_P"] = "S_PLAYER_2"
df.loc[(df["NEW_RBI"] == "RBI_2") & (df["NEW_RUNS"] == "RUNS_1") & (df["NEW_WALKS"] == "WALKS_1"), "NEW_STRATEGIC_P"] = "S_PLAYER_2"
df.loc[(df["NEW_RBI"] == "RBI_1") & (df["NEW_RUNS"] == "RUNS_2") & (df["NEW_WALKS"] == "WALKS_2"), "NEW_STRATEGIC_P"] = "S_PLAYER_3"
df.loc[(df["NEW_RBI"] == "RBI_2") & (df["NEW_RUNS"] == "RUNS_1") & (df["NEW_WALKS"] == "WALKS_2"), "NEW_STRATEGIC_P"] = "S_PLAYER_3"
df.loc[(df["NEW_RBI"] == "RBI_2") & (df["NEW_RUNS"] == "RUNS_2") & (df["NEW_WALKS"] == "WALKS_1"), "NEW_STRATEGIC_P"] = "S_PLAYER_3"

df.loc[(df["NEW_PUT_OUTS"] == "PUT_OUTS_1") & (df["NEW_ASSİSTS"] == "ASSİSTS_1"), "NEW_STRATEGIC_P_D"] = "D_TALENTED_PLAYER_1"
df.loc[(df["NEW_PUT_OUTS"] == "PUT_OUTS_2") & (df["NEW_ASSİSTS"] == "ASSİSTS_1"), "NEW_STRATEGIC_P_D"] = "D_TALENTED_PLAYER_2"
df.loc[(df["NEW_PUT_OUTS"] == "PUT_OUTS_1") & (df["NEW_ASSİSTS"] == "ASSİSTS_2"), "NEW_STRATEGIC_P_D"] = "D_TALENTED_PLAYER_2"

df.loc[(df["NEW_HITS"] == "HITS_1") & (df["NEW_HM_RUN"] == "HM_RUN_1"), "NEW_STRATEGIC_P_O"] = "O_TALENTED_PLAYER_1"
df.loc[(df["NEW_HITS"] == "HITS_2") & (df["NEW_HM_RUN"] == "HM_RUN_1"), "NEW_STRATEGIC_P_O"] = "O_TALENTED_PLAYER_2"
df.loc[(df["NEW_HITS"] == "HITS_1") & (df["NEW_HM_RUN"] == "HM_RUN_2"), "NEW_STRATEGIC_P_O"] = "O_TALENTED_PLAYER_2"


    # Label encoder
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']
binary_cols
for col in binary_cols:
   df = label_encoder(df, col)

    # One Hot encoder
ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols)

    # Rare encoder
rare_analyser(df, "Salary", 0.01)
df = rare_encoder(df, 0.01)

df.columns = [col.upper() for col in df.columns]

df.head()
check_df(df)

# LightGBM

y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

#######################################
# LightGBM: Model & Tahmin
#######################################

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # İlkel test hatamız -->250

#######################################
# Model Tuning
#######################################

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.001,0.002],
               "n_estimators": [9500,10000],
               "max_depth": [6,7,8],
               "colsample_bytree": [0.1,0.2]}

lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=10,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_ # {'colsample_bytree': 0.2, 'learning_rate': 0.001, 'max_depth': 7, 'n_estimators': 9500}

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))  # Model hatamız --> 238

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_tuned, X_train)


##########################################################
# GBM : Model & Tahmin
#*********************************************************

gbm_model=GradientBoostingRegressor().fit(X_train,y_train)
y_pred=gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) # İlkel test hatamız --> 215

##########################################################
# Model Tuning
#*********************************************************

gbm_params={"learning_rate":[0.01, 0.02],
            "max_depth":[3,5,10],
            "n_estimators":[800,1000,1500],
            "subsample":[0.6,0.7,0.8]}
gbm_model=GradientBoostingRegressor(random_state=17)
gbm_cv_model=GridSearchCV(gbm_model,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
gbm_cv_model.best_params_ # {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.7} {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1500, 'subsample': 0.6}

#Algoritmaların tadını alamazsınız küçük verilerde istediği performansı sergileyemez.Bu tarz algoritmalarda çok küçük veri kullanılmaz

#########################################################################
# Final Model
#************************************************************************

gbm_tuned=GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train,y_train)
y_pred=gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) # --> 210


#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(gbm_tuned, X_train)

########################################################
# XGBoost : Model & Tahmin
#******************************************************

xgb=XGBRegressor().fit(X_train,y_train)
y_pred= xgb.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) # ilkel test hatamız -->278

#########################################################
# Model Tuning
#*******************************************************

# colsample_bytree= her bir iterasyonda ağaçlardan alınacak gözlem sayısı
xgb_params={"learning_rate":[0.1,0.01],
            "max_depth":[5,8],
            "n_estimators":[100,1000],
            "colsample_bytree":[0.7,1]}

xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=1).fit(X_train,y_train)
xgb_cv_model.best_params_

########################################################
# Final Model
#*******************************************************

xgb_tuned=XGBRegressor(**xgb_cv_model.best_params_).fit(X_train,y_train)
y_pred=xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred)) # Model hatamız --> 264

plot_importance(xgb_tuned,X_train)
