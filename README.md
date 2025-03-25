# Hitters Dataset Model
This project applies machine learning models to predict player salaries based on their performance during the 1986-1987 baseball season. The main objective is to analyze the performance data, preprocess the data, engineer relevant features, and build predictive models to forecast the players' salaries. The project demonstrates various data science techniques, including missing data handling, feature engineering, model building, and hyperparameter optimization.

## Dataset
The dataset contains various performance metrics for baseball players from the 1986-1987 season. The key features include:

AtBat: Number of at-bats in 1986-1987.
Hits: Number of hits in 1986-1987.
HmRun: Number of home runs in 1986-1987.
Runs: Number of runs contributed by the player.
RBI: Number of runs driven in by the player.
Walks: Number of walks (free passes) given to the player.
Years: Number of years the player has played in the major league.
Salary: Player’s salary for the 1986-1987 season.

## Requirements
Python 3.x

## The following libraries:
numpy
pandas
matplotlib
seaborn
lightgbm
xgboost
catboost
sklearn
missingno

To install the required libraries, use:
 - pip install numpy pandas matplotlib seaborn lightgbm xgboost catboost scikit-learn missingno

 - Data Loading and Exploration
 - Missing Data Handling
 - Feature Engineering
 - Model Building (LightGBM)
 - Model Tuning (Hyperparameter Optimization)
 - Feature Importance Visualization
 - Model Evaluation (RMSE)

## Model Results
The LightGBM model successfully reduced prediction error, achieving an RMSE of approximately 238%.

# Hitters Dataset Modeli
Bu proje, 1986-1987 beyzbol sezonundaki oyuncu performanslarına dayalı olarak oyuncuların maaşlarını tahmin etmek için makine öğrenimi modelleri kullanmaktadır. Ana hedef, oyuncu performans verilerini analiz etmek, veriyi ön işlemek, ilgili özellikleri mühendislik etmek ve maaş tahminini yapmak için çeşitli makine öğrenimi modelleri oluşturmak. Proje, eksik veri temizleme, özellik mühendisliği, model oluşturma ve hiperparametre optimizasyonu gibi veri bilimi tekniklerini göstermektedir.

## Veri Kümesi
Veri kümesi, 1986-1987 sezonuna ait beyzbol oyuncularının çeşitli performans verilerini içermektedir. Anahtar özellikler şunlardır:

AtBat: 1986-1987 sezonunda atılan top sayısı.
Hits: 1986-1987 sezonunda yapılan isabetli vuruş sayısı.
HmRun: 1986-1987 sezonunda yapılan ev sahibi vuruş sayısı.
Runs: Oyuncunun kazandırdığı sayı sayısı.
RBI: Oyuncunun yaptığı vuruşlarla koşu sayısını arttırma.
Walks: Oyuncuya yapılan yürüyüş (serbest geçiş) sayısı.
Years: Oyuncunun büyük ligde oynama süresi (yıl).
Salary: Oyuncunun 1986-1987 sezonundaki maaşı.

## Gereksinimler
Python 3.x

## Aşağıdaki kütüphaneler:
numpy
pandas
matplotlib
seaborn
lightgbm
xgboost
catboost
sklearn
missingno

Gerekli kütüphaneleri yüklemek için:
pip install numpy pandas matplotlib seaborn lightgbm xgboost catboost scikit-learn missingno

## Adımlar
 - Veri Yükleme ve İnceleme
 - Eksik Verilerin Temizlenmesi
 - Özellik Mühendisliği
 - Model Oluşturma (LightGBM)
 - Model Ayarlama (Hiperparametre Optimizasyonu)
 - Özellik Önemliliği Görselleştirmesi
 - Model Değerlendirmesi (RMSE)

## Model Sonuçları
LightGBM modeli başarılı bir şekilde tahmin hatasını %238 seviyelerine indirdi ve RMSE değeri elde edilmiştir.
