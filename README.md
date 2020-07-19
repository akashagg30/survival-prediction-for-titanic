# survival-prediction-for-titanic

### abstract 
Used data of titanic ship of the night it sank, data like information about passenger's name, age, sex, etc (source: Kaggle) to predict whether they survived the tradegy or not. used Random forest algorithm to achive the accuracy of 76% in this prediction.

### methodology
firstly i dropped the columns *Ticket* and *Cabin* as they were of no use in this analysis. after which i extracted the titles of passengers from there name and made a separate column named *Title*. then i replaced the titles (those with less people sharing them) by 'Rare' and mapped them accordingly. now as i had titles so i no longer needed the *Name* and *PassengerId* column so i dropped them too.
then i converted categorical values to numerical values for *sex*. then i created *AgeBand* to reduce the feature size. i also created *FamilySize* and dropped *SibSp*(number of siblings or spouse on board) and *Parch*(number of parents and childrens on board) and then used *FamilySize* to create *IsAlone* feature.
now to fill the null values in *Embarked* i calculated mode of it and replaced all null values with the value with most frequencies and changed the values of *embarked* from categorical to  numerical values.
now all that left was to feed this data to ML algorithms. so i choose some commonly used algorithms for categorization such as Logistic Regression, KNN, SVM, Random Forest, etc and trained them on this data. then i compared the results and Random Forest was seen to be best working on this data, so i choose it as my final algorithm.

### result
i was able to achive 76% accuracy on test dataset using Random Forest algorithm.
