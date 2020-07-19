# survival-prediction-for-titanic

### abstract 
Used data of titanic ship of the night it sank, data like information about passenger's name, age, sex, etc (source: Kaggle) to predict whether they survived the tradegy or not. used Random forest algorithm to achive the accuracy of 76% in this prediction.

### methodology
firstly i dropped the columns *Ticket* and *Cabin* as they were of no use in this analysis. after which i extracted the titles of passengers from there name and made a separate column named *Title*. then i replaced the titles (those with less people sharing them) by 'Rare' and mapped them accordingly. now as i had titles so i no longer needed the *Name* and *PassengerId* column so i dropped them too.
then i converted categorical valuees to integer values for *sex*. then i created *AgeBand* to reduce the feature size and also *FamilySize* and dropped *SibSp*(number of siblings or spouse on board) and *Parch*(number of parents and childrens on board). 
