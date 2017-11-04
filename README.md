# SYS6018-Kagglecompetition4-4
Team 4-4

## ----------------------Team Members

**Sally Gao:** Open track, data exploration, variable significance, support vector machine (SVM), one-page reflection

**Adrian Mead:** Random forest, optimization of linear model with K-fold CV, one-page reflection

**Caitlin Dreisbach:** Github owner, first pass submission, linear model with K-fold CV, one-page reflection

## ----------------------Approaches

First pass: Imputation with the mode
  Score: -0.006

Linear model: General linear model with K-fold cross-validation
  Score: 0.169
  
Nonlinear model: Random Forest with Out-Of-Bag
  Score: 0.255
  
Open Track: Support Vector Machine (SVM)
  Score: 0.266
  

# Reflection

## Who might care about this problem and why?

All states require auto insurance or proof of the ability to afford damage caused in an at-fault accident (Department of Motor Vehicles, 2017). To help cover costs in the event of an accident, most drivers are forced or opt to purchase automobile insurance which can cost over $815 a year depending on driver characteristics, vehicle, and setting (Rocky Mountain Insurance Information Association, 2017). Unfortunately for most drivers, their insurance money is never recognized and it contributes to increased payments without coverage of normal car maintenance. Consumers are often discouraged from using their insurance power due to the high potential of an increased premium. The opportunity for a company to be able to identify drivers who will file a claim would decrease costs for drivers who won’t use their benefits and likely have higher consumer satisfaction overall.

## Why might this problem be challenging?

Car claims are a tricky issue. You can have the best driver in the world, someone you'd never expect to get into an accident, and have them be hit by someone else. They aren't at fault, but they will file a claim. When you're collecting data as an insurance company, you can collect information on individual drivers easily but not who will be driving next to them each day. So you have to deal with a prediction problem dominated by factors external what you're actually collecting data on. Furthermore, this prediction problem is challenging because of the large number of individuals who won’t submit a claim. Because the frequency of submitted claims is significantly lower (~3% of the data), sharp identification of those who will submit a claim is nigh impossible. As illustrated in the Kaggle competition, having high predictive power is extremely difficult in this setting with the highest score being less than a 0.29 Gini coefficient. 

## What other problems resemble this problem?

Many companies may have interest in this issue including insurance companies in the automobile and home industry. Another timely and politically-charged topic is health insurance. Recent research presented at the IEEE Engineering in Medicine and Biology Society (EMBC) conference attempted to predict the number of hospitalization days based on health insurance claims. The connection between this research and the Kaggle competition is the desire to lower cost for subsets of the consumer base and increase it for those who utilize the services to a higher extent. In an attempt to provide better clinical and administrative decision making, decision trees were used to improve accuracy in a binary classification in “no hospitalization” or “one day in the hospital” (Xie, Schreier, Chang, Neubauer, Redmond & Lovell, 2014). One last field would be credit card fraud, where typically fraud accounts for <<1% of all transactions, so that there's a huge class imbalance. In addition, credit card numbers are also usually stolen by some sort of breach external to the actual cardholder.

## References

Department of Motor Vehicles. How Car Insurance Works. Retrieved November 1, 2017. Accessed at https://www.dmv.org/insurance/how-car-insurance-works.php

Rocky Mountain Insurance Information Association (RMIIA). Cost of Auto Insurance. Retrieved November 1, 2017. Accessed at http://www.rmiia.org/auto/steering_through_your_ auto_policy/Cost_of_Auto_Insurance.asp. 

Xie, Y., Schreier, G., Chang, D., Neubauer, S., Redmond, S. & Lovell N. (2014). Predicting number of hospitalization days based on health insurance claims data using bagged regression trees. Conf Proc IEEE Eng Med Biol Soc.,  2706-9. doi: 10.1109/EMBC.2014.6944181.
